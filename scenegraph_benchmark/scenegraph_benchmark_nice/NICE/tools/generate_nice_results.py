# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime
import numpy as np
import copy
import torch
from torch.nn.utils import clip_grad_norm_
import os
import shutil
import tempfile
import urllib
from pos_nsd_clustering import PosNSDClustering, LDCOF
import torch
import pandas as pd
from scipy.spatial import distance_matrix

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from tqdm import tqdm

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def extract(cfg, local_rank, distributed, logger, new_data_path):
    debug_print(logger, 'prepare training')
    model = build_detection_model(cfg)
    debug_print(logger, 'end model construction')

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)

    fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor", ]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"}

    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping[
            "roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT,
                                                  with_optim=False,
                                                  update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
    debug_print(logger, 'end load checkpointer')
    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(cfg, mode='extract', is_distributed=distributed)
    dataset = data_loaders_val[0].dataset
    print('dataset len', len(dataset))

    # print(len(data_loaders_val))
    model.eval()
    results_dict = {}

    name = 'motif_predcls_nsd_filter_nonoverlap_reweight_tail_0.6'

    if not os.path.exists(new_data_path + '/' + 'results_dict_{}.npy'.format(name)):
        for _, batch in enumerate(tqdm(data_loaders_val[0])):
            with torch.no_grad():
                images, targets, image_ids = batch
                tgt_rel_matrix = targets[0].get_field("relation")  # [tgt, tgt]
                tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
                tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
                tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
                tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)
                if len(tgt_pair_idxs) == 0:
                    print('no relation', image_ids[0])
                    continue
                targets = [target.to(device) for target in targets]
                outputs = model(images.to(device), targets)
                outputs[0].add_field('target_rels', tgt_rel_labs.cpu())
                outputs[0].add_field('target_rels_pair_idxs', tgt_pair_idxs.cpu())
                results_dict.update({image_id: output for image_id, output in zip(image_ids, outputs)})

        np.save(new_data_path + '/' + 'results_dict_{}.npy'.format(name), results_dict)

    results_dict_load = np.load(new_data_path + '/' + 'results_dict_{}.npy'.format(name), allow_pickle=True)

    rel_ids = []
    rel_reps = []
    rel_labels = []
    rel_pairs = []
    rel_pair_idxs_dict = {}
    rel_pairs_ids_generated = []

    for image_id in results_dict_load.item().keys():
        target_label = results_dict_load.item()[image_id].get_field('target_rels')
        rel_rep = results_dict_load.item()[image_id].get_field('pred_rel_rep')
        rel_pair_ids = results_dict_load.item()[image_id].get_field('target_rels_pair_idxs')
        objs_labels = (results_dict_load.item()[image_id].get_field('labels'))

        rel_ids.extend([str(image_id) + '_' + str(x) for x in range(results_dict_load.item()[image_id].get_field('pred_rel_rep').shape[0])])
        rel_reps.extend(rel_rep)

        rel_labels.extend(target_label)
        rel_pair_idxs_dict[image_id] = rel_pair_ids
        for rel_pair_id in rel_pair_ids:
            rel_pairs.append(str(objs_labels[rel_pair_id[0]].item()) + '_' + str(objs_labels[rel_pair_id[1]].item()))
            rel_pairs_ids_generated.append(str(rel_pair_id[0].item()) + '_' + str(rel_pair_id[1].item()))

    rel_reps = torch.stack(rel_reps, dim=0)

    torch.save(rel_reps, new_data_path + '/' + 'rel_reps_{}.pth'.format(name))
    with open(new_data_path + '/' + 'rel_ids_{}.txt'.format(name), 'w') as f:
        for rel_id in rel_ids:
            f.write(rel_id)
            f.write('\n')

    with open(new_data_path + '/' + 'rel_pairs_{}.txt'.format(name), 'w') as f:
        for rel_pair in rel_pairs:
            f.write(rel_pair)
            f.write('\n')

    with open(new_data_path + '/' + 'rel_pairs_ids_{}.txt'.format(name), 'w') as f:
        for rel_pair in rel_pairs_ids_generated:
            f.write(rel_pair)
            f.write('\n')

    with open(new_data_path + '/' + 'rel_labels_{}.txt'.format(name), 'w') as f:
        print(len(rel_pairs))
        for rel_label in rel_labels:
            f.write(str(rel_label.item()))
            f.write('\n')
        f.close()

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False


def test_pos_nsd_cluster(cfg, model_name, method_name, subset_n, new_data_path, head_density_t=0.125, body_density_t=0.25, tail_density_t=0.5):
    '''
    Load and generate Pos-NSD clustering results based on local density
    Args:
        model_name: The name of the model used to extract features
        method_name: Clustering methods
        subset_n: Number of subsets

    Returns: None

    '''
    X, y, metadata = load_data(model_name, new_data_path)
    cc = PosNSDClustering(cfg=cfg, n_subsets=subset_n, head_density_t=0.125, body_density_t=0.25, tail_density_t=0.5, verbose=True, random_state=0, dim_reduce=4096)
    cc.fit(X, y)

    test_dir = new_data_path + '/' + 'pos_nsd_clustering_results_{}_{}'.format(model_name, method_name)
    generate_clusters(labels=cc.output_labels, densities=cc.densities, n_subsets=cc.n_subsets,
                                       metadata=metadata, test_dir=test_dir)


def load_data(model_name, new_data_path):
    '''
    Load the data for clustering
    Args:
        model_name: The name of the model used to extract features

    Returns:
        None
    '''

    # X: feature of each triplet
    features_file = new_data_path + '/' + 'rel_reps_{}.pth'.format(model_name)
    X = np.array(torch.load(features_file, map_location=torch.device('cpu')).cpu())

    # y: relation labels
    cluster_list = new_data_path + '/' + 'rel_labels_{}.txt'.format(model_name)
    with open(cluster_list) as f:
        y = [str(x).strip() for x in f]
        f.close()

    # metadata: ids
    cluster_id_list = new_data_path + '/' + 'rel_ids_{}.txt'.format(model_name)
    with open(cluster_id_list) as f:
        metadata = [str(x) for x in f]

    return X, y, metadata


def generate_clusters(labels, densities, n_subsets, metadata, test_dir):
    '''
    Generate clustering results
    Args:
        labels: Cluster labels
        densities: local densities
        n_subsets: Number of subsets
        metadata: Sample ids
        test_dir: Result saving path

    Returns: None

    '''

    if os.path.isdir(test_dir):
        pass
    else:
        os.makedirs(test_dir)

    clustered_by_levels = [list() for _ in range(n_subsets)]
    clustered_densities = [list() for _ in range(n_subsets)]

    for idx, image_rel_id in enumerate(metadata):
        clustered_by_levels[labels[idx]].append(idx)
        if densities is None:
            pass
        else:
            clustered_densities[labels[idx]].append(densities[idx])

    for idx, level_output in enumerate(clustered_by_levels):
        with open("{}/{}.txt".format(test_dir, idx), 'w') as f:
            for i in level_output:
                f.write("{}".format(str(metadata[i])))

    if densities is None:
        pass
    else:
        for idx, level_output in enumerate(clustered_densities):
            with open("{}/densities_{}.txt".format(test_dir, idx), 'w') as f:
                for i in level_output:
                    f.write("{}\n".format(str(i)))


def generate_rel_subset(model_name, method_name, change_noisy_name, subset_n, new_data_path):
    '''
    Generate data sets for SGG training, rel_clean_set stores clean data (sample ids not in noisiest subset),
    rel_noisy_set stores noisy data (sample ids in nosiest subset), changed_rel_noisy_labels stores relation labels by NSC.
    Args:
        model_name: The name of the model used to extract features
        method_name: Clustering methods
        change_noisy_name: Name of the method used for NSC
        subset_n: Number of subsets

    Returns: None
    '''
    cluster_results_dir = 'pos_nsd_clustering_results_{}_{}'.format(model_name, method_name)
    changes_noist_label_path = new_data_path + '/' + 'changed_noisy_common_pair_{}_{}_{}.txt'.format(model_name, method_name, change_noisy_name)
    weight_path = new_data_path + '/' + 'changed_noisy_weight_{}_{}_{}.txt'.format(model_name, method_name, change_noisy_name)

    with open(changes_noist_label_path) as f:
        changed_rel_labels = [str(x).strip() for x in f]
    with open(weight_path) as f:
        changed_rel_label_weights = [float(str(x).strip()) for x in f]

    cluster_level_dict = {}
    image_clean_dict = {}
    image_noisy_dict = {}
    image_noisy_changed_label_dict = {}
    image_noisy_changed_label_weight_dict = {}

    with open(new_data_path + '/' + '{}/{}.txt'.format(cluster_results_dir, subset_n - 1)) as f:
        noisy_rel_ids = [str(x).strip() for x in f]

    for cluster_subset_filename in os.listdir(new_data_path + '/' + cluster_results_dir):
        if 'den' not in cluster_subset_filename:
            path = os.path.join(new_data_path + '/' + cluster_results_dir, cluster_subset_filename)
            with open(path) as f:
                rel_ids = [str(x).strip() for x in f]
                cluster_level_dict.update(
                    zip(rel_ids, [cluster_subset_filename.split('.')[0] for i in range(len(rel_ids))]))
                f.close()

    for image_rel_id in cluster_level_dict.keys():
        image_id, rel_id = image_rel_id.split('_')
        if cluster_level_dict[image_rel_id] != str(subset_n - 1):
            if image_id not in image_clean_dict.keys():
                image_clean_dict[image_id] = []
            image_clean_dict[image_id].append(rel_id)
        elif cluster_level_dict[image_rel_id] == str(subset_n - 1):
            if image_id not in image_noisy_dict.keys():
                image_noisy_dict[image_id] = []
                image_noisy_changed_label_dict[image_id] = []
                image_noisy_changed_label_weight_dict[image_id] = []
            image_noisy_dict[image_id].append(rel_id)
            changed_rel_label = changed_rel_labels[noisy_rel_ids.index(image_rel_id)]
            image_noisy_changed_label_dict[image_id].append(int(changed_rel_label))

            changed_rel_label_weight = changed_rel_label_weights[noisy_rel_ids.index(image_rel_id)]
            image_noisy_changed_label_weight_dict[image_id].append(changed_rel_label_weight)

    image_label_dict = {}
    image_pair_dict = {}
    image_id_dict = {}
    path = new_data_path + '/' + 'rel_ids_{}.txt'.format(model_name)
    rel_pair_path = new_data_path + '/' + 'rel_pairs_ids_{}.txt'.format(model_name)
    rel_label_path = new_data_path + '/' + 'rel_labels_{}.txt'.format(model_name)

    with open(path) as f:
        rel_ids = [str(x).strip() for x in f]
    with open(rel_pair_path) as f:
        rel_pairs = [str(x).strip() for x in f]
    with open(rel_label_path) as f:
        rel_labels = [str(x).strip() for x in f]

    for i in range(len(rel_ids)):
        image_rel_id = rel_ids[i]
        image_id, rel_id = image_rel_id.split('_')
        if image_id not in image_pair_dict.keys():
            image_id_dict[image_id] = []
            image_pair_dict[image_id] = []
            image_label_dict[image_id] = []
        image_id_dict[image_id].append(rel_id)
        image_pair_dict[image_id].append(rel_pairs[i])
        image_label_dict[image_id].append(rel_labels[i])


    np.save(new_data_path + '/' + 'rel_clean_set_{}_{}.npy'.format(model_name, method_name), image_clean_dict)
    np.save(new_data_path + '/' + 'rel_noisy_set_{}_{}.npy'.format(model_name, method_name), image_noisy_dict)
    np.save(new_data_path + '/' + 'changed_rel_noisy_labels_{}_{}_{}.npy'.format(model_name, method_name, change_noisy_name),
            image_noisy_changed_label_dict)
    np.save(new_data_path + '/' + 'changed_rel_weights_{}_{}_{}.npy'.format(model_name, method_name, change_noisy_name),
            image_noisy_changed_label_weight_dict)
    np.save(new_data_path + '/' + 'rel_ids_{}_{}.npy'.format(model_name, method_name), image_id_dict)
    np.save(new_data_path + '/' + 'rel_pairs_ids_{}_{}.npy'.format(model_name, method_name), image_pair_dict)
    np.save(new_data_path + '/' + 'rel_labels_{}_{}.npy'.format(model_name, method_name), image_label_dict)


def test_ldcof_cluster(cfg, model_name, method_name, new_data_path):
    '''
    Pos-NSD clustering results are generated by LDCOF method
    Args:
        model_name: The name of the model used to extract features
        method_name: Clustering methods

    Returns: None
    '''
    features_file = new_data_path + '/' + 'rel_reps_{}.pth'.format(model_name)
    X = np.array(torch.load(features_file, map_location=torch.device('cpu')).cpu())
    cluster_list = new_data_path + '/' + 'rel_labels_{}.txt'.format(model_name)

    with open(cluster_list) as f:
        rel_labels = [(str(x).strip()).split('_')[-1] for x in f]
        f.close()

    cluster_id_list = new_data_path + '/' + 'rel_ids_{}.txt'.format(model_name)
    test_dir = new_data_path + '/' + 'pos_nsd_clustering_results_{}_{}'.format(model_name, method_name)

    with open(cluster_id_list) as f:
        metadata = [str(x) for x in f]

    ldcof = LDCOF(cfg, alpha=0.6, n_clusters=10)
    cc = ldcof.fit(features=X, rel_labels=rel_labels)

    generate_clusters(labels=cc.output_labels, densities=None, n_subsets=cc.n_subsets,
                                       metadata=metadata, test_dir=test_dir)


def change_noisy_common_pair(model_name, method_name, change_noisy_name, k, subset_n, new_data_path):
    '''
    Generate the relation labels for noisy samples by NSC
    Args:
        model_name: The name of the model used to extract features
        method_name: Clustering methods
        change_noisy_name: Name of the method used for NSC
        k: K in weighted K-Nearest-Neighbors
        subset_n: Number of subsets
        base_prior: Whether the model used to extract features is based on the results of prior model
        prior_model_name:Prior model name

    Returns: None

    '''
    features_file = new_data_path + '/' + 'rel_reps_{}.pth'.format(model_name)
    X = np.array(torch.load(features_file, map_location=torch.device('cpu')).cpu())

    cluster_path = 'pos_nsd_clustering_results_{}_{}'.format(model_name,method_name)
    noisy_path = new_data_path + '/' + '{}/{}.txt'.format(cluster_path, subset_n - 1)
    clean_path = new_data_path + '/' + '{}/0.txt'.format(cluster_path)

    all_labels_path = new_data_path + '/' + 'rel_labels_{}.txt'.format(model_name)
    all_ids_path = new_data_path + '/' + 'rel_ids_{}.txt'.format(model_name)
    pair_path = new_data_path + '/' + 'rel_pairs_{}.txt'.format(model_name)

    result_path = new_data_path + '/' + 'changed_noisy_common_pair_{}_{}_{}.txt'.format(model_name, method_name, change_noisy_name)
    weight_path = new_data_path + '/' + 'changed_noisy_weight_{}_{}_{}.txt'.format(model_name, method_name, change_noisy_name)

    with open(clean_path) as f:
        clean_rel_ids = [str(x).strip() for x in f]
        f.close()

    with open(noisy_path) as f:
        noisy_rel_ids = [str(x).strip() for x in f]
        f.close()

    with open(all_ids_path) as f:
        all_rel_ids = [str(x).strip() for x in f]
        print(len(all_rel_ids))
        f.close()

    with open(all_labels_path) as f:
        all_labels_ids = [int(str(x).strip()) for x in f]
        f.close()

    with open(pair_path) as f:
        pairs = [str(x).strip() for x in f]
        print(len(pairs))
        f.close()

    clean_indexs = []
    noisy_indexs = []
    clean_pair_dict = dict()

    for clean_rel_id in clean_rel_ids:
        clean_index = all_rel_ids.index(clean_rel_id)
        if pairs[clean_index] not in clean_pair_dict.keys():
            clean_pair_dict[pairs[clean_index]] = []
        clean_pair_dict[pairs[clean_index]].append(clean_index)
        clean_indexs.append(clean_index)

    for noisy_rel_id in noisy_rel_ids:
        noisy_indexs.append(all_rel_ids.index(noisy_rel_id))
    new_perception_weight_list = []

    with open(result_path, 'w') as f:
        for i in tqdm(range(len(noisy_indexs))):
            noisy_index = noisy_indexs[i]
            noisy_obj_pair = pairs[noisy_index]
            noisy_feature = X[noisy_index]
            noisy_feature = noisy_feature.reshape(1, -1)
            all_labels_ids = np.array(all_labels_ids)
            if noisy_obj_pair in clean_pair_dict.keys():
                common_pair_clean_indexs = clean_pair_dict[noisy_obj_pair]
                clean_common_pair_features = X[common_pair_clean_indexs]
                distances = distance_matrix(noisy_feature, clean_common_pair_features)
                distances = distances.reshape(-1)

                if len(common_pair_clean_indexs) > k:
                    nearest_indexs = np.argsort(distances)[:k]
                    common_pair_clean_indexs = np.array(common_pair_clean_indexs)
                    nearest_labels = all_labels_ids[common_pair_clean_indexs[nearest_indexs]]
                    sigma = 10
                    nearest_distances = distances[nearest_indexs]
                    nearest_weights = np.exp(-nearest_distances / (2 * sigma ** 2))
                    weight_count = np.bincount(nearest_labels.flatten(), weights=nearest_weights)
                    nearest_label = np.argmax(np.bincount(nearest_labels.flatten(), weights=nearest_weights))
                    if nearest_label != all_labels_ids[noisy_index]:
                        if all_labels_ids[noisy_index] in nearest_labels.flatten():
                            new_perception_weight = weight_count[np.argmax(weight_count)] / (
                                        weight_count[np.argmax(weight_count)] + weight_count[
                                    all_labels_ids[noisy_index]])
                        else:
                            new_perception_weight = weight_count[np.argmax(weight_count)] / (
                                        weight_count[np.argmax(weight_count)] + 1)
                    else:
                        new_perception_weight = 1.0
                else:
                    nearest_index = np.argmin(distances, axis=-1)
                    nearest_label = all_labels_ids[common_pair_clean_indexs[nearest_index]]
                    if nearest_label != all_labels_ids[noisy_index]:
                        new_perception_weight = 0.5
                    else:
                        new_perception_weight = 1.0

            else:
                nearest_label = all_labels_ids[noisy_index]
                new_perception_weight = 1.0
            new_perception_weight_list.append(new_perception_weight)
            print(all_labels_ids[noisy_index], nearest_label, new_perception_weight)
            f.write("{}\n".format(nearest_label))

        with open(weight_path, 'w') as f:
            for weight in new_perception_weight_list:
                f.write("{}\n".format(weight))

def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    # YACS
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    extract(cfg, args.local_rank, args.distributed, logger, cfg.NEW_DATA_PATH)


    densities_t = cfg.DENSITIES_T
    head_density_t, body_density_t, tail_density_t = densities_t[0], densities_t[1], densities_t[2]
    k = cfg.WKNN_K
    subset_n = cfg.SUBSET_N
    model_name = 'motif_predcls_nsd_filter_nonoverlap_reweight_tail_0.6'
    method_name = 'density_thresh_{}_{}_{}_n_{}'.format(head_density_t,body_density_t,tail_density_t,subset_n)
    change_noisy_name = 'weighted_k_{}_nearest_perception'.format(k)

    if 'ldcof' in method_name:
        print('ldcof')
        test_ldcof_cluster(cfg, model_name, method_name, new_data_path=cfg.NEW_DATA_PATH)
    else:
        print('density based')
        test_pos_nsd_cluster(cfg, model_name, method_name, subset_n=subset_n, new_data_path=cfg.NEW_DATA_PATH, head_density_t=head_density_t, body_density_t=body_density_t, tail_density_t=tail_density_t)
    change_noisy_common_pair(model_name, method_name, change_noisy_name, k=k, subset_n=subset_n, new_data_path=cfg.NEW_DATA_PATH)
    generate_rel_subset(model_name, method_name, change_noisy_name, subset_n=subset_n, new_data_path=cfg.NEW_DATA_PATH)

if __name__ == "__main__":
    main()
