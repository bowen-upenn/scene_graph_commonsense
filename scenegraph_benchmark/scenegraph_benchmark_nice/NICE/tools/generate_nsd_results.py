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


def generate_neg_nsd_results(cfg, local_rank, distributed, logger, groups, groups_t, new_data_path):
    debug_print(logger, 'prepare training')
    model = build_detection_model(cfg)
    debug_print(logger, 'end model construction')
    groups_dict = dict(zip(groups, groups_t))
    head_rel_ids = cfg.HEAD_IDS
    body_rel_ids = cfg.BODY_IDS
    tail_rel_ids = cfg.TAIL_IDS
    filter_rel_ids_dicts = dict(zip(['head', 'body', 'tail'], [head_rel_ids, body_rel_ids, tail_rel_ids]))
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

    model.eval()
    name = 'motif_predcls_nsd_pred_filter_nonoverlap_reweight'
    if not os.path.exists(new_data_path + '/' + 'results_dict_{}.npy'.format(name)):
        results_dict = {}
        for _, batch in enumerate(tqdm(data_loaders_val[0])):
            with torch.no_grad():
                images, targets, image_ids = batch

                tgt_rel_matrix = targets[0].get_field("relation")  # [tgt, tgt]
                tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
                tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
                tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
                tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

                if len(tgt_pair_idxs) <= 0:
                    print('no relation', image_ids[0])
                    continue
                if len(tgt_pair_idxs[0]) <= 0:
                    print('no relation', image_ids[0])
                    continue

                targets = [target.to(device) for target in targets]
                outputs = model(images.to(device), targets)
                if outputs == None:
                    continue
                # print(targets[0].get_field("labels"))
                outputs[0].add_field('target_rels', tgt_rel_labs.cpu())
                outputs[0].add_field('target_rels_pair_idxs', tgt_pair_idxs.cpu())
                results_dict.update({image_id: output for image_id, output in zip(image_ids, outputs)})

        np.save(new_data_path + '/' + 'results_dict_{}.npy'.format(name), results_dict)
    results_dict_load = np.load(new_data_path + '/' + 'results_dict_{}.npy'.format(name), allow_pickle=True)

    for group in groups_dict.keys():
        group_t = groups_dict[group]
        bg_rel_pairs_ids_generated = []
        bg_ids = []
        bg_rel_labels = []
        bg_rel_pairs = []
        filter_rel_ids = filter_rel_ids_dicts[group]
        print(group, group_t, filter_rel_ids)


        for image_id in results_dict_load.item().keys():
            sorted_pred_label = results_dict_load.item()[image_id].get_field('pred_rel_labels')
            sorted_rel_pair_ids = results_dict_load.item()[image_id].get_field('rel_pair_idxs')
            ood_scores = results_dict_load.item()[image_id].get_field('ood_score')
            ood_scores = ood_scores.view(-1)
            no_rel_pair_ids = sorted_rel_pair_ids[ood_scores >= group_t]
            no_rel_label = sorted_pred_label[ood_scores >= group_t] + 1
            filter_ids = []
            for i in range(no_rel_label.shape[0]):
                if no_rel_label[i] in filter_rel_ids:
                    filter_ids.append(i)

            no_rel_pair_ids = no_rel_pair_ids[filter_ids]
            no_rel_label = no_rel_label[filter_ids]
            bg_rel_len = no_rel_label.shape[0]
            objs_labels = (results_dict_load.item()[image_id].get_field('labels'))
            bg_ids.extend([str(image_id) + '_' + str(x) for x in range(bg_rel_len)])
            bg_rel_labels.extend(no_rel_label)

            for bg_rel_pair_id in no_rel_pair_ids:
                bg_rel_pairs.append(str(objs_labels[bg_rel_pair_id[0]].item() + 1) + '_' + str(objs_labels[bg_rel_pair_id[1]].item() + 1))
                bg_rel_pairs_ids_generated.append(str(bg_rel_pair_id[0].item()) + '_' + str(bg_rel_pair_id[1].item()))

        name = 'motif_predcls_nsd_filter_nonoverlap_reweight_{}_{}'.format(group, group_t)

        with open(new_data_path + '/' + 'bg_ids_{}.txt'.format(name), 'w') as f:
            for rel_id in bg_ids:
                f.write(rel_id)
                f.write('\n')

        with open(new_data_path + '/' + 'bg_rel_pairs_{}.txt'.format(name), 'w') as f:
            for rel_pair in bg_rel_pairs:
                f.write(rel_pair)
                f.write('\n')

        with open(new_data_path + '/' + 'bg_rel_pairs_ids_{}.txt'.format(name), 'w') as f:
            for rel_pair in bg_rel_pairs_ids_generated:
                f.write(rel_pair)
                f.write('\n')

        with open(new_data_path + '/' + 'bg_labels_{}.txt'.format(name), 'w') as f:
            for rel_label in bg_rel_labels:
                f.write(str(rel_label.item()))
                f.write('\n')

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False

def generate(groups, groups_t, new_data_path):
    groups_dict = dict(zip(groups, groups_t))
    original_name = 'motif_predcls_with_bg_256'

    original_rel_ids_filename = new_data_path + '/' + 'has_rel_ids_{}.txt'.format(original_name)
    original_rel_pairs_ids_filename = new_data_path + '/' + 'has_rel_pairs_ids_{}.txt'.format(original_name)
    original_rel_labels_filename = new_data_path + '/' + 'has_rel_labels_{}.txt'.format(original_name)

    with open(original_rel_ids_filename) as f:
        original_rel_ids = [str(x).strip() for x in f]
        f.close()

    with open(original_rel_pairs_ids_filename) as f:
        original_rel_pairs_ids = [str(x).strip() for x in f]
        f.close()

    with open(original_rel_labels_filename) as f:
        original_rel_labels = [str(x).strip() for x in f]
        f.close()

    if 'head' in groups_dict.keys():
        bg_head_name = 'motif_predcls_nsd_filter_nonoverlap_reweight_head_{}'.format(groups_dict['head'])
        bg_head_rel_ids_filename = new_data_path + '/' + 'bg_ids_{}.txt'.format(bg_head_name)
        bg_head_rel_pairs_ids_filename = new_data_path + '/' + 'bg_rel_pairs_ids_{}.txt'.format(bg_head_name)
        bg_head_rel_labels_filename = new_data_path + '/' + 'bg_labels_{}.txt'.format(bg_head_name)

        with open(bg_head_rel_ids_filename) as f:
            bg_head_rel_ids = [str(x).strip() for x in f]
            f.close()

        with open(bg_head_rel_pairs_ids_filename) as f:
            bg_head_rel_pairs_ids = [str(x).strip() for x in f]
            f.close()

        with open(bg_head_rel_labels_filename) as f:
            bg_head_rel_labels = [str(x).strip() for x in f]

    if 'body' in groups_dict.keys():
        bg_body_name = 'motif_predcls_nsd_filter_nonoverlap_reweight_body_{}'.format(groups_dict['body'])
        bg_body_rel_ids_filename = new_data_path + '/' + 'bg_ids_{}.txt'.format(bg_body_name)
        bg_body_rel_pairs_ids_filename = new_data_path + '/' + 'bg_rel_pairs_ids_{}.txt'.format(bg_body_name)
        bg_body_rel_labels_filename = new_data_path + '/' + 'bg_labels_{}.txt'.format(bg_body_name)

        with open(bg_body_rel_ids_filename) as f:
            bg_body_rel_ids = [str(x).strip() for x in f]
            f.close()

        with open(bg_body_rel_pairs_ids_filename) as f:
            bg_body_rel_pairs_ids = [str(x).strip() for x in f]
            f.close()

        with open(bg_body_rel_labels_filename) as f:
            bg_body_rel_labels = [str(x).strip() for x in f]

    if 'tail' in groups_dict.keys():
        bg_tail_name = 'motif_predcls_nsd_filter_nonoverlap_reweight_tail_{}'.format(groups_dict['tail'])
        bg_tail_rel_ids_filename = new_data_path + '/' + 'bg_ids_{}.txt'.format(bg_tail_name)
        bg_tail_rel_pairs_ids_filename = new_data_path + '/' + 'bg_rel_pairs_ids_{}.txt'.format(bg_tail_name)
        bg_tail_rel_labels_filename = new_data_path + '/' + 'bg_labels_{}.txt'.format(bg_tail_name)

        with open(bg_tail_rel_ids_filename) as f:
            bg_tail_rel_ids = [str(x).strip() for x in f]
            f.close()

        with open(bg_tail_rel_pairs_ids_filename) as f:
            bg_tail_rel_pairs_ids = [str(x).strip() for x in f]
            f.close()

        with open(bg_tail_rel_labels_filename) as f:
            bg_tail_rel_labels = [str(x).strip() for x in f]

    generate_rel_labels_dict = {}
    generate_rel_pairs_ids_dict = {}


    for i in range(len(original_rel_ids)):
        image_rel_id = original_rel_ids[i]
        image_id, rel_id = image_rel_id.split('_')
        if image_id not in generate_rel_labels_dict.keys():
            generate_rel_labels_dict[image_id] = []
            generate_rel_pairs_ids_dict[image_id] = []
        generate_rel_labels_dict[image_id].append(original_rel_labels[i])
        generate_rel_pairs_ids_dict[image_id].append(original_rel_pairs_ids[i])

    if 'head' in groups_dict.keys():
        for i in range(len(bg_head_rel_ids)):
            image_rel_id = bg_head_rel_ids[i]
            image_id, rel_id = image_rel_id.split('_')
            if image_id not in generate_rel_labels_dict.keys():
                generate_rel_labels_dict[image_id] = []
                generate_rel_pairs_ids_dict[image_id] = []
            generate_rel_labels_dict[image_id].append(bg_head_rel_labels[i])
            generate_rel_pairs_ids_dict[image_id].append(bg_head_rel_pairs_ids[i])
        print('head bg num', len(bg_head_rel_ids))

    if 'body' in groups_dict.keys():
        for i in range(len(bg_body_rel_ids)):
            image_rel_id = bg_body_rel_ids[i]
            image_id, rel_id = image_rel_id.split('_')
            if image_id not in generate_rel_labels_dict.keys():
                generate_rel_labels_dict[image_id] = []
                generate_rel_pairs_ids_dict[image_id] = []
            generate_rel_labels_dict[image_id].append(bg_body_rel_labels[i])
            generate_rel_pairs_ids_dict[image_id].append(bg_body_rel_pairs_ids[i])
        print('body bg num', len(bg_body_rel_ids))

    if 'tail' in groups_dict.keys():
        for i in range(len(bg_tail_rel_ids)):
            image_rel_id = bg_tail_rel_ids[i]
            image_id, rel_id = image_rel_id.split('_')
            if image_id not in generate_rel_labels_dict.keys():
                generate_rel_labels_dict[image_id] = []
                generate_rel_pairs_ids_dict[image_id] = []
            generate_rel_labels_dict[image_id].append(bg_tail_rel_labels[i])
            generate_rel_pairs_ids_dict[image_id].append(bg_tail_rel_pairs_ids[i])
        print('tail bg num', len(bg_tail_rel_ids))


    new_name = 'motif_predcls_nsd_filter_nonoverlap_reweight'
    if 'head' in groups_dict.keys():
        new_name += '_head_{}'.format(groups_dict['head'])
    if 'body' in groups_dict.keys():
        new_name += '_body_{}'.format(groups_dict['body'])
    if 'tail' in groups_dict.keys():
        new_name += '_tail_{}'.format(groups_dict['tail'])
    np.save(new_data_path + '/' + 'generate_rel_labels_{}.npy'.format(new_name), generate_rel_labels_dict)
    np.save(new_data_path + '/' + 'generate_rel_pairs_ids_{}.npy'.format(new_name), generate_rel_pairs_ids_dict)


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
    groups = cfg.GROUPS.split(',')
    groups_t = cfg.GROUPS_T
    # print(groups, groups_t)
    assert len(groups) == len(groups_t)
    generate_neg_nsd_results(cfg, args.local_rank, args.distributed, logger, groups, groups_t, cfg.NEW_DATA_PATH)
    generate(groups, groups_t, cfg.NEW_DATA_PATH)

if __name__ == "__main__":
    main()
