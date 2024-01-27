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


def generate_has_rels_results(cfg, local_rank, distributed, logger, new_data_path):
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

    model.eval()
    name = 'motif_predcls_has_rels'
 for _, batch in enumerate(tqdm(data_loaders_val[0])):
            with torch.no_grad():
                images, targets, image_ids = batch
                for i in range(len(image_ids)):
                    tgt_rel_matrix = targets[i].get_field("relation")  # [tgt, tgt]
                    tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
                    tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
                    tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
                    tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)
                    if len(tgt_pair_idxs) <= 0:
                        continue
                    if len(targets[i]) <= 0:
                        continue
                    needed_key = {}
                    needed_key.update({'target_rels': tgt_rel_labs.cpu()})
                    needed_key.update({'target_rels_pair_idxs': tgt_pair_idxs.cpu()})
                    needed_key.update({'labels': targets[i].get_field("labels").cpu()})
                    results_dict.update({image_ids[i]:needed_key})

        torch.save(results_dict, new_data_path + '/' + 'results_dict_{}.pth'.format(name))

    results_dict_load = dict(torch.load(new_data_path + '/' + 'results_dict_{}.pth'.format(name)))

    rel_ids = []
    rel_labels = []
    rel_pairs = []
    rel_pair_idxs_dict = {}
    rel_pairs_ids_generated = []


    for image_id in results_dict_load.keys():
        target_label = results_dict_load[image_id]['target_rels']
        rel_pair_ids = results_dict_load[image_id]['target_rels_pair_idxs']
        objs_labels = results_dict_load[image_id]['labels']

        rel_ids.extend([str(image_id) + '_' + str(x) for x in range(results_dict_load[image_id]['target_rels_pair_idxs'].shape[0])])
        rel_labels.extend(target_label)
        rel_pair_idxs_dict[image_id] = rel_pair_ids
        for rel_pair_id in rel_pair_ids:
            rel_pairs.append(str(objs_labels[rel_pair_id[0]].item()) + '_' + str(objs_labels[rel_pair_id[1]].item()))
            rel_pairs_ids_generated.append(str(rel_pair_id[0].item()) + '_' + str(rel_pair_id[1].item()))

    name = 'motif_predcls_with_bg_256'
    with open(new_data_path + '/' + 'has_rel_ids_{}.txt'.format(name), 'w') as f:
        for rel_id in rel_ids:
            f.write(rel_id)
            f.write('\n')

    with open(new_data_path + '/' + 'has_rel_pairs_{}.txt'.format(name), 'w') as f:
        for rel_pair in rel_pairs:
            f.write(rel_pair)
            f.write('\n')

    with open(new_data_path + '/' + 'has_rel_pairs_ids_{}.txt'.format(name), 'w') as f:
        for rel_pair in rel_pairs_ids_generated:
            f.write(rel_pair)
            f.write('\n')

    with open(new_data_path + '/' + 'has_rel_labels_{}.txt'.format(name), 'w') as f:
        print(len(rel_pairs))
        for rel_label in rel_labels:
            f.write(str(rel_label.item()))
            f.write('\n')
        f.close()

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False

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
    generate_has_rels_results(cfg, args.local_rank, args.distributed, logger, cfg.NEW_DATA_PATH)

if __name__ == "__main__":
    main()
