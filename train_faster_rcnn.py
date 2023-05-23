#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import torch
import json
from collections import OrderedDict
import random

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

from dataset_utils import object_class_int2str

SUBSET = 1     # global variable of value 0-1 to select a subset from the dataset for debugging


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = ("vg_train",)
    cfg.DATASETS.TEST = ("vg_test",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.OUTPUT_DIR = "checkpoints/faster_rcnn/"

    cfg.MODEL.DEVICE = "cuda"
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 32  # number of images per batch across all machines, may be adjusted automatically if REFERENCE_WORLD_SIZE is set.
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 4  # the reference number of workers (GPUs) this config is meant to train with.

    # we follow mostly the same setting in [https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch]
    # but migrate it from maskrcnn-benchmark to Detectron2
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
    cfg.MODEL.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    cfg.INPUT.MIN_SIZE_TRAIN = (600,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1000
    cfg.INPUT.MIN_SIZE_TEST = (600,)
    cfg.INPUT.MAX_SIZE_TEST = 1000
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.FORMAT = "BGR"
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True

    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.KEYPOINT_ON = False
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.STRIDE = [[4, 8, 16, 32, 64]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.23232838, 0.63365731, 1.28478321, 3.15089189]]
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 151
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.3]
    cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 151
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
    cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 4096
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.BIAS_LR_FACTOR = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.MAX_ITER = int(50000 * SUBSET)
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.NESTEROV = False
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.STEPS = [30000, 45000]
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 5.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.WARMUP_FACTOR = 0.1
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_BIAS = None

    cfg.EVAL_PERIOD = 20000
    cfg.TEST.AUG.ENABLED = True
    cfg.TEST.AUG.MAX_SIZE = 4000
    cfg.TEST.AUG.FLIP = False
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    default_setup(cfg, args)
    return cfg


def my_train_dataset_function():
    with open('/tmp/datasets/vg/annotations/instances_vg_train_coco.json') as f:
        instances_vg_train_coco = json.load(f)

    # use only 1% of the data for debugging
    num_instances = len(instances_vg_train_coco)
    instances_vg_train_coco = instances_vg_train_coco[:int(SUBSET * num_instances)]

    return instances_vg_train_coco


def my_test_dataset_function():
    with open('/tmp/datasets/vg/annotations/instances_vg_test_coco.json') as f:
        instances_vg_test_coco = json.load(f)

    # use only 1% of the data for debugging
    num_instances = len(instances_vg_test_coco)
    instances_vg_test_coco = instances_vg_test_coco[:int(SUBSET * num_instances)]

    return instances_vg_test_coco


def main(args):
    # register visual genome dataset from coco
    DatasetCatalog.register("vg_train", my_train_dataset_function)
    DatasetCatalog.register("vg_test", my_test_dataset_function)

    object_class_int2str_dict = object_class_int2str()
    all_class_names = [object_class_int2str_dict[key] for key in object_class_int2str_dict]
    MetadataCatalog.get("vg_train").set(thing_classes=all_class_names, evaluator_type="coco")
    MetadataCatalog.get("vg_test").set(thing_classes=all_class_names, evaluator_type="coco")

    # register_coco_instances("vg_train", {}, "/tmp/datasets/vg_coco_annot/train.json", "/tmp/datasets/vg/images")
    # register_coco_instances("vg_test", {}, "/tmp/datasets/vg_coco_annot/test.json", "/tmp/datasets/vg/images")
    # register_coco_instances("vg_val", {}, "/tmp/datasets/vg_coco_annot/val.json", "/tmp/datasets/vg/images")
    # data = DatasetCatalog.get("vg_val")

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
