# NICE for SGG in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.7.0-%237732a8)

Our paper [The Devil is in the Labels:
Noisy Label Correction for Robust Scene Graph Generation](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_The_Devil_Is_in_the_Labels_Noisy_Label_Correction_for_CVPR_2022_paper.pdf) has been accepted by CVPR 2022.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Training Predictor in NSD

```base
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifConfidencePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR checkpoints/motif-predcls-non-bg-reweight TYPE None ADD_BG False NEW_DATA_PATH new_data SOLVER.SCHEDULE.TYPE WarmupMultiStepLR SOLVER.BASE_LR 0.001
```
## Extract Information from Original Dataset
```base
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --master_port 10037 --nproc_per_node=1 tools/generate_has_rels_results.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/motif-precls-exmp OUTPUT_DIR checkpoints/motif-precls-exmp GROUPS tail GROUPS_T [0.6] ADD_BG True NEW_DATA_PATH new_data 
```

## Completing Missing Annotated Triplets by NSD
```base
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --master_port 10031 --nproc_per_node=1 tools/generate_nsd_results.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifConfidencePredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/motif-predcls-non-bg-reweight OUTPUT_DIR checkpoints/motif-predcls-non-bg-reweight GROUPS tail GROUPS_T [0.6] TYPE complete_bg ADD_BG False  NEW_DATA_PATH new_data
```

## Generating New Dataset by NICE
```base
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --master_port 10032 --nproc_per_node=1 tools/generate_nice_results.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/motif-precls-exmp OUTPUT_DIR checkpoints/motif-precls-exmp GROUPS tail GROUPS_T [0.6] TYPE extract_pos ADD_BG True NEW_DATA_PATH new_data 
```

## Training Models with NICE
```base
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR checkpoints/model_predcls_nice TYPE nice ADD_BG True NEW_DATA_PATH new_data
```

## Comments for Parameters in Command
To make it easier for you to run our code, the Parameters in the command are explained here:

- `--master_port`: It represents the port on which the command is run.
- `CUDA_VISIBLE_DEVICES`: It means the the GPUs that you are going to use. For example, `CUDA_VISIBLE_DEVICES=0,1` use the first two GPUs.
- `--nproc_per_node`: It is the number of GPUs you are going to use.
- `SOLVER.IMS_PER_BATCH`: It is the training batch size.
- `TEST.IMS_PER_BATCH`: It is the testing batch size.
- `SOLVER.MAX_ITER`: It is the maximum iteration.
- `SOLVER.STEPS`: It is the steps where we decay the learning rate
- `SOLVER.VAL_PERIOD`: It is the period of conducting val.
- `SOLVER.CHECKPOINT_PERIOD`: It is the period of saving checkpoint.
- `MODEL.RELATION_ON` It means turning on the relationship head or not (since this is the pretraining phase for Faster R-CNN only, we turn off the relationship head), OUTPUT_DIR is the output directory to save checkpoints.
- `MODEL.ROI_RELATION_HEAD.USE_GT_BOX` and `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL`: They used to select the protocols, (1) **PredCls**: They are all set as `True`. (2) **SGCls**: `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL` is set to `False`, while `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True` is set to `True`. (3) **SGDet**: They are all set to `False`.
- `MODEL.ROI_RELATION_HEAD.PREDICTOR`: It is the backbobe you are going to use., and the MOTIFS SGG backbone (`MotifPredictor`) is used by default. 
- `GROUPS`: It represents the predicate in which group to complete. The options are `head`, `body`, `tail`, or their combinations, separated by commas.  
- `TYPE`: The type of dataset loaded. If it set to 'complete_bg', it represents the loading of triplet samples of '_background_' predicates. If it set to 'extract_pos', it means the loading of the positive triplet samples. If it set to 'nice', it means the loading of the samples generated by NICE.  
- `GROUPS_T`: Thresholds of confidence score of groups in 'GROUPS'.
- `ADD_BG`: It means whether or not the sample loaded contains '_background_'.
- `NEW_DATA_PATH`: The path where NICE generates new datasets.
## Models and Generated Files
For the Motifs Predictor in NSD and Motifs-NICE, we provide the trained models (checkpoint) for verification purpose. Please download from [here*](https://drive.google.com/drive/folders/1hfeqruVM99Bk1q3O_5mkibJSHXbWxDEK?usp=sharing) and unzip to checkpoints. Besides, we provide the files generated in new_data, you can download from [here*](https://drive.google.com/drive/folders/1JDNeUdug2ewLlRID3yepKVgjw5PenE50?usp=sharing).) The NSD checkpoint can be download from [here*](https://drive.google.com/file/d/1X_hXO3O6Z_VJYyiJcuRHvCgZYxB2ZmxN/view?usp=sharing).


## Citations

If you find this project helps your research, please kindly consider citing our project or papers in your publications.

<!-- ```
@misc{tang2020sggcode,
title = {A Scene Graph Generation Codebase in PyTorch},
author = {Tang, Kaihua},
year = {2020},
note = {\url{https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch}},
}
``` -->
## Credits

Our codebase is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
