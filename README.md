# This repository contains the implementation of the algorithm proposed in the paper "Scene Graph Generation from Hierarchical Relationship Reasoning".

## Dependencies:
  - python >= 3.6.9
  - torch >= 1.10.1+cu102 
  - torchvision >= 0.11.2+cu102
  - numpy
  - json
  - yaml
  - os
  - PIL
  - string
  - tqdm
  - collections
  - math
  - copy
  - typing
  - transformers


## Dataset:
  Please refer to [datasets/DATASET_README.md](datasets/DATASET_README.md) to download and prepare the dataset.


## Pretrained Models
  We provide the pretrained models in our paper. Please download and put them under the [checkpoints/](checkpoints/) directory.

  - DETR-101 object detection module: https://drive.google.com/file/d/1fnTP1VXhFzwPFLYqQAEtjuENg2nQUFJ_/view?usp=sharing
  - Local prediction modules on the third epoch: https://drive.google.com/file/d/1z9XNCPCZgCIkPFy54oEWsG-saYnLny-J/view?usp=sharing
  - Optional transformer encoder on the fifth epoch: https://drive.google.com/file/d/1GK1zV9TODI44rSqk1MRLYODlfGp4rnSA/view?usp=sharing
  - Prediction head on the fifth epoch if the optional transformer encoder is used: https://drive.google.com/file/d/1MNcaD7UlRpzQ3ad4gL9qpO43yBGurHqP/view?usp=sharing


## Quick Start:
  All hyper-parameters are listed in the [config.yaml](config.yaml) file.
  We train our code using four NVIDIA V100 GPUs with 32GB memory: ```export CUDA_VISIBLE_DEVICES=0,1,2,3```.
  Training and evaluation results will be automatically recorded and saved in the [results/](results/) directory.
  Please modify ```start_epoch```, ```test_epoch```, and ```continue_train``` based on your own experiment, where ```continue_train``` allows you to stop and resume the training process of the local prediction module.

  ### To train the local prediction module:
    In config.yaml, set
      training:
        run_mode: 'train'
        train_mode: 'local'
        continue_train: False
        start_epoch: 0

Execute ```python main.py```.

  ### To evaluate the local prediction module on predicate classification (PredCLS) tasks:
    In config.yaml, set
      training:
        run_mode: 'eval'
        train_mode: 'local'
        eval_mode: 'pc'
        test_epoch: 2

Execute ```python main.py```.

  ### To evaluate the local prediction module on scene graph detection (SGDET) tasks:
    In config.yaml, set
      training:
        run_mode: 'eval'
        train_mode: 'local'
        eval_mode: 'sgd'
        test_epoch: 2

    Execute ```python main.py```.

  ### To train the model with the optional transformer encoder:
    In config.yaml, set
      training:
        run_mode: 'train'
        train_mode: 'global'
        continue_train: True
        start_epoch: 3

Execute ```python main.py```.

  ### To evaluate the model with the optional transformer encoder on predicate classification (PredCLS) tasks:
    In config.yaml, set
      training:
        run_mode: 'eval'
        train_mode: 'global'
        eval_mode: 'pc'
        test_epoch: 5

Execute ```python main.py```.

  ### To evaluate the local prediction module on scene graph detection (SGDET) tasks:
    In config.yaml, set
      training:
        run_mode: 'eval'
        train_mode: 'global'
        eval_mode: 'sgd'
        test_epoch: 5

Execute ```python main.py```.
