## This repository contains the implementation of the algorithm proposed in the paper "Scene Graph Generation from Hierarchical Relationship Reasoning".
This paper describes a novel approach to deducing relationships between objects in a scene. It explicitly exploits an informative hierarchical structure that can be imposed to divide the object and relationship categories into disjoint super-categories. Our proposed scheme implements a Bayesian approach to jointly predicts the super-category or type of relationship between the two objects, along with the specific relationship within that super-category. We present results on the Visual Genome dataset showing that this factorized approach offers significant performance benefits.


![Figure1](figures/flow.png)
Illustration of our scene graph construction scheme. 


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
  Please refer to [datasets/README.md](datasets/README.md) or [DATASET_README.md](DATASET_README.md) to download and prepare the dataset.


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


## Training results
Predicate classification (PredCLS)
| R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
| ---- | ---- | ----- | ----- | ----- | ------ |
| 60.5 | 73.2 | 77.9  | 14.8  | 21.5  |  24.9  |

| R@20* | R@50* | R@100* | mR@20* | mR@50* | mR@100* |
| ----- | ----- | ------ | ------ | ------ | ------- |
| 66.9  | 77.5  | 80.3   | 20.5   | 25.8   |  27.1   |

Scene graph detection (SGDET)
| R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
| ---- | ---- | ----- | ----- | ----- | ------ |
| 21.8 | 28.0 | 30.7  |  4.6  |  7.4  |  9.7   |

| R@20* | R@50* | R@100* | mR@20* | mR@50* | mR@100* |
| ----- | ----- | ------ | ------ | ------ | ------- |
| 24.8  | 30.2  |  31.8  |  7.1   |  9.4   |  10.1   |


## Examples of the generated scene graphs
![Figure2](figures/plot.png)

All samples come from the Visual Genome test dataset. Each edge is annotated with 3 possible labels, one for each relationship super-category. The numbers show the systems belief that that super-category estimate is the correct one. We show all edges that contain predicates among the top 5 most confident predictions and other true positive predictions among the top 20. We sketch four types of arrows: (1) solid pink arrow: contains true positive predicates. (2) dotted pink arrow: false negative predicates, relationships that are in the ground truth annotation that are missed. (3) solid blue arrow: represent reasonable true positives that are not annotated in the dataset. (4) solid gray arrow: represent false positive predicates.
