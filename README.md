## This repository contains the implementation of the algorithm proposed in the paper "Scene Graph Generation from Hierarchical Relationship Reasoning".
This paper describes a novel approach to deducing relationships between objects in a visual scene. It explicitly exploits an informative hierarchical structure that can be imposed to divide the object and relationship categories into disjoint super-categories. Specifically, our proposed scheme implements a Bayes prediction head to jointly predict the super-category or type of relationship between the two objects, along with the detailed relationship within that super-category. This design reduces the impact of class imbalance problems. We present experimental results on the Visual Genome and OpenImage V6 datasets showing that this factorized approach allows a relatively simple model to achieve competitive performance, especially on predicate classification and zero-shot tasks.

## Dependencies
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


## Dataset
  Please refer to [datasets/README.md](datasets/README.md) or [DATASET_README.md](DATASET_README.md) to download and prepare the dataset.


## Pretrained Models
  We provide the pretrained models in our paper. Please download and put them under the [checkpoints/](checkpoints/) directory.

  - DETR-101 object detection module pretrained on Visual Genome: https://drive.google.com/file/d/1fnTP1VXhFzwPFLYqQAEtjuENg2nQUFJ_/view?usp=sharing
  - DETR-101 object detection module pretrained on OpenImage V6: https://drive.google.com/file/d/1WgssZUXkSU1SKXHRNuBG35iGd5I-00QB/view?usp=sharing
  - Local prediction module trained on Visual Genome for three epoches: https://drive.google.com/file/d/1z9XNCPCZgCIkPFy54oEWsG-saYnLny-J/view?usp=sharing
  - Local prediction module trained on OpenImage V6 for one epoch: https://drive.google.com/file/d/1OxM97iE9hm4OWOIZPc8suW5hxkQSqWfg/view?usp=sharing
  - Optional transformer encoder trained on Visual Genome for two more epoches: https://drive.google.com/file/d/1GK1zV9TODI44rSqk1MRLYODlfGp4rnSA/view?usp=sharing
  - Prediction head trained on Visual Genome for two more epoches if the optional transformer encoder is used: https://drive.google.com/file/d/1MNcaD7UlRpzQ3ad4gL9qpO43yBGurHqP/view?usp=sharing


## Quick Start
  All hyper-parameters are listed in the [config.yaml](config.yaml) file.
  We train our code using four NVIDIA V100 GPUs with 32GB memory: ```export CUDA_VISIBLE_DEVICES=0,1,2,3```.
  Training and evaluation results will be automatically recorded and saved in the [results/](results/) directory.
  
  Please modify ```start_epoch```, ```test_epoch```, and ```continue_train``` based on your own experiment, where ```continue_train``` allows you to stop and resume the training process of the local prediction module.
  
  We currently support training and evaluation on Predicate Classification (PredCLS), Scene Graph Classification (SGCLS), and Scene Graph Detection (SGDET) tasks for Visual Genome, including zero-shot evaluation and the PredCLS with the optional transformer encoder. We also support the PredCLS for OpenImage V6.

  ### To train the model on Visual Genome or OpenImage V6:
    In config.yaml, set
      dataset:
        dataset: 'vg' or 'oiv6'
      training:
        run_mode: 'train'
        train_mode: 'local'
        continue_train: False
        start_epoch: 0

Execute ```python main.py```.

  ### To evaluate the model on Visual Genome for PredCLS, SGCLS, or SGDET:
    In config.yaml, set
      dataset:
        dataset: 'vg'
      training:
        run_mode: 'eval'
        train_mode: 'local'
        eval_mode: 'pc' or 'sgc' or 'sgd'
        test_epoch: 2

Execute ```python main.py```.

### To evaluate the model on OpenImage V6 for PredCLS:
    In config.yaml, set
      dataset:
        dataset: 'oiv6'
      training:
        run_mode: 'eval'
        train_mode: 'local'
        eval_mode: 'pc'
        test_epoch: 0

Execute ```python main.py```.

  ### To train the model with the optional transformer encoder on Visual Genome:
    In config.yaml, set
      dataset:
        dataset: 'vg'
      training:
        run_mode: 'train'
        train_mode: 'global'
        continue_train: True
        start_epoch: 3

Execute ```python main.py```.

  ### To evaluate the model with the optional transformer encoder on PredCLS:
    In config.yaml, set
      dataset:
        dataset: 'vg'
      training:
        run_mode: 'eval'
        train_mode: 'global'
        eval_mode: 'pc'
        test_epoch: 5

Execute ```python main.py```.


## Training results
Please refer our paper for the full experimental results.


![Figure1](figures/flow.png)
Illustration of our scene graph construction scheme. The DETR detection backbone generates image features, instance bounding boxes and labels. The local predictor simply predicts pairwise relationships. Given two object proposals $i$ and $j$, we do two separate passes through our relationship network, one with $i$ as the subject and $j$ as the object, and the other with $j$ as the subject and $i$ as the object. Their feature maps are concatenated as $X_{ij}$ and $X_{ji}$ as two possibilities evaluated individually by the Bayes prediction head. The head predicts the super-category distribution and the conditional probabilities under each super-category. It produces three hypotheses for the relationship, one for each super-category. Each of these hypotheses is scored by computing the product of the edge connectivity score and the maximum entry in each of the three final vectors.
In the results, pink predicates are true positive, blue predicates are reasonably true predictions but not annotated in the dataset, and gray predicates are false positive

![Figure2](figures/plot.png)

Illustrations of scene graphs produced by our model for the PredCLS task. All samples come from the Visual Genome test dataset. Each edge is annotated with 3 possible labels, one for each relationship super-category. The numbers show the system's belief that that super-category estimate is the correct one. We show all edges that contain predicates among the top 5 most confident predictions and other true positive predictions among the top 20. We sketch four types of arrows: (1) solid pink arrow: contains true positive predicates. (2) dotted pink arrow: false negative predicates, relationships that are in the ground truth annotation that are missed. (3) solid blue arrow: represent reasonable true positives that are not annotated in the dataset. (4) solid gray arrow: represent false positive predicates.
