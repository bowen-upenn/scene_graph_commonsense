## This is the official implementation of the paper [Scene Graph Generation from Hierarchical Relationship Reasoning](https://arxiv.org/abs/2303.06842) in PyTorch.
This paper presents a novel approach for inferring relationships between objects in visual scenes. It explicitly exploits an informative hierarchical structure that can be imposed to divide the object and relationship categories into disjoint super-categories. Specifically, our proposed method incorporates a Bayes prediction head, enabling joint predictions of the super-category as the type of relationship between the two objects, along with the detailed relationship within that super-category. This design reduces the impact of class imbalance problems. Furthermore, we also modify the supervised contrastive learning to adapt our hierarchical classification scheme. Experimental evaluations on the Visual Genome and OpenImage V6 datasets demonstrate that this factorized approach allows a relatively simple model to achieve competitive performance, particularly in predicate classification and zero-shot tasks.
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
  
  Like what we did in our [config.yaml](config.yaml), you can also add a soft link to your own datasets/ folder which stores large-size images and annotations
  by going to OS tmp folder ```cd ~/tmp``` and then
  ```ln -s /path/to/your/datasets/ .``` Otherwise, please remove the /tmp/ header from all paths in the provided [config.yaml](config.yaml).

## Pretrained Models
  We provide the pretrained models in our paper. Please download and put them under the [checkpoints/](checkpoints/) directory.

  - DETR-101 object detection module pretrained on Visual Genome: https://drive.google.com/file/d/1fnTP1VXhFzwPFLYqQAEtjuENg2nQUFJ_/view?usp=sharing
  - DETR-101 object detection module pretrained on OpenImage V6: https://drive.google.com/file/d/1WgssZUXkSU1SKXHRNuBG35iGd5I-00QB/view?usp=sharing
  - Local prediction module trained on Visual Genome for three epoches: https://drive.google.com/file/d/1U5X3jn3iM8PM1oNcn5-PsCU9AqCCXB9J/view?usp=sharing
  - Local prediction module trained on OpenImage V6 for one epoch: https://drive.google.com/file/d/1OxM97iE9hm4OWOIZPc8suW5hxkQSqWfg/view?usp=sharing

## Quick Start
  All hyper-parameters are listed in the [config.yaml](config.yaml) file.
  We train our code using four NVIDIA V100 GPUs with 32GB memory: ```export CUDA_VISIBLE_DEVICES=0,1,2,3```.
  Training and evaluation results will be automatically recorded and saved in the [results/](results/) directory.
  
  Please modify ```num_epoch```, ```start_epoch```, ```test_epoch```, and ```continue_train``` based on your own experiment, where ```continue_train``` allows you to stop and resume the training process.
  
  We currently support training and evaluation on Predicate Classification (PredCLS), Scene Graph Classification (SGCLS), and Scene Graph Detection (SGDET) tasks for Visual Genome, including zero-shot evaluation and the PredCLS with the optional transformer encoder. We also support the PredCLS for OpenImage V6.

  ### To train the model on Visual Genome:
    In config.yaml, set
      dataset:
        dataset: 'vg'
      training:
        num_epoch: 3
        run_mode: 'train'
        continue_train: False
        start_epoch: 0

Execute ```python main.py```.

  ### To train the model on OpenImage V6:
    In config.yaml, set
      dataset:
        dataset: 'oiv6'
      training:
        num_epoch: 1
        run_mode: 'train'
        continue_train: False
        start_epoch: 0

Execute ```python main.py```.

  ### To evaluate the model on Visual Genome for PredCLS, SGCLS, or SGDET:
    In config.yaml, set
      dataset:
        dataset: 'vg'
      training:
        run_mode: 'eval'
        eval_mode: 'pc' or 'sgc' or 'sgd'
        test_epoch: 2

Execute ```python main.py```.

### To evaluate the model on OpenImage V6 for PredCLS:
    In config.yaml, set
      dataset:
        dataset: 'oiv6'
      training:
        run_mode: 'eval'
        eval_mode: 'pc'
        test_epoch: 0

Execute ```python main.py```.
