## This is the official implementation of the paper [Enhancing Scene Graph Generation with Hierarchical Relationships and Commonsense Knowledge](https://arxiv.org/abs/2311.12889) in PyTorch. 

A presentation slide can be accessed [here](https://docs.google.com/presentation/d/14MvudT8IP5zATr3GYfPXrDRyDENsBl-xV6E17Ffhr7s/edit?usp=sharing). The work is based on the preliminary work [Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation](https://arxiv.org/abs/2303.06842) accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop ([GLFrontiers](https://glfrontiers.github.io/)) and NeurIPS 2023 [Queer in AI](https://www.queerinai.com/neurips-2023).

Abstract: This work presents an enhanced approach to generating scene graphs by incorporating a relationship hierarchy and commonsense knowledge. Specifically, we propose a Bayesian classification head that exploits an informative hierarchical structure. It jointly predicts the super-category or type of relationship between the two objects, along with the detailed relationship under each super-category. We design a commonsense validation pipeline that uses a large language model to critique the results from the scene graph prediction system and then use that feedback to enhance the model performance. The system requires no external large language model assistance at test time, making it more convenient for practical applications. Experiments on the Visual Genome and the OpenImage V6 datasets demonstrate that harnessing hierarchical relationships enhances the model performance by a large margin. The proposed Bayesian head can also be incorporated as a portable module in existing scene graph generation algorithms to improve their results. In addition, the commonsense validation enables the model to generate an extensive set of reasonable predictions beyond dataset annotations.

To summarize our contributions:
1. We observe that a scene graph model can achieve superior performance by leveraging relationship hierarchies, and therefore, propose a Bayesian classification head to replace flat classification.
   to jointly predicts relationship super-category probabilities and detailed relationships within each super-category.
2. Dataset annotations like those in Visual Genome are sparse, but we can generate extensive predictions beyond the sparse annotations, with strong zero-shot performance and generalization abilities.
3. We design a commonsense validation pipeline that bakes commonsense knowledge from large language models into our model during training, while eliminating the necessity to access large language models at testing time.
4. We show that these techniques can also be used to enhance the performance of other existing scene graph generation algorithms.

## TODOs
1. Clean up the codes for efficient single-image inference.
2. Clean up the codes for integrating the Bayesian classification head into other scene graph generation algorithms, starting from a common [code framework](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
3. Clean up the codes for running experiments on the OpenImage V6 dataset.

## Dependencies
Please check [requirements.txt](requirements.txt). You can run the following commands to create a virtual environment and install all the requirements:
    
    python -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt

## Dataset
  Please refer to [datasets/README.md](datasets/README.md) or [DATASET_README.md](DATASET_README.md) to download and prepare the dataset.
  
  Like what we did in our [config.yaml](config.yaml), you can also add a soft link to your own datasets/ folder which stores large-size images and annotations
  by going to the tmp folder of your operating system```cd ~/tmp``` and then
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
  
  Please modify ```dataset```, ```num_epoch```, ```start_epoch```, and ```test_epoch``` in [config.py](config.py) based on your own experiment. We currently support training and evaluation on Predicate Classification (PredCLS), Scene Graph Classification (SGCLS), and Scene Graph Detection (SGDET) tasks for Visual Genome. We also support the PredCLS for OpenImage V6.
  
  We allow command-line argparser for the following arguments: ```run_mode```: train, eval, prepare_cs, train_cs, eval_cs. ```eval_mode```: pc, sgc, sgd. ```continue_train```: set continue_train to True, which allows you to stop and resume the training process.
  ```start_epoch```: specify the start epoch if continue_train is True. ```hierar```: set hierarchical_pred to True to apply our Bayesian head in the relationship classification model. Its default value is False, which is an ablation study of using a flat classification head instead.

  ### To set the dataset:
  set ```dataset: 'vg'``` in [config.yaml](config.yaml) to run experiments on the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) dataset (recommended),  
  or set ```dataset: 'oiv6'``` in [config.yaml](config.yaml) to choose the [OpenImage V6](https://storage.googleapis.com/openimages/web/download.html) dataset.

  ### To train the model on Visual Genome:

    python main.py --run_mode train --eval_mode pc --hierar

  ### To evaluate the model on predicate classification:

    python main.py --run_mode eval --eval_mode pc --hierar

  ### To evaluate the model on scene graph classification:

    python main.py --run_mode eval --eval_mode sgc --hierar

  ### To evaluate the model on scene graph classification:

    python main.py --run_mode eval --eval_mode sgd --hierar

  ### To run the commonsense validation pipeline:
  So far, we have trained a baseline relationship classification model as an ablation study. 
  The commonsense validation pipeline will involve three steps: 
  1. prepare_cs, which collects commonsense-aligned and violated sets from the large language model, OpenAI GPT3.5-turbo-instruct. Please add a ```openai_key.txt``` file to your top directory, follow [OpenAI instructions](https://platform.openai.com/docs/quickstart?context=python) to set up your OpenAI API, 
and copy and paste your [API key](https://platform.openai.com/api-keys) into your txt file.
  2. train_cs, which re-trains the relationship classification model. 
  3. eval_cs, which evaluates the re-trained relationship classification model, and you have the options to set --eval_mode as pc, sgc, or sgd.

    python main.py --run_mode prepare_cs --eval_mode pc --hierar
    python main.py --run_mode train_cs --eval_mode pc --hierar
    python main.py --run_mode eval_cs --eval_mode pc --hierar

  ### Ablation: To train and evaluate the model on Visual Genome without the Bayesian classification head, but a flat one:

    python main.py --run_mode train --eval_mode pc
    python main.py --run_mode eval --eval_mode pc




