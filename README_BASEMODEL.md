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

Object detection backbones: 
  - DETR-101 object detection module pretrained on Visual Genome: https://drive.google.com/file/d/1fnTP1VXhFzwPFLYqQAEtjuENg2nQUFJ_/view?usp=sharing
  - DETR-101 object detection module pretrained on OpenImage V6: https://drive.google.com/file/d/1WgssZUXkSU1SKXHRNuBG35iGd5I-00QB/view?usp=sharing

All the following models are trained on Visual Genome for three epochs:
  - [Ablation] Flat relation classification model without commonsense validation:
  - [Ablation] Flat relation classification model retrained with commonsense validation: https://drive.google.com/file/d/1nwN8ToqfcRfabtf5PcJLAzd0J-Ky-6s3/view?usp=sharing
  - [Ablation] Hierarchical relation classification model without commonsense validation: https://drive.google.com/file/d/1ilguUyMlAf4_q-nNUpXOf3cEUQyA5k5d/view?usp=sharing
  - [Final] Hierarchical relation classification model retrained with commonsense validation: https://drive.google.com/file/d/1gBRD4VOU530WXhbyf4XFy_cYGhwLPnXS/view?usp=sharing

## Quick Start
  All hyper-parameters are listed in the [config.yaml](config.yaml) file.
  We train our code using four NVIDIA V100 GPUs with 32GB memory: ```export CUDA_VISIBLE_DEVICES=0,1,2,3```.
  Training and evaluation results will be automatically recorded and saved in the [results/](results/) directory.
  
  Please modify ```dataset```, ```num_epoch```, ```start_epoch```, and ```test_epoch``` in [config.py](config.py) based on your own experiment. We currently support training and evaluation on Predicate Classification (PredCLS), Scene Graph Classification (SGCLS), and Scene Graph Detection (SGDET) tasks for Visual Genome.
  
  We allow command-line argparser for the following arguments: ```--run_mode```: ``train, eval, prepare_cs, train_cs, eval_cs``. ```--eval_mode```: ``pc, sgc, sgd``.
  ```--hierar```: set hierarchical_pred to True to apply our Bayesian head in the relationship classification model. Its default value is False, which is an ablation study of using a flat classification head instead.
  ```--cluster```: ``motif, gpt2, bert, clip``, where the default value ``motif`` uses manually defined relationship hierarchy in both our paper and [Neural Motifs](https://arxiv.org/abs/1711.06640) and ``clip`` achieves comparable performance if you prefer unsupervised relation clustering without manual effort.

  ### To set the dataset:
  set ```dataset: 'vg'``` in [config.yaml](config.yaml) to run experiments on the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) dataset.


  ### To train the baseline model on Visual Genome:

    python main.py --run_mode train --eval_mode pc --hierar

  ### To evaluate the baseline model on predicate classification:

    python main.py --run_mode eval --eval_mode pc --hierar

  ### To evaluate the baseline model on scene graph classification:

    python main.py --run_mode eval --eval_mode sgc --hierar

  ### To evaluate the baseline model on scene graph detection:

    python main.py --run_mode eval --eval_mode sgd --hierar

  ### To run the commonsense validation pipeline:
  So far, we have trained a baseline relationship classification model as an ablation study. 
  The training process with the commonsense validation pipeline involves two steps: 
  1. ```prepare_cs```, which collects commonsense-aligned and violated sets from the large language model, OpenAI GPT3.5-turbo-instruct. Please add a ```openai_key.txt``` file to your top directory, follow [OpenAI instructions](https://platform.openai.com/docs/quickstart?context=python) to set up your OpenAI API, 
and copy and paste your [API key](https://platform.openai.com/api-keys) into your txt file.
You can skip this step by leveraging the provided [triplets/commonsense_aligned_triplets.pt](triplets/commonsense_aligned_triplets.pt) and [triplets/commonsense_violated_triplets.pt](triplets/commonsense_violated_triplets.pt). 
These two sets are collected based on our baseline relationship classification model trained on Visual Genome for three epochs. 
We strongly suggest running the ```prepare_cs``` step yourself if you are using a different model.
  2. ```train_cs```, which re-trains the relationship classification model. 

    python main.py --run_mode prepare_cs --eval_mode pc --hierar
    python main.py --run_mode train_cs --eval_mode pc --hierar

  ### To evaluate the final model after the commonsense validation:
  Similar to the baseline model, you can evaluate the final model on predicate classification, scene graph classification, or scene graph detection by running one of the following commands.

     python main.py --run_mode eval_cs --eval_mode pc --hierar
     python main.py --run_mode eval_cs --eval_mode sgc --hierar
     python main.py --run_mode eval_cs --eval_mode sgd --hierar

  ### Ablation: To train and evaluate the baseline model without the Bayesian classification head, but a flat one:

    python main.py --run_mode train --eval_mode pc
    python main.py --run_mode eval --eval_mode pc

## Visual results
<p align="center">
<img src=figures/plot_new.png />
</p>
<p>
    <em>Illustration of generated scene graphs on predicate classification. All examples are from the testing dataset of Visual Genome. The first row displays images and objects, while the second row displays the final scene graphs. The third row shows an ablation without commonsense validation, where there exist many unreasonable predictions with high confidence. For each image, we display the top 10 most confident predictions, and each edge is annotated with its relation label and relation super-category. It is possible for an edge to have multiple predicted relationships, but they must come from disjoint super-categories. Blue edges represent incorrect edges based on our observations. Pink edges are true positives in the dataset. Interestingly, all black edges are reasonable predictions we believe but not annotated, which should not be regarded as false positives.</em>
</p>

<p align="center">
<img src=figures/response.png />
</p>
<p>
    <em>Example responses with reasoning from the large language model in the commonsense validation process. (a) shows a successful case. (b) shows an unexpected response but the reasoning provided by the model shows that it sometimes considers irrelevant and misleading contexts. (c) and (d) shows how asking for the reasoning behind the answers could help us develop better prompt engineering.</em>
</p>

