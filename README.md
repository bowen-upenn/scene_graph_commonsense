## This is the official implementation of the paper [Enhancing Scene Graph Generation with Hierarchical Relationships and Commonsense Knowledge](https://arxiv.org/abs/2311.12889) in PyTorch. 

The work is developed from the preliminary work [Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation](https://arxiv.org/abs/2303.06842) accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop ([GLFrontiers](https://glfrontiers.github.io/)) and NeurIPS 2023 [Queer in AI](https://www.queerinai.com/neurips-2023), which shares the same GitHub repository. A presentation slide can be accessed [here](https://docs.google.com/presentation/d/14MvudT8IP5zATr3GYfPXrDRyDENsBl-xV6E17Ffhr7s/edit?usp=sharing). 

This work considers the scene graph generation problem. Different from object detection or segmentation, it represents each image as a graph, where each object instance is a node and each relation between a pair of nodes is a directed edge. Our main contributions are:
1. We observe that a scene graph model can achieve superior performance by leveraging relationship hierarchies. Therefore, we propose a Bayesian classification head to replace flat classification, which jointly predicts relationship super-category probabilities and detailed relationships within each super-category.
2. Dataset annotations in scene graph generation are highly sparse and biased, but we can generate extensive predictions beyond the sparse annotations, with strong zero-shot performance and generalization abilities.
3. We design a commonsense validation pipeline that bakes commonsense knowledge from ***large language models*** into our model during training without the need to access any LLMs at test time, making the algorithm more efficient to deploy in practice.
4. ***Our proposed methods can be used as plug-and-play modules into many other existing works, continuing to push these SOTA works to new SOTA levels of performance.***

## Repository Structure

### This repository consists of two parts.
- :purple_heart: **Plug-and-play to existing SOTA works in [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) framework**:

   Follow the ***[Plug-and-Play README](scenegraph_benchmark/README.md)*** for step-by-step instructions on how to integrate our methods (hierarchical relationships & commonsense validation) into your own work. We hope our methods can support your work in achieving even better performance :raised_hands:
Our implementations on [Neural Motifs](https://arxiv.org/abs/1711.06640), 
[VTransE](https://arxiv.org/abs/1702.08319), 
[VCTree](https://arxiv.org/abs/1812.01880), 
[TDE](https://arxiv.org/pdf/2002.11949.pdf), 
[NICE](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_The_Devil_Is_in_the_Labels_Noisy_Label_Correction_for_CVPR_2022_paper.pdf), 
[IETrans](https://arxiv.org/abs/2203.11654) can be found under our [scenegraph_benchmark/](scenegraph_benchmark/) directory.

- :blue_heart: **A light-weighted standalone framework**: Please continue reading.

<p align="center">
<img src=figures/flow_new.png />
</p>

## TODOs
- [x] 1. Clean up the codes for integrating hierarchical classification as a portable module to other existing SOTA works in [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
- [x] 2. Release pretrained model weights.
- [x] 3. Add unsupervised relation super-category clustering from pretrained GPT2, BERT, and CLIP token embeddings, eliminating the need for humans to classify relation labels.
- [ ] 4. Clean up the codes for efficient single-image inference.
- [ ] 5. Clean up the codes for running experiments on the OpenImage V6 dataset.
- [ ] 6. Refine all function headers and comments for better readability.
- [ ] 7. :rainbow: We are currently working on a zero-shot LVM-for-SGG algorithm for open-world scenarios. We expect to release a preliminary manuscript and codes within the next few months, so please stay tuned for updates!

## Citation
If you believe our work has inspired your research, please kindly cite our work. Thank you!

    @article{jiang2023enhancing,
      title={Enhancing Scene Graph Generation with Hierarchical Relationships and Commonsense Knowledge},
      author={Jiang, Bowen and Zhuang, Zhijun and Taylor, Camillo Jose},
      journal={arXiv preprint arXiv:2311.12889},
      year={2023}
    }

    @inproceedings{
      jiang2023hierarchical,
      title={Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation},
      author={Bowen Jiang and Camillo Taylor},
      booktitle={NeurIPS 2023 Workshop: New Frontiers in Graph Learning},
      year={2023},
      url={https://openreview.net/forum?id=T40bRpEd6P}
    }

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

