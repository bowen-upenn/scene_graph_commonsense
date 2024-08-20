## This is the official implementation of the paper [Enhancing Scene Graph Generation with Hierarchical Relationships and Commonsense Knowledge](https://arxiv.org/abs/2311.12889) in PyTorch. 

[![Ranking on PaperWithCode](https://img.shields.io/badge/PaperWithCode-PredCLS_Visual_Genome_Ranking_1-5DD9DB)](https://paperswithcode.com/sota/predicate-classification-on-visual-genome)
[![Ranking on PaperWithCode](https://img.shields.io/badge/PaperWithCode-SGCLS_Visual_Genome_Ranking_1-5DD9DB)](https://paperswithcode.com/sota/scene-graph-classification-on-visual-genome)
[![Arxiv](https://img.shields.io/badge/ArXiv-Full_Paper-B31B1B)](https://arxiv.org/abs/2311.12889)

The work is developed from the preliminary work [Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation](https://arxiv.org/abs/2303.06842) accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop ([GLFrontiers](https://glfrontiers.github.io/)) and NeurIPS 2023 [Queer in AI](https://www.queerinai.com/neurips-2023), which shares the same GitHub repository.

This work considers the scene graph generation problem. Different from object detection or segmentation, it represents each image as a graph, where each object instance is a node and each relation between a pair of nodes is a directed edge. Our highlights are:
1. We leverage the commonsense knowledge from small-scale, open-source, and on-device **LLMs/VLMs**, filtering out **commonsense-violated** predictions made by baseline scene graph generation models.
3. **Hierarchical relationships work surprisingly well:** Replacing the flat relation classification head with a Bayesian head, which jointly predicts relationship super-category probabilities and detailed relationships within each super-category, can improve model performance by a large margin.
4. Both of our proposed methods are **model-agnostic:** They can be easily used as **plug-and-play** modules into **existing SOTA** works, continuing pushing their SOTA performance to **new levels**.

We dub our methods as **HIERCOM**, an acronym for **HIE**rarchical **R**elationships and **COM**monsense Validation.

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
- [x] 1. Clean up the codes for integrating hierarchical classification and commonsense validation as portable modules to other existing SOTA works in [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
- [x] 2. Release pretrained model weights.
- [x] 3. Add unsupervised relation super-category clustering from pretrained GPT2, BERT, and CLIP token embeddings, eliminating the need for humans to classify relation labels.
- [ ] 4. Clean up the codes for efficient single-image inference.
- [ ] 5. Clean up the codes for running experiments on the OpenImage V6 dataset.
- [ ] 6. Refine all function headers and comments for better readability.
- [x] 7. :rainbow: We are currently working on a zero-shot Visual Question Answering in open-world scenarios. VQA is a commonly explored downstream task in the scene graph generation community, and we expect to release a preliminary manuscript and codes within the next few months, so please stay tuned for updates! **[Update] It is now released and available at [Multi-Agent VQA](https://github.com/bowen-upenn/Multi-Agent-VQA)**.
- [ ] 8. :rainbow: Integrate hierarchical classification and commonsense validation to an existing scene graph algorithm on **3D point-cloud datasets for robotic applications**.

## Citation
If you believe our work has inspired your research, please kindly cite our work. Thank you!

    @article{jiang2023enhancing,
     title={Enhancing Scene Graph Generation with Hierarchical Relationships and Commonsense Knowledge},
     author={Jiang, Bowen and Zhuang, Zhijun and Taylor, Camillo Jose},
     journal={arXiv preprint arXiv:2311.12889},
     year={2023}
    }

    @article{jiang2023scene,
     title={Scene graph generation from hierarchical relationship reasoning},
     author={Jiang, Bowen and Taylor, Camillo J},
     journal={arXiv preprint arXiv:2303.06842},
     year={2023}
    }

There is a subtle problem on Google Scholar if the authors change their paper title, so the second BibTeX actually refers to the preliminary work [Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation](https://arxiv.org/abs/2303.06842) accepted at NeurIPS 2023 Workshop.

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

