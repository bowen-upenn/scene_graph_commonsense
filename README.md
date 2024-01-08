## This is the official implementation of the paper [Enhancing Scene Graph Generation with Hierarchical Relationships and Commonsense Knowledge](https://arxiv.org/abs/2311.12889) in PyTorch. 

A presentation slide can be accessed [here](https://docs.google.com/presentation/d/14MvudT8IP5zATr3GYfPXrDRyDENsBl-xV6E17Ffhr7s/edit?usp=sharing). The work is based on the preliminary work [Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation](https://arxiv.org/abs/2303.06842) accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop ([GLFrontiers](https://glfrontiers.github.io/)) and NeurIPS 2023 [Queer in AI](https://www.queerinai.com/neurips-2023).

Abstract: This work presents an enhanced approach to generating scene graphs by incorporating a relationship hierarchy and commonsense knowledge. Specifically, we propose a Bayesian classification head that exploits an informative hierarchical structure. It jointly predicts the super-category or type of relationship between the two objects, along with the detailed relationship under each super-category. We design a commonsense validation pipeline that uses a large language model to critique the results from the scene graph prediction system and then use that feedback to enhance the model performance. The system requires no external large language model assistance at test time, making it more convenient for practical applications. Experiments on the Visual Genome and the OpenImage V6 datasets demonstrate that harnessing hierarchical relationships enhances the model performance by a large margin. The proposed Bayesian head can also be incorporated as a portable module in existing scene graph generation algorithms to improve their results. In addition, the commonsense validation enables the model to generate an extensive set of reasonable predictions beyond dataset annotations.

To summarize our contributions:
1. We observe that a scene graph model can achieve superior performance by leveraging relationship hierarchies and, therefore, propose a Bayesian classification head to replace flat classification.
   to jointly predict relationship super-category probabilities and detailed relationships within each super-category.
2. Dataset annotations like those in Visual Genome are sparse, but we can generate extensive predictions beyond the sparse annotations, with strong zero-shot performance and generalization abilities.
3. We design a commonsense validation pipeline that bakes commonsense knowledge from large language models into our model during training. This eliminates the necessity to access any large language models at testing time, making the algorithm more efficient for practical use.
4. We show that these techniques can also be integrated into other existing scene graph generation algorithms as a portable module, further enhancing their state-of-the-art performance.

<p align="center">
<img src=figures/flow_new.png />
</p>

## TODOs
1. Release all pretrained model weights.
2. Clean up the codes for efficient single-image inference.
3. Clean up the codes for running experiments on the OpenImage V6 dataset.
4. Refine all function headers and comments for better readability.

## Citation
Please cite our works if you believe they have inspired your research. Thank you!

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
  - [Ablation] Flat relation classification model retrained with commonsense validation:
  - [Ablation] Hierarchical relation classification model without commonsense validation: https://drive.google.com/file/d/1ilguUyMlAf4_q-nNUpXOf3cEUQyA5k5d/view?usp=sharing
  - [Final] Hierarchical relation classification model retrained with commonsense validation: https://drive.google.com/file/d/1gBRD4VOU530WXhbyf4XFy_cYGhwLPnXS/view?usp=sharing

## Quick Start
  All hyper-parameters are listed in the [config.yaml](config.yaml) file.
  We train our code using four NVIDIA V100 GPUs with 32GB memory: ```export CUDA_VISIBLE_DEVICES=0,1,2,3```.
  Training and evaluation results will be automatically recorded and saved in the [results/](results/) directory.
  
  Please modify ```dataset```, ```num_epoch```, ```start_epoch```, and ```test_epoch``` in [config.py](config.py) based on your own experiment. We currently support training and evaluation on Predicate Classification (PredCLS), Scene Graph Classification (SGCLS), and Scene Graph Detection (SGDET) tasks for Visual Genome. We also support the PredCLS for OpenImage V6.
  
  We allow command-line argparser for the following arguments: ```run_mode```: train, eval, prepare_cs, train_cs, eval_cs. ```eval_mode```: pc, sgc, sgd. ```continue_train```: set continue_train to True, which allows you to stop and resume the training process.
  ```start_epoch```: specify the start epoch if continue_train is True. ```hierar```: set hierarchical_pred to True to apply our Bayesian head in the relationship classification model. Its default value is False, which is an ablation study of using a flat classification head instead.

  ### To set the dataset:
  set ```dataset: 'vg'``` in [config.yaml](config.yaml) to run experiments on the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) dataset (recommended),  
  or set ```dataset: 'oiv6'``` in [config.yaml](config.yaml) to choose the [OpenImage V6](https://storage.googleapis.com/openimages/web/download.html) dataset (limited supports).

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

## Extension to existing works as a portable module

Our Bayesian classification head can also serve as a portable module to furthermore improve the performance of other existing SOTA works. 
Implementations based on the popular code framework [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) can be found under the [scenegraph_benchmark](scenegraph_benchmark/) folder.

<p align="center">
<img src=figures/extension.png />
</p>
<p style="font-size: x-small;"> 
   <em>This image is altered from its original version by Tang, Kaihua, Hanwang Zhang, Baoyuan Wu, Wenhan Luo, and Wei Liu. "Learning to compose dynamic tree structures for visual contexts." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6619-6628. 2019.</p>

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

