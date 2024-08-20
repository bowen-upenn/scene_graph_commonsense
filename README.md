## This is the official implementation of the paper [Enhancing Scene Graph Generation with Hierarchical Relationships and Commonsense Knowledge](https://arxiv.org/abs/2311.12889) in PyTorch. 

[![Ranking on PaperWithCode](https://img.shields.io/badge/PaperWithCode-PredCLS_Visual_Genome_Ranking_1-5DD9DB)](https://paperswithcode.com/sota/predicate-classification-on-visual-genome)
[![Ranking on PaperWithCode](https://img.shields.io/badge/PaperWithCode-SGCLS_Visual_Genome_Ranking_1-5DD9DB)](https://paperswithcode.com/sota/scene-graph-classification-on-visual-genome)
[![Arxiv](https://img.shields.io/badge/ArXiv-Full_Paper-B31B1B)](https://arxiv.org/abs/2311.12889)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Cite_Our_Paper-4085F4)](https://scholar.googleusercontent.com/scholar.bib?q=info:YqVjr0NJOdkJ:scholar.google.com/&output=citation&scisdr=ClH8C6kREL_MzHQJtOk:AFWwaeYAAAAAZsUPrOm7uaRLtHwVQlhhD44iVZQ&scisig=AFWwaeYAAAAAZsUPrKhfzxmAUv-0ByPP_10kTv0&scisf=4&ct=citation&cd=-1&hl=en)

The work is developed from the preliminary work [Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation](https://arxiv.org/abs/2303.06842) accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop ([GLFrontiers](https://glfrontiers.github.io/)) and NeurIPS 2023 [Queer in AI](https://www.queerinai.com/neurips-2023), which shares the same GitHub repository.

This work considers the scene graph generation problem. Different from object detection or segmentation, it represents each image as a graph, where each object instance is a node and each relation between a pair of nodes is a directed edge. Our highlights are:
1. We leverage the commonsense knowledge from small-scale, open-source, and on-device **LLMs/VLMs**, filtering out **commonsense-violated** predictions made by baseline scene graph generation models.
3. **Hierarchical relationships work surprisingly well:** Replacing the flat relation classification head with a Bayesian head, which jointly predicts relationship super-category probabilities and detailed relationships within each super-category, can improve model performance by a large margin.
4. Both of our proposed methods are **model-agnostic:** They can be easily used as **plug-and-play** modules into **existing SOTA** works, continuing pushing their SOTA performance to **new levels**.

We dub our methods as **HIERCOM**, an acronym for **HIE**rarchical **R**elationships and **COM**monsense Validation.

## Repository Structure

### This repository consists of two parts.
- :purple_heart: **Plug-and-play to existing SOTA works in [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) framework**:

   Visit the [scenegraph_benchmark/](scenegraph_benchmark/) directory and follow the ***[Plug-and-Play README](README_PLUGANDPLAY.md)*** for step-by-step instructions on how to integrate our methods (hierarchical relationships & commonsense validation) into your own work. 
We have experimented with on [Neural Motifs](https://arxiv.org/abs/1711.06640), 
[VTransE](https://arxiv.org/abs/1702.08319), 
[VCTree](https://arxiv.org/abs/1812.01880), 
[TDE](https://arxiv.org/pdf/2002.11949.pdf), 
[NICE](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_The_Devil_Is_in_the_Labels_Noisy_Label_Correction_for_CVPR_2022_paper.pdf), 
[IETrans](https://arxiv.org/abs/2203.11654). We hope our methods can support your work in achieving even better performance :raised_hands:

- :blue_heart: **A light-weighted standalone baseline model**:

  Follow the ***[Plug-and-Play README](README_BASEMODEL.md)*** to install dependencies, load datasets and pretrained weights, and start training and evaluation processes. Codes are organized under the top directory.

<p align="center">
<img src=figures/framework.png />
</p>

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

## TODOs
- [x] 1. Clean up the codes for integrating hierarchical classification and commonsense validation as portable modules to other existing SOTA works in [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
- [x] 2. Add unsupervised relation super-category clustering from pretrained GPT2, BERT, and CLIP token embeddings, eliminating the need for humans to classify relation labels.\
- [ ] 3. Release all pretrained model weights. (partially done)
- [ ] 4. Clean up the codes for efficient single-image inference.
- [ ] 5. Clean up the codes for running experiments on the OpenImage V6 dataset.
- [ ] 6. VQA is a commonly explored downstream task in the scene graph generation community, and we expect to release a preliminary manuscript and codes within the next few months, so please stay tuned for updates! **[Update] It is now released and available at [Multi-Agent VQA](https://github.com/bowen-upenn/Multi-Agent-VQA)**.


## Visual results
<p align="center">
<img src=figures/plot_new.png />
</p>
<p>
    <em>Illustration of generated scene graphs on predicate classification. All examples are from the testing dataset of Visual Genome. The first row displays images and objects, while the second row displays the final scene graphs. The third row shows an ablation without commonsense validation, where there exist many unreasonable predictions with high confidence. For each image, we display the top 10 most confident predictions, and each edge is annotated with its relation label and relation super-category. It is possible for an edge to have multiple predicted relationships, but they must come from disjoint super-categories. Blue edges represent incorrect edges based on our observations. Pink edges are true positives in the dataset. Interestingly, all black edges are reasonable predictions we believe but not annotated, which should not be regarded as false positives.</em>
</p>
