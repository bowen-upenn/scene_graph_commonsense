## Plug-and-Play Introduction

Our proposed relationship hierarchy and commonsense validation as plug-and-play methods could continue pushing existing SOTA works to new SOTA levels of performance by a large margin. 
This documentation will guide you on how to integrate these two methods as plug-and-play modules into your work.

We assume that you have already built your own repository using the [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) framework.
Our methods are generalizable to many current SOTA works, including but not limited to 
[Neural Motifs](https://arxiv.org/abs/1711.06640), 
[Transformer](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), 
[VCTree](https://arxiv.org/abs/1812.01880), 
[TDE](https://arxiv.org/pdf/2002.11949.pdf), 
[NICE](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_The_Devil_Is_in_the_Labels_Noisy_Label_Correction_for_CVPR_2022_paper.pdf), 
[IETrans](https://arxiv.org/abs/2203.11654), etc. 
The current version does not support methods that modify loss function on the relation predictions,
or modify the final classification layer of the relation head (not a flat classification layer) to plug in the relationship hierarchy, but the commonsense validation should work for all methods.

In our provided repository here, [Scene-Graph-Benchmark.pytorch/](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/tree/5544610cfed0be574f6d34aa8d15f063a637a806)
plugs our methods into [Neural Motifs](https://arxiv.org/abs/1711.06640), 
[Transformer](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), and
[VCTree](https://arxiv.org/abs/1812.01880), with additional [TDE](https://arxiv.org/pdf/2002.11949.pdf) that addresses long-tailed distribution problems.

Here is an illustration of how to plug our hierarchical relationships into [VCTree](https://arxiv.org/abs/1812.01880). We hope it is straightforward but highly effective.
<p align="center">
<img src=../figures/extension.png />
</p>
<p style="font-size: x-small;"> 
   <em>This image is altered from its original version by Tang, Kaihua, Hanwang Zhang, Baoyuan Wu, Wenhan Luo, and Wei Liu. "Learning to compose dynamic tree structures for visual contexts." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6619-6628. 2019.</em>
</p>

## Step-by-Step Instructions

These steps may appear complicated at first glance, but they are actually quite easy to implement compared to many other plug-and-play methods!
Basically, you need to modify [model_motifs_hierarchical.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/model_motifs_hierarchical.py),
[utils_motifs_hierarchical.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/utils_motifs_hierarchical.py),
[roi_relation_predictors.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py),
[inference.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/inference.py),
[loss.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py),
[paths_catalog.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/5544610cfed0be574f6d34aa8d15f063a637a806/maskrcnn_benchmark/config/paths_catalog.py)
[sgg_eval.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py),
[vg_eval.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/data/datasets/evaluation/vg/vg_eval.py),
and the command line or bash script to train or evaluate the model, most of which are under the paths ``/maskrcnn_benchmark/modeling/roi_heads/relation_head/``
or ``/maskrcnn_benchmark/data/datasets/evaluation/vg/``.
The results are worth it! You will notice a significant performance boost after applying our methods.

We take [Neural Motifs](https://arxiv.org/abs/1711.06640) on the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) dataset as an example for illustration.

### Instructions for plugging in the relationship hierarchy
**Step 1**: copy our provided [model_motifs_hierarchical.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/model_motifs_hierarchical.py)
and [utils_motifs_hierarchical.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/utils_motifs_hierarchical.py)
we provided to your repository under the path ``/maskrcnn_benchmark/modeling/roi_heads/relation_head/``
This step follows the [README.md](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/README.md) of the [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) repository to customize your own model.

**Step 2**: in the file ``/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`` in your repository,
add the line 

    from .model_motifs_hierarchical.py import BayesHead, BayesHeadProd

at the file top,
and then add a new class named ``MotifHierarchicalPredictor`` we provided in our [roi_relation_predictors.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py).
Make sure you have registered the new class by adding the line 

    @registry.ROI_RELATION_PREDICTOR.register("MotifHierarchicalPredictor")
    class MotifHierarchicalPredictor(nn.Module):
        ...

right above the class definition.

**Step 3**: in the file ``/maskrcnn_benchmark/modeling/roi_heads/relation_head/relation_head.py`` in your repository,
follow our [relation_head.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/relation_head.py)
to add the 

    class ROIRelationHead(torch.nn.Module):
        ...
        def forward(...):
            ...
            if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "MotifHierarchicalPredictor":
                ...

condition in the class ``ROIRelationHead``,
so that we extend the original ``relation_logits`` variable to ``rel1_prob, rel2_prob, rel3_prob, super_rel_prob``.

**Step 4**: in the file ``/maskrcnn_benchmark/modeling/roi_heads/relation_head/inference.py`` in your repository,
follow our [inference.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/inference.py)
to add a new class named ``HierarchPostProcessor`` and then update the ``make_roi_relation_post_processor`` function to include the condition

    class HierarchPostProcessor(nn.Module):
        ...

    def make_roi_relation_post_processor(cfg):
        ...
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "MotifHierarchicalPredictor":
            ...

**Step 5**: in the file ``/maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py`` in your repository,
follow our [loss.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py)
to add a new class named ``RelationHierarchicalLossComputation`` and then update the ``make_roi_relation_loss_evaluator`` function to include the condition 

    class RelationHierarchicalLossComputation(object):
        ...

    def make_roi_relation_loss_evaluator(cfg):
        ...
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "MotifHierarchicalPredictor":
            ...

**Step 6**: in the file ``/maskrcnn_benchmark/config/paths_catalog.py``, update ``DATA_DIR`` in the class ``DatasetCatalog`` to your own VG data path.

**Step 7**: copy our provided [sgg_eval.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) and 
[vg_eval.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/data/datasets/evaluation/vg/vg_eval.py) 
to your repository under the path ``/maskrcnn_benchmark/data/datasets/evaluation/vg/``.
You might need to modify the ``zeroshot_triplet`` path in ``vg_eval.py``.

**Step 8**: In either the command line or bash script to train or evaluate the model,
specify ``MODEL.ROI_RELATION_HEAD.PREDICTOR`` as ``MotifHierarchicalPredictor``,
and update your ``GLOVE_DIR, PRETRAINED_DETECTOR_CKPT, OUTPUT_DIR`` to your own paths. 
We also suggest reducing your learning rate ``SOLVER.BASE_LR`` for a more stable training, and we use ``SOLVER.BASE_LR = 0.001``or``0.0025`` in our experiments.
If you encounter an error of ``non-existent key``, add the line ``cfg.set_new_allowed(True)`` to
`` tools/relation_train_net.py`` on the line before ``cfg.merge_from_file(args.config_file)``.

### Instructions for plugging in the commonsense validation
We have added support for validating commonsense using GPT-3.5 during inference. 
If you want to bake commonsense knowledge in your model without using any external LLM at inference time, 
you can refer to our [standalone model codes](../train_test.py) with the flag ``--train_mode`` being ``prepare_cs`` or ``train_cs``, which 
will query GPT-3.5 for commonsense validation during training, save the results as commonsense-aligned and violated triplets, and then use them for re-training the model from scratch. 
Additionally, please refer to the [standalone model code](../query_llm.py) for examples on how to use GPT-4V instead of GPT-3.5, 
where you will need access to each image and subject-object bounding boxes.

**Step 1**: copy our provided [query_llm.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/query_llm.py)
to your repository under the path ``/maskrcnn_benchmark/modeling/roi_heads/relation_head/``

**Step 2**: in the file ``/maskrcnn_benchmark/modeling/roi_heads/relation_head/inference.py`` in your repository,
add 

    from .query_llm import CommonsenseValidator
    import math

at the file top, add 

    self.llm = CommonsenseValidator()

in the class ``__init__`` function,
and then follow our [inference.py](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/inference.py)
to add the code block

    # query llm about top k triplets for commonsense validation
    llm_responses = self.llm.query(rel_pair_idx[:self.llm.top_k, :], rel_labels[:self.llm.top_k])
    rel_class_prob[:self.llm.top_k, :][llm_responses == -1] = -math.inf
    
    # resort the triplets
    _, sorting_idx = torch.sort(rel_class_prob, dim=0, descending=True)
    rel_pair_idx = rel_pair_idx[sorting_idx]
    rel_class_prob = rel_class_prob[sorting_idx]
    rel_labels = rel_labels[sorting_idx]

to the ``forward`` function, usually above the line ``boxlist.add_field``.

**Step 3**: run the inference script as usual.


## Experimental Results
| Methods                       | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
| ----------------------------- | ---- | ---- | ----- | ----- | ----- | ------ |
| [Neural Motifs](https://arxiv.org/abs/1711.06640)                        | **58.5** | 65.2 | 67.0  | 11.7  | 14.8  | 16.1   |
| [Neural Motifs](https://arxiv.org/abs/1711.06640)+**Ours**               | 53.8 | **68.3** | **74.6** | **15.9**  | **24.3** | **29.9** |

| Methods                       | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
| ----------------------------- | ---- | ---- | ----- | ----- | ----- | ------ |
| [VTransE](https://arxiv.org/abs/1702.08319)                       | **59.1** | 65.6 | 67.3  | 12.8 | 16.3 | 17.6   |
| [VTransE](https://arxiv.org/abs/1702.08319)+**Ours**              | 53.8  | **68.1** | **74.5**  | **18.1**  | **26.2** | **31.5**  |

| Methods                       | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
| ----------------------------- | ---- | ---- | ----- | ----- | ----- | ------ |
| [VCTree](https://arxiv.org/abs/1812.01880)                        | **59.0** | 65.4 | 67.2  | 13.1  |16.7  |18.2  |
| [VCTree](https://arxiv.org/abs/1812.01880)+**Ours**               | 54.5  | **69.1**  | **75.4** | **16.7**  | **26.3** |**32.2** |

| Methods                       | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
| ----------------------------- | ---- | ---- | ----- | ----- | ----- | ------ |
| [Motifs](https://arxiv.org/abs/1711.06640)+[TDE](https://arxiv.org/pdf/2002.11949.pdf)                    | 33.6 | 46.2 | 51.4  | 18.5  | 25.5  | 29.1   |
| [Motifs](https://arxiv.org/abs/1711.06640)+[TDE](https://arxiv.org/pdf/2002.11949.pdf)+**Ours**           | **39.7** | **56.9** | **66.7** | **20.1** | **28.8** | **34.9** |

| Methods                       | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
| ----------------------------- | ---- | ---- | ----- | ----- | ----- | ------ |
| [VCTree](https://arxiv.org/abs/1812.01880)+[TDE](https://arxiv.org/pdf/2002.11949.pdf)                    | 36.2 | 47.2 | 51.6  | 18.4  | 25.4  | 28.7   |
| [VCTree](https://arxiv.org/abs/1812.01880)+[TDE](https://arxiv.org/pdf/2002.11949.pdf)+[EBM](https://arxiv.org/abs/2103.02221)                | **41.6** | 51.2 | 54.3  | **19.9** | 26.7  | 30.0   |
| [VCTree](https://arxiv.org/abs/1812.01880)+[TDE](https://arxiv.org/pdf/2002.11949.pdf)+**Ours**           | 39.6 | **56.9** | **66.6** | 19.6  | **28.6** | **35.2** |

| Methods                       | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
| ----------------------------- | ---- | ---- | ----- | ----- | ----- | ------ |
| [Motifs](https://arxiv.org/abs/1711.06640)+[NICE](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_The_Devil_Is_in_the_Labels_Noisy_Label_Correction_for_CVPR_2022_paper.pdf)                   | -    | 55.1 | 57.2  | -     | 29.9  | 32.3   |
| [Motifs](https://arxiv.org/abs/1711.06640)+[NICE](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_The_Devil_Is_in_the_Labels_Noisy_Label_Correction_for_CVPR_2022_paper.pdf)+**Ours**          | **43.1** | **58.2** | **65.4** | **22.6** | **33.1** | **39.8** |

| Methods                       | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
| ----------------------------- | ---- | ---- | ----- | ----- | ----- | ------ |
| [Motifs](https://arxiv.org/abs/1711.06640)+[IETrans](https://arxiv.org/abs/2203.11654)                | -    | 48.6 | 50.5  | -     | 35.8  | 39.1   |
| [Motifs](https://arxiv.org/abs/1711.06640)+[IETrans](https://arxiv.org/abs/2203.11654)+**Ours**       | **47.9** | **60.4** | **66.4** | **26.4** | **38.0** | **44.1** |

[Motifs](https://arxiv.org/abs/1711.06640)+[IETrans](https://arxiv.org/abs/2203.11654)+**Ours** achieves the SOTA performance in terms of mR@k scores :rocket:
