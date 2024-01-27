OUTPATH=/raid0/docker-raid/bwjiang/scene_graph/checkpoints/benchmark/motif-hierarch-ietrans-newdata

#cd $SG
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port 10070 --nproc_per_node=1 \
        tools/relation_test_net.py --config-file "configs/sup-50.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR MotifHierarchicalPredictor \
        TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR $OUTPATH/../glove \
        MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH/../pretrained_faster_rcnn/model_final.pth  \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True
