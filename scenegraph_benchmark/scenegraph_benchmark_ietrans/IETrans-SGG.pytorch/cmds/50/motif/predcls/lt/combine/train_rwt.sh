OUTPATH=/raid0/docker-raid/bwjiang/scene_graph/checkpoints/benchmark
mkdir -p $OUTPATH

#CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch \
#  --master_port 10091 --nproc_per_node=2 \
#  tools/relation_train_net.py --config-file "configs/wsup-50.yaml" \
#  DATASETS.TRAIN \(\"50DS_VG_VGKB_train\",\) \
#  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
#  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
#  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
#  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 \
#  DTYPE "float16" SOLVER.MAX_ITER 50000 \
#  SOLVER.VAL_PERIOD 2000 \
#  SOLVER.CHECKPOINT_PERIOD 2000 \
#  GLOVE_DIR $OUTPATH/glove \
#  MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH/pretrained_faster_rcnn/model_final.pth  \
#  OUTPUT_DIR $OUTPATH/motif-hierarch-ietrans \
#  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
#  SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
#  TEST.INFERENCE "SOFTMAX"  IETRANS.RWT True \
#  WSUPERVISE.LOSS_TYPE  ce_rwt WSUPERVISE.DATASET InTransDataset  WSUPERVISE.SPECIFIED_DATA_FILE   $OUTPATH/motif-hierarch-ietrans/ietrans-em_E.pk_0.7


CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch \
  --master_port 10094 --nproc_per_node=2 \
  tools/relation_train_net.py --config-file "configs/wsup-50.yaml" \
  DATASETS.TRAIN \(\"50DS_VG_VGKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifHierarchicalPredictor \
  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  SOLVER.BASE_LR 0.001 \
  GLOVE_DIR $OUTPATH/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH/pretrained_faster_rcnn/model_final.pth  \
  OUTPUT_DIR $OUTPATH/motif-hierarch-ietrans \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
  SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  TEST.INFERENCE "SOFTMAX"  IETRANS.RWT False \
  WSUPERVISE.LOSS_TYPE  ce WSUPERVISE.DATASET InTransDataset  WSUPERVISE.SPECIFIED_DATA_FILE   $OUTPATH/motif-hierarch-ietrans/ietrans-em_E.pk_0.7

