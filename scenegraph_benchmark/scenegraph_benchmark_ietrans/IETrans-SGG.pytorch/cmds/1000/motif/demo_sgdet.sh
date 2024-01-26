OUTPATH=./visualization/demo
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10081 --nproc_per_node=1 \
  tools/demo.py --config-file "configs/sup-1000.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.PRE_VAL False \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" \
  SOLVER.MAX_ITER 40000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  TEST.METRIC "R" TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH $1 \
  DETECTED_SGG_DIR $OUTPATH  TEST.ONLY_ACC False

