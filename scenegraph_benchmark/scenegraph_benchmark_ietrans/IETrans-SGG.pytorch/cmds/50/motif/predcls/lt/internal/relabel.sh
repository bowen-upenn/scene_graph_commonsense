OUTPATH=/raid0/docker-raid/bwjiang/scene_graph/checkpoints/benchmark/
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
  --master_port 10083 --nproc_per_node=2 \
  tools/internal_relabel.py --config-file "configs/wsup-50.yaml"  \
  DATASETS.TRAIN \(\"50DS_VG_VGKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifHierarchicalPredictor \
  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $OUTPATH/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH/pretrained_faster_rcnn/model_final.pth  \
  OUTPUT_DIR $OUTPATH/motif-hierarch-ietrans  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
  SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  WSUPERVISE.DATASET InTransDataset  EM.MODE E  WSUPERVISE.SPECIFIED_DATA_FILE  /tmp/datasets/vg/50/vg_sup_data.pk


cd $OUTPATH
cp $SG/tools/ietrans/internal_cut.py ./
cp /tmp/datasets/vg/50/VG-SGG-dicts-with-attri.json ./
python internal_cut.py 0.7
