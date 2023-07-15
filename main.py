import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Subset
import yaml
import os
import json
import torch.multiprocessing as mp
import detectron2

from dataset import VisualGenomeDataset, VisualGenomeDatasetEfficient, OpenImageV6Dataset
# from train_test_local_concat import train_local
from train_test_efficient import train_local
# from train_test_global import train_global
from evaluate import eval_pc, eval_sgc, eval_sgd

# from train_faster_rcnn import setup
# from detectron2.engine import default_argument_parser


if __name__ == "__main__":
    print('Torch', torch.__version__, 'Torchvision', torchvision.__version__)
    # load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))
    print("Running model", args['models']['detr_or_faster_rcnn'])
    # print("detectron2:", detectron2.__version__)

    # prepare datasets
    if args['dataset']['dataset'] == 'vg':
        args['models']['num_classes'] = 150
        args['models']['num_relations'] = 50
        args['models']['num_super_classes'] = 17
        args['models']['num_geometric'] = 15
        args['models']['num_possessive'] = 11
        args['models']['num_semantic'] = 24
        args['models']['detr101_pretrained'] = 'checkpoints/detr101_vg_ckpt.pth'

        print("Loading the datasets...")
        train_dataset = VisualGenomeDatasetEfficient(args, device, args['dataset']['annotation_train'])
        test_dataset = VisualGenomeDatasetEfficient(args, device, args['dataset']['annotation_test'])
        # train_dataset = VisualGenomeDatasetNonDynamic(args, device, args['dataset']['annotation_train'])
        # test_dataset = VisualGenomeDatasetNonDynamic(args, device, args['dataset']['annotation_test'])

    elif args['dataset']['dataset'] == 'oiv6':
        args['models']['num_classes'] = 601
        args['models']['num_relations'] = 30
        args['models']['num_geometric'] = 4
        args['models']['num_possessive'] = 2
        args['models']['num_semantic'] = 24
        args['models']['detr101_pretrained'] = 'checkpoints/detr101_oiv6_ckpt.pth'

        print("Loading the datasets...")
        train_dataset = OpenImageV6Dataset(args, device, '../datasets/open_image_v6/annotations/oiv6-adjust/vrd-train-anno.json')
        test_dataset = OpenImageV6Dataset(args, device, '../datasets/open_image_v6/annotations/oiv6-adjust/vrd-test-anno.json')
    else:
        print('Unknown dataset.')

    torch.manual_seed(0)
    train_subset_idx = torch.randperm(len(train_dataset))[:int(args['dataset']['percent_train'] * len(train_dataset))]
    train_subset = Subset(train_dataset, train_subset_idx)
    test_subset_idx = torch.randperm(len(test_dataset))[:int(args['dataset']['percent_test'] * len(test_dataset))]
    test_subset = Subset(test_dataset, test_subset_idx)
    print('num of train, test:', len(train_subset), len(test_subset))

    # prepare faster rcnn configs
    faster_rcnn_cfg = None
    if args['models']['detr_or_faster_rcnn'] == 'faster':
        faster_rcnn_args = default_argument_parser().parse_args()
        faster_rcnn_cfg = setup(faster_rcnn_args)

    # select training or evaluation
    if args['training']['run_mode'] == 'train':
        # local prediction module or the model with optional transformer encoder
        if args['training']['train_mode'] == 'local':
            mp.spawn(train_local, nprocs=world_size, args=(args, train_subset, test_subset, faster_rcnn_cfg))
        # elif args['training']['train_mode'] == 'global' and args['dataset']['dataset'] == 'vg':
        #     mp.spawn(train_global, nprocs=world_size, args=(args, train_subset, test_subset))
        else:
            print('Invalid arguments or not implemented.')

    elif args['training']['run_mode'] == 'eval':
        # if args['training']['train_mode'] == 'global':
        #     print('Not implemented.')
        # else:
        # select evaluation mode
        if args['training']['eval_mode'] == 'pc':          # predicate classification
            mp.spawn(eval_pc, nprocs=world_size, args=(args, test_subset, faster_rcnn_cfg))
        elif args['training']['eval_mode'] == 'sgc' and args['dataset']['dataset'] == 'vg':       # scene graph classification
            args['models']['topk_cat'] = 1
            mp.spawn(eval_sgc, nprocs=world_size, args=(args, test_subset))
        elif args['training']['eval_mode'] == 'sgd' and args['dataset']['dataset'] == 'vg':       # scene graph detection
            args['models']['topk_cat'] = 2
            mp.spawn(eval_sgd, nprocs=world_size, args=(args, test_subset))
        else:
            print('Invalid arguments or not implemented.')
    else:
        print('Invalid arguments.')
