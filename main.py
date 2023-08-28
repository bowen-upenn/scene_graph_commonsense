import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Subset
import yaml
import os
import json
import torch.multiprocessing as mp
import argparse

from dataset import VisualGenomeDataset, OpenImageV6Dataset
from train_test import train_local
from evaluate import eval_pc, eval_sgc, eval_sgd
from downstream_tasks import image_captioning
from query_large_model import eval_open_flamingo

if __name__ == "__main__":
    print('Torch', torch.__version__, 'Torchvision', torchvision.__version__)
    # load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--run_mode', type=str, default=None, help='Override run_mode (train, eval, caption)')
    parser.add_argument('--eval_mode', type=str, default=None, help='Override eval_mode (pc, sgc, sgd)')
    parser.add_argument('--continue_train', type=bool, default=None, help='Override continue_train (True/False)')
    parser.add_argument('--start_epoch', type=int, default=None, help='Override start_epoch value')
    parser.add_argument('--hierar', type=bool, default=None, help='Override hierarchical_pred value')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['training']['run_mode'] = cmd_args.run_mode if cmd_args.run_mode is not None else args['training']['run_mode']
    args['training']['eval_mode'] = cmd_args.eval_mode if cmd_args.eval_mode is not None else args['training']['eval_mode']
    args['training']['continue_train'] = cmd_args.continue_train if cmd_args.continue_train is not None else args['training']['continue_train']
    args['training']['start_epoch'] = cmd_args.start_epoch if cmd_args.start_epoch is not None else args['training']['start_epoch']
    args['models']['hierarchical_pred'] = cmd_args.hierar if cmd_args.hierar is not None else args['models']['hierarchical_pred']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))
    print("Running model", args['models']['detr_or_faster_rcnn'])

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
        train_dataset = VisualGenomeDataset(args, device, args['dataset']['annotation_train'])
        test_dataset = VisualGenomeDataset(args, device, args['dataset']['annotation_test'])

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

    # select training or evaluation
    if args['training']['run_mode'] == 'train':
         mp.spawn(train_local, nprocs=world_size, args=(args, train_subset, test_subset))
    elif args['training']['run_mode'] == 'eval':
        # select evaluation mode
        if args['training']['eval_mode'] == 'pc':          # predicate classification
            mp.spawn(eval_pc, nprocs=world_size, args=(args, test_subset))
        elif args['training']['eval_mode'] == 'sgc' and args['dataset']['dataset'] == 'vg':       # scene graph classification
            args['models']['topk_cat'] = 1
            mp.spawn(eval_sgc, nprocs=world_size, args=(args, test_subset))
        elif args['training']['eval_mode'] == 'sgd' and args['dataset']['dataset'] == 'vg':       # scene graph detection
            args['models']['topk_cat'] = 2
            mp.spawn(eval_sgd, nprocs=world_size, args=(args, test_subset))
        else:
            print('Invalid arguments or not implemented.')
    elif args['training']['run_mode'] == 'caption':
        # inference(device, world_size, args, test_dataset, file_idx=0)
        # args['training']['eval_freq_test'] = 1
        image_captioning(device, world_size, args, test_dataset)
    elif args['training']['run_mode'] == 'question':
        eval_open_flamingo(device, world_size, args, test_dataset)
    else:
        print('Invalid arguments.')
