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
import pickle

from dataloader import VisualGenomeDataset, OpenImageV6Dataset
from train_test import training
from evaluate import eval_pc, eval_sgc, eval_sgd


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
    parser.add_argument('--run_mode', type=str, default=None, help='Override run_mode (train, eval, prepare_cs, train_cs, eval_cs)')
    parser.add_argument('--eval_mode', type=str, default=None, help='Override eval_mode (pc, sgc, sgd)')
    parser.add_argument('--cluster', type=str, default=None, help='Override supcat_clustering (motif, gpt2, bert, clip)')
    parser.add_argument('--hierar', dest='hierar', action='store_true', help='Set hierarchical_pred to True')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['training']['run_mode'] = cmd_args.run_mode if cmd_args.run_mode is not None else args['training']['run_mode']
    args['training']['eval_mode'] = cmd_args.eval_mode if cmd_args.eval_mode is not None else args['training']['eval_mode']
    args['dataset']['supcat_clustering'] = cmd_args.cluster if cmd_args.cluster is not None else args['dataset']['supcat_clustering']
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

        # we still use the name num_geometric, num_possessive, num_semantic for name consistency in this code repo
        # but they are actually the number of clusters for each super category in the GPT-2, BERT, or CLIP clustering
        if args['dataset']['supcat_clustering'] == 'gpt2':
            args['models']['num_geometric'] = 9
            args['models']['num_possessive'] = 32
            args['models']['num_semantic'] = 9
        elif args['dataset']['supcat_clustering'] == 'bert':
            args['models']['num_geometric'] = 12
            args['models']['num_possessive'] = 25
            args['models']['num_semantic'] = 13
        elif args['dataset']['supcat_clustering'] == 'clip':
            args['models']['num_geometric'] = 27
            args['models']['num_possessive'] = 15
            args['models']['num_semantic'] = 8
        else:   # if 'supcat_clustering' is 'motif', we follow the super-category definitions in the paper Neural Motifs
            args['models']['num_geometric'] = 15
            args['models']['num_possessive'] = 11
            args['models']['num_semantic'] = 24

        args['models']['detr101_pretrained'] = 'checkpoints/detr101_vg_ckpt.pth'

        print("Loading the datasets...")
        train_dataset = VisualGenomeDataset(args, device, args['dataset']['annotation_train'], training=True)
        test_dataset = VisualGenomeDataset(args, device, args['dataset']['annotation_test'], training=False)    # always evaluate on the original testing dataset

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

    print(args)
    # select training or evaluation
    if args['training']['run_mode'] == 'train' or args['training']['run_mode'] == 'train_cs':
         mp.spawn(training, nprocs=world_size, args=(args, train_subset, test_subset))

    elif args['training']['run_mode'] == 'prepare_cs':
        """
        we have to collect commonsense-aligned and violated triplets only from the training dataset to prevent data leakage
        the process is divided into two steps to avoid unexpected interrupts from OpenAI API connections
        the first step requires model inference, but the second step only requires calling the __getitem__ function in dataloader
        """
        # step 1: collect and save commonsense-aligned and violated triplets on the current baseline model for each image
        mp.spawn(eval_pc, nprocs=world_size, args=(args, train_subset, train_dataset, 1))
        # step 2: rerun it again but to accumulate all collected triplets from the two sets and save them into two .pt files
        mp.spawn(eval_pc, nprocs=world_size, args=(args, train_subset, train_dataset, 2))

    elif args['training']['run_mode'] == 'eval' or args['training']['run_mode'] == 'eval_cs':
        # select evaluation mode
        if args['training']['eval_mode'] == 'pc':    # predicate classification
            mp.spawn(eval_pc, nprocs=world_size, args=(args, test_subset))
        elif args['training']['eval_mode'] == 'sgc' and args['dataset']['dataset'] == 'vg':     # scene graph classification
            mp.spawn(eval_sgc, nprocs=world_size, args=(args, test_subset))
        elif args['training']['eval_mode'] == 'sgd' and args['dataset']['dataset'] == 'vg':     # scene graph detection
            mp.spawn(eval_sgd, nprocs=world_size, args=(args, test_subset))
        else:
            print('Invalid arguments or not implemented.')
    else:
        print('Invalid arguments.')

