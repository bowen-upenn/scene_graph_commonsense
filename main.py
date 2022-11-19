import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Subset
import yaml
import os
import torch.multiprocessing as mp

from dataset import VisualGenomeDataset
from train_test_local import train_local
from train_test_global import train_global


if __name__ == "__main__":
    print('Torch', torch.__version__, 'Torchvision', torchvision.__version__)
    # load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    if args['training']['train_mode'] == 'local':
        from evaluate_local import eval_pc, eval_sgd
    elif args['training']['train_mode'] == 'global':
        from evaluate_global import eval_pc, eval_sgd
    else:
        print('Invalid arguments.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Prepare datasets
    print("Loading the datasets...")
    train_dataset = VisualGenomeDataset(args, device, args['dataset']['annotation_train'])
    test_dataset = VisualGenomeDataset(args, device, args['dataset']['annotation_test'])

    torch.manual_seed(0)
    train_subset_idx = torch.randperm(len(train_dataset))[:int(args['dataset']['percent_train'] * len(train_dataset))]
    train_subset = Subset(train_dataset, train_subset_idx)
    test_subset_idx = torch.randperm(len(test_dataset))[:int(args['dataset']['percent_test'] * len(test_dataset))]
    test_subset = Subset(test_dataset, test_subset_idx)
    print('num of train, test:', len(train_subset), len(test_subset))

    # select training or evaluation
    if args['training']['run_mode'] == 'train':
        # local prediction module or the model with optional transformer encoder
        if args['training']['train_mode'] == 'local':
            mp.spawn(train_local, nprocs=world_size, args=(args, train_subset, test_subset))
        elif args['training']['train_mode'] == 'global':
            mp.spawn(train_global, nprocs=world_size, args=(args, train_subset, test_subset))
        else:
            print('Invalid arguments.')

    elif args['training']['run_mode'] == 'eval':
        # select evaluation mode
        if args['training']['eval_mode'] == 'pc':          # predicate classification
            mp.spawn(eval_pc, nprocs=world_size, args=(args, test_subset))
        elif args['training']['eval_mode'] == 'sgd':       # scene graph detection
            mp.spawn(eval_sgd, nprocs=world_size, args=(args, test_subset))
        else:
            print('Invalid arguments.')
    else:
        print('Invalid arguments.')
