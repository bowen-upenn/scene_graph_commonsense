import torch
import numpy as np
import json
import yaml
import torchvision
from torchvision import transforms
from torch.utils.data import Subset

from utils import collate_fn
from dataset import *
from dataset_utils import *


# load hyper-parameters
try:
    with open ('config.yaml', 'r') as file:
        args = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config file')

if args['dataset']['dataset'] == 'vg':
    train_dataset = PrepareVisualGenomeDataset(args['dataset']['annotation_train'])
    test_dataset  = PrepareVisualGenomeDataset(args['dataset']['annotation_test'])
else:
    train_dataset = PrepareOpenImageV6Dataset(args, '../datasets/open_image_v6/annotations/oiv6-adjust/vrd-train-anno.json')
    test_dataset = PrepareOpenImageV6Dataset(args, '../datasets/open_image_v6/annotations/oiv6-adjust/vrd-test-anno.json')
    val_dataset = PrepareOpenImageV6Dataset(args, '../datasets/open_image_v6/annotations/oiv6-adjust/vrd-val-anno.json')
    val_subset_idx = torch.arange(len(val_dataset))
    val_subset = Subset(val_dataset, val_subset_idx)
    val_loader = torch.utils.data.DataLoader(val_subset,  batch_size=1, shuffle=False)

torch.manual_seed(0)
test_start, train_start = 0, 0
train_subset_idx = torch.arange(len(train_dataset))[train_start:]
train_subset = Subset(train_dataset, train_subset_idx)
test_subset_idx = torch.arange(len(test_dataset))[test_start:]
test_subset = Subset(test_dataset, test_subset_idx)
print('num of train, test:', len(train_subset), len(test_subset))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_subset,  batch_size=1, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
world_size = torch.cuda.device_count()
print('device', device, world_size)

# find_zero_shot_triplet(args['dataset']['annotation_train'], args['dataset']['annotation_test'])

image_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((args['models']['image_size'], args['models']['image_size']))])

model_type = args['models']['depth_model_type']
depth_estimator = torch.hub.load("intel-isl/MiDaS", model_type)

if args['dataset']['dataset'] == 'vg':
    print('Start gathering testing annotations...')
    prepare_data_offline(args, test_loader, device, args['dataset']['annotation_test'], image_transform, depth_estimator, test_start)
    print('Finished gathering testing annotations...')

    print('Start gathering training annotations...')
    prepare_data_offline(args, train_loader, device, args['dataset']['annotation_train'], image_transform, depth_estimator, train_start)
    print('Finished gathering training annotations...')

else:
    prepare_depth_oiv6_offline(args, test_loader, device, depth_estimator)
    prepare_depth_oiv6_offline(args, train_loader, device, depth_estimator)