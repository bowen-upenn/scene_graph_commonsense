import torch
import os
from PIL import Image
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluate import inference, eval_pc
from utils import *


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def process_sgg_results(rank, sgg_results):
    top_k_predictions = sgg_results['top_k_predictions']
    print('top_k_predictions', top_k_predictions)


def query_clip(gpu, args, test_dataset):
    rank = gpu
    world_size = torch.cuda.device_count()
    setup(rank, world_size)
    print('rank', rank, 'torch.distributed.is_initialized', torch.distributed.is_initialized())

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=test_sampler)
    print("Finished loading the datasets...")

    # receive current SGG predictions from a baseline model
    sgg_results = eval_pc(rank, args, test_loader, return_sgg_results=True, top_k=5)

    # iterate through the generator to receive results
    for batch_idx, batch_sgg_results in enumerate(sgg_results):
        print('batch_idx', batch_idx)
        process_sgg_results(rank, batch_sgg_results)

    dist.destroy_process_group()  # clean up

