# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


# rel_label_mapping = torch.tensor([50,  0,  1,  2,  3,  4,  5, 26,  6, 15,  7, 27, 28, 29, 30, 31, 16, 17,
#                         32, 33, 18, 34,  8,  9, 35, 36, 37, 19, 38, 10, 20, 11, 12, 13, 39, 40,
#                         21, 41, 42, 43, 44, 45, 22, 14, 46, 47, 48, 49, 23, 24, 25, 51])
# rel_label_mapping = torch.tensor([0,  1,  2,  3,  4,  5,  6, 27,  7, 16,  8, 28, 29, 30, 31, 32, 17, 18,
#         33, 34, 19, 35,  9, 10, 36, 37, 38, 20, 39, 11, 21, 12, 13, 14, 40, 41,
#         22, 42, 43, 44, 45, 46, 23, 15, 47, 48, 49, 50, 24, 25, 26, 51])
map = torch.tensor([ 0,  1,  2,  3,  4,  5,  6, 27,  7, 16,  8, 28, 29, 30, 31, 32, 17, 18,
        33, 34, 19, 35,  9, 10, 36, 37, 38, 20, 39, 11, 21, 12, 13, 14, 40, 41,
        22, 42, 43, 44, 45, 46, 23, 15, 47, 48, 49, 50, 24, 25, 26, 51])


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--ann_path', default='./data/vg/', type=str)
    parser.add_argument('--img_folder', default='/home/cong/Dokumente/tmp/data/visualgenome/images/', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")

    ## ADD-ON #################################################
    parser.add_argument('--hierar', action='store_true')
    parser.add_argument('--num_rel_prior', default=4, type=int)
    parser.add_argument('--num_rel_geometric', default=15, type=int)
    parser.add_argument('--num_rel_possessive', default=11, type=int)
    parser.add_argument('--num_rel_semantic', default=24, type=int)
    parser.add_argument('--resume_from_flat', action='store_true')
    ###########################################################
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    ## ADD-ON #################################################
    # if args.hierar and args.resume_from_flat:
    #     # freeze all parameters
    #     for param in model_without_ddp.parameters():
    #         param.requires_grad = False
    #     # unfreeze the parameters of the specified layers
    #     layers_to_unfreeze = ['fc_rel', 'fc_rel_prior', 'fc_rel_geo', 'fc_rel_pos', 'fc_rel_sem']
    #     for name, param in model_without_ddp.named_parameters():
    #         if any(layer in name for layer in layers_to_unfreeze):
    #             param.requires_grad = True
    ###########################################################

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume or args.eval:
        checkpoint = torch.load(args.resume, map_location='cpu')
        ## ADD-ON #################################################
        model_without_ddp.load_state_dict(checkpoint['model'], strict=not args.resume_from_flat)
        # del checkpoint['optimizer']
        if not args.resume_from_flat and not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        ###########################################################
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

        ## ADD-ON #################################################
        # model_without_ddp.rel_class_embed[0].weight.data.copy_(checkpoint['model']['rel_class_embed.layers.0.weight'])
        # model_without_ddp.rel_class_embed[0].bias.data.copy_(checkpoint['model']['rel_class_embed.layers.0.bias'])
        # model_without_ddp.rel_class_embed[2].weight.data.copy_(checkpoint['model']['rel_class_embed.layers.1.weight'])
        # model_without_ddp.rel_class_embed[2].bias.data.copy_(checkpoint['model']['rel_class_embed.layers.1.bias'])

        if args.resume_from_flat:
            # model_without_ddp.fc_rel.weight.data.copy_(checkpoint['model']['rel_class_embed.layers.0.weight'])
            # model_without_ddp.fc_rel.bias.data.copy_(checkpoint['model']['rel_class_embed.layers.0.bias'])
            # model_without_ddp.rel_class_embed[0].weight.data.copy_(checkpoint['model']['rel_class_embed.layers.0.weight'])
            # model_without_ddp.rel_class_embed[0].bias.data.copy_(checkpoint['model']['rel_class_embed.layers.0.bias'])
            model_without_ddp.fc_rel.weight.data.copy_(checkpoint['model']['rel_class_embed.layers.0.weight'])
            model_without_ddp.fc_rel.bias.data.copy_(checkpoint['model']['rel_class_embed.layers.0.bias'])

            # reorder rows in the pretrained weights of the final layer
            pretrained_layers_1_weight = checkpoint['model']['rel_class_embed.layers.1.weight']  # size 52, 256
            pretrained_layers_1_weight = pretrained_layers_1_weight[map, :]
            model_without_ddp.fc_rel_prior.weight[-2].data.copy_(pretrained_layers_1_weight[0])
            print('model_without_ddp.fc_rel_prior.weight', model_without_ddp.fc_rel_prior.weight.shape, 'pretrained_layers_1_weight', pretrained_layers_1_weight.shape)
            model_without_ddp.fc_rel_prior.weight[-1].data.copy_(pretrained_layers_1_weight[-1])
            model_without_ddp.fc_rel_geo.weight.data.copy_(pretrained_layers_1_weight[1:16])
            model_without_ddp.fc_rel_pos.weight.data.copy_(pretrained_layers_1_weight[16:27])
            model_without_ddp.fc_rel_sem.weight.data.copy_(pretrained_layers_1_weight[27:-1])

            pretrained_layers_1_bias = checkpoint['model']['rel_class_embed.layers.1.bias']
            pretrained_layers_1_bias = pretrained_layers_1_bias[map]
            model_without_ddp.fc_rel_prior.bias[-2].data.copy_(pretrained_layers_1_bias[0])
            model_without_ddp.fc_rel_prior.bias[-1].data.copy_(pretrained_layers_1_bias[-1])
            model_without_ddp.fc_rel_geo.bias.data.copy_(pretrained_layers_1_bias[1:16])
            model_without_ddp.fc_rel_pos.bias.data.copy_(pretrained_layers_1_bias[16:27])
            model_without_ddp.fc_rel_sem.bias.data.copy_(pretrained_layers_1_bias[27:-1])

            # model_without_ddp.rel_class_embed[2].weight.data.copy_(pretrained_layers_1_weight)
            # model_without_ddp.rel_class_embed[2].bias.data.copy_(pretrained_layers_1_bias)
        ###########################################################

    if args.eval:
        print('It is the {}th checkpoint'.format(checkpoint['epoch']))
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,
                                      postprocessors, data_loader_val, base_ds, args, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth'] # anti-crash
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
