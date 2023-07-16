import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import os
import math
import torchvision
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluator import Evaluator_PC, Evaluator_PC_Top3
from model import TransformerEncoder, EdgeHead, EdgeHeadHier
from utils import *


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_local(gpu, args, train_subset, test_subset, faster_rcnn_cfg=None):
    """
    This function trains and evaluates the local prediction module on predicate classification tasks.
    :param gpu: current gpu index
    :param args: input arguments in config.yaml
    :param train_subset: training dataset
    :param test_subset: testing dataset
    """
    if args['models']['detr_or_faster_rcnn'] == 'faster':
        from detectron2.modeling import build_model
        from detectron2.structures.image_list import ImageList

    rank = gpu
    world_size = torch.cuda.device_count()
    setup(rank, world_size)
    print('rank', rank, 'torch.distributed.is_initialized', torch.distributed.is_initialized())

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=test_sampler)
    print("Finished loading the datasets...")

    start = []
    if not args['training']['continue_train']:
        record = []
        test_record = []
        with open(args['training']['result_path'] + 'train_results_' + str(rank) + '.json', 'w') as f:  # clear history logs
            json.dump(start, f)
        with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # clear history logs
            json.dump(start, f)
    else:
        with open(args['training']['result_path'] + 'train_results_' + str(rank) + '.json', 'r') as f:
            record = json.load(f)
        with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'r') as f:
            test_record = json.load(f)

    # if args['models']['hierarchical_pred']:
    # edge_head = DDP(EdgeHeadHier(args=args, input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
    #                              num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'],
    #                              num_geometric=args['models']['num_geometric'], num_possessive=args['models']['num_possessive'], num_semantic=args['models']['num_semantic'])).to(rank)
    # else:
    #     edge_head = DDP(EdgeHead(args=args, input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'], feature_size=args['models']['feature_size'],
    #                              num_classes=args['models']['num_classes'])).to(rank)
    transformer_encoder = DDP(TransformerEncoder(args)).to(rank)
    transformer_encoder.train()

    if args['models']['detr_or_faster_rcnn'] == 'detr':
        detr = DDP(build_detr101(args))
        detr.eval()
    elif args['models']['detr_or_faster_rcnn'] == 'faster':
        faster_rcnn = build_model(faster_rcnn_cfg).to(rank)
        faster_rcnn.load_state_dict(torch.load(os.path.join(faster_rcnn_cfg.OUTPUT_DIR, "model_final.pth"))['model'], strict=True)
        faster_rcnn = DDP(faster_rcnn)
        faster_rcnn.eval()
    else:
        print('Unknown model.')

    map_location = {'cuda:%d' % rank: 'cuda:%d' % 0}
    if args['training']['continue_train']:
        if args['models']['hierarchical_pred']:
            transformer_encoder.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'TransEncoderHier' + str(args['training']['start_epoch'] - 1) + '_0' + '.pth', map_location=map_location))
        else:
            transformer_encoder.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'TransEncoderFlat' + str(args['training']['start_epoch'] - 1) + '_0' + '.pth', map_location=map_location))
    #     if args['models']['hierarchical_pred']:
    #         edge_head.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHeadHier' + str(args['training']['start_epoch'] - 1) + '_0' + '.pth', map_location=map_location))
    #     else:
    #         edge_head.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHead' + str(args['training']['start_epoch'] - 1) + '_0' + '.pth', map_location=map_location))
    #
    # optimizer = optim.SGD([{'params': edge_head.parameters(), 'initial_lr': args['training']['learning_rate']}],
    #                       lr=args['training']['learning_rate'], momentum=0.9, weight_decay=args['training']['weight_decay'])

    optimizer = optim.SGD([{'params': transformer_encoder.parameters(), 'initial_lr': args['training']['learning_rate']}],
                            lr=args['training']['learning_rate'], momentum=0.9, weight_decay=args['training']['weight_decay'])
    original_lr = optimizer.param_groups[0]["lr"]

    relation_count = get_num_each_class_reordered(args)
    class_weight = 1 - relation_count / torch.sum(relation_count)

    if args['models']['hierarchical_pred']:
        criterion_relationship_1 = torch.nn.NLLLoss(weight=class_weight[:args['models']['num_geometric']].to(rank))  # log softmax already applied
        criterion_relationship_2 = torch.nn.NLLLoss(weight=class_weight[args['models']['num_geometric']:args['models']['num_geometric']+args['models']['num_possessive']].to(rank))
        criterion_relationship_3 = torch.nn.NLLLoss(weight=class_weight[args['models']['num_geometric']+args['models']['num_possessive']:].to(rank))
        criterion_super_relationship = torch.nn.NLLLoss()
    else:
        criterion_relationship = torch.nn.CrossEntropyLoss(weight=class_weight.to(rank))
    criterion_connectivity = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([20]).to(rank)

    running_losses, running_loss_connectivity, running_loss_relationship, connectivity_recall, connectivity_precision, \
    num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    recall_top3, recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs, wmap_rel, wmap_phrase = None, None, None, None, None, None, None, None

    Recall = Evaluator_PC(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])
    if args['dataset']['dataset'] == 'vg':
        Recall_top3 = Evaluator_PC_Top3(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])

    lr_decay = 1
    for epoch in range(args['training']['start_epoch'], args['training']['num_epoch']):
        print('Start Training... EPOCH %d / %d\n' % (epoch, args['training']['num_epoch']))
        if epoch == args['training']['scheduler_param1'] or epoch == args['training']['scheduler_param2']:  # lr scheduler
            lr_decay *= 0.1

        for batch_count, data in enumerate(tqdm(train_loader), 0):
            """
            PREPARE INPUT DATA
            """
            try:
                images, image_depth, triplets_list = data
            except:
                continue

            with torch.no_grad():
                detr.to(rank)
                images = torch.stack(images).to(rank)
                image_feature, pos_embed = detr.module.backbone(nested_tensor_from_tensor_list(images))
                src, mask = image_feature[-1].decompose()
                src = detr.module.input_proj(src).flatten(2).permute(2, 0, 1)
                pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
                image_feature = detr.module.transformer.encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)
                image_feature = image_feature.permute(1, 2, 0)
                image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])
                detr.to('cpu')
            del images

            # append corresponding image index to each triplet
            all_triplets = []
            for i, triplets in enumerate(triplets_list):
                for triplet in triplets:
                    triplet.append(i)
                    all_triplets.append(triplet)

            all_image_idx = torch.tensor([triplet[-1] for triplet in all_triplets]).to(rank)
            # index_counts = torch.bincount(all_image_idx)

            image_depth = torch.stack(image_depth).to(rank)
            image_feature = torch.cat((image_feature, image_depth), dim=1)

            # # duplicate the image features based on the index tensor
            # new_image_feature = torch.empty(len(all_image_idx), *image_feature.shape[1:]).to(rank)
            # current_idx = 0
            # for idx, count in enumerate(index_counts):
            #     new_image_feature[current_idx:current_idx + count] = image_feature[idx].repeat(count, 1, 1, 1)
            #     current_idx += count
            # image_feature = new_image_feature

            all_sub_categories = torch.tensor([triplet[0] for triplet in all_triplets]).to(rank)
            all_obj_categories = torch.tensor([triplet[5] for triplet in all_triplets]).to(rank)
            all_sub_super_categories = [[sc.to(rank) for sc in triplet[1]] for triplet in all_triplets]
            all_obj_super_categories = [[sc.to(rank) for sc in triplet[6]] for triplet in all_triplets]
            all_sub_bboxes = torch.stack([triplet[2] for triplet in all_triplets]).long().to(rank)
            all_obj_bboxes = torch.stack([triplet[7] for triplet in all_triplets]).long().to(rank)

            all_sub_masks = torch.zeros(len(all_triplets), args['models']['feature_size'], args['models']['feature_size'], dtype=torch.uint8).to(rank)
            all_obj_masks = torch.zeros(len(all_triplets), args['models']['feature_size'], args['models']['feature_size'], dtype=torch.uint8).to(rank)
            for i in range(len(all_triplets)):
                all_sub_masks[i, int(all_sub_bboxes[i][2]):int(all_sub_bboxes[i][3]), int(all_sub_bboxes[i][0]):int(all_sub_bboxes[i][1])] = 1
                all_obj_masks[i, int(all_obj_bboxes[i][2]):int(all_obj_bboxes[i][3]), int(all_obj_bboxes[i][0]):int(all_obj_bboxes[i][1])] = 1

            all_img_features = torch.zeros(len(all_triplets), args['models']['num_img_feature'] + 1, args['models']['feature_size'], args['models']['feature_size']).to(rank)
            all_sub_features = torch.zeros(len(all_triplets), args['models']['num_img_feature'] + 1, args['models']['feature_size'], args['models']['feature_size']).to(rank)
            all_obj_features = torch.zeros(len(all_triplets), args['models']['num_img_feature'] + 1, args['models']['feature_size'], args['models']['feature_size']).to(rank)
            for i in range(len(all_triplets)):
                image_idx = all_triplets[i][-1]
                all_img_features[i] = image_feature[image_idx]
                all_sub_features[i] = all_img_features[i] * all_sub_masks[i]
                all_obj_features[i] = all_img_features[i] * all_obj_masks[i]
            del image_depth, image_feature, all_img_features

            all_relations_target = torch.stack([triplet[4].to(rank) for triplet in all_triplets])

            loss_connectivity, loss_relationship = 0.0, 0.0
            if args['models']['hierarchical_pred']:
                # relation_1, relation_2, relation_3, super_relation, connectivity = transformer_encoder(image_feature, all_sub_bboxes, all_obj_bboxes,
                #                                     all_sub_categories, all_obj_categories, all_sub_super_categories, all_obj_super_categories, rank)
                relation_1, relation_2, relation_3, super_relation, connectivity = transformer_encoder(all_sub_features, all_obj_features, all_sub_categories, all_obj_categories,
                                                                                             all_sub_super_categories, all_obj_super_categories, rank)
                relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
            else:
                relation, connectivity = transformer_encoder(all_sub_features, all_obj_features, all_sub_categories, all_obj_categories,
                                                   all_sub_super_categories, all_obj_super_categories, rank)
                super_relation = None

            # print("relation", relation.shape, "all_relations_target", all_relations_target.shape, "all_sub_categories", all_sub_categories.shape,
            #       "all_sub_masks", all_sub_masks.shape, "all_sub_bboxes", all_sub_bboxes.shape)

            if args['models']['hierarchical_pred']:
                super_relation_target = all_relations_target.clone()
                super_relation_target[super_relation_target < args['models']['num_geometric']] = 0
                super_relation_target[torch.logical_and(super_relation_target >= args['models']['num_geometric'],
                                                        super_relation_target < args['models']['num_geometric'] + args['models']['num_possessive'])] = 1
                super_relation_target[super_relation_target >= args['models']['num_geometric'] + args['models']['num_possessive']] = 2
                loss_relationship += criterion_super_relationship(super_relation, super_relation_target)

                connected_1 = torch.nonzero(all_relations_target < args['models']['num_geometric']).flatten()  # geometric
                connected_2 = torch.nonzero(torch.logical_and(all_relations_target >= args['models']['num_geometric'],
                                                              all_relations_target < args['models']['num_geometric'] + args['models']['num_possessive'])).flatten()  # possessive
                connected_3 = torch.nonzero(all_relations_target >= args['models']['num_geometric'] + args['models']['num_possessive']).flatten()  # semantic
                if len(connected_1) > 0:
                    loss_relationship += criterion_relationship_1(relation_1[connected_1], all_relations_target[connected_1])
                if len(connected_2) > 0:
                    loss_relationship += criterion_relationship_2(relation_2[connected_2], all_relations_target[connected_2] - args['models']['num_geometric'])
                if len(connected_3) > 0:
                    loss_relationship += criterion_relationship_3(relation_3[connected_3], all_relations_target[connected_3] - args['models']['num_geometric'] - args['models']['num_possessive'])
            else:
                loss_relationship += criterion_relationship(relation, all_relations_target)
            running_losses += loss_relationship.item()

            if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                Recall.accumulate(all_image_idx, relation, all_relations_target, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                  all_sub_categories, all_obj_categories, all_sub_categories, all_obj_categories,
                                  all_sub_bboxes, all_obj_bboxes, all_sub_bboxes, all_obj_bboxes)
                if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                    Recall_top3.accumulate(all_image_idx, relation, all_relations_target, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                           all_sub_categories, all_obj_categories, all_sub_categories, all_obj_categories,
                                           all_sub_bboxes, all_obj_bboxes, all_sub_bboxes, all_obj_bboxes)

            """
            EVALUATE AND PRINT CURRENT TRAINING RESULTS
            """
            if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                if args['dataset']['dataset'] == 'vg':
                    recall, _, mean_recall, recall_zs, _, mean_recall_zs = Recall.compute(per_class=True)
                    if args['models']['hierarchical_pred']:
                        recall_top3, _, mean_recall_top3 = Recall_top3.compute(per_class=True)
                        Recall_top3.clear_data()
                else:
                    recall, _, mean_recall, _, _, _ = Recall.compute(per_class=True)
                    wmap_rel, wmap_phrase = Recall.compute_precision()
                Recall.clear_data()

            if (batch_count % args['training']['print_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                record_train_results(args, record, rank, epoch, batch_count, original_lr, lr_decay, recall_top3, recall, mean_recall_top3, mean_recall,
                                     recall_zs, mean_recall_zs, running_losses, running_loss_relationship, running_loss_connectivity, connectivity_recall,
                                     num_connected, num_not_connected, connectivity_precision, num_connected_pred, wmap_rel, wmap_phrase)
                dist.monitored_barrier()

            running_losses, running_loss_connectivity, running_loss_relationship, connectivity_recall, connectivity_precision, num_connected, num_not_connected = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if args['models']['hierarchical_pred']:
            torch.save(transformer_encoder.state_dict(), args['training']['checkpoint_path'] + 'TransEncoderHier' + str(epoch) + '_' + str(rank) + '.pth')
        else:
            torch.save(transformer_encoder.state_dict(), args['training']['checkpoint_path'] + 'TransEncoderFlat' + str(epoch) + '_' + str(rank) + '.pth')
        dist.monitored_barrier()

        if args['models']['detr_or_faster_rcnn'] == 'detr':
            test_local(args, detr, transformer_encoder, test_loader, test_record, epoch, rank)
        else:
            test_local(args, faster_rcnn, transformer_encoder, test_loader, test_record, epoch, rank)

    dist.destroy_process_group()  # clean up
    print('FINISHED TRAINING\n')


def test_local(args, backbone, transformer_encoder, test_loader, test_record, epoch, rank):
    backbone.eval()

    connectivity_recall, connectivity_precision, num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0
    recall_top3, recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs, wmap_rel, wmap_phrase = None, None, None, None, None, None, None, None
    Recall = Evaluator_PC(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])
    if args['dataset']['dataset'] == 'vg':
        Recall_top3 = Evaluator_PC_Top3(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])

    print('Start Testing PC...')
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            """
            PREPARE INPUT DATA
            """
            try:
                images, image_depth, triplets_list = data
            except:
                continue

            images = torch.stack(images).to(rank)
            image_feature, pos_embed = backbone.module.backbone(nested_tensor_from_tensor_list(images))
            src, mask = image_feature[-1].decompose()
            src = backbone.module.input_proj(src).flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
            image_feature = backbone.module.transformer.encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)
            image_feature = image_feature.permute(1, 2, 0)
            image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])
            del images

            # append corresponding image index to each triplet
            all_triplets = []
            for i, triplets in enumerate(triplets_list):
                for triplet in triplets:
                    triplet.append(i)
                    all_triplets.append(triplet)

            all_image_idx = torch.tensor([triplet[-1] for triplet in all_triplets]).to(rank)

            all_sub_categories = torch.tensor([triplet[0] for triplet in all_triplets]).to(rank)
            all_obj_categories = torch.tensor([triplet[5] for triplet in all_triplets]).to(rank)
            all_sub_super_categories = [[sc.to(rank) for sc in triplet[1]] for triplet in all_triplets]
            all_obj_super_categories = [[sc.to(rank) for sc in triplet[6]] for triplet in all_triplets]
            all_sub_bboxes = torch.stack([triplet[2] for triplet in all_triplets]).to(rank)
            all_obj_bboxes = torch.stack([triplet[7] for triplet in all_triplets]).to(rank)

            all_sub_masks = torch.zeros(len(all_triplets), args['models']['feature_size'], args['models']['feature_size'], dtype=torch.uint8).to(rank)
            all_obj_masks = torch.zeros(len(all_triplets), args['models']['feature_size'], args['models']['feature_size'], dtype=torch.uint8).to(rank)
            for i in range(len(all_triplets)):
                all_sub_masks[i, int(all_sub_bboxes[i][2]):int(all_sub_bboxes[i][3]), int(all_sub_bboxes[i][0]):int(all_sub_bboxes[i][1])] = 1
                all_obj_masks[i, int(all_obj_bboxes[i][2]):int(all_obj_bboxes[i][3]), int(all_obj_bboxes[i][0]):int(all_obj_bboxes[i][1])] = 1

            all_img_features = torch.zeros(len(all_triplets), args['models']['num_img_feature'] + 1, args['models']['feature_size'], args['models']['feature_size']).to(rank)
            all_sub_features = torch.zeros(len(all_triplets), args['models']['num_img_feature'] + 1, args['models']['feature_size'], args['models']['feature_size']).to(rank)
            all_obj_features = torch.zeros(len(all_triplets), args['models']['num_img_feature'] + 1, args['models']['feature_size'], args['models']['feature_size']).to(rank)
            for i in range(len(all_triplets)):
                image_idx = all_triplets[i][-1]
                all_img_features[i, :-1] = image_feature[image_idx].to(rank)
                all_img_features[i, -1] = image_depth[image_idx].to(rank)
                all_sub_features[i] = all_img_features[i] * all_sub_masks[i]
                all_obj_features[i] = all_img_features[i] * all_obj_masks[i]

            all_relations_target = torch.stack([triplet[4].to(rank) for triplet in all_triplets])

            """
            FIRST DIRECTION
            """
            if args['models']['hierarchical_pred']:
                relation_1, relation_2, relation_3, super_relation, connectivity = transformer_encoder(all_sub_features, all_obj_features, all_sub_categories, all_obj_categories,
                                                                                             all_sub_super_categories, all_obj_super_categories, rank)
                relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
            else:
                relation, connectivity = transformer_encoder(all_sub_features, all_obj_features, all_sub_categories, all_obj_categories,
                                                   all_sub_super_categories, all_obj_super_categories, rank)
                super_relation = None

            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                Recall.accumulate(all_image_idx, relation, all_relations_target, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                  all_sub_categories, all_obj_categories, all_sub_categories, all_obj_categories,
                                  all_sub_bboxes, all_obj_bboxes, all_sub_bboxes, all_obj_bboxes)
                if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                    Recall_top3.accumulate(all_image_idx, relation, all_relations_target, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                           all_sub_categories, all_obj_categories, all_sub_categories, all_obj_categories,
                                           all_sub_bboxes, all_obj_bboxes, all_sub_bboxes, all_obj_bboxes)
            """
            SECOND DIRECTION
            """
            if args['models']['hierarchical_pred']:
                relation_1, relation_2, relation_3, super_relation, connectivity = transformer_encoder(all_obj_features, all_sub_features, all_obj_categories, all_sub_categories,
                                                                                             all_obj_super_categories, all_sub_super_categories, rank)
                relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
            else:
                relation, connectivity = transformer_encoder(all_obj_features, all_sub_features, all_obj_categories, all_sub_categories,
                                                   all_obj_super_categories, all_sub_super_categories, rank)
                super_relation = None

            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                Recall.accumulate(all_image_idx, relation, all_relations_target, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                  all_obj_categories, all_sub_categories, all_obj_categories, all_sub_categories,
                                  all_obj_bboxes, all_sub_bboxes, all_obj_bboxes, all_sub_bboxes)
                if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                    Recall_top3.accumulate(all_image_idx, relation, all_relations_target, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                           all_obj_categories, all_sub_categories, all_obj_categories, all_sub_categories,
                                           all_obj_bboxes, all_sub_bboxes, all_obj_bboxes, all_sub_bboxes)

            """
            EVALUATE AND PRINT CURRENT RESULTS
            """
            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                if args['dataset']['dataset'] == 'vg':
                    if batch_count + 1 == len(test_loader):
                        recall, recall_k_per_class, mean_recall, recall_zs, _, mean_recall_zs = Recall.compute(per_class=True)
                        print("recall_k_per_class", recall_k_per_class)
                    else:
                        recall, _, mean_recall, recall_zs, _, mean_recall_zs = Recall.compute(per_class=True)
                    if args['models']['hierarchical_pred']:
                        recall_top3, _, mean_recall_top3 = Recall_top3.compute(per_class=True)
                        Recall_top3.clear_data()
                else:
                    recall, _, mean_recall, _, _, _ = Recall.compute(per_class=True)
                    wmap_rel, wmap_phrase = Recall.compute_precision()
                Recall.clear_data()

            if (batch_count % args['training']['print_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                record_test_results(args, test_record, rank, epoch, recall_top3, recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs,
                                    connectivity_recall, num_connected, num_not_connected, connectivity_precision, num_connected_pred, wmap_rel, wmap_phrase)
                dist.monitored_barrier()
    print('FINISHED EVALUATING\n')