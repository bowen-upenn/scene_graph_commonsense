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
from model import EdgeHead, EdgeHeadHier
from utils import *


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_local(gpu, args, train_subset, test_subset):
    """
    Model Inputs:
        image_feature:     the image feature map from Mask R-CNN
                              size [num_img_feature, feature_size, feature_size]
        image_depth:       the image depth map from single-image depth estimation
                              size [1, feature_size, feature_size]
        list_seg_masks:    list of segmentation masks for each instance detected by Mask R-CNN
                           size [num_obj][1, feature_size, feature_size]
    """
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

    if args['models']['hierarchical_pred']:
        edge_head = DDP(EdgeHeadHier(input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
                                     num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'])).to(rank)
    else:
        edge_head = DDP(EdgeHead(input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'], feature_size=args['models']['feature_size'],
                                 num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'])).to(rank)

    detr = DDP(build_detr101(args)).to(rank)
    backbone = DDP(detr.module.backbone).to(rank)
    input_proj = DDP(detr.module.input_proj).to(rank)
    feature_encoder = DDP(detr.module.transformer.encoder).to(rank)

    detr.eval()
    edge_head.train()
    backbone.eval()
    input_proj.eval()
    feature_encoder.eval()

    map_location = {'cuda:%d' % rank: 'cuda:%d' % rank}
    if args['training']['continue_train']:
        if args['models']['hierarchical_pred']:
            edge_head.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHeadHier' + str(args['training']['start_epoch'] - 1) + '_' + str(rank) + '.pth', map_location=map_location))
        else:
            edge_head.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHead' + str(args['training']['start_epoch'] - 1) + '_' + str(rank) + '.pth', map_location=map_location))

    optimizer = optim.SGD([{'params': edge_head.parameters(), 'initial_lr': args['training']['learning_rate']}],
                          lr=args['training']['learning_rate'], momentum=0.9, weight_decay=args['training']['weight_decay'])
    original_lr = optimizer.param_groups[0]["lr"]

    relation_count = get_num_each_class_reordered()
    class_weight = 1 - relation_count / torch.sum(relation_count)
    criterion_relationship_1 = torch.nn.NLLLoss(weight=class_weight[:15].to(rank))      # log softmax already applied
    criterion_relationship_2 = torch.nn.NLLLoss(weight=class_weight[15:26].to(rank))
    criterion_relationship_3 = torch.nn.NLLLoss(weight=class_weight[26:].to(rank))
    criterion_super_relationship = torch.nn.NLLLoss()
    criterion_relationship = torch.nn.CrossEntropyLoss(weight=class_weight.to(rank))
    criterion_connectivity = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([20]).to(rank)

    running_losses, running_loss_connectivity, running_loss_relationship, connectivity_recall, connectivity_precision, \
        num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    Recall = Evaluator_PC(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])
    Recall_top3 = Evaluator_PC_Top3(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])

    lr_decay = 1
    for epoch in range(args['training']['start_epoch'], args['training']['num_epoch']):
        print('Start Training... EPOCH %d / %d\n' % (epoch, args['training']['num_epoch']))
        if epoch == args['training']['scheduler_param1'] or epoch == args['training']['scheduler_param2']:  # lr scheduler
            lr_decay *= 0.1

        for batch_count, data in enumerate(tqdm(train_loader), 0):
            '''
            PREPARE INPUT DATA
            '''
            _, images, image_depth, categories, super_categories, masks, bbox, relationships, subj_or_obj = data

            images = [image.to(rank) for image in images]
            with torch.no_grad():
                image_feature, pos_embed = backbone(nested_tensor_from_tensor_list(images))
                src, mask = image_feature[-1].decompose()
                src = input_proj(src).flatten(2).permute(2, 0, 1)
                pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
                image_feature = feature_encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)
            image_feature = image_feature.permute(1, 2, 0)
            image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])

            image_depth = torch.stack([depth.to(rank) for depth in image_depth])
            categories = [category.to(rank) for category in categories]  # [batch_size][curr_num_obj, 1]
            super_categories = [[sc.to(rank) for sc in super_category] for super_category in super_categories]  # [batch_size][curr_num_obj, [1 or more]]
            bbox = [box.to(rank) for box in bbox]  # [batch_size][curr_num_obj, 4]
            optimizer.param_groups[0]["lr"] = original_lr

            masks = []
            for i in range(len(bbox)):
                mask = torch.zeros(bbox[i].shape[0], 32, 32, dtype=torch.uint8).to(rank)
                for j, box in enumerate(bbox[i]):
                    mask[j, int(bbox[i][j][2]):int(bbox[i][j][3]), int(bbox[i][j][0]):int(bbox[i][j][1])] = 1
                masks.append(mask)

            '''
            PREPARE TARGETS
            Different data samples in a batch have different num of RNN iterations, i.e., different number of objects in images. To do mini-batch parallel training,
            need to convert size [batch_size][curr_num_obj][from 1 to curr_num_obj-1] to [max_num_obj][from 1 to max_num_obj-1][<=batch_size]
            '''
            relations_target = []
            direction_target = []
            num_graph_iter = torch.as_tensor([len(mask) for mask in masks]) - 1
            for graph_iter in range(max(num_graph_iter)):
                which_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)
                relations_target.append(torch.vstack([relationships[i][graph_iter] for i in which_in_batch]).T.to(rank))  # integer labels
                direction_target.append(torch.vstack([subj_or_obj[i][graph_iter] for i in which_in_batch]).T.to(rank))

            '''
            FORWARD PASS
            Different data samples in a batch have different num of RNN iterations, i.e., different number of objects in images.
            To do mini-batch parallel training, in each graph-level iteration,
            only stack the data samples whose total num of RNN iterations not exceeding the current iteration.
            '''
            num_graph_iter = torch.as_tensor([len(mask) for mask in masks])
            for graph_iter in range(max(num_graph_iter)):
                which_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)
                optimizer.param_groups[0]["lr"] = original_lr * lr_decay * math.sqrt(len(which_in_batch) / len(num_graph_iter))  # dynamic batch size needs dynamic learning rate

                curr_graph_masks = torch.stack([torch.unsqueeze(masks[i][graph_iter], dim=0) for i in which_in_batch])
                h_graph = torch.cat((image_feature[which_in_batch] * curr_graph_masks, image_depth[which_in_batch] * curr_graph_masks), dim=1)  # (bs, 256, 64, 64), (bs, 1, 64, 64)
                cat_graph = torch.tensor([torch.unsqueeze(categories[i][graph_iter], dim=0) for i in which_in_batch]).to(rank)
                scat_graph = [super_categories[i][graph_iter] for i in which_in_batch]
                bbox_graph = torch.stack([bbox[i][graph_iter] for i in which_in_batch]).to(rank)

                for edge_iter in range(graph_iter):
                    curr_edge_masks = torch.stack([torch.unsqueeze(masks[i][edge_iter], dim=0) for i in which_in_batch])  # seg mask of every prev obj
                    h_edge = torch.cat((image_feature[which_in_batch] * curr_edge_masks, image_depth[which_in_batch] * curr_edge_masks), dim=1)
                    cat_edge = torch.tensor([torch.unsqueeze(categories[i][edge_iter], dim=0) for i in which_in_batch]).to(rank)
                    scat_edge = [super_categories[i][edge_iter] for i in which_in_batch]
                    bbox_edge = torch.stack([bbox[i][edge_iter] for i in which_in_batch]).to(rank)

                    """
                    FIRST DIRECTION
                    """
                    loss_connectivity, loss_relationship = 0.0, 0.0
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity = edge_head(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)    # <graph subject, rel, edge object>
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = edge_head(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
                        super_relation = None

                    not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 1)[0]  # which data samples in curr which_in_batch are not connected
                    num_not_connected += len(not_connected)
                    temp = criterion_connectivity(connectivity[not_connected, 0], torch.zeros(len(not_connected)).to(rank))
                    loss_connectivity += 0.0 if torch.isnan(temp) else args['training']['lambda_not_connected'] * temp

                    connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 1)[0]  # which data samples in curr which_in_batch are connected
                    num_connected += len(connected)
                    connected_pred = torch.nonzero(torch.sigmoid(connectivity[:, 0]) >= 0.5).flatten()
                    connectivity_precision += torch.sum(relations_target[graph_iter - 1][edge_iter][connected_pred] != -1)
                    num_connected_pred += len(connected_pred)

                    if len(connected) > 0:
                        temp = criterion_connectivity(connectivity[connected, 0], torch.ones(len(connected)).to(rank))
                        loss_connectivity += 0.0 if torch.isnan(temp) else temp
                        connectivity_recall += torch.sum(torch.round(torch.sigmoid(connectivity[connected, 0])))

                        if args['models']['hierarchical_pred']:
                            super_relation_target = relations_target[graph_iter - 1][edge_iter][connected].clone()
                            super_relation_target[super_relation_target < 15] = 0
                            super_relation_target[torch.logical_and(super_relation_target >= 15, super_relation_target < 26)] = 1
                            super_relation_target[super_relation_target >= 26] = 2
                            loss_relationship += criterion_super_relationship(super_relation[connected], super_relation_target)

                            connected_1 = torch.nonzero(relations_target[graph_iter - 1][edge_iter][connected] < 15).flatten()                       # geometric
                            connected_2 = torch.nonzero(torch.logical_and(relations_target[graph_iter - 1][edge_iter][connected] >= 15,
                                                                          relations_target[graph_iter - 1][edge_iter][connected] < 26)).flatten()    # possessive
                            connected_3 = torch.nonzero(relations_target[graph_iter - 1][edge_iter][connected] >= 26).flatten()                      # semantic
                            if len(connected_1) > 0:
                                loss_relationship += criterion_relationship_1(relation_1[connected][connected_1], relations_target[graph_iter - 1][edge_iter][connected][connected_1])
                            if len(connected_2) > 0:
                                loss_relationship += criterion_relationship_2(relation_2[connected][connected_2], relations_target[graph_iter - 1][edge_iter][connected][connected_2] - 15)
                            if len(connected_3) > 0:
                                loss_relationship += criterion_relationship_3(relation_3[connected][connected_3], relations_target[graph_iter - 1][edge_iter][connected][connected_3] - 26)
                        else:
                            loss_relationship += criterion_relationship(relation[connected], relations_target[graph_iter - 1][edge_iter][connected])

                    # evaluate recall@k scores
                    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                    relations_target_directed[not_connected] = -1

                    if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                        Recall.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                          cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                        if args['models']['hierarchical_pred']:
                            Recall_top3.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                                   cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

                    losses = loss_relationship + args['training']['lambda_connectivity'] * (loss_connectivity + args['training']['lambda_sparsity'] * torch.linalg.norm(torch.sigmoid(connectivity), ord=1))
                    running_loss_connectivity += args['training']['lambda_connectivity'] * (loss_connectivity + args['training']['lambda_sparsity'] * torch.linalg.norm(torch.sigmoid(connectivity), ord=1))
                    running_loss_relationship += loss_relationship
                    running_losses += losses.item()

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    """
                    SECOND DIRECTION
                    """
                    loss_connectivity, loss_relationship = 0.0, 0.0
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity = edge_head(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)    # <edge subject, rel, graph object>
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = edge_head(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)
                        super_relation = None

                    not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 0)[0]  # which data samples in curr which_in_batch are not connected
                    num_not_connected += len(not_connected)
                    temp = criterion_connectivity(connectivity[not_connected, 0], torch.zeros(len(not_connected)).to(rank))
                    loss_connectivity += 0.0 if torch.isnan(temp) else args['training']['lambda_not_connected'] * temp

                    connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 0)[0]  # which data samples in curr which_in_batch are connected
                    num_connected += len(connected)
                    connected_pred = torch.nonzero(torch.sigmoid(connectivity[:, 0]) >= 0.5).flatten()
                    connectivity_precision += torch.sum(relations_target[graph_iter - 1][edge_iter][connected_pred] != -1)
                    num_connected_pred += len(connected_pred)

                    if len(connected) > 0:
                        temp = criterion_connectivity(connectivity[connected, 0], torch.ones(len(connected)).to(rank))
                        loss_connectivity += 0.0 if torch.isnan(temp) else temp
                        connectivity_recall += torch.sum(torch.round(torch.sigmoid(connectivity[connected, 0])))

                        if args['models']['hierarchical_pred']:
                            super_relation_target = relations_target[graph_iter - 1][edge_iter][connected].clone()
                            super_relation_target[super_relation_target < 15] = 0
                            super_relation_target[torch.logical_and(super_relation_target >= 15, super_relation_target < 26)] = 1
                            super_relation_target[super_relation_target >= 26] = 2
                            loss_relationship += criterion_super_relationship(super_relation[connected], super_relation_target)

                            connected_1 = torch.nonzero(relations_target[graph_iter - 1][edge_iter][connected] < 15).flatten()                     # geometric
                            connected_2 = torch.nonzero(torch.logical_and(relations_target[graph_iter - 1][edge_iter][connected] >= 15,
                                                                          relations_target[graph_iter - 1][edge_iter][connected] < 26)).flatten()  # possessive
                            connected_3 = torch.nonzero(relations_target[graph_iter - 1][edge_iter][connected] >= 26).flatten()                    # semantic
                            if len(connected_1) > 0:
                                loss_relationship += criterion_relationship_1(relation_1[connected][connected_1], relations_target[graph_iter - 1][edge_iter][connected][connected_1])
                            if len(connected_2) > 0:
                                loss_relationship += criterion_relationship_2(relation_2[connected][connected_2], relations_target[graph_iter - 1][edge_iter][connected][connected_2] - 15)
                            if len(connected_3) > 0:
                                loss_relationship += criterion_relationship_3(relation_3[connected][connected_3], relations_target[graph_iter - 1][edge_iter][connected][connected_3] - 26)
                        else:
                            loss_relationship += criterion_relationship(relation[connected], relations_target[graph_iter - 1][edge_iter][connected])

                    # evaluate recall@k scores
                    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                    relations_target_directed[not_connected] = -1

                    if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                        Recall.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                          cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                        if args['models']['hierarchical_pred']:
                            Recall_top3.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                                   cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

                    losses = loss_relationship + args['training']['lambda_connectivity'] * (loss_connectivity + args['training']['lambda_sparsity'] * torch.linalg.norm(torch.sigmoid(connectivity), ord=1))
                    running_loss_connectivity += args['training']['lambda_connectivity'] * (loss_connectivity + args['training']['lambda_sparsity'] * torch.linalg.norm(torch.sigmoid(connectivity), ord=1))
                    running_loss_relationship += loss_relationship
                    running_losses += losses.item()

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

            '''
            EVALUATE AND PRINT CURRENT TRAINING RESULTS
            '''
            if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                recall, _, mean_recall = Recall.compute(args['models']['hierarchical_pred'], per_class=True)
                if args['models']['hierarchical_pred']:
                    recall_top3, _, mean_recall_top3 = Recall_top3.compute(args['models']['hierarchical_pred'], per_class=True)
                Recall.clear_data()
                Recall_top3.clear_data()

            if (batch_count % args['training']['print_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                if args['models']['hierarchical_pred']:
                    print('TRAINING, rank %d, epoch %d, batch %d, lr: %.4f, R@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), mR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), loss: %.4f, %.4f, conn: %.4f, %.4f.'
                          % (rank, epoch, batch_count, original_lr * lr_decay,
                             recall_top3[0], recall_top3[1], recall_top3[2], recall[0], recall[1], recall[2],
                             mean_recall_top3[0], mean_recall_top3[1], mean_recall_top3[2], mean_recall[0], mean_recall[1], mean_recall[2],
                             running_loss_relationship / (args['training']['print_freq'] * args['training']['batch_size']),
                             running_loss_connectivity / (args['training']['print_freq'] * args['training']['batch_size']),
                             connectivity_recall / (num_connected+1e-5), connectivity_precision / (num_connected_pred+1e-5)))
                else:
                    print('TRAINING, rank %d, epoch %d, batch %d, lr: %.4f, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, loss: %.4f, %.4f, conn: %.4f, %.4f.'
                          % (rank, epoch, batch_count, original_lr * lr_decay,
                             recall[0], recall[1], recall[2],
                             mean_recall[0], mean_recall[1], mean_recall[2],
                             running_loss_relationship / (args['training']['print_freq'] * args['training']['batch_size']),
                             running_loss_connectivity / (args['training']['print_freq'] * args['training']['batch_size']),
                             connectivity_recall / (num_connected + 1e-5), connectivity_precision / (num_connected_pred + 1e-5)))

                record.append({'rank': rank, 'epoch': epoch, 'batch': batch_count, 'lr': original_lr * lr_decay,
                               'recall_relationship': [recall[0], recall[1], recall[2]],
                               'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                               'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                               'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                               'connectivity_recall': connectivity_recall.item() / (num_connected+1e-5), 'connectivity_precision': connectivity_precision.item() / (num_connected_pred+1e-5),
                               'total_losses': running_losses / (args['training']['print_freq'] * args['training']['batch_size']),
                               'relationship_loss': running_loss_relationship.item() / (args['training']['print_freq'] * args['training']['batch_size']),
                               'connectivity_loss': running_loss_connectivity.item() / (args['training']['print_freq'] * args['training']['batch_size']),
                               'num_objs_average': torch.mean(num_graph_iter.float()).item(), 'num_objs_max': max(num_graph_iter).item(),
                               'num_connected': num_connected, 'num_not_connected': num_not_connected})
                with open(args['training']['result_path'] + 'train_results_' + str(rank) + '.json', 'w') as f:  # append current logs, must use "w" not "a"
                    json.dump(record, f)
                dist.monitored_barrier()

            running_losses, running_loss_connectivity, running_loss_relationship, connectivity_recall, connectivity_precision, num_connected, num_not_connected = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if args['models']['hierarchical_pred']:
            torch.save(edge_head.state_dict(), args['training']['checkpoint_path'] + 'EdgeHeadHier' + str(epoch) + '_' + str(rank) + '.pth')
        else:
            torch.save(edge_head.state_dict(), args['training']['checkpoint_path'] + 'EdgeHead' + str(epoch) + '_' + str(rank) + '.pth')
        dist.monitored_barrier()

        test_local(args, edge_head, backbone, input_proj, feature_encoder, test_loader, test_record, epoch, rank)

    dist.destroy_process_group()  # clean up
    print('FINISHED TRAINING\n')


def test_local(args, edge_head, backbone, input_proj, feature_encoder, test_loader, test_record, epoch, rank):
    """
    Model Inputs:
        image_feature:     the image feature map from Mask R-CNN
                              size [num_img_feature, feature_size, feature_size]
        image_depth:       the image depth map from single-image depth estimation
                              size [1, feature_size, feature_size]
        list_seg_masks:    list of segmentation masks for each instance detected by Mask R-CNN
                           size [num_obj][1, feature_size, feature_size]
    """
    edge_head.eval()
    feature_encoder.eval()

    connectivity_recall, connectivity_precision, num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0
    Recall = Evaluator_PC(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])
    Recall_top3 = Evaluator_PC_Top3(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])

    print('Start Testing PC...')
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            '''
            PREPARE INPUT DATA
            '''
            _, images_s, image_depth, categories, super_categories, masks, bbox, relationships, subj_or_obj = data

            images_s = [image.to(rank) for image in images_s]
            image_feature, pos_embed = backbone(nested_tensor_from_tensor_list(images_s))
            src, mask = image_feature[-1].decompose()
            src = input_proj(src).flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
            image_feature = feature_encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)
            image_feature = image_feature.permute(1, 2, 0)
            image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])

            categories = [category.to(rank) for category in categories]  # [batch_size][curr_num_obj, 1]
            super_categories = [[sc.to(rank) for sc in super_category] for super_category in super_categories]  # [batch_size][curr_num_obj, [1 or more]]
            image_depth = torch.stack([depth.to(rank) for depth in image_depth])
            bbox = [box.to(rank) for box in bbox]  # [batch_size][curr_num_obj, 4]

            masks = []
            for i in range(len(bbox)):
                mask = torch.zeros(bbox[i].shape[0], 32, 32, dtype=torch.uint8).to(rank)
                for j, box in enumerate(bbox[i]):
                    mask[j, int(bbox[i][j][2]):int(bbox[i][j][3]), int(bbox[i][j][0]):int(bbox[i][j][1])] = 1
                masks.append(mask)

            '''
            PREPARE TARGETS
            Different data samples in a batch have different num of RNN iterations, i.e., different number of objects in images. To do mini-batch parallel training, 
            need to convert size [batch_size][curr_num_obj][from 1 to curr_num_obj-1] to [max_num_obj][from 1 to max_num_obj-1][<=batch_size]
            '''
            relations_target = []
            direction_target = []
            num_graph_iter = torch.as_tensor([len(mask) for mask in masks]) - 1
            for graph_iter in range(max(num_graph_iter)):
                which_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)
                relations_target.append(torch.vstack([relationships[i][graph_iter] for i in which_in_batch]).T.to(rank))  # integer labels
                direction_target.append(torch.vstack([subj_or_obj[i][graph_iter] for i in which_in_batch]).T.to(rank))

            '''
            FORWARD PASS
            Different data samples in a batch have different num of RNN iterations, i.e., different number of objects in images.
            To do mini-batch parallel training, in each graph-level iteration, 
            only stack the data samples whose total num of RNN iterations not exceeding the current iteration.
            '''
            num_graph_iter = torch.as_tensor([len(mask) for mask in masks])
            for graph_iter in range(max(num_graph_iter)):
                which_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)

                curr_graph_masks = torch.stack([torch.unsqueeze(masks[i][graph_iter], dim=0) for i in which_in_batch])
                h_graph = torch.cat((image_feature[which_in_batch] * curr_graph_masks, image_depth[which_in_batch] * curr_graph_masks), dim=1)  # (bs, 256, 64, 64), (bs, 1, 64, 64)
                cat_graph = torch.tensor([torch.unsqueeze(categories[i][graph_iter], dim=0) for i in which_in_batch]).to(rank)
                scat_graph = [super_categories[i][graph_iter] for i in which_in_batch]
                bbox_graph = torch.stack([bbox[i][graph_iter] for i in which_in_batch]).to(rank)

                for edge_iter in range(graph_iter):
                    curr_edge_masks = torch.stack([torch.unsqueeze(masks[i][edge_iter], dim=0) for i in which_in_batch])  # seg mask of every prev obj
                    h_edge = torch.cat((image_feature[which_in_batch] * curr_edge_masks, image_depth[which_in_batch] * curr_edge_masks), dim=1)
                    cat_edge = torch.tensor([torch.unsqueeze(categories[i][edge_iter], dim=0) for i in which_in_batch]).to(rank)
                    scat_edge = [super_categories[i][edge_iter] for i in which_in_batch]
                    bbox_edge = torch.stack([bbox[i][edge_iter] for i in which_in_batch]).to(rank)

                    """
                    FIRST DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity = edge_head(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = edge_head(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
                        super_relation = None

                    not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 1)[0]  # which data samples in curr which_in_batch are not connected
                    num_not_connected += len(not_connected)
                    connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 1)[0]  # which data samples in curr which_in_batch are connected
                    num_connected += len(connected)
                    connected_pred = torch.nonzero(torch.sigmoid(connectivity[:, 0]) >= 0.5).flatten()
                    connectivity_precision += torch.sum(relations_target[graph_iter - 1][edge_iter][connected_pred] != -1)
                    num_connected_pred += len(connected_pred)
                    if len(connected) > 0:
                        connectivity_recall += torch.sum(torch.round(torch.sigmoid(connectivity[connected, 0])))

                    # evaluate recall@k scores
                    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                    relations_target_directed[not_connected] = -1

                    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                        Recall.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                          cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                        if args['models']['hierarchical_pred']:
                            Recall_top3.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                                   cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

                    """
                    SECOND DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity = edge_head(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = edge_head(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)
                        super_relation = None

                    not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 0)[0]  # which data samples in curr which_in_batch are not connected
                    num_not_connected += len(not_connected)
                    connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 0)[0]  # which data samples in curr which_in_batch are connected
                    num_connected += len(connected)
                    connected_pred = torch.nonzero(torch.sigmoid(connectivity[:, 0]) >= 0.5).flatten()
                    connectivity_precision += torch.sum(relations_target[graph_iter - 1][edge_iter][connected_pred] != -1)
                    num_connected_pred += len(connected_pred)
                    if len(connected) > 0:
                        connectivity_recall += torch.sum(torch.round(torch.sigmoid(connectivity[connected, 0])))

                    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                    relations_target_directed[not_connected] = -1

                    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                        Recall.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                          cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                        if args['models']['hierarchical_pred']:
                            Recall_top3.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                                   cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

            '''
            EVALUATE AND PRINT CURRENT TRAINING RESULTS
            '''
            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                recall, _, mean_recall = Recall.compute(args['models']['hierarchical_pred'], per_class=True)
                if args['models']['hierarchical_pred']:
                    recall_top3, _, mean_recall_top3 = Recall_top3.compute(args['models']['hierarchical_pred'], per_class=True)
                Recall.clear_data()
                Recall_top3.clear_data()

            if (batch_count % args['training']['print_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                if args['models']['hierarchical_pred']:
                    print('VALIDATING, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), mR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), conn: %.4f, %.4f.'
                          % (rank, epoch, recall_top3[0], recall_top3[1], recall_top3[2], recall[0], recall[1], recall[2],
                             mean_recall_top3[0], mean_recall_top3[1], mean_recall_top3[2], mean_recall[0], mean_recall[1], mean_recall[2],
                             connectivity_recall/(num_connected + 1e-5), connectivity_precision / (num_connected_pred+1e-5)))
                else:
                    print('VALIDATING, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, conn: %.4f, %.4f.'
                          % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2],
                             connectivity_recall / (num_connected + 1e-5), connectivity_precision / (num_connected_pred + 1e-5)))

                test_record.append({'rank': rank, 'epoch': epoch, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                   'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                                   'connectivity_recall': connectivity_recall.item() / (num_connected + 1e-5),
                                   'connectivity_precision': connectivity_precision.item() / (num_connected_pred + 1e-5),
                                   'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                   'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                                   'num_objs_average': torch.mean(num_graph_iter.float()).item(), 'num_objs_max': max(num_graph_iter).item(),
                                   'num_connected': num_connected, 'num_not_connected': num_not_connected})
                with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # append current logs
                    json.dump(test_record, f)

    dist.monitored_barrier()
    print('FINISHED EVALUATING\n')
