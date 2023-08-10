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
from model import *
from utils import *
from sup_contrast.losses import SupConLoss, SupConLossHierar


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_local(gpu, args, train_subset, test_subset):
    """
    This function trains and evaluates the local prediction module on predicate classification tasks.
    :param gpu: current gpu index
    :param args: input arguments in config.yaml
    :param train_subset: training dataset
    :param test_subset: testing dataset
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
        local_predictor = DDP(HierMotif(args=args, input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
                                        num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'],
                                        num_geometric=args['models']['num_geometric'], num_possessive=args['models']['num_possessive'],
                                        num_semantic=args['models']['num_semantic'])).to(rank)
        if args['models']['add_transformer']:
            transformer_encoder = DDP(TransformerEncoder(d_model=args['models']['d_model'], nhead=args['models']['nhead'], num_layers=args['models']['num_layers'],
                                                         dim_feedforward=args['models']['dim_feedforward'], dropout=args['models']['dropout'], num_geometric=args['models']['num_geometric'],
                                                         num_possessive=args['models']['num_possessive'], num_semantic=args['models']['num_semantic'], hierar=True)).to(rank)
    else:
        local_predictor = DDP(FlatMotif(args=args, input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'], feature_size=args['models']['feature_size'],
                                        num_classes=args['models']['num_classes'])).to(rank)
        if args['models']['add_transformer']:
            transformer_encoder = DDP(TransformerEncoder(d_model=args['models']['d_model'], nhead=args['models']['nhead'], num_layers=args['models']['num_layers'],
                                                         dim_feedforward=args['models']['dim_feedforward'], dropout=args['models']['dropout'],
                                                         output_dim=args['models']['num_relations'], hierar=False)).to(rank)

    if args['models']['detr_or_faster_rcnn'] == 'detr':
        detr = DDP(build_detr101(args)).to(rank)
        detr.eval()
    else:
        print('Unknown model.')

    map_location = {'cuda:%d' % rank: 'cuda:%d' % 0}
    if args['training']['continue_train']:
        if args['models']['hierarchical_pred']:
            local_predictor.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'HierMotif' + str(args['training']['start_epoch'] - 1) + '_0' + '.pth', map_location=map_location))
            if args['models']['add_transformer']:
                transformer_encoder.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'TransEncoder' + str(args['training']['start_epoch'] - 1) + '_0' + '.pth', map_location=map_location))
        else:
            local_predictor.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'FlatMotif' + str(args['training']['start_epoch'] - 1) + '_0' + '.pth', map_location=map_location))
            if args['models']['add_transformer']:
                transformer_encoder.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'FlatTransEncoder' + str(args['training']['start_epoch'] - 1) + '_0' + '.pth', map_location=map_location))

    if args['models']['add_transformer']:
        optimizer = optim.SGD([{'params': list(local_predictor.parameters()) + list(transformer_encoder.parameters()), 'initial_lr': args['training']['learning_rate']}],
                              lr=args['training']['learning_rate'], momentum=0.9, weight_decay=args['training']['weight_decay'])
    else:
        optimizer = optim.SGD([{'params': local_predictor.parameters(), 'initial_lr': args['training']['learning_rate']}],
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
    criterion_contrast = SupConLossHierar()
    criterion_connectivity = torch.nn.BCEWithLogitsLoss()

    running_losses, running_loss_connectivity, running_loss_relationship, running_loss_contrast, running_loss_transformer, connectivity_recall, connectivity_precision, \
    num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
            images, images_aug, image_depth, categories, super_categories, bbox, relationships, subj_or_obj = data
            batch_size = len(images)

            with torch.no_grad():
                images = torch.stack(images).to(rank)
                image_feature, pos_embed = detr.module.backbone(nested_tensor_from_tensor_list(images))
                src, mask = image_feature[-1].decompose()
                src = detr.module.input_proj(src).flatten(2).permute(2, 0, 1)
                pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
                image_feature = detr.module.transformer.encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)
                image_feature = image_feature.permute(1, 2, 0)
                image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])

                images_aug = torch.stack(images_aug).to(rank)
                image_feature_aug, pos_embed = detr.module.backbone(nested_tensor_from_tensor_list(images_aug))
                src, mask = image_feature_aug[-1].decompose()
                src = detr.module.input_proj(src).flatten(2).permute(2, 0, 1)
                pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
                image_feature_aug = detr.module.transformer.encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)
                image_feature_aug = image_feature_aug.permute(1, 2, 0)
                image_feature_aug = image_feature_aug.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])
                del images, images_aug

            categories = [category.to(rank) for category in categories]  # [batch_size][curr_num_obj, 1]
            if super_categories[0] is not None:
                super_categories = [[sc.to(rank) for sc in super_category] for super_category in super_categories]  # [batch_size][curr_num_obj, [1 or more]]
            image_depth = torch.stack([depth.to(rank) for depth in image_depth])
            bbox = [box.to(rank) for box in bbox]  # [batch_size][curr_num_obj, 4]
            optimizer.param_groups[0]["lr"] = original_lr

            masks = []
            for i in range(len(bbox)):
                mask = torch.zeros(bbox[i].shape[0], args['models']['feature_size'], args['models']['feature_size'], dtype=torch.uint8).to(rank)
                for j, box in enumerate(bbox[i]):
                    mask[j, int(bbox[i][j][2]):int(bbox[i][j][3]), int(bbox[i][j][0]):int(bbox[i][j][1])] = 1
                masks.append(mask)

            """
            PREPARE TARGETS
            """
            relations_target = []
            direction_target = []
            num_graph_iter = torch.as_tensor([len(mask) for mask in masks]) - 1
            for graph_iter in range(max(num_graph_iter)):
                keep_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)
                relations_target.append(torch.vstack([relationships[i][graph_iter] for i in keep_in_batch]).T.to(rank))  # integer labels
                direction_target.append(torch.vstack([subj_or_obj[i][graph_iter] for i in keep_in_batch]).T.to(rank))

            """
            FORWARD PASS
            """
            hidden_cat_accumulated = [[] for _ in range(batch_size)]
            hidden_cat_labels_accumulated = [[] for _ in range(batch_size)]
            losses, loss_connectivity, loss_relationship, loss_contrast, loss_transformer = 0.0, 0.0, 0.0, 0.0, 0.0

            num_graph_iter = torch.as_tensor([len(mask) for mask in masks])
            for graph_iter in range(max(num_graph_iter)):
                keep_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)
                optimizer.param_groups[0]["lr"] = original_lr * lr_decay * math.sqrt(len(keep_in_batch) / len(num_graph_iter))  # dynamic batch size needs dynamic learning rate

                curr_graph_masks = torch.stack([torch.unsqueeze(masks[i][graph_iter], dim=0) for i in keep_in_batch])
                h_graph = torch.cat((image_feature[keep_in_batch] * curr_graph_masks, image_depth[keep_in_batch] * curr_graph_masks), dim=1)  # (bs, 256, 64, 64), (bs, 1, 64, 64)
                h_graph_aug = torch.cat((image_feature_aug[keep_in_batch] * curr_graph_masks, image_depth[keep_in_batch] * curr_graph_masks), dim=1)
                cat_graph = torch.tensor([torch.unsqueeze(categories[i][graph_iter], dim=0) for i in keep_in_batch]).to(rank)
                scat_graph = [super_categories[i][graph_iter] for i in keep_in_batch] if super_categories[0] is not None else None
                bbox_graph = torch.stack([bbox[i][graph_iter] for i in keep_in_batch]).to(rank)

                for edge_iter in range(graph_iter):
                    curr_edge_masks = torch.stack([torch.unsqueeze(masks[i][edge_iter], dim=0) for i in keep_in_batch])  # seg mask of every prev obj
                    h_edge = torch.cat((image_feature[keep_in_batch] * curr_edge_masks, image_depth[keep_in_batch] * curr_edge_masks), dim=1)
                    h_edge_aug = torch.cat((image_feature_aug[keep_in_batch] * curr_edge_masks, image_depth[keep_in_batch] * curr_edge_masks), dim=1)
                    cat_edge = torch.tensor([torch.unsqueeze(categories[i][edge_iter], dim=0) for i in keep_in_batch]).to(rank)
                    scat_edge = [super_categories[i][edge_iter] for i in keep_in_batch] if super_categories[0] is not None else None
                    bbox_edge = torch.stack([bbox[i][edge_iter] for i in keep_in_batch]).to(rank)

                    """
                    FIRST DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity, hidden, hidden_aug = local_predictor(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank, h_graph_aug, h_edge_aug)
                        hidden_cat = torch.cat((hidden.unsqueeze(1), hidden_aug.unsqueeze(1)), dim=1)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = local_predictor(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
                        super_relation = None

                    not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 1)[0]  # which data samples in curr keep_in_batch are not connected
                    num_not_connected += len(not_connected)
                    temp = criterion_connectivity(connectivity[not_connected, 0], torch.zeros(len(not_connected)).to(rank))
                    loss_connectivity += 0.0 if torch.isnan(temp) else args['training']['lambda_not_connected'] * temp

                    connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 1)[0]  # which data samples in curr keep_in_batch are connected
                    num_connected += len(connected)
                    connected_pred = torch.nonzero(torch.sigmoid(connectivity[:, 0]) >= 0.5).flatten()
                    connectivity_precision += torch.sum(relations_target[graph_iter - 1][edge_iter][connected_pred] != -1)
                    num_connected_pred += len(connected_pred)

                    hidden_cat = hidden_cat[connected]

                    if len(connected) > 0:
                        temp = criterion_connectivity(connectivity[connected, 0], torch.ones(len(connected)).to(rank))
                        loss_connectivity += 0.0 if torch.isnan(temp) else temp
                        connectivity_recall += torch.sum(torch.round(torch.sigmoid(connectivity[connected, 0])))

                        if args['models']['hierarchical_pred']:
                            super_relation_target = relations_target[graph_iter - 1][edge_iter][connected].clone()
                            super_relation_target[super_relation_target < args['models']['num_geometric']] = 0
                            super_relation_target[torch.logical_and(super_relation_target >= args['models']['num_geometric'], super_relation_target < args['models']['num_geometric']+args['models']['num_possessive'])] = 1
                            super_relation_target[super_relation_target >= args['models']['num_geometric']+args['models']['num_possessive']] = 2
                            loss_relationship += criterion_super_relationship(super_relation[connected], super_relation_target)

                            connected_1 = torch.nonzero(relations_target[graph_iter - 1][edge_iter][connected] < args['models']['num_geometric']).flatten()  # geometric
                            connected_2 = torch.nonzero(torch.logical_and(relations_target[graph_iter - 1][edge_iter][connected] >= args['models']['num_geometric'],
                                                                          relations_target[graph_iter - 1][edge_iter][connected] < args['models']['num_geometric']+args['models']['num_possessive'])).flatten()  # possessive
                            connected_3 = torch.nonzero(relations_target[graph_iter - 1][edge_iter][connected] >= args['models']['num_geometric']+args['models']['num_possessive']).flatten()  # semantic
                            if len(connected_1) > 0:
                                loss_relationship += criterion_relationship_1(relation_1[connected][connected_1], relations_target[graph_iter - 1][edge_iter][connected][connected_1])
                            if len(connected_2) > 0:
                                loss_relationship += criterion_relationship_2(relation_2[connected][connected_2], relations_target[graph_iter - 1][edge_iter][connected][connected_2] - args['models']['num_geometric'])
                            if len(connected_3) > 0:
                                loss_relationship += criterion_relationship_3(relation_3[connected][connected_3], relations_target[graph_iter - 1][edge_iter][connected][connected_3] - args['models']['num_geometric'] - args['models']['num_possessive'])
                        else:
                            loss_relationship += criterion_relationship(relation[connected], relations_target[graph_iter - 1][edge_iter][connected])

                        hidden_cat_labels = relations_target[graph_iter - 1][edge_iter][connected]
                        for index, batch_index in enumerate(keep_in_batch[connected]):
                            hidden_cat_accumulated[batch_index].append(hidden_cat[index])
                            hidden_cat_labels_accumulated[batch_index].append(hidden_cat_labels[index])
                        # hidden_cat_accumulated.append(hidden_cat)
                        # hidden_cat_labels_accumulated.append(hidden_cat_labels)

                    # evaluate recall@k scores
                    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                    relations_target_directed[not_connected] = -1

                    if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                        Recall.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                          cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                        if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                            Recall_top3.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                                   cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

                    # print('loss_contrast', loss_contrast, 'loss_relationship', loss_relationship)
                    losses += loss_relationship + args['training']['lambda_connectivity'] * (
                                loss_connectivity + args['training']['lambda_sparsity'] * torch.linalg.norm(torch.sigmoid(connectivity), ord=1))
                    running_loss_connectivity += args['training']['lambda_connectivity'] * (
                                loss_connectivity + args['training']['lambda_sparsity'] * torch.linalg.norm(torch.sigmoid(connectivity), ord=1))
                    running_loss_relationship += loss_relationship

                    """
                    SECOND DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity, hidden2, hidden_aug2 = local_predictor(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank, h_edge_aug, h_graph_aug)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                        hidden_cat2 = torch.cat((hidden2.unsqueeze(1), hidden_aug2.unsqueeze(1)), dim=1)
                    else:
                        relation, connectivity = local_predictor(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)
                        super_relation = None

                    not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 0)[0]  # which data samples in curr keep_in_batch are not connected
                    num_not_connected += len(not_connected)
                    temp = criterion_connectivity(connectivity[not_connected, 0], torch.zeros(len(not_connected)).to(rank))
                    loss_connectivity += 0.0 if torch.isnan(temp) else args['training']['lambda_not_connected'] * temp

                    connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 0)[0]  # which data samples in curr keep_in_batch are connected
                    num_connected += len(connected)
                    connected_pred = torch.nonzero(torch.sigmoid(connectivity[:, 0]) >= 0.5).flatten()
                    connectivity_precision += torch.sum(relations_target[graph_iter - 1][edge_iter][connected_pred] != -1)
                    num_connected_pred += len(connected_pred)

                    hidden_cat2 = hidden_cat2[connected]

                    if len(connected) > 0:
                        temp = criterion_connectivity(connectivity[connected, 0], torch.ones(len(connected)).to(rank))
                        loss_connectivity += 0.0 if torch.isnan(temp) else temp
                        connectivity_recall += torch.sum(torch.round(torch.sigmoid(connectivity[connected, 0])))

                        if args['models']['hierarchical_pred']:
                            super_relation_target = relations_target[graph_iter - 1][edge_iter][connected].clone()
                            super_relation_target[super_relation_target < args['models']['num_geometric']] = 0
                            super_relation_target[torch.logical_and(super_relation_target >= args['models']['num_geometric'], super_relation_target < args['models']['num_geometric']+args['models']['num_possessive'])] = 1
                            super_relation_target[super_relation_target >= args['models']['num_geometric']+args['models']['num_possessive']] = 2
                            loss_relationship += criterion_super_relationship(super_relation[connected], super_relation_target)

                            connected_1 = torch.nonzero(relations_target[graph_iter - 1][edge_iter][connected] < args['models']['num_geometric']).flatten()  # geometric
                            connected_2 = torch.nonzero(torch.logical_and(relations_target[graph_iter - 1][edge_iter][connected] >= args['models']['num_geometric'],
                                                                          relations_target[graph_iter - 1][edge_iter][connected] < args['models']['num_geometric']+args['models']['num_possessive'])).flatten()  # possessive
                            connected_3 = torch.nonzero(relations_target[graph_iter - 1][edge_iter][connected] >= args['models']['num_geometric']+args['models']['num_possessive']).flatten()  # semantic
                            if len(connected_1) > 0:
                                loss_relationship += criterion_relationship_1(relation_1[connected][connected_1], relations_target[graph_iter - 1][edge_iter][connected][connected_1])
                            if len(connected_2) > 0:
                                loss_relationship += criterion_relationship_2(relation_2[connected][connected_2], relations_target[graph_iter - 1][edge_iter][connected][connected_2] - args['models']['num_geometric'])
                            if len(connected_3) > 0:
                                loss_relationship += criterion_relationship_3(relation_3[connected][connected_3], relations_target[graph_iter - 1][edge_iter][connected][connected_3] - args['models']['num_geometric'] - args['models']['num_possessive'])
                        else:
                            loss_relationship += criterion_relationship(relation[connected], relations_target[graph_iter - 1][edge_iter][connected])

                        hidden_cat_labels2 = relations_target[graph_iter - 1][edge_iter][connected]
                        for index, batch_index in enumerate(keep_in_batch[connected]):
                            hidden_cat_accumulated[batch_index].append(hidden_cat2[index])
                            hidden_cat_labels_accumulated[batch_index].append(hidden_cat_labels2[index])
                        # hidden_cat_accumulated.append(hidden_cat2)
                        # hidden_cat_labels_accumulated.append(hidden_cat_labels2)

                    # evaluate recall@k scores
                    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                    relations_target_directed[not_connected] = -1

                    if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                        Recall.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                          cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                        if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                            Recall_top3.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                                   cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

                    # print('loss_contrast', loss_contrast, 'loss_relationship', loss_relationship)
                    losses += loss_relationship + args['training']['lambda_connectivity'] * (
                                loss_connectivity + args['training']['lambda_sparsity'] * torch.linalg.norm(torch.sigmoid(connectivity), ord=1))
                    running_loss_connectivity += args['training']['lambda_connectivity'] * (
                                loss_connectivity + args['training']['lambda_sparsity'] * torch.linalg.norm(torch.sigmoid(connectivity), ord=1))
                    running_loss_relationship += loss_relationship


            if len(hidden_cat_accumulated) > 0:
                # concatenate all hidden_cat and hidden_cat_labels along the 0th dimension
                print('hidden_cat_accumulated', len(hidden_cat_accumulated), [len(sublist) for sublist in hidden_cat_accumulated])
                hidden_cat_accumulated = [torch.stack(sublist) for sublist in hidden_cat_accumulated if len(sublist) > 0]
                hidden_cat_labels_accumulated = [torch.stack(sublist) for sublist in hidden_cat_labels_accumulated if len(sublist) > 0]
                print('hidden_cat_accumulated', len(hidden_cat_accumulated), [sublist.shape for sublist in hidden_cat_accumulated], 'hidden_cat_labels_all', [sublist.shape for sublist in hidden_cat_labels_accumulated])

                hidden_cat_all = torch.cat(hidden_cat_accumulated, dim=0)
                hidden_cat_labels_all = torch.cat(hidden_cat_labels_accumulated, dim=0)
                print('hidden_cat_all', hidden_cat_all.shape, 'hidden_cat_labels_all', hidden_cat_labels_all.shape)

                temp = criterion_contrast(rank, hidden_cat_all, hidden_cat_labels_all)
                loss_contrast += 0.0 if torch.isnan(temp) else args['training']['lambda_contrast'] * temp
                print('loss_contrast', loss_contrast)


                # use a transformer encoder network to fuse global information about all relation triplets in the scene
                seq_lens = [len(sublist) for sublist in hidden_cat_accumulated]
                max_length = max(seq_lens)
                print('seq_lens', seq_lens, 'max_length', max_length)
                padded_hidden_cat_all = torch.stack([torch.cat([sublist[:, 0, :], torch.zeros(max_length - len(sublist), args['models']['d_model']).to(rank)], dim=0)
                                                     for sublist in hidden_cat_accumulated])
                print('padded_hidden_cat_all2', padded_hidden_cat_all.shape)
                padded_hidden_cat_all = torch.permute(padded_hidden_cat_all, (1, 0, 2))  # (S, N, E)
                print('padded_hidden_cat_all3', padded_hidden_cat_all.shape)
                src_key_padding_mask = torch.zeros((padded_hidden_cat_all.shape[1], padded_hidden_cat_all.shape[0]), dtype=torch.bool).to(rank)  # (N, S)
                for i, length in enumerate(seq_lens):
                    src_key_padding_mask[i, length:] = 1

                if args['models']['hierarchical_pred']:
                    print('padded_hidden_cat_all', padded_hidden_cat_all.shape, 'src_key_padding_mask', src_key_padding_mask.shape)
                    refined_relation_1, refined_relation_2, refined_relation_3, refined_super_relation = transformer_encoder(padded_hidden_cat_all, src_key_padding_mask)
                    print('before relation', refined_relation_1.shape, 'relation_2', refined_relation_2.shape, 'relation_3', refined_relation_3.shape, 'super_relation', refined_super_relation.shape)
                else:
                    refined_relation = transformer_encoder(padded_hidden_cat_all, src_key_padding_mask)
                    refined_relation = refined_relation[not src_key_padding_mask]

                if args['models']['hierarchical_pred']:
                    super_relation_target = hidden_cat_labels_all.clone()
                    super_relation_target[super_relation_target < args['models']['num_geometric']] = 0
                    super_relation_target[
                        torch.logical_and(super_relation_target >= args['models']['num_geometric'], super_relation_target < args['models']['num_geometric'] + args['models']['num_possessive'])] = 1
                    super_relation_target[super_relation_target >= args['models']['num_geometric'] + args['models']['num_possessive']] = 2
                    print('refined_super_relation', refined_super_relation.shape, 'super_relation_target', super_relation_target.shape)
                    loss_transformer += criterion_super_relationship(refined_super_relation, super_relation_target)
                    print('loss_transformer0', loss_transformer)

                    connected_1 = torch.nonzero(hidden_cat_labels_all < args['models']['num_geometric']).flatten()  # geometric
                    connected_2 = torch.nonzero(torch.logical_and(hidden_cat_labels_all >= args['models']['num_geometric'],
                                                                  hidden_cat_labels_all < args['models']['num_geometric'] + args['models']['num_possessive'])).flatten()  # possessive
                    connected_3 = torch.nonzero(hidden_cat_labels_all >= args['models']['num_geometric'] + args['models']['num_possessive']).flatten()  # semantic

                    print('hidden_cat_labels_all', hidden_cat_labels_all)
                    print('hidden_cat_labels_all[connected_1]', hidden_cat_labels_all[connected_1])
                    print('hidden_cat_labels_all[connected_2]', hidden_cat_labels_all[connected_2] - args['models']['num_geometric'])
                    print('hidden_cat_labels_all[connected_3]', hidden_cat_labels_all[connected_3] - args['models']['num_geometric'] - args['models']['num_possessive'])

                    if len(connected_1) > 0:
                        print('relation_1[connected_1]', refined_relation_1[connected_1].shape, 'hidden_cat_labels_all[connected_1]', hidden_cat_labels_all[connected_1].shape)
                        loss_transformer += criterion_relationship_1(refined_relation_1[connected_1], hidden_cat_labels_all[connected_1])
                        print('loss_transformer1', loss_transformer)
                    if len(connected_2) > 0:
                        print('relation_2[connected_2]', refined_relation_2[connected_2].shape, 'hidden_cat_labels_all[connected_1]', hidden_cat_labels_all[connected_2].shape)
                        loss_transformer += criterion_relationship_2(refined_relation_2[connected_2], hidden_cat_labels_all[connected_2]-args['models']['num_geometric'])
                        print('loss_transformer2', loss_transformer)
                    if len(connected_3) > 0:
                        print('relation_3[connected_3]', refined_relation_3[connected_3].shape, 'hidden_cat_labels_all[connected_3]', hidden_cat_labels_all[connected_3].shape)
                        loss_transformer += criterion_relationship_3(refined_relation_3[connected_3], hidden_cat_labels_all[connected_3]-args['models']['num_geometric']-args['models']['num_possessive'])
                        print('loss_transformer3', loss_transformer)
                else:
                    loss_transformer += criterion_relationship(refined_relation, hidden_cat_labels_all)

                print('loss_transformer', loss_transformer)

            print('done transformer!!!!!')

            running_loss_contrast += args['training']['lambda_contrast'] * loss_contrast
            running_loss_transformer += loss_transformer
            losses += loss_contrast + loss_transformer
            running_losses += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

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
                                     recall_zs, mean_recall_zs, running_losses, running_loss_relationship, running_loss_contrast, running_loss_connectivity, connectivity_recall,
                                     num_connected, num_not_connected, connectivity_precision, num_connected_pred, wmap_rel, wmap_phrase)
                dist.monitored_barrier()

            running_losses, running_loss_connectivity, running_loss_relationship, running_loss_contrast, running_loss_transformer, connectivity_precision, \
                num_connected, num_not_connected = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if args['models']['hierarchical_pred']:
            torch.save(local_predictor.state_dict(), args['training']['checkpoint_path'] + 'HierMotif' + str(epoch) + '_' + str(rank) + '.pth')
            torch.save(transformer_encoder.state_dict(), args['training']['checkpoint_path'] + 'TransEncoder' + str(epoch) + '_' + str(rank) + '.pth')
        else:
            torch.save(local_predictor.state_dict(), args['training']['checkpoint_path'] + 'FlatMotif' + str(epoch) + '_' + str(rank) + '.pth')
            torch.save(transformer_encoder.state_dict(), args['training']['checkpoint_path'] + 'FlatTransEncoder' + str(epoch) + '_' + str(rank) + '.pth')
        dist.monitored_barrier()

        test_local(args, detr, local_predictor, transformer_encoder, test_loader, test_record, epoch, rank)

    dist.destroy_process_group()  # clean up
    print('FINISHED TRAINING\n')


def test_local(args, backbone, local_predictor, transformer_encoder, test_loader, test_record, epoch, rank):
    backbone.eval()
    local_predictor.eval()
    transformer_encoder.eval()

    connectivity_recall, connectivity_precision, num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0
    recall_top3, recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs, wmap_rel, wmap_phrase = None, None, None, None, None, None, None, None
    Recall = Evaluator_PC(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])
    if args['dataset']['dataset'] == 'vg':
        Recall_top3 = Evaluator_PC_Top3(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])

    print('Start Testing PC...')
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            if epoch < 2 and batch_count > 100:
                break
            """
            PREPARE INPUT DATA
            """
            images, _, image_depth, categories, super_categories, bbox, relationships, subj_or_obj = data

            images = torch.stack(images).to(rank)
            image_feature, pos_embed = backbone.module.backbone(nested_tensor_from_tensor_list(images))
            src, mask = image_feature[-1].decompose()
            src = backbone.module.input_proj(src).flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
            image_feature = backbone.module.transformer.encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)
            image_feature = image_feature.permute(1, 2, 0)
            image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])
            del images

            categories = [category.to(rank) for category in categories]  # [batch_size][curr_num_obj, 1]
            if super_categories[0] is not None:
                super_categories = [[sc.to(rank) for sc in super_category] for super_category in super_categories]  # [batch_size][curr_num_obj, [1 or more]]
            image_depth = torch.stack([depth.to(rank) for depth in image_depth])
            bbox = [box.to(rank) for box in bbox]  # [batch_size][curr_num_obj, 4]

            masks = []
            for i in range(len(bbox)):
                mask = torch.zeros(bbox[i].shape[0], args['models']['feature_size'], args['models']['feature_size'], dtype=torch.uint8).to(rank)
                for j, box in enumerate(bbox[i]):
                    mask[j, int(bbox[i][j][2]):int(bbox[i][j][3]), int(bbox[i][j][0]):int(bbox[i][j][1])] = 1
                masks.append(mask)

            """
            PREPARE TARGETS
            """
            relations_target = []
            direction_target = []
            num_graph_iter = torch.as_tensor([len(mask) for mask in masks]) - 1
            for graph_iter in range(max(num_graph_iter)):
                keep_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)
                relations_target.append(torch.vstack([relationships[i][graph_iter] for i in keep_in_batch]).T.to(rank))  # integer labels
                direction_target.append(torch.vstack([subj_or_obj[i][graph_iter] for i in keep_in_batch]).T.to(rank))

            """
            FORWARD PASS THROUGH THE LOCAL PREDICTOR
            """
            num_graph_iter = torch.as_tensor([len(mask) for mask in masks])
            for graph_iter in range(max(num_graph_iter)):
                keep_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)

                curr_graph_masks = torch.stack([torch.unsqueeze(masks[i][graph_iter], dim=0) for i in keep_in_batch])
                h_graph = torch.cat((image_feature[keep_in_batch] * curr_graph_masks, image_depth[keep_in_batch] * curr_graph_masks), dim=1)  # (bs, 256, 64, 64), (bs, 1, 64, 64)
                cat_graph = torch.tensor([torch.unsqueeze(categories[i][graph_iter], dim=0) for i in keep_in_batch]).to(rank)
                scat_graph = [super_categories[i][graph_iter] for i in keep_in_batch] if super_categories[0] is not None else None
                bbox_graph = torch.stack([bbox[i][graph_iter] for i in keep_in_batch]).to(rank)

                for edge_iter in range(graph_iter):
                    curr_edge_masks = torch.stack([torch.unsqueeze(masks[i][edge_iter], dim=0) for i in keep_in_batch])  # seg mask of every prev obj
                    h_edge = torch.cat((image_feature[keep_in_batch] * curr_edge_masks, image_depth[keep_in_batch] * curr_edge_masks), dim=1)
                    cat_edge = torch.tensor([torch.unsqueeze(categories[i][edge_iter], dim=0) for i in keep_in_batch]).to(rank)
                    scat_edge = [super_categories[i][edge_iter] for i in keep_in_batch] if super_categories[0] is not None else None
                    bbox_edge = torch.stack([bbox[i][edge_iter] for i in keep_in_batch]).to(rank)

                    """
                    FIRST DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity, hidden, _ = local_predictor(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
                        hidden_cat = torch.cat((hidden.unsqueeze(1), hidden_aug.unsqueeze(1)), dim=1)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = local_predictor(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
                        super_relation = None

                    not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 1)[0]  # which data samples in curr keep_in_batch are not connected
                    num_not_connected += len(not_connected)
                    connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 1)[0]  # which data samples in curr keep_in_batch are connected
                    num_connected += len(connected)
                    connected_pred = torch.nonzero(torch.sigmoid(connectivity[:, 0]) >= 0.5).flatten()
                    connectivity_precision += torch.sum(relations_target[graph_iter - 1][edge_iter][connected_pred] != -1)
                    num_connected_pred += len(connected_pred)

                    hidden_cat = hidden_cat[torch.sigmoid(connectivity[:, 0]) > 0.5]

                    if len(connected) > 0:
                        connectivity_recall += torch.sum(torch.round(torch.sigmoid(connectivity[connected, 0])))

                    # evaluate recall@k scores
                    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                    relations_target_directed[not_connected] = -1

                    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                        Recall.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                          cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                        if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                            Recall_top3.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                                   cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

                    """
                    SECOND DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity, hidden2, _ = local_predictor(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)
                        hidden_cat2 = torch.cat((hidden2.unsqueeze(1), hidden_aug2.unsqueeze(1)), dim=1)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = local_predictor(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)
                        super_relation = None

                    not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 0)[0]  # which data samples in curr keep_in_batch are not connected
                    num_not_connected += len(not_connected)
                    connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 0)[0]  # which data samples in curr keep_in_batch are connected
                    num_connected += len(connected)
                    connected_pred = torch.nonzero(torch.sigmoid(connectivity[:, 0]) >= 0.5).flatten()
                    connectivity_precision += torch.sum(relations_target[graph_iter - 1][edge_iter][connected_pred] != -1)
                    num_connected_pred += len(connected_pred)

                    hidden_cat2 = hidden_cat2[torch.sigmoid(connectivity[:, 0]) > 0.5]

                    if len(connected) > 0:
                        connectivity_recall += torch.sum(torch.round(torch.sigmoid(connectivity[connected, 0])))

                    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                    relations_target_directed[not_connected] = -1

                    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                        Recall.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                          cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                        if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                            Recall_top3.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                                   cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

            """
            EVALUATE AND PRINT CURRENT RESULTS
            """
            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                if args['dataset']['dataset'] == 'vg':
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