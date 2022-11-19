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
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup

from evaluator import Evaluator_PC, Evaluator_PC_Top3
from model import *
from utils import *
from transformer import build_transformer


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_global(gpu, args, train_subset, test_subset):
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
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True, drop_last=True, sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True, drop_last=True, sampler=test_sampler)
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
        motif_embedding = DDP(MotifEmbedHier(input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
                                             num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'])).to(rank)
                                             # T1=args['training']['temperature_1'], T2=args['training']['temperature_2'], T3=args['training']['temperature_3'])).to(rank)
        motif_head = DDP(MotifHeadHier(input_dim=args['models']['embed_hidden_dim'])).to(rank)
    else:
        motif_embedding = DDP(MotifEmbed(input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'], feature_size=args['models']['feature_size'],
                                         num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'])).to(rank)
        motif_head = DDP(MotifHead(input_dim=args['models']['embed_hidden_dim'], output_dim=args['models']['num_relations'])).to(rank)

    transformer_encoder = DDP(build_transformer(args)).to(rank)
    transformer_encoder.train()

    detr = DDP(build_detr101(args)).to(rank)
    backbone = DDP(detr.module.backbone).to(rank)
    input_proj = DDP(detr.module.input_proj).to(rank)
    feature_encoder = DDP(detr.module.transformer.encoder).to(rank)

    detr.eval()
    motif_embedding.eval()
    motif_head.train()
    backbone.eval()
    input_proj.eval()
    feature_encoder.eval()

    map_location = {'cuda:%d' % rank: 'cuda:%d' % rank}
    if args['models']['hierarchical_pred']:
        motif_embedding.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHeadHier' + str(2) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)
        motif_head.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHeadHier' + str(2) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)
    else:
        motif_embedding.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHead' + str(2) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)
        motif_head.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHead' + str(2) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)

    params = list(transformer_encoder.parameters()) + list(motif_head.parameters())
    optimizer = optim.AdamW([{'params': params, 'initial_lr': args['training']['learning_rate']}], lr=args['training']['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.5*len(train_loader), num_training_steps=(args['training']['num_epoch']-args['training']['start_epoch'])*len(train_loader))

    relation_count = get_num_each_class()
    class_weight = 1 - relation_count / torch.sum(relation_count)
    criterion_relationship_1 = torch.nn.NLLLoss(weight=class_weight[:15].to(rank))
    criterion_relationship_2 = torch.nn.NLLLoss(weight=class_weight[15:26].to(rank))
    criterion_relationship_3 = torch.nn.NLLLoss(weight=class_weight[26:].to(rank))
    criterion_super_relationship = torch.nn.NLLLoss()
    criterion_relationship = torch.nn.CrossEntropyLoss(weight=class_weight.to(rank))  # reduction='sum'
    criterion_connectivity = torch.nn.BCEWithLogitsLoss()

    Recall = Evaluator_PC(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])
    Recall_top3 = Evaluator_PC_Top3(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])

    lr_decay = 1
    for epoch in range(args['training']['start_epoch'], args['training']['num_epoch']):
        print('Start Training... EPOCH %d / %d\n' % (epoch, args['training']['num_epoch']))
        if epoch == args['training']['scheduler_param1'] or epoch == args['training']['scheduler_param2']:  # lr scheduler
            lr_decay *= 0.1

        for batch_count, data in enumerate(tqdm(train_loader), 0):
            connectivity_recall, connectivity_precision, running_losses, running_loss_connectivity, running_loss_relationship, \
                num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            '''
            PREPARE INPUT DATA
            '''
            try:
                _, images, image_depth, categories, super_categories, masks, bbox, relationships, subj_or_obj = data
            except:
                continue

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
            del images, relationships, subj_or_obj

            '''
            COLLECT ALL MOTIFS HIDDEN STATES
            Different data samples in a batch have different num of RNN iterations, i.e., different number of objects in images.
            To do mini-batch parallel training, in each graph-level iteration,
            only stack the data samples whose total num of RNN iterations not exceeding the current iteration.
            '''
            hidden_states = None
            relations_targets = None
            which_in_batch_all = None
            cat_subjects = None
            cat_objects = None
            bbox_subjects = None
            bbox_objects = None

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
                    with torch.no_grad():
                        hidden_state = motif_embedding(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
                        if not args['models']['hierarchical_pred']:
                            hidden_state = torch.hstack((hidden_state[:, :-1], torch.zeros(hidden_state.shape[0], 4).to(rank), hidden_state[:, -1].view(-1, 1)))

                    not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 1)[0]  # which data samples in curr which_in_batch are not connected
                    num_not_connected += len(not_connected)

                    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                    relations_target_directed[not_connected] = -1

                    if hidden_states is None:
                        hidden_states = hidden_state
                    else:
                        hidden_states = torch.vstack((hidden_states, hidden_state))  # (bs, 4480)
                    if relations_targets is None:
                        relations_targets = relations_target_directed
                    else:
                        relations_targets = torch.hstack((relations_targets, relations_target_directed))  # (bs, 50)
                    if which_in_batch_all is None:
                        which_in_batch_all = which_in_batch
                    else:
                        which_in_batch_all = torch.hstack((which_in_batch_all, which_in_batch))

                    if cat_subjects is None:
                        cat_subjects = cat_graph
                    else:
                        cat_subjects = torch.hstack((cat_subjects, cat_graph))
                    if cat_objects is None:
                        cat_objects = cat_edge
                    else:
                        cat_objects = torch.hstack((cat_objects, cat_edge))
                    if bbox_subjects is None:
                        bbox_subjects = bbox_graph
                    else:
                        bbox_subjects = torch.vstack((bbox_subjects, bbox_graph))
                    if bbox_objects is None:
                        bbox_objects = bbox_edge
                    else:
                        bbox_objects = torch.vstack((bbox_objects, bbox_edge))

                    """
                    SECOND DIRECTION
                    """
                    with torch.no_grad():
                        hidden_state = motif_embedding(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)  # <edge subject, rel, graph object>
                        if not args['models']['hierarchical_pred']:
                            hidden_state = torch.hstack((hidden_state[:, :-1], torch.zeros(hidden_state.shape[0], 4).to(rank), hidden_state[:, -1].view(-1, 1)))

                    not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 0)[0]  # which data samples in curr which_in_batch are not connected
                    num_not_connected += len(not_connected)

                    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                    relations_target_directed[not_connected] = -1

                    hidden_states = torch.vstack((hidden_states, hidden_state))  # (bs, 4480)
                    relations_targets = torch.hstack((relations_targets, relations_target_directed))  # (bs, 50)
                    which_in_batch_all = torch.hstack((which_in_batch_all, which_in_batch))

                    cat_subjects = torch.hstack((cat_subjects, cat_edge))
                    cat_objects = torch.hstack((cat_objects, cat_graph))
                    bbox_subjects = torch.vstack((bbox_subjects, bbox_edge))
                    bbox_objects = torch.vstack((bbox_objects, bbox_graph))

            '''
            FORWARD PASS THROUGH TRANSFORMER ENCODER
            '''
            del image_feature, masks, bbox, categories, super_categories, image_depth, relations_target, direction_target, h_graph, h_edge

            # Add binary mask to filter out only top k most confident predictions into the rn
            confidence_all = torch.max(torch.vstack((torch.max(hidden_states[:, -54:-39], dim=1)[0],
                                                     torch.max(hidden_states[:, -39:-28], dim=1)[0],
                                                     torch.max(hidden_states[:, -28:-4], dim=1)[0])), dim=0)[0]  # values

            topk_mask = torch.zeros(confidence_all.shape[0], dtype=bool).to(rank)
            all_imgs = torch.unique(which_in_batch_all)
            for img in all_imgs:
                curr_img = torch.nonzero(which_in_batch_all == img).flatten()
                curr_confidence = confidence_all[curr_img]
                curr_confidence_sorted = torch.sort(curr_confidence, descending=True)[0]
                top_confidence = curr_confidence_sorted[int(args['transformers']['sparsity'] * len(curr_confidence))]   #curr_confidence_sorted[min(50, len(curr_confidence)-1)]
                topk_mask[curr_img[torch.nonzero(curr_confidence >= top_confidence)]] = 1

            topk_mask[torch.isinf(confidence_all)] = 0
            topk_mask[torch.sigmoid(hidden_states[:, -1]) >= 0.5] = 1

            hidden_states_filtered = hidden_states[topk_mask]
            which_in_batch_all_filtered = which_in_batch_all[topk_mask]
            hidden_states_batched = pad_sequence([hidden_states_filtered[which_in_batch_all_filtered == img] for img in torch.unique(which_in_batch_all_filtered, sorted=True)])
            num_motifs = hidden_states_batched.shape[0]

            pad_mask = torch.ones(hidden_states_batched.shape[1], num_motifs, dtype=torch.bool).to(rank)
            for img_idx, img in enumerate(torch.unique(which_in_batch_all_filtered, sorted=True)):
                pad_mask[img_idx, torch.sum(which_in_batch_all_filtered == img):] = 0
            pad_mask = pad_mask.T

            hidden_states_batched = transformer_encoder(hidden_states_batched, num_motifs, rank)
            hidden_states_out = hidden_states[topk_mask, :512] + hidden_states_batched[pad_mask]      # skip connection

            new_relation_pred = hidden_states[:, -54:-4].detach()
            new_super_relation_pred = hidden_states[:, -4:-1].detach()
            new_connectivity_pred = hidden_states[:, -1].view(-1, 1).detach()
            if args['models']['hierarchical_pred']:
                new_relation_pred_1, new_relation_pred_2, new_relation_pred_3, new_super_relation_pred_temp, new_connectivity_pred_temp = motif_head(hidden_states_out)
                new_relation_pred[topk_mask] = torch.cat((new_relation_pred_1, new_relation_pred_2, new_relation_pred_3), dim=1)
                new_super_relation_pred[topk_mask] = new_super_relation_pred_temp
                new_connectivity_pred[topk_mask] = new_connectivity_pred_temp
            else:
                new_relation_pred_temp, new_connectivity_pred_temp = motif_head(hidden_states_out)
                new_relation_pred[topk_mask] = new_relation_pred_temp
                new_connectivity_pred[topk_mask] = new_connectivity_pred_temp

            loss_connectivity, loss_relationship = 0.0, 0.0
            connected = torch.nonzero(relations_targets[topk_mask] != -1).flatten()
            connected_pred = torch.nonzero(torch.sigmoid(new_connectivity_pred[topk_mask, 0]) >= 0.5).flatten()
            connectivity_precision += torch.sum(relations_targets[topk_mask][connected_pred] != -1)
            num_connected += len(connected)
            num_connected_pred += len(connected_pred)
            not_connected = torch.nonzero(relations_targets[topk_mask] == -1).flatten()

            temp = criterion_connectivity(new_connectivity_pred[topk_mask][not_connected, 0], torch.zeros(len(not_connected)).to(rank))
            loss_connectivity += 0.0 if torch.isnan(temp) else args['training']['lambda_not_connected'] * temp
            if len(connected) > 0:
                temp = criterion_connectivity(new_connectivity_pred[topk_mask][connected, 0], torch.ones(len(connected)).to(rank))
                loss_connectivity += 0.0 if torch.isnan(temp) else temp
                connectivity_recall += torch.sum((torch.sigmoid(new_connectivity_pred[topk_mask][connected, 0]) >= 0.5) == torch.ones(len(connected)).to(rank))

                if args['models']['hierarchical_pred']:
                    super_relation_target = relations_targets[topk_mask][connected].clone()
                    super_relation_target[super_relation_target < 15] = 0
                    super_relation_target[torch.logical_and(super_relation_target >= 15, super_relation_target < 26)] = 1
                    super_relation_target[super_relation_target >= 26] = 2
                    loss_relationship += 5 * criterion_super_relationship(new_super_relation_pred[topk_mask][connected], super_relation_target)

                    connected_1 = torch.nonzero(relations_targets[topk_mask][connected] < 15).flatten()  # geometric
                    connected_2 = torch.nonzero(torch.logical_and(relations_targets[topk_mask][connected] >= 15, relations_targets[topk_mask][connected] < 26)).flatten()  # possessive
                    connected_3 = torch.nonzero(relations_targets[topk_mask][connected] >= 26).flatten()  # semantic
                    if len(connected_1) > 0:
                        loss_relationship += criterion_relationship_1(new_relation_pred[topk_mask, :15][connected][connected_1], relations_targets[topk_mask][connected][connected_1])
                    if len(connected_2) > 0:
                        loss_relationship += criterion_relationship_2(new_relation_pred[topk_mask, 15:26][connected][connected_2], relations_targets[topk_mask][connected][connected_2] - 15)
                    if len(connected_3) > 0:
                        loss_relationship += criterion_relationship_3(new_relation_pred[topk_mask, 26:][connected][connected_3], relations_targets[topk_mask][connected][connected_3] - 26)
                else:
                    loss_relationship += criterion_relationship(new_relation_pred[topk_mask][connected], relations_targets[topk_mask][connected])

            if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                Recall.accumulate(which_in_batch_all, new_relation_pred, relations_targets, new_super_relation_pred, torch.log(torch.sigmoid(new_connectivity_pred[:, 0])),
                                  cat_subjects, cat_objects, cat_subjects, cat_objects, bbox_subjects, bbox_objects, bbox_subjects, bbox_objects)
                Recall_top3.accumulate(which_in_batch_all, new_relation_pred, relations_targets, new_super_relation_pred, torch.log(torch.sigmoid(new_connectivity_pred[:, 0])),
                                       cat_subjects, cat_objects, cat_subjects, cat_objects, bbox_subjects, bbox_objects, bbox_subjects, bbox_objects)

            losses = loss_relationship + loss_connectivity
            running_loss_relationship += loss_relationship
            running_losses += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

            '''
            EVALUATE AND PRINT CURRENT TRAINING RESULTS
            '''
            if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                recall, _, mean_recall = Recall.compute(args['models']['hierarchical_pred'], per_class=True)
                recall_top3, _, mean_recall_top3 = Recall_top3.compute(args['models']['hierarchical_pred'], per_class=True)
                Recall.clear_data()
                Recall_top3.clear_data()

            if (batch_count % args['training']['print_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                print('TRAINING, rank %d, epoch %d, batch %d, lr: %.4f, R@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), mR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), loss: %.4f, conn: %.4f, %.4f, sparsity: %.4f.'
                      % (rank, epoch, batch_count, optimizer.param_groups[0]["lr"],
                         recall_top3[0], recall_top3[1], recall_top3[2], recall[0], recall[1], recall[2],
                         mean_recall_top3[0], mean_recall_top3[1], mean_recall_top3[2], mean_recall[0], mean_recall[1], mean_recall[2],
                         running_loss_relationship / (args['training']['print_freq'] * args['training']['batch_size']),
                         connectivity_recall / num_connected, connectivity_precision / num_connected_pred, torch.sum(topk_mask) / len(topk_mask)))

                record.append({'rank': rank, 'epoch': epoch, 'batch': batch_count, 'lr': optimizer.param_groups[0]["lr"],
                               'recall_relationship': [recall[0], recall[1], recall[2]],
                               'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                               'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                               'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                               'total_losses': running_losses / (args['training']['print_freq'] * args['training']['batch_size']),
                               'relationship_loss': running_loss_relationship.item() / (args['training']['print_freq'] * args['training']['batch_size']),
                               'num_objs_average': torch.mean(num_graph_iter.float()).item(), 'num_objs_max': max(num_graph_iter).item(),
                               'num_connected': num_connected, 'num_not_connected': num_not_connected})

                with open(args['training']['result_path'] + 'train_results_' + str(rank) + '.json', 'w') as f:  # append current logs, must use "w" not "a"
                    json.dump(record, f)
                dist.monitored_barrier()

        torch.save(transformer_encoder.state_dict(), args['training']['checkpoint_path'] + 'TransformerEncoder' + str(epoch) + '_' + str(rank) + '.pth')
        if args['models']['hierarchical_pred']:
            torch.save(motif_head.state_dict(), args['training']['checkpoint_path'] + 'MotifHeadHier' + str(epoch) + '_' + str(rank) + '.pth')
        else:
            torch.save(motif_head.state_dict(), args['training']['checkpoint_path'] + 'MotifHead' + str(epoch) + '_' + str(rank) + '.pth')
        dist.monitored_barrier()

        test_global(args, transformer_encoder, motif_embedding, motif_head, backbone, input_proj, feature_encoder, test_loader, test_record, epoch, rank)

    dist.destroy_process_group()  # clean up
    print('FINISHED TRAINING\n')


def test_global(args, transformer_encoder, motif_embedding, motif_head, backbone, input_proj, feature_encoder, test_loader, test_record, epoch, rank):
    """
    Model Inputs:
        image_feature:     the image feature map from Mask R-CNN
                              size [num_img_feature, feature_size, feature_size]
        image_depth:       the image depth map from single-image depth estimation
                              size [1, feature_size, feature_size]
        list_seg_masks:    list of segmentation masks for each instance detected by Mask R-CNN
                           size [num_obj][1, feature_size, feature_size]
    """
    motif_embedding.eval()
    transformer_encoder.eval()
    motif_head.eval()
    feature_encoder.eval()

    connectivity_recall, connectivity_precision, num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0
    Recall = Evaluator_PC(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])
    Recall_top3 = Evaluator_PC_Top3(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])

    print('Start Evaluating...')
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            '''
            PREPARE INPUT DATA
            '''
            try:
                _, images, image_depth, categories, super_categories, masks, bbox, relationships, subj_or_obj = data
            except:
                continue

            images = [image.to(rank) for image in images]
            images = nested_tensor_from_tensor_list(images)
            image_feature, pos_embed = backbone(images)

            src, mask = image_feature[-1].decompose()
            src = input_proj(src).flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)

            image_feature = feature_encoder(src, src_key_padding_mask=mask, pos=pos_embed)
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
            del images, relationships, subj_or_obj

            '''
            COLLECT ALL MOTIFS HIDDEN STATES
            Different data samples in a batch have different num of RNN iterations, i.e., different number of objects in images.
            To do mini-batch parallel training, in each graph-level iteration, 
            only stack the data samples whose total num of RNN iterations not exceeding the current iteration.
            '''
            hidden_states = None
            relations_targets = None
            which_in_batch_all = None
            cat_subjects = None
            cat_objects = None
            bbox_subjects = None
            bbox_objects = None

            with torch.no_grad():
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
                        hidden_state = motif_embedding(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)  # <edge subject, rel, graph object>
                        if not args['models']['hierarchical_pred']:
                            hidden_state = torch.hstack((hidden_state[:, :-1], torch.zeros(hidden_state.shape[0], 4).to(rank), hidden_state[:, -1].view(-1, 1)))

                        not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 1)[0]  # which data samples in curr which_in_batch are not connected
                        num_not_connected += len(not_connected)

                        relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                        relations_target_directed[not_connected] = -1

                        if hidden_states is None:
                            hidden_states = hidden_state
                        else:
                            hidden_states = torch.vstack((hidden_states, hidden_state))  # (bs, 4480)
                        if relations_targets is None:
                            relations_targets = relations_target_directed
                        else:
                            relations_targets = torch.hstack((relations_targets, relations_target_directed))  # (bs, 50)
                        if which_in_batch_all is None:
                            which_in_batch_all = which_in_batch
                        else:
                            which_in_batch_all = torch.hstack((which_in_batch_all, which_in_batch))

                        if cat_subjects is None:
                            cat_subjects = cat_graph
                        else:
                            cat_subjects = torch.hstack((cat_subjects, cat_graph))
                        if cat_objects is None:
                            cat_objects = cat_edge
                        else:
                            cat_objects = torch.hstack((cat_objects, cat_edge))
                        if bbox_subjects is None:
                            bbox_subjects = bbox_graph
                        else:
                            bbox_subjects = torch.vstack((bbox_subjects, bbox_graph))
                        if bbox_objects is None:
                            bbox_objects = bbox_edge
                        else:
                            bbox_objects = torch.vstack((bbox_objects, bbox_edge))

                        """
                        SECOND DIRECTION
                        """
                        hidden_state = motif_embedding(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)  # <edge subject, rel, graph object>
                        if not args['models']['hierarchical_pred']:
                            hidden_state = torch.hstack((hidden_state[:, :-1], torch.zeros(hidden_state.shape[0], 4).to(rank), hidden_state[:, -1].view(-1, 1)))

                        not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 0)[0]  # which data samples in curr which_in_batch are not connected
                        num_not_connected += len(not_connected)

                        relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
                        relations_target_directed[not_connected] = -1

                        hidden_states = torch.vstack((hidden_states, hidden_state))  # (bs, 4480)
                        relations_targets = torch.hstack((relations_targets, relations_target_directed))  # (bs, 50)
                        which_in_batch_all = torch.hstack((which_in_batch_all, which_in_batch))

                        cat_subjects = torch.hstack((cat_subjects, cat_edge))
                        cat_objects = torch.hstack((cat_objects, cat_graph))
                        bbox_subjects = torch.vstack((bbox_subjects, bbox_edge))
                        bbox_objects = torch.vstack((bbox_objects, bbox_graph))

            '''
            FORWARD PASS THROUGH TRANSFORMER ENCODER
            '''
            del image_feature, masks, bbox, categories, super_categories, image_depth, relations_target, direction_target, h_graph, h_edge

            # Add binary mask to filter out only top k most confident predictions into the rnn
            confidence_all = torch.max(torch.vstack((torch.max(hidden_states[:, -54:-39], dim=1)[0],
                                                     torch.max(hidden_states[:, -39:-28], dim=1)[0],
                                                     torch.max(hidden_states[:, -28:-4], dim=1)[0])), dim=0)[0]  # values

            topk_mask = torch.zeros(confidence_all.shape[0], dtype=bool).to(rank)
            all_imgs = torch.unique(which_in_batch_all)
            for img in all_imgs:
                curr_img = torch.nonzero(which_in_batch_all == img).flatten()
                curr_confidence = confidence_all[curr_img]
                curr_confidence_sorted = torch.sort(curr_confidence, descending=True)[0]
                top_confidence = curr_confidence_sorted[int(args['transformers']['sparsity'] * len(curr_confidence))]   #curr_confidence_sorted[min(50, len(curr_confidence)-1)]
                topk_mask[curr_img[torch.nonzero(curr_confidence >= top_confidence)]] = 1

            topk_mask[torch.isinf(confidence_all)] = 0
            topk_mask[torch.sigmoid(hidden_states[:, -1]) >= 0.5] = 1

            hidden_states_filtered = hidden_states[topk_mask]
            which_in_batch_all_filtered = which_in_batch_all[topk_mask]
            hidden_states_batched = pad_sequence([hidden_states_filtered[which_in_batch_all_filtered == img] for img in torch.unique(which_in_batch_all_filtered, sorted=True)])
            num_motifs = hidden_states_batched.shape[0]

            pad_mask = torch.ones(hidden_states_batched.shape[1], num_motifs, dtype=torch.bool).to(rank)
            for img_idx, img in enumerate(torch.unique(which_in_batch_all_filtered, sorted=True)):
                pad_mask[img_idx, torch.sum(which_in_batch_all_filtered == img):] = 0
            pad_mask = pad_mask.T

            hidden_states_batched = transformer_encoder(hidden_states_batched, num_motifs, rank)
            hidden_states_out = hidden_states[topk_mask, :512] + hidden_states_batched[pad_mask]

            new_relation_pred = hidden_states[:, -54:-4]
            new_super_relation_pred = hidden_states[:, -4:-1].detach()
            new_connectivity_pred = hidden_states[:, -1].view(-1, 1)
            if args['models']['hierarchical_pred']:
                new_relation_pred_1, new_relation_pred_2, new_relation_pred_3, new_super_relation_pred_temp, new_connectivity_pred_temp = motif_head(hidden_states_out)
                new_relation_pred[topk_mask] = torch.cat((new_relation_pred_1, new_relation_pred_2, new_relation_pred_3), dim=1)
                new_super_relation_pred[topk_mask] = new_super_relation_pred_temp
                new_connectivity_pred[topk_mask] = new_connectivity_pred_temp
            else:
                new_relation_pred_temp, new_connectivity_pred_temp = motif_head(hidden_states_out)
                new_relation_pred[topk_mask] = new_relation_pred_temp
                new_connectivity_pred[topk_mask] = new_connectivity_pred_temp

            connected = torch.nonzero(relations_targets[topk_mask] != -1).flatten()
            connected_pred = torch.nonzero(torch.sigmoid(new_connectivity_pred[topk_mask, 0]) >= 0.5).flatten()
            connectivity_precision += torch.sum(relations_targets[topk_mask][connected_pred] != -1)
            num_connected += len(connected)
            num_connected_pred += len(connected_pred)
            if len(connected) > 0:
                connectivity_recall += torch.sum((torch.sigmoid(new_connectivity_pred[topk_mask][connected, 0]) >= 0.5) == torch.ones(len(connected)).to(rank))

            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                Recall.accumulate(which_in_batch_all, new_relation_pred, relations_targets, new_super_relation_pred, torch.log(torch.sigmoid(new_connectivity_pred[:, 0])),
                                  cat_subjects, cat_objects, cat_subjects, cat_objects, bbox_subjects, bbox_objects, bbox_subjects, bbox_objects)
                Recall_top3.accumulate(which_in_batch_all, new_relation_pred, relations_targets, new_super_relation_pred, torch.log(torch.sigmoid(new_connectivity_pred[:, 0])),
                                       cat_subjects, cat_objects, cat_subjects, cat_objects, bbox_subjects, bbox_objects, bbox_subjects, bbox_objects)

            '''
            EVALUATE AND PRINT CURRENT TRAINING RESULTS
            '''
            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                recall, _, mean_recall = Recall.compute(args['models']['hierarchical_pred'], per_class=True)
                recall_top3, _, mean_recall_top3 = Recall_top3.compute(args['models']['hierarchical_pred'], per_class=True)
                Recall.clear_data()
                Recall_top3.clear_data()

            if (batch_count % args['training']['print_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                print('VALIDATING, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), mR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), conn: %.4f, %.4f, sparsity: %.4f.'
                      % (rank, epoch, recall_top3[0], recall_top3[1], recall_top3[2], recall[0], recall[1], recall[2],
                         mean_recall_top3[0], mean_recall_top3[1], mean_recall_top3[2], mean_recall[0], mean_recall[1], mean_recall[2],
                         connectivity_recall/num_connected, connectivity_precision / num_connected_pred, torch.sum(topk_mask) / len(topk_mask)))

                test_record.append({'rank': rank, 'epoch': epoch, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                   'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                                   'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                   'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                                   'num_objs_average': torch.mean(num_graph_iter.float()).item(), 'num_objs_max': max(num_graph_iter).item(),
                                   'num_connected': num_connected, 'num_not_connected': num_not_connected})

                with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # append current logs
                    json.dump(test_record, f)
                dist.monitored_barrier()

    dist.monitored_barrier()
    print('FINISHED EVALUATING\n')
