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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluator import Evaluator_PC, Evaluator_SGD, Evaluator_PC_Top3, Evaluator_SGD_Top3
from model import MotifEmbed, MotifEmbedHier, MotifHead, MotifHeadHier
from utils import *
from dataset import object_class_alp2fre
from transformer import build_transformer
from torch.nn.utils.rnn import pad_sequence


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def eval_pc(gpu, args, test_subset):
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

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn,
                                              num_workers=0, pin_memory=True, drop_last=True, sampler=test_sampler)
    print("Finished loading the datasets...")

    start = []
    test_record = []
    with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # clear history logs
        json.dump(start, f)

    if args['models']['hierarchical_pred']:
        motif_embedding = DDP(MotifEmbedHier(input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
                                             num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'])).to(rank)
        motif_head = DDP(MotifHeadHier(input_dim=args['models']['embed_hidden_dim'])).to(rank)
    else:
        motif_embedding = DDP(MotifEmbed(input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'], feature_size=args['models']['feature_size'],
                                         num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'])).to(rank)
        motif_head = DDP(MotifHead(input_dim=args['models']['embed_hidden_dim'], output_dim=args['models']['num_relations'])).to(rank)

    transformer_encoder = DDP(build_transformer(args)).to(rank)
    transformer_encoder.eval()

    detr = DDP(build_detr101(args)).to(rank)
    backbone = DDP(detr.module.backbone).to(rank)
    input_proj = DDP(detr.module.input_proj).to(rank)
    feature_encoder = DDP(detr.module.transformer.encoder).to(rank)

    detr.eval()
    motif_embedding.eval()
    motif_head.eval()
    backbone.eval()
    input_proj.eval()
    feature_encoder.eval()

    map_location = {'cuda:%d' % rank: 'cuda:%d' % rank}
    if args['models']['hierarchical_pred']:
        motif_embedding.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHeadHier' + str(2) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)
        motif_head.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'MotifHeadHier' + str(args['training']['test_epoch']) + '_' + str(rank) + '.pth', map_location=map_location), strict=True)
    else:
        motif_embedding.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHead' + str(2) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)
        motif_head.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'MotifHead' + str(args['training']['test_epoch']) + '_' + str(rank) + '.pth', map_location=map_location), strict=True)
    transformer_encoder.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'TransformerEncoder' + str(args['training']['test_epoch']) + '_' + str(rank) + '.pth', map_location=map_location), strict=True)

    connectivity_recall, connectivity_precision, num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0
    Recall = Evaluator_PC(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])
    Recall_top3 = Evaluator_PC_Top3(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])

    print('Start Testing PC...')
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            '''
            PREPARE INPUT DATA
            '''
            try:
                images_o, images, image_depth, categories, super_categories, masks, bbox, relationships, subj_or_obj = data
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
            FORWARD PASS THROUGH BI-DIRECTIONAL RECURRENT NETWORKS
            '''
            del image_feature, masks, bbox, categories, super_categories, image_depth, relations_target, direction_target, h_graph, h_edge

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
                print('TESTING, rank: %d, R@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), mR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), conn: %.4f, %.4f, %d, sparsity: %.4f.'
                      % (rank, recall_top3[0], recall_top3[1], recall_top3[2], recall[0], recall[1], recall[2],
                         mean_recall_top3[0], mean_recall_top3[1], mean_recall_top3[2], mean_recall[0], mean_recall[1], mean_recall[2],
                         connectivity_recall / num_connected, connectivity_precision / num_connected_pred, num_connected, torch.sum(topk_mask) / len(topk_mask)))

                test_record.append({'rank': rank, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                   'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                                   'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                   'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                                   'num_objs_average': torch.mean(num_graph_iter.float()).item(), 'num_objs_max': max(num_graph_iter).item(),
                                   'num_connected': num_connected, 'num_not_connected': num_not_connected})

                with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # append current logs
                    json.dump(test_record, f)
                dist.monitored_barrier()

    dist.destroy_process_group()  # clean up
    print('FINISHED TESTING PC\n')


def eval_sgd(gpu, args, test_subset):
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

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn,
                                              num_workers=0, pin_memory=True, drop_last=True, sampler=test_sampler)
    print("Finished loading the datasets...")

    start = []
    test_record = []
    with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # clear history logs
        json.dump(start, f)

    if args['models']['hierarchical_pred']:
        motif_embedding = DDP(MotifEmbedHier(input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
                                             num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'])).to(rank)
        motif_head = DDP(MotifHeadHier(input_dim=args['models']['embed_hidden_dim'])).to(rank)
    else:
        motif_embedding = DDP(MotifEmbed(input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'], feature_size=args['models']['feature_size'],
                                         num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'])).to(rank)
        motif_head = DDP(MotifHead(input_dim=args['models']['embed_hidden_dim'], output_dim=args['models']['num_relations'])).to(rank)

    transformer_encoder = DDP(build_transformer(args)).to(rank)
    transformer_encoder.eval()

    detr = DDP(build_detr101(args)).to(rank)
    backbone = DDP(detr.module.backbone).to(rank)
    input_proj = DDP(detr.module.input_proj).to(rank)
    feature_encoder = DDP(detr.module.transformer.encoder).to(rank)

    detr.eval()
    motif_embedding.eval()
    motif_head.eval()
    backbone.eval()
    input_proj.eval()
    feature_encoder.eval()

    map_location = {'cuda:%d' % rank: 'cuda:%d' % rank}
    if args['models']['hierarchical_pred']:
        motif_embedding.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHeadHier' + str(2) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)
        motif_head.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'MotifHeadHier' + str(args['training']['test_epoch']) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)
    else:
        motif_embedding.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'EdgeHead' + str(2) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)
        motif_head.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'MotifHead' + str(args['training']['test_epoch']) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)
    transformer_encoder.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'TransformerEncoder' + str(args['training']['test_epoch']) + '_' + str(rank) + '.pth', map_location=map_location), strict=False)

    Recall = Evaluator_SGD(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])
    Recall_top3 = Evaluator_SGD_Top3(num_classes=50, iou_thresh=0.5, top_k=[20, 50, 100])

    sub2super_cat_dict = torch.load(args['dataset']['sub2super_cat_dict'])
    object_class_alp2fre_dict = object_class_alp2fre()

    print('Start Testing SGD...')
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            '''
            PREPARE INPUT DATA
            '''
            try:
                images, images_s, image_depth, categories_target, super_categories_target, masks_target, bbox_target, relationships, subj_or_obj = data
            except:
                continue

            images_s = [image.to(rank) for image in images_s]
            image_feature, pos_embed = backbone(nested_tensor_from_tensor_list(images_s))
            src, mask = image_feature[-1].decompose()
            src = input_proj(src).flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
            image_feature = feature_encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)

            image_feature = image_feature.permute(1, 2, 0)
            image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])
            image_depth = torch.stack([depth.to(rank) for depth in image_depth])
            categories_target = [category.to(rank) for category in categories_target]  # [batch_size][curr_num_obj, 1]
            bbox_target = [box.to(rank) for box in bbox_target]  # [batch_size][curr_num_obj, 4]

            images = [image.to(rank) for image in images]
            out_dict = detr(nested_tensor_from_tensor_list(images))

            logits_pred = torch.argmax(F.softmax(out_dict['pred_logits'], dim=2), dim=2)
            has_object_pred = logits_pred < 150
            logits_pred = torch.topk(F.softmax(out_dict['pred_logits'], dim=2), dim=2, k=args['models']['topk_cat'])[1].view(-1, 100, args['models']['topk_cat'])

            logits_pred_value = torch.topk(F.softmax(out_dict['pred_logits'], dim=2), dim=2, k=args['models']['topk_cat'])[0].view(-1, 100, args['models']['topk_cat'])
            cat_pred_confidence = [logits_pred_value[i, has_object_pred[i], :].flatten() for i in range(logits_pred_value.shape[0]) if torch.sum(has_object_pred[i]) > 0]

            categories_pred = [logits_pred[i, has_object_pred[i], :].flatten() for i in range(logits_pred.shape[0]) if torch.sum(has_object_pred[i]) > 0]  # (batch_size, num_obj, 150)
            # object category indices in pretrained DETR are different from our indices
            for i in range(len(categories_pred)):
                for j in range(len(categories_pred[i])):
                    categories_pred[i][j] = object_class_alp2fre_dict[categories_pred[i][j].item()]     # at this moment, keep cat whose top2 == 150 for convenience
            cat_mask = [categories_pred[i] != 150 for i in range(len(categories_pred))]

            bbox_pred = [out_dict['pred_boxes'][i, has_object_pred[i]] for i in range(logits_pred.shape[0]) if torch.sum(has_object_pred[i]) > 0]  # convert from 0-1 to 0-32
            for i in range(len(bbox_pred)):
                bbox_pred_c = bbox_pred[i].clone()
                bbox_pred[i][:, 0] = bbox_pred_c[:, 0] - 1.2 * bbox_pred_c[:, 2] / 2
                bbox_pred[i][:, 1] = bbox_pred_c[:, 0] + 1.2 * bbox_pred_c[:, 2] / 2
                bbox_pred[i][:, 2] = bbox_pred_c[:, 1] - 1.2 * bbox_pred_c[:, 3] / 2
                bbox_pred[i][:, 3] = bbox_pred_c[:, 1] + 1.2 * bbox_pred_c[:, 3] / 2

                bbox_pred[i][:, 0][bbox_pred[i][:, 0] < 0] = 0
                bbox_pred[i][:, 1][bbox_pred[i][:, 1] > 1] = 1
                bbox_pred[i][:, 2][bbox_pred[i][:, 2] < 0] = 0
                bbox_pred[i][:, 3][bbox_pred[i][:, 3] > 1] = 1
                bbox_pred[i] *= 32

                bbox_pred[i] = bbox_pred[i].repeat_interleave(args['models']['topk_cat'], dim=0)

            masks_pred = []
            for i in range(len(bbox_pred)):
                mask_pred = torch.zeros(bbox_pred[i].shape[0], 32, 32, dtype=torch.uint8).to(rank)
                for j, box in enumerate(bbox_pred[i]):
                    mask_pred[j, int(bbox_pred[i][j][2]):int(bbox_pred[i][j][3]), int(bbox_pred[i][j][0]):int(bbox_pred[i][j][1])] = 1
                masks_pred.append(mask_pred)

            for i in range(len(categories_pred)):
                categories_pred[i] = categories_pred[i][cat_mask[i]]
                cat_pred_confidence[i] = cat_pred_confidence[i][cat_mask[i]]
                bbox_pred[i] = bbox_pred[i][cat_mask[i]]
                masks_pred[i] = masks_pred[i][cat_mask[i]]

            # non-maximum suppression
            for i in range(len(bbox_pred)):
                bbox_pred[i] = bbox_pred[i][:, [0, 2, 1, 3]]
                nms_keep_idx = None
                for cls in torch.unique(categories_pred[i]):  # per class nms
                    curr_class_idx = categories_pred[i] == cls
                    curr_nms_keep_idx = torchvision.ops.nms(boxes=bbox_pred[i][curr_class_idx], scores=cat_pred_confidence[i][curr_class_idx],
                                                            iou_threshold=args['models']['nms'])       # requires (x1, y1, x2, y2)
                    if nms_keep_idx is None:
                        nms_keep_idx = (torch.nonzero(curr_class_idx).flatten())[curr_nms_keep_idx]
                    else:
                        nms_keep_idx = torch.hstack((nms_keep_idx, (torch.nonzero(curr_class_idx).flatten())[curr_nms_keep_idx]))
                bbox_pred[i] = bbox_pred[i][:, [0, 2, 1, 3]]       # convert back to (x1, x2, y1, y2)

                categories_pred[i] = categories_pred[i][nms_keep_idx]
                cat_pred_confidence[i] = cat_pred_confidence[i][nms_keep_idx]
                bbox_pred[i] = bbox_pred[i][nms_keep_idx]
                masks_pred[i] = masks_pred[i][nms_keep_idx]

            # after nms
            super_categories_pred = [[sub2super_cat_dict[c.item()] for c in categories_pred[i]] for i in range(len(categories_pred))]
            super_categories_pred = [[torch.as_tensor(sc).to(rank) for sc in super_category] for super_category in super_categories_pred]

            '''
            PREPARE TARGETS
            Different data samples in a batch have different num of RNN iterations, i.e., different number of objects in images. To do mini-batch parallel training,
            need to convert size [batch_size][curr_num_obj][from 1 to curr_num_obj-1] to [max_num_obj][from 1 to max_num_obj-1][<=batch_size]
            relations_target and direction_target: matched targets for each prediction
            cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target, relation_target_origin: sets of original unmatched targets
            '''
            cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target, relation_target \
                = match_target_sgd(rank, relationships, subj_or_obj, categories_target, bbox_target)

            '''
            FORWARD PASS
            Different data samples in a batch have different num of RNN iterations, i.e., different number of objects in images.
            To do mini-batch parallel training, in each graph-level iteration,
            only stack the data samples whose total num of RNN iterations not exceeding the current iteration.
            '''
            hidden_states = None
            which_in_batch_all = None
            cat_subjects_pred = None
            cat_objects_pred = None
            cat_subject_conf = None
            cat_object_conf = None
            bbox_subjects_pred = None
            bbox_objects_pred = None

            num_graph_iter = torch.as_tensor([len(mask) for mask in masks_pred])
            for graph_iter in range(max(num_graph_iter)):
                which_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)

                curr_graph_masks = torch.stack([torch.unsqueeze(masks_pred[i][graph_iter], dim=0) for i in which_in_batch])
                h_graph = torch.cat((image_feature[which_in_batch] * curr_graph_masks, image_depth[which_in_batch] * curr_graph_masks), dim=1)  # (bs, 256, 64, 64), (bs, 1, 64, 64)
                cat_graph_pred = torch.tensor([torch.unsqueeze(categories_pred[i][graph_iter], dim=0) for i in which_in_batch]).to(rank)
                bbox_graph_pred = torch.stack([bbox_pred[i][graph_iter] for i in which_in_batch]).to(rank)
                cat_graph_confidence = torch.hstack([cat_pred_confidence[i][graph_iter] for i in which_in_batch])

                for edge_iter in range(graph_iter):
                    curr_edge_masks = torch.stack([torch.unsqueeze(masks_pred[i][edge_iter], dim=0) for i in which_in_batch])  # seg mask of every prev obj
                    h_edge = torch.cat((image_feature[which_in_batch] * curr_edge_masks, image_depth[which_in_batch] * curr_edge_masks), dim=1)
                    cat_edge_pred = torch.tensor([torch.unsqueeze(categories_pred[i][edge_iter], dim=0) for i in which_in_batch]).to(rank)
                    bbox_edge_pred = torch.stack([bbox_pred[i][edge_iter] for i in which_in_batch]).to(rank)
                    cat_edge_confidence = torch.hstack([cat_pred_confidence[i][edge_iter] for i in which_in_batch])

                    # filter out subject-object pairs whose iou=0
                    joint_intersect = torch.logical_or(curr_graph_masks, curr_edge_masks)
                    joint_union = torch.logical_and(curr_graph_masks, curr_edge_masks)
                    joint_iou = (torch.sum(torch.sum(joint_intersect, dim=-1), dim=-1) / torch.sum(torch.sum(joint_union, dim=-1), dim=-1)).flatten()
                    joint_iou[torch.isinf(joint_iou)] = 0
                    iou_mask = joint_iou > 0

                    if torch.sum(iou_mask) == 0:
                        continue

                    scat_graph_pred = []    # they are not tensors but lists, which requires special mask manipulations
                    scat_edge_pred = []
                    for count, i in enumerate(which_in_batch):
                        if iou_mask[count]:
                            scat_graph_pred.append(super_categories_pred[i][graph_iter])
                            scat_edge_pred.append(super_categories_pred[i][edge_iter])

                    """
                    FIRST DIRECTION
                    """
                    hidden_state = motif_embedding(h_graph[iou_mask], h_edge[iou_mask], cat_graph_pred[iou_mask], cat_edge_pred[iou_mask], scat_graph_pred, scat_edge_pred, rank)
                    if not args['models']['hierarchical_pred']:
                        hidden_state = torch.hstack((hidden_state[:, :-1], torch.zeros(hidden_state.shape[0], 4).to(rank), hidden_state[:, -1].view(-1, 1)))

                    if hidden_states is None:
                        hidden_states = hidden_state
                    else:
                        hidden_states = torch.vstack((hidden_states, hidden_state))  # (bs, 4480)
                    if which_in_batch_all is None:
                        which_in_batch_all = which_in_batch[iou_mask]
                    else:
                        which_in_batch_all = torch.hstack((which_in_batch_all, which_in_batch[iou_mask]))

                    if cat_subjects_pred is None:
                        cat_subjects_pred = cat_graph_pred[iou_mask]
                    else:
                        cat_subjects_pred = torch.hstack((cat_subjects_pred, cat_graph_pred[iou_mask]))
                    if cat_objects_pred is None:
                        cat_objects_pred = cat_edge_pred[iou_mask]
                    else:
                        cat_objects_pred = torch.hstack((cat_objects_pred, cat_edge_pred[iou_mask]))
                    if cat_subject_conf is None:
                        cat_subject_conf = cat_graph_confidence[iou_mask]
                    else:
                        cat_subject_conf = torch.hstack((cat_subject_conf, cat_graph_confidence[iou_mask]))
                    if cat_object_conf is None:
                        cat_object_conf = cat_edge_confidence[iou_mask]
                    else:
                        cat_object_conf = torch.hstack((cat_object_conf, cat_edge_confidence[iou_mask]))

                    if bbox_subjects_pred is None:
                        bbox_subjects_pred = bbox_graph_pred[iou_mask]
                    else:
                        bbox_subjects_pred = torch.vstack((bbox_subjects_pred, bbox_graph_pred[iou_mask]))
                    if bbox_objects_pred is None:
                        bbox_objects_pred = bbox_edge_pred[iou_mask]
                    else:
                        bbox_objects_pred = torch.vstack((bbox_objects_pred, bbox_edge_pred[iou_mask]))

                    """
                    SECOND DIRECTION
                    """
                    hidden_state = motif_embedding(h_edge[iou_mask], h_graph[iou_mask], cat_edge_pred[iou_mask], cat_graph_pred[iou_mask], scat_edge_pred, scat_graph_pred, rank)
                    if not args['models']['hierarchical_pred']:
                        hidden_state = torch.hstack((hidden_state[:, :-1], torch.zeros(hidden_state.shape[0], 4).to(rank), hidden_state[:, -1].view(-1, 1)))

                    hidden_states = torch.vstack((hidden_states, hidden_state))  # (bs, 4480)
                    which_in_batch_all = torch.hstack((which_in_batch_all, which_in_batch[iou_mask]))

                    cat_subjects_pred = torch.hstack((cat_subjects_pred, cat_edge_pred[iou_mask]))
                    cat_objects_pred = torch.hstack((cat_objects_pred, cat_graph_pred[iou_mask]))
                    cat_subject_conf = torch.hstack((cat_subject_conf, cat_edge_confidence[iou_mask]))
                    cat_object_conf = torch.hstack((cat_object_conf, cat_graph_confidence[iou_mask]))
                    bbox_subjects_pred = torch.vstack((bbox_subjects_pred, bbox_edge_pred[iou_mask]))
                    bbox_objects_pred = torch.vstack((bbox_objects_pred, bbox_graph_pred[iou_mask]))

            '''
            FORWARD PASS THROUGH TRANSFORMER ENCODER
            '''
            # Add binary mask to filter out only top k most confident predictions into the rn
            confidence_all = torch.max(torch.vstack((torch.max(hidden_states[:, -54:-39], dim=1)[0],
                                                     torch.max(hidden_states[:, -39:-28], dim=1)[0],
                                                     torch.max(hidden_states[:, -28:-4], dim=1)[0])), dim=0)[0]  # values

            topk_mask = torch.zeros(confidence_all.shape[0], dtype=bool).to(rank)
            for i in range(hidden_states.shape[0]):
                if iou(bbox_subjects_pred[i], bbox_objects_pred[i]) == 0:
                    confidence_all[i] = -math.inf

            all_imgs = torch.unique(which_in_batch_all)
            for img in all_imgs:
                curr_img = torch.nonzero(which_in_batch_all == img).flatten()
                curr_confidence = confidence_all[curr_img]
                curr_confidence_sorted = torch.sort(curr_confidence, descending=True)[0]
                top_confidence = curr_confidence_sorted[int(args['transformers']['sparsity'] * len(curr_confidence))]  # curr_confidence_sorted[min(50, len(curr_confidence)-1)]
                topk_mask[curr_img[torch.nonzero(curr_confidence >= top_confidence)]] = 1

            topk_mask[torch.isinf(confidence_all)] = 0
            topk_mask[torch.sigmoid(hidden_states[:, -1]) >= 0.5] = 1

            hidden_states_filtered = hidden_states[topk_mask]
            which_in_batch_all_filtered = which_in_batch_all[topk_mask]

            hidden_states_batched = pad_sequence([hidden_states_filtered[which_in_batch_all_filtered == img] for img in torch.unique(which_in_batch_all_filtered, sorted=True)])
            num_motifs = hidden_states_batched.shape[0]
            # print('hidden_states_batched', hidden_states.shape, hidden_states_batched.shape)     # size (max seq length, num of img in batch, feature dim)

            pad_mask = torch.ones(hidden_states_batched.shape[1], num_motifs, dtype=torch.bool).to(rank)
            for img_idx, img in enumerate(torch.unique(which_in_batch_all_filtered, sorted=True)):
                pad_mask[img_idx, torch.sum(which_in_batch_all_filtered == img):] = 0
            pad_mask = pad_mask.T

            hidden_states_batched = transformer_encoder(hidden_states_batched, num_motifs, rank)
            hidden_states_out = hidden_states[topk_mask, :512] + hidden_states_batched[pad_mask]
            # print('hidden_states_batched_out', hidden_states_batched.shape)

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

            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                Recall.accumulate_pred(which_in_batch_all, new_relation_pred, new_super_relation_pred, cat_subjects_pred, cat_objects_pred, bbox_subjects_pred, bbox_objects_pred,
                                       cat_subject_conf, cat_object_conf, torch.log(torch.sigmoid(new_connectivity_pred[:, 0])))
                Recall_top3.accumulate_pred(which_in_batch_all, new_relation_pred, new_super_relation_pred, cat_subjects_pred, cat_objects_pred, bbox_subjects_pred, bbox_objects_pred,
                                            cat_subject_conf, cat_object_conf, torch.log(torch.sigmoid(new_connectivity_pred[:, 0])))

            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                Recall.accumulate_target(relation_target, cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target)
                Recall_top3.accumulate_target(relation_target, cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target)

            '''
            EVALUATE AND PRINT CURRENT TRAINING RESULTS
            '''
            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                recall, _, mean_recall = Recall.compute(args['models']['hierarchical_pred'], per_class=True)
                recall_top3, _, mean_recall_top3 = Recall_top3.compute(args['models']['hierarchical_pred'], per_class=True)
                Recall.clear_data()
                Recall_top3.clear_data()

            if (batch_count % args['training']['print_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                print('VALIDATING, rank: %d, R@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), mR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f)'
                      % (rank, recall_top3[0], recall_top3[1], recall_top3[2], recall[0], recall[1], recall[2],
                         mean_recall_top3[0], mean_recall_top3[1], mean_recall_top3[2], mean_recall[0], mean_recall[1], mean_recall[2]))

                test_record.append({'rank': rank, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                   'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                                   'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                   'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                                   'num_objs_average': torch.mean(num_graph_iter.float()).item(), 'num_objs_max': max(num_graph_iter).item()})
                with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # append current logs
                    json.dump(test_record, f)

    dist.monitored_barrier()
    print('FINISHED TESTING SGD\n')
