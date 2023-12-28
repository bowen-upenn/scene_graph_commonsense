import torch
import numpy as np
from tqdm import tqdm
import os
import json
from utils import *


def process_image_features(args, images, detr, rank):
    images = torch.stack(images).to(rank)
    image_feature, pos_embed = detr.module.backbone(nested_tensor_from_tensor_list(images))
    src, mask = image_feature[-1].decompose()
    src = detr.module.input_proj(src).flatten(2).permute(2, 0, 1)
    pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
    image_feature = detr.module.transformer.encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)
    image_feature = image_feature.permute(1, 2, 0)
    image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])
    return image_feature


def train_one_direction(relation_classifier, args, h_sub, h_obj, cat_sub, cat_obj, spcat_sub, spcat_obj, bbox_sub, bbox_obj, h_sub_aug, h_obj_aug, iou_mask, rank, graph_iter, edge_iter, keep_in_batch,
                        Recall, Recall_top3, criterion_relationship, criterion_connectivity, relations_target, direction_target, batch_count, hidden_cat_accumulated, hidden_cat_labels_accumulated,
                        commonsense_yes_triplets, commonsense_no_triplets, len_train_loader, first_direction=True):

    if args['models']['hierarchical_pred']:
        relation_1, relation_2, relation_3, super_relation, connectivity, hidden, hidden_aug \
            = relation_classifier(h_sub, h_obj, cat_sub, cat_obj, spcat_sub, spcat_obj, rank, h_sub_aug, h_obj_aug)
        relation = [relation_1, relation_2, relation_3]
        hidden_cat = torch.cat((hidden.unsqueeze(1), hidden_aug.unsqueeze(1)), dim=1)

        # match with the commonsense filtering pool
        relation_pred = torch.hstack((torch.argmax(relation_1, dim=1),
                                      torch.argmax(relation_2, dim=1) + args['models']['num_geometric'],
                                      torch.argmax(relation_3, dim=1) + args['models']['num_geometric'] + args['models']['num_possessive']))
        triplets = torch.hstack((cat_sub.repeat(3).unsqueeze(1), relation_pred.unsqueeze(1), cat_obj.repeat(3).unsqueeze(1)))

    else:
        relation, connectivity, hidden, hidden_aug = relation_classifier(h_sub, h_obj, cat_sub, cat_obj, spcat_sub, spcat_obj, rank, h_sub_aug, h_obj_aug)
        hidden_cat = torch.cat((hidden.unsqueeze(1), hidden_aug.unsqueeze(1)), dim=1)
        super_relation = None

        # match with the commonsense filtering pool
        relation_pred = torch.argmax(relation, dim=1)
        triplets = torch.hstack((cat_sub.unsqueeze(1), relation_pred.unsqueeze(1), cat_obj.unsqueeze(1)))

    # evaluate on the commonsense for all predictions, regardless of whether they match with the ground truth or not
    not_in_yes_dict = args['training']['lambda_cs_weak'] * torch.tensor([tuple(triplets[i].cpu().tolist()) not in commonsense_yes_triplets for i in range(len(triplets))], dtype=torch.float).to(rank)
    is_in_no_dict = args['training']['lambda_cs_strong'] * torch.tensor([tuple(triplets[i].cpu().tolist()) in commonsense_no_triplets for i in range(len(triplets))], dtype=torch.float).to(rank)
    loss_commonsense = (not_in_yes_dict + is_in_no_dict).mean()

    # evaluate on the connectivity
    if first_direction:
        not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 1)[0]  # which data samples in curr keep_in_batch are not connected
    else:
        not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 0)[0]
    num_not_connected = len(not_connected)
    temp = criterion_connectivity(connectivity[not_connected, 0], torch.zeros(len(not_connected)).to(rank))
    loss_connectivity = 0.0 if torch.isnan(temp) else args['training']['lambda_not_connected'] * temp

    if first_direction:
        connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 1)[0]  # which data samples in curr keep_in_batch are connected
    else:
        connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 0)[0]
    num_connected = len(connected)
    connected_pred = torch.nonzero(torch.sigmoid(connectivity[:, 0]) >= 0.5).flatten()
    connectivity_precision = torch.sum(relations_target[graph_iter - 1][edge_iter][connected_pred] != -1)
    num_connected_pred = len(connected_pred)

    connected_indices = torch.zeros(len(hidden_cat), dtype=torch.bool).to(rank)
    hidden_cat = hidden_cat[connected]
    connected_indices[connected] = 1

    # evaluate on the relationships
    loss_relationship = 0.0
    connectivity_recall = 0.0
    if len(connected) > 0:
        temp = criterion_connectivity(connectivity[connected, 0], torch.ones(len(connected)).to(rank))
        loss_connectivity = 0.0 if torch.isnan(temp) else temp
        connectivity_recall = torch.sum(torch.round(torch.sigmoid(connectivity[connected, 0])))

        loss_relationship = calculate_losses_on_relationships(args, relation, super_relation, connected, relations_target[graph_iter - 1][edge_iter], criterion_relationship)

        hidden_cat_labels = relations_target[graph_iter - 1][edge_iter][connected]
        for index, batch_index in enumerate(keep_in_batch[connected]):
            hidden_cat_accumulated[batch_index].append(hidden_cat[index])
            hidden_cat_labels_accumulated[batch_index].append(hidden_cat_labels[index])

    # evaluate recall@k scores
    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
    relations_target_directed[not_connected] = -1

    if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len_train_loader):
        if args['models']['hierarchical_pred']:
            relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
        Recall.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                          cat_sub, cat_obj, cat_sub, cat_obj, bbox_sub, bbox_obj, bbox_sub, bbox_obj, iou_mask)
        if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
            Recall_top3.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                   cat_sub, cat_obj, cat_sub, cat_obj, bbox_sub, bbox_obj, bbox_sub, bbox_obj, iou_mask)

    return loss_relationship, loss_connectivity, loss_commonsense, num_not_connected, num_connected, num_connected_pred, connectivity_precision, \
           connectivity_recall, hidden_cat_accumulated, hidden_cat_labels_accumulated


def calculate_losses_on_relationships(args, relation, super_relation, connected, curr_relations_target, criterion_relationship, pseudo_label_mask=None, lambda_pseudo=1):
    loss_relationship = 0.0
    is_hierarchical_pred = args['models']['hierarchical_pred']
    is_train_semi = args['training']['run_mode'] == 'train_semi'

    # Only proceed if there are any connected edges to evaluate
    if connected.numel() == 0:
        return loss_relationship

    # Compute super category losses if hierarchical_pred is enabled
    if is_hierarchical_pred:
        criterion_relationship_1, criterion_relationship_2, criterion_relationship_3, criterion_super_relationship = criterion_relationship
        super_relation_target = super_relation_processing(args, connected, curr_relations_target)

        # Compute losses for super relationships
        if is_train_semi:
            # Get the mask for pseudo labels in the super relation context
            curr_pseudo_labels = pseudo_label_mask[connected]
            # Calculate super relation loss separately for pseudo and true labels
            loss_pseudo, loss_true = calculate_semi_supervised_loss(super_relation[connected], super_relation_target, curr_pseudo_labels, criterion_super_relationship, lambda_pseudo)
            loss_relationship += loss_pseudo + loss_true
        else:
            loss_relationship += criterion_super_relationship(super_relation[connected], super_relation_target)

        # Compute sub category losses
        connected_1 = torch.nonzero(curr_relations_target[connected] < args['models']['num_geometric']).flatten()  # geometric
        connected_2 = torch.nonzero(torch.logical_and(curr_relations_target[connected] >= args['models']['num_geometric'],
                                                      curr_relations_target[connected] < args['models']['num_geometric'] + args['models']['num_possessive'])).flatten()  # possessive
        connected_3 = torch.nonzero(curr_relations_target[connected] >= args['models']['num_geometric'] + args['models']['num_possessive']).flatten()  # semantic
        connected_sub = [connected_1, connected_2, connected_3]

        for i, (criterion_rel, offset) in enumerate(zip(
                [criterion_relationship_1, criterion_relationship_2, criterion_relationship_3],
                [0, args['models']['num_geometric'], args['models']['num_geometric'] + args['models']['num_possessive']]
        )):
            connected_i = connected_sub[i]

            if is_train_semi and connected_i.numel() > 0:
                # Calculate losses for the current category
                loss_pseudo, loss_true = calculate_semi_supervised_loss(
                    relation[i][connected][connected_i], curr_relations_target[connected][connected_i] - offset, curr_pseudo_labels[connected_i], criterion_rel, lambda_pseudo)
                loss_relationship += loss_pseudo + loss_true
            elif connected_i.numel() > 0:  # Non-semi-supervised or non-empty connected indices
                loss_relationship += criterion_rel(relation[i][connected][connected_i], curr_relations_target[connected][connected_i] - offset)

    # Compute losses if not using hierarchical predictions
    else:
        if is_train_semi:
            curr_pseudo_labels = pseudo_label_mask[connected]
            loss_pseudo, loss_true = calculate_semi_supervised_loss(relation[connected], curr_relations_target[connected], curr_pseudo_labels, criterion_relationship, lambda_pseudo)
            loss_relationship += loss_pseudo + loss_true
        else:
            # temp = criterion_relationship(relation[connected], curr_relations_target[connected])
            # loss_relationship += 0.0 if torch.isnan(temp) else temp
            loss_relationship += criterion_relationship(relation[connected], curr_relations_target[connected])

    return loss_relationship


def evaluate_one_direction(relation_classifier, args, h_sub, h_obj, cat_sub, cat_obj, spcat_sub, spcat_obj, bbox_sub, bbox_obj, iou_mask, rank, graph_iter, edge_iter, keep_in_batch,
                           Recall, Recall_top3, relations_target, direction_target, batch_count, len_test_loader, first_direction=True):
    if args['models']['hierarchical_pred']:
        relation_1, relation_2, relation_3, super_relation, connectivity, _, _ = relation_classifier(h_sub, h_obj, cat_sub, cat_obj, spcat_sub, spcat_obj, rank)
        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
    else:
        relation, connectivity, _, _ = relation_classifier(h_sub, h_obj, cat_sub, cat_obj, spcat_sub, spcat_obj, rank)
        super_relation = None

    if first_direction:
        not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 1)[0]  # which data samples in curr keep_in_batch are not connected
        connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 1)[0]  # which data samples in curr keep_in_batch are connected
    else:
        not_connected = torch.where(direction_target[graph_iter - 1][edge_iter] != 0)[0]
        connected = torch.where(direction_target[graph_iter - 1][edge_iter] == 0)[0]  # which data samples in curr keep_in_batch are connected
    num_not_connected = len(not_connected)
    num_connected = len(connected)
    connected_pred = torch.nonzero(torch.sigmoid(connectivity[:, 0]) >= 0.5).flatten()
    connectivity_precision = torch.sum(relations_target[graph_iter - 1][edge_iter][connected_pred] != -1)
    num_connected_pred = len(connected_pred)

    connectivity_recall = 0.0
    if len(connected) > 0:
        connectivity_recall = torch.sum(torch.round(torch.sigmoid(connectivity[connected, 0])))

    # evaluate recall@k scores
    relations_target_directed = relations_target[graph_iter - 1][edge_iter].clone()
    relations_target_directed[not_connected] = -1

    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len_test_loader):
        Recall.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                          cat_sub, cat_obj, cat_sub, cat_obj, bbox_sub, bbox_obj, bbox_sub, bbox_obj, iou_mask)
        if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
            Recall_top3.accumulate(keep_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                   cat_sub, cat_obj, cat_sub, cat_obj, bbox_sub, bbox_obj, bbox_sub, bbox_obj, iou_mask)

    return num_not_connected, num_connected, num_connected_pred, connectivity_precision, connectivity_recall
