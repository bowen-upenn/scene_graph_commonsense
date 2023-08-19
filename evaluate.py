import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import yaml
import os
import math
import torchvision
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluator import *
from model import FlatMotif, HierMotif
from utils import *
from dataset_utils import object_class_alp2fre


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def inference(rank, args, test_dataset, file_name=None, file_idx=None):
    """
    This function inference the module on predicate classification tasks for a single image.
    :param rank: current gpu index
    :param args: input arguments in config.yaml
    :param test_dataset: testing dataset
    """
    if args['models']['hierarchical_pred']:
        local_predictor = HierMotif(args=args, input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
                                    num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'],
                                    num_geometric=args['models']['num_geometric'], num_possessive=args['models']['num_possessive'], num_semantic=args['models']['num_semantic']).to(rank)
    else:
        local_predictor = FlatMotif(args=args, input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'], feature_size=args['models']['feature_size'],
                                    num_classes=args['models']['num_classes']).to(rank)

    detr = build_detr101(args).to(rank)
    detr.eval()
    local_predictor.eval()

    if args['models']['hierarchical_pred']:
        saved_state_dict = torch.load(args['training']['checkpoint_path'] + 'HierMotif' + str(args['training']['test_epoch']) + '_0' + '.pth')
    else:
        saved_state_dict = torch.load(args['training']['checkpoint_path'] + 'FlatMotif' + str(args['training']['test_epoch']) + '_0' + '.pth')
    renamed_state_dict = remove_ddp_module_in_weights(saved_state_dict)
    local_predictor.load_state_dict(renamed_state_dict)

    connectivity_recall, connectivity_precision, num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0
    recall_top3, recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs, wmap_rel, wmap_phrase = None, None, None, None, None, None, None, None

    Recall = Evaluator_PC(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])
    if args['dataset']['dataset'] == 'vg':
        Recall_top3 = Evaluator_PC_Top3(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])

    print('Start Testing PC...')
    with torch.no_grad():
        """
        PREPARE INPUT DATA
        """
        image, image_depth, categories, super_categories, bbox, relationships, subj_or_obj = test_dataset.load_one_image(file_name, idx=file_idx, return_annot=True)

        image = torch.stack(image).to(rank)
        image_feature, pos_embed = detr.backbone(nested_tensor_from_tensor_list(image))
        src, mask = image_feature[-1].decompose()
        src = detr.input_proj(src).flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
        image_feature = detr.transformer.encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)
        image_feature = image_feature.permute(1, 2, 0)
        image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])


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
            which_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)
            relations_target.append(torch.vstack([relationships[i][graph_iter] for i in which_in_batch]).T.to(rank))  # integer labels
            direction_target.append(torch.vstack([subj_or_obj[i][graph_iter] for i in which_in_batch]).T.to(rank))

        """
        FORWARD PASS THROUGH THE LOCAL PREDICTOR
        """
        num_graph_iter = torch.as_tensor([len(mask) for mask in masks])
        for graph_iter in range(max(num_graph_iter)):
            which_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)

            curr_graph_masks = torch.stack([torch.unsqueeze(masks[i][graph_iter], dim=0) for i in which_in_batch])
            h_graph = torch.cat((image_feature[which_in_batch] * curr_graph_masks, image_depth[which_in_batch] * curr_graph_masks), dim=1)  # (bs, 256, 64, 64), (bs, 1, 64, 64)
            cat_graph = torch.tensor([torch.unsqueeze(categories[i][graph_iter], dim=0) for i in which_in_batch]).to(rank)
            scat_graph = [super_categories[i][graph_iter] for i in which_in_batch] if super_categories[0] is not None else None
            bbox_graph = torch.stack([bbox[i][graph_iter] for i in which_in_batch]).to(rank)

            for edge_iter in range(graph_iter):
                curr_edge_masks = torch.stack([torch.unsqueeze(masks[i][edge_iter], dim=0) for i in which_in_batch])  # seg mask of every prev obj
                h_edge = torch.cat((image_feature[which_in_batch] * curr_edge_masks, image_depth[which_in_batch] * curr_edge_masks), dim=1)
                cat_edge = torch.tensor([torch.unsqueeze(categories[i][edge_iter], dim=0) for i in which_in_batch]).to(rank)
                scat_edge = [super_categories[i][edge_iter] for i in which_in_batch] if super_categories[0] is not None else None
                bbox_edge = torch.stack([bbox[i][edge_iter] for i in which_in_batch]).to(rank)

                """
                FIRST DIRECTION
                """
                if args['models']['hierarchical_pred']:
                    relation_1, relation_2, relation_3, super_relation, connectivity, _, _ = local_predictor(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
                    relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                else:
                    relation, connectivity = local_predictor(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
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

                Recall.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                  cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                    Recall_top3.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                           cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

                """
                SECOND DIRECTION
                """
                if args['models']['hierarchical_pred']:
                    relation_1, relation_2, relation_3, super_relation, connectivity, _, _= local_predictor(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)
                    relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                else:
                    relation, connectivity = local_predictor(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)
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

                Recall.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                  cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                    Recall_top3.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                           cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

        """
        EVALUATE AND PRINT CURRENT RESULTS
        """
        if args['dataset']['dataset'] == 'vg':
            recall, _, mean_recall, recall_zs, _, mean_recall_zs = Recall.compute(per_class=True)
            if args['models']['hierarchical_pred']:
                recall_top3, _, mean_recall_top3 = Recall_top3.compute(per_class=True)
                Recall_top3.clear_data()
        else:
            recall, _, mean_recall, _, _, _ = Recall.compute(per_class=True)

        top_k_predictions = Recall.get_top_k_predictions(top_k=10)
        Recall.clear_data()

    print('FINISHED INFERENCE PC\n')
    return top_k_predictions


def eval_pc(gpu, args, test_subset):
    """
    This function evaluates the module on predicate classification tasks.
    :param gpu: current gpu index
    :param args: input arguments in config.yaml
    :param test_subset: testing dataset
    """
    rank = gpu
    world_size = torch.cuda.device_count()
    setup(rank, world_size)
    print('rank', rank, 'torch.distributed.is_initialized', torch.distributed.is_initialized())

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=test_sampler)
    print("Finished loading the datasets...")

    start = []
    test_record = []
    with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # clear history logs
        json.dump(start, f)

    if args['models']['hierarchical_pred']:
        local_predictor = DDP(HierMotif(args=args, input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
                                     num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'],
                                     num_geometric=args['models']['num_geometric'], num_possessive=args['models']['num_possessive'], num_semantic=args['models']['num_semantic'])).to(rank)
    else:
        local_predictor = DDP(FlatMotif(args=args, input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'], feature_size=args['models']['feature_size'],
                                 num_classes=args['models']['num_classes'])).to(rank)

    if args['models']['detr_or_faster_rcnn'] == 'detr':
        detr = DDP(build_detr101(args)).to(rank)
        local_predictor.eval()
    else:
        print('Unknown model.')

    map_location = {'cuda:%d' % rank: 'cuda:%d' % 0}
    if args['models']['hierarchical_pred']:
        local_predictor.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'HierMotif' + str(args['training']['test_epoch']) + '_0' + '.pth', map_location=map_location))
    else:
        local_predictor.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'FlatMotif' + str(args['training']['test_epoch']) + '_0' + '.pth', map_location=map_location))

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
                images, _, image_depth, categories, super_categories, bbox, relationships, subj_or_obj = data
            except:
                continue

            images = torch.stack(images).to(rank)
            image_feature, pos_embed = detr.module.backbone(nested_tensor_from_tensor_list(images))
            src, mask = image_feature[-1].decompose()
            src = detr.module.input_proj(src).flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
            image_feature = detr.module.transformer.encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)
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
                which_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)
                relations_target.append(torch.vstack([relationships[i][graph_iter] for i in which_in_batch]).T.to(rank))  # integer labels
                direction_target.append(torch.vstack([subj_or_obj[i][graph_iter] for i in which_in_batch]).T.to(rank))

            """
            FORWARD PASS THROUGH THE LOCAL PREDICTOR
            """
            num_graph_iter = torch.as_tensor([len(mask) for mask in masks])
            for graph_iter in range(max(num_graph_iter)):
                which_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)

                curr_graph_masks = torch.stack([torch.unsqueeze(masks[i][graph_iter], dim=0) for i in which_in_batch])
                h_graph = torch.cat((image_feature[which_in_batch] * curr_graph_masks, image_depth[which_in_batch] * curr_graph_masks), dim=1)  # (bs, 256, 64, 64), (bs, 1, 64, 64)
                cat_graph = torch.tensor([torch.unsqueeze(categories[i][graph_iter], dim=0) for i in which_in_batch]).to(rank)
                scat_graph = [super_categories[i][graph_iter] for i in which_in_batch] if super_categories[0] is not None else None
                bbox_graph = torch.stack([bbox[i][graph_iter] for i in which_in_batch]).to(rank)

                for edge_iter in range(graph_iter):
                    curr_edge_masks = torch.stack([torch.unsqueeze(masks[i][edge_iter], dim=0) for i in which_in_batch])  # seg mask of every prev obj
                    h_edge = torch.cat((image_feature[which_in_batch] * curr_edge_masks, image_depth[which_in_batch] * curr_edge_masks), dim=1)
                    cat_edge = torch.tensor([torch.unsqueeze(categories[i][edge_iter], dim=0) for i in which_in_batch]).to(rank)
                    scat_edge = [super_categories[i][edge_iter] for i in which_in_batch] if super_categories[0] is not None else None
                    bbox_edge = torch.stack([bbox[i][edge_iter] for i in which_in_batch]).to(rank)

                    """
                    FIRST DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity, _, _ = local_predictor(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = local_predictor(h_graph, h_edge, cat_graph, cat_edge, scat_graph, scat_edge, rank)
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

                    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_subset)):
                        Recall.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                          cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                        if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                            Recall_top3.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                                   cat_graph, cat_edge, cat_graph, cat_edge, bbox_graph, bbox_edge, bbox_graph, bbox_edge)

                    """
                    SECOND DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity, _, _= local_predictor(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = local_predictor(h_edge, h_graph, cat_edge, cat_graph, scat_edge, scat_graph, rank)
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

                    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_subset)):
                        Recall.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
                                          cat_edge, cat_graph, cat_edge, cat_graph, bbox_graph, bbox_edge, bbox_graph, bbox_edge)
                        if args['dataset']['dataset'] == 'vg' and args['models']['hierarchical_pred']:
                            Recall_top3.accumulate(which_in_batch, relation, relations_target_directed, super_relation, torch.log(torch.sigmoid(connectivity[:, 0])),
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
                record_test_results(args, test_record, rank, args['training']['test_epoch'], recall_top3, recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs,
                                    connectivity_recall, num_connected, num_not_connected, connectivity_precision, num_connected_pred, wmap_rel, wmap_phrase)

    dist.destroy_process_group()  # clean up
    print('FINISHED TESTING PC\n')



def eval_sgd(gpu, args, test_subset):
    """
    This function evaluates the module on scene graph detection tasks.
    :param gpu: current gpu index
    :param args: input arguments in config.yaml
    :param test_subset: testing dataset
    """
    rank = gpu
    world_size = torch.cuda.device_count()
    setup(rank, world_size)
    print('rank', rank, 'torch.distributed.is_initialized', torch.distributed.is_initialized())

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=test_sampler)
    print("Finished loading the datasets...")

    start = []
    test_record = []
    with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # clear history logs
        json.dump(start, f)

    if args['models']['hierarchical_pred']:
        local_predictor = DDP(HierMotif(args=args, input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
                                     num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'],
                                     num_geometric=args['models']['num_geometric'], num_possessive=args['models']['num_possessive'], num_semantic=args['models']['num_semantic'])).to(rank)
    else:
        local_predictor = DDP(FlatMotif(args=args, input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'], feature_size=args['models']['feature_size'],
                                 num_classes=args['models']['num_classes'])).to(rank)

    detr = DDP(build_detr101(args)).to(rank)
    backbone = DDP(detr.module.backbone).to(rank)
    input_proj = DDP(detr.module.input_proj).to(rank)
    feature_encoder = DDP(detr.module.transformer.encoder).to(rank)

    local_predictor.eval()
    backbone.eval()
    input_proj.eval()
    feature_encoder.eval()

    map_location = {'cuda:%d' % rank: 'cuda:%d' % 0}
    if args['models']['hierarchical_pred']:
        local_predictor.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'HierMotif' + str(args['training']['test_epoch']) + '_0' + '.pth', map_location=map_location))
    else:
        local_predictor.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'FlatMotif' + str(args['training']['test_epoch']) + '_0' + '.pth', map_location=map_location))

    Recall = Evaluator_SGD(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])
    Recall_top3 = Evaluator_SGD_Top3(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])

    sub2super_cat_dict = torch.load(args['dataset']['sub2super_cat_dict'])
    object_class_alp2fre_dict = object_class_alp2fre()

    print('Start Testing SGD...')
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            """
            PREPARE INPUT DATA WITH PREDICTED OBJECT BOUNDING BOXES AND LABELS
            """
            try:
                images, image2, image_depth, categories_target, super_categories_target, bbox_target, relationships, subj_or_obj = data
            except:
                continue

            images = torch.stack(images).to(rank)
            image_feature, pos_embed = backbone(nested_tensor_from_tensor_list(images))
            src, mask = image_feature[-1].decompose()
            src = input_proj(src).flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
            image_feature = feature_encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)

            image_feature = image_feature.permute(1, 2, 0)
            image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])
            image_depth = torch.stack([depth.to(rank) for depth in image_depth])
            categories_target = [category.to(rank) for category in categories_target]  # [batch_size][curr_num_obj, 1]
            bbox_target = [box.to(rank) for box in bbox_target]  # [batch_size][curr_num_obj, 4]

            image2 = [image.to(rank) for image in image2]
            out_dict = detr(nested_tensor_from_tensor_list(image2))

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
                bbox_pred[i][:, 0] = bbox_pred_c[:, 0] - 1 * bbox_pred_c[:, 2] / 2
                bbox_pred[i][:, 1] = bbox_pred_c[:, 0] + 1 * bbox_pred_c[:, 2] / 2
                bbox_pred[i][:, 2] = bbox_pred_c[:, 1] - 1 * bbox_pred_c[:, 3] / 2
                bbox_pred[i][:, 3] = bbox_pred_c[:, 1] + 1 * bbox_pred_c[:, 3] / 2

                bbox_pred[i][:, 0][bbox_pred[i][:, 0] < 0] = 0
                bbox_pred[i][:, 1][bbox_pred[i][:, 1] > 1] = 1
                bbox_pred[i][:, 2][bbox_pred[i][:, 2] < 0] = 0
                bbox_pred[i][:, 3][bbox_pred[i][:, 3] > 1] = 1
                bbox_pred[i] *= 32
                bbox_pred[i] = bbox_pred[i].repeat_interleave(args['models']['topk_cat'], dim=0)

            masks_pred = []
            for i in range(len(bbox_pred)):
                mask_pred = torch.zeros(bbox_pred[i].shape[0], args['models']['feature_size'], args['models']['feature_size'], dtype=torch.uint8).to(rank)
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

            """
            PREPARE TARGETS
            relations_target and direction_target: matched targets for each prediction
            cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target, relation_target_origin: sets of original unmatched targets
            """
            cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target, relation_target \
                = match_target_sgd(rank, relationships, subj_or_obj, categories_target, bbox_target)

            """
            FORWARD PASS THROUGH THE LOCAL PREDICTOR
            """
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
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity, _, _ = local_predictor(h_graph[iou_mask], h_edge[iou_mask], cat_graph_pred[iou_mask], cat_edge_pred[iou_mask],
                                                                                                     scat_graph_pred, scat_edge_pred, rank)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = local_predictor(h_graph[iou_mask], h_edge[iou_mask], cat_graph_pred[iou_mask], cat_edge_pred[iou_mask], scat_graph_pred, scat_edge_pred, rank)

                    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                        Recall.accumulate_pred(which_in_batch[iou_mask], relation, super_relation, cat_graph_pred[iou_mask], cat_edge_pred[iou_mask], bbox_graph_pred[iou_mask], bbox_edge_pred[iou_mask],
                                               cat_graph_confidence[iou_mask], cat_edge_confidence[iou_mask], torch.log(torch.sigmoid(connectivity[:, 0])))
                        Recall_top3.accumulate_pred(which_in_batch[iou_mask], relation, super_relation, cat_graph_pred[iou_mask], cat_edge_pred[iou_mask], bbox_graph_pred[iou_mask], bbox_edge_pred[iou_mask],
                                                    cat_graph_confidence[iou_mask], cat_edge_confidence[iou_mask], torch.log(torch.sigmoid(connectivity[:, 0])))
                    """
                    SECOND DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity, _, _ = local_predictor(h_edge[iou_mask], h_graph[iou_mask], cat_edge_pred[iou_mask], cat_graph_pred[iou_mask],
                                                                                                     scat_edge_pred, scat_graph_pred, rank)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = local_predictor(h_edge[iou_mask], h_graph[iou_mask], cat_edge_pred[iou_mask], cat_graph_pred[iou_mask], scat_edge_pred, scat_graph_pred, rank)

                    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                        Recall.accumulate_pred(which_in_batch[iou_mask], relation, super_relation, cat_edge_pred[iou_mask], cat_graph_pred[iou_mask], bbox_edge_pred[iou_mask], bbox_graph_pred[iou_mask],
                                               cat_edge_confidence[iou_mask], cat_graph_confidence[iou_mask], torch.log(torch.sigmoid(connectivity[:, 0])))
                        Recall_top3.accumulate_pred(which_in_batch[iou_mask], relation, super_relation, cat_edge_pred[iou_mask], cat_graph_pred[iou_mask], bbox_edge_pred[iou_mask], bbox_graph_pred[iou_mask],
                                                    cat_edge_confidence[iou_mask], cat_graph_confidence[iou_mask], torch.log(torch.sigmoid(connectivity[:, 0])))

            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                Recall.accumulate_target(relation_target, cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target)
                Recall_top3.accumulate_target(relation_target, cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target)

            """
            EVALUATE AND PRINT CURRENT RESULTS
            """
            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                recall, _, mean_recall, recall_k_wrong_label_corr_rel = Recall.compute(per_class=True)
                recall_top3, _, mean_recall_top3 = Recall_top3.compute(per_class=True)
                Recall.clear_data()
                Recall_top3.clear_data()

            if (batch_count % args['training']['print_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                print('TESTING, rank: %d, R@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), mR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), %.4f, %.4f, %.4f'
                      % (rank, recall_top3[0], recall_top3[1], recall_top3[2], recall[0], recall[1], recall[2],
                         mean_recall_top3[0], mean_recall_top3[1], mean_recall_top3[2], mean_recall[0], mean_recall[1], mean_recall[2],
                         recall_k_wrong_label_corr_rel[0], recall_k_wrong_label_corr_rel[1], recall_k_wrong_label_corr_rel[2]))

                test_record.append({'rank': rank, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                   'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                                   'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                   'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                                   'num_objs_average': torch.mean(num_graph_iter.float()).item(), 'num_objs_max': max(num_graph_iter).item()})
                with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # append current logs
                    json.dump(test_record, f)

    dist.monitored_barrier()
    print('FINISHED TESTING SGD\n')


def eval_sgc(gpu, args, test_subset):
    """
    This function evaluates the module on scene graph classification tasks.
    :param gpu: current gpu index
    :param args: input arguments in config.yaml
    :param test_subset: testing dataset
    """
    rank = gpu
    world_size = torch.cuda.device_count()
    setup(rank, world_size)
    print('rank', rank, 'torch.distributed.is_initialized', torch.distributed.is_initialized())

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=test_sampler)
    print("Finished loading the datasets...")

    start = []
    test_record = []
    with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # clear history logs
        json.dump(start, f)

    if args['models']['hierarchical_pred']:
        local_predictor = DDP(HierMotif(args=args, input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
                                     num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'],
                                     num_geometric=args['models']['num_geometric'], num_possessive=args['models']['num_possessive'], num_semantic=args['models']['num_semantic'])).to(rank)
    else:
        local_predictor = DDP(FlatMotif(args=args, input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'], feature_size=args['models']['feature_size'],
                                 num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'],
                                 num_geometric=args['models']['num_geometric'], num_possessive=args['models']['num_possessive'], num_semantic=args['models']['num_semantic'])).to(rank)

    detr = DDP(build_detr101(args)).to(rank)
    backbone = DDP(detr.module.backbone).to(rank)
    input_proj = DDP(detr.module.input_proj).to(rank)
    feature_encoder = DDP(detr.module.transformer.encoder).to(rank)

    local_predictor.eval()
    backbone.eval()
    input_proj.eval()
    feature_encoder.eval()

    map_location = {'cuda:%d' % rank: 'cuda:%d' % 0}
    if args['models']['hierarchical_pred']:
        local_predictor.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'HierMotif' + str(args['training']['test_epoch']) + '_0' + '.pth', map_location=map_location))
    else:
        local_predictor.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'FlatMotif' + str(args['training']['test_epoch']) + '_0' + '.pth', map_location=map_location))

    Recall = Evaluator_SGD(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])
    Recall_top3 = Evaluator_SGD_Top3(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])

    sub2super_cat_dict = torch.load(args['dataset']['sub2super_cat_dict'])
    object_class_alp2fre_dict = object_class_alp2fre()

    print('Start Testing SGC...')
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            """
            PREPARE INPUT DATA WITH PREDICTED OBJECT BOUNDING BOXES AND LABELS
            """
            try:
                images, image2, image_depth, categories_target, super_categories_target, bbox_target, relationships, subj_or_obj = data
            except:
                continue

            images = torch.stack(images).to(rank)
            image_feature, pos_embed = backbone(nested_tensor_from_tensor_list(images))
            src, mask = image_feature[-1].decompose()
            src = input_proj(src).flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed[-1].flatten(2).permute(2, 0, 1)
            image_feature = feature_encoder(src, src_key_padding_mask=mask.flatten(1), pos=pos_embed)

            image_feature = image_feature.permute(1, 2, 0)
            image_feature = image_feature.view(-1, args['models']['num_img_feature'], args['models']['feature_size'], args['models']['feature_size'])
            image_depth = torch.stack([depth.to(rank) for depth in image_depth])
            categories_target = [category.to(rank) for category in categories_target]  # [batch_size][curr_num_obj, 1]
            bbox_target = [box.to(rank) for box in bbox_target]  # [batch_size][curr_num_obj, 4]

            image2 = [image.to(rank) for image in image2]
            out_dict = detr(nested_tensor_from_tensor_list(image2))

            logits_pred = torch.argmax(F.softmax(out_dict['pred_logits'], dim=2), dim=2)
            has_object_pred = logits_pred < 150
            logits_pred = torch.topk(F.softmax(out_dict['pred_logits'], dim=2), dim=2, k=2)[1].view(-1, 100, 2)

            logits_pred_value = torch.topk(F.softmax(out_dict['pred_logits'], dim=2), dim=2, k=2)[0].view(-1, 100, 2)
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
                bbox_pred[i][:, 0] = bbox_pred_c[:, 0] - 1 * bbox_pred_c[:, 2] / 2
                bbox_pred[i][:, 1] = bbox_pred_c[:, 0] + 1 * bbox_pred_c[:, 2] / 2
                bbox_pred[i][:, 2] = bbox_pred_c[:, 1] - 1 * bbox_pred_c[:, 3] / 2
                bbox_pred[i][:, 3] = bbox_pred_c[:, 1] + 1 * bbox_pred_c[:, 3] / 2

                bbox_pred[i][:, 0][bbox_pred[i][:, 0] < 0] = 0
                bbox_pred[i][:, 1][bbox_pred[i][:, 1] > 1] = 1
                bbox_pred[i][:, 2][bbox_pred[i][:, 2] < 0] = 0
                bbox_pred[i][:, 3][bbox_pred[i][:, 3] > 1] = 1
                bbox_pred[i] *= 32
                bbox_pred[i] = bbox_pred[i].repeat_interleave(2, dim=0)

            for i in range(len(categories_pred)):
                categories_pred[i] = categories_pred[i][cat_mask[i]]
                cat_pred_confidence[i] = cat_pred_confidence[i][cat_mask[i]]
                bbox_pred[i] = bbox_pred[i][cat_mask[i]]

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

            """
            PREPARE TARGETS
            relations_target and direction_target: matched targets for each prediction
            cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target, relation_target_origin: sets of original unmatched targets
            """
            cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target, relation_target \
                = match_target_sgd(rank, relationships, subj_or_obj, categories_target, bbox_target)

            """
            MATCH PREDICTED OBJECT LABELS FROM BOUNDING BOX IOUS
            """
            categories_pred_matched, cat_pred_confidence_matched, bbox_target = match_object_categories(categories_pred, cat_pred_confidence, bbox_pred, bbox_target)
            if categories_pred_matched is None or cat_pred_confidence_matched is None:
                continue

            # bbox_target = [torch.repeat_interleave(box, repeats=2, dim=0) for box in bbox_target]
            assert len(categories_pred_matched[0]) == len(bbox_target[0])

            # after nms
            super_categories_pred = [[sub2super_cat_dict[c.item()] for c in categories_pred_matched[i]] for i in range(len(categories_pred_matched))]
            super_categories_pred = [[torch.as_tensor(sc).to(rank) for sc in super_category] for super_category in super_categories_pred]

            masks_target = []
            for i in range(len(bbox_target)):
                mask = torch.zeros(bbox_target[i].shape[0], args['models']['feature_size'], args['models']['feature_size'], dtype=torch.uint8).to(rank)
                for j, box in enumerate(bbox_target[i]):
                    mask[j, int(bbox_target[i][j][2]):int(bbox_target[i][j][3]), int(bbox_target[i][j][0]):int(bbox_target[i][j][1])] = 1
                masks_target.append(mask)

            """
            FORWARD PASS THROUGH THE LOCAL PREDICTOR
            """
            num_graph_iter = torch.as_tensor([len(mask) for mask in masks_target])
            for graph_iter in range(max(num_graph_iter)):
                which_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1)

                curr_graph_masks = torch.stack([torch.unsqueeze(masks_target[i][graph_iter], dim=0) for i in which_in_batch])
                h_graph = torch.cat((image_feature[which_in_batch] * curr_graph_masks, image_depth[which_in_batch] * curr_graph_masks), dim=1)  # (bs, 256, 64, 64), (bs, 1, 64, 64)
                cat_graph_pred = torch.tensor([torch.unsqueeze(categories_pred_matched[i][graph_iter], dim=0) for i in which_in_batch]).to(rank)
                bbox_graph_pred = torch.stack([bbox_target[i][graph_iter] for i in which_in_batch]).to(rank)    # use ground-truth bounding boxes
                cat_graph_confidence = torch.hstack([cat_pred_confidence_matched[i][graph_iter] for i in which_in_batch])

                for edge_iter in range(graph_iter):
                    curr_edge_masks = torch.stack([torch.unsqueeze(masks_target[i][edge_iter], dim=0) for i in which_in_batch])  # seg mask of every prev obj
                    h_edge = torch.cat((image_feature[which_in_batch] * curr_edge_masks, image_depth[which_in_batch] * curr_edge_masks), dim=1)
                    cat_edge_pred = torch.tensor([torch.unsqueeze(categories_pred_matched[i][edge_iter], dim=0) for i in which_in_batch]).to(rank)
                    bbox_edge_pred = torch.stack([bbox_target[i][edge_iter] for i in which_in_batch]).to(rank)    # use ground-truth bounding boxes
                    cat_edge_confidence = torch.hstack([cat_pred_confidence_matched[i][edge_iter] for i in which_in_batch])

                    scat_graph_pred = []
                    scat_edge_pred = []
                    for count, i in enumerate(which_in_batch):
                        scat_graph_pred.append(super_categories_pred[i][graph_iter])
                        scat_edge_pred.append(super_categories_pred[i][edge_iter])

                    """
                    FIRST DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity, _, _ = local_predictor(h_graph, h_edge, cat_graph_pred, cat_edge_pred,
                                                                                                     scat_graph_pred, scat_edge_pred, rank)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = local_predictor(h_graph, h_edge, cat_graph_pred, cat_edge_pred, scat_graph_pred, scat_edge_pred, rank)

                    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                        Recall.accumulate_pred(which_in_batch, relation, super_relation, cat_graph_pred, cat_edge_pred, bbox_graph_pred, bbox_edge_pred,
                                               cat_graph_confidence, cat_edge_confidence, torch.log(torch.sigmoid(connectivity[:, 0])))
                        Recall_top3.accumulate_pred(which_in_batch, relation, super_relation, cat_graph_pred, cat_edge_pred, bbox_graph_pred, bbox_edge_pred,
                                                    cat_graph_confidence, cat_edge_confidence, torch.log(torch.sigmoid(connectivity[:, 0])))
                    """
                    SECOND DIRECTION
                    """
                    if args['models']['hierarchical_pred']:
                        relation_1, relation_2, relation_3, super_relation, connectivity, _, _ = local_predictor(h_edge, h_graph, cat_edge_pred, cat_graph_pred,
                                                                                                     scat_edge_pred, scat_graph_pred, rank)
                        relation = torch.cat((relation_1, relation_2, relation_3), dim=1)
                    else:
                        relation, connectivity = local_predictor(h_edge, h_graph, cat_edge_pred, cat_graph_pred, scat_edge_pred, scat_graph_pred, rank)

                    if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                        Recall.accumulate_pred(which_in_batch, relation, super_relation, cat_edge_pred, cat_graph_pred, bbox_edge_pred, bbox_graph_pred,
                                               cat_edge_confidence, cat_graph_confidence, torch.log(torch.sigmoid(connectivity[:, 0])))
                        Recall_top3.accumulate_pred(which_in_batch, relation, super_relation, cat_edge_pred, cat_graph_pred, bbox_edge_pred, bbox_graph_pred,
                                                    cat_edge_confidence, cat_graph_confidence, torch.log(torch.sigmoid(connectivity[:, 0])))

            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                Recall.accumulate_target(relation_target, cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target)
                Recall_top3.accumulate_target(relation_target, cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target)

            """
            EVALUATE AND PRINT CURRENT RESULTS
            """
            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                recall, _, mean_recall, recall_k_wrong_label_corr_rel = Recall.compute(per_class=True)
                recall_top3, _, mean_recall_top3 = Recall_top3.compute(per_class=True)
                Recall.clear_data()
                Recall_top3.clear_data()

            if (batch_count % args['training']['print_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                print('TESTING, rank: %d, R@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), mR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), %.4f, %.4f, %.4f'
                      % (rank, recall_top3[0], recall_top3[1], recall_top3[2], recall[0], recall[1], recall[2],
                         mean_recall_top3[0], mean_recall_top3[1], mean_recall_top3[2], mean_recall[0], mean_recall[1], mean_recall[2],
                         recall_k_wrong_label_corr_rel[0], recall_k_wrong_label_corr_rel[1], recall_k_wrong_label_corr_rel[2]))

                test_record.append({'rank': rank, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                   'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                                   'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                   'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                                   'num_objs_average': torch.mean(num_graph_iter.float()).item(), 'num_objs_max': max(num_graph_iter).item()})
                with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # append current logs
                    json.dump(test_record, f)

    dist.monitored_barrier()
    print('FINISHED TESTING SGC\n')
