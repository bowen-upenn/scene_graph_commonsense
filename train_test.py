import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import os
import math
import shutil
import torchvision
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from evaluator import Evaluator, Evaluator_Top3
from model import *
from utils import *
from train_utils import *
from dataset_utils import *
from sup_contrast.losses import SupConLoss, SupConLossHierar


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def training(gpu, args, train_subset, test_subset):
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

    writer = None
    if rank == 0:
        log_dir = 'runs/train_sg'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)  # remove the old log directory if it exists
        writer = SummaryWriter(log_dir)

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
        relation_classifier = DDP(BayesianRelationClassifier(args=args, input_dim=args['models']['hidden_dim'], feature_size=args['models']['feature_size'],
                                                             num_classes=args['models']['num_classes'], num_super_classes=args['models']['num_super_classes'],
                                                             num_geometric=args['models']['num_geometric'], num_possessive=args['models']['num_possessive'],
                                                             num_semantic=args['models']['num_semantic'])).to(rank)
    else:
        relation_classifier = DDP(FlatRelationClassifier(args=args, input_dim=args['models']['hidden_dim'], output_dim=args['models']['num_relations'],
                                                         feature_size=args['models']['feature_size'], num_classes=args['models']['num_classes'])).to(rank)

    detr = DDP(build_detr101(args)).to(rank)
    detr.eval()

    map_location = {'cuda:%d' % rank: 'cuda:%d' % 0}
    if args['training']['continue_train']:
        if args['models']['hierarchical_pred']:
            load_model_name = 'HierMotif_CS' if args['dataset']['run_mode'] == 'train_cs' else 'HierMotif_Baseline'
            load_model_name = args['training']['checkpoint_path'] + load_model_name + str(args['training']['start_epoch'] - 1) + '_0' + '.pth'
        else:
            load_model_name = 'FlatMotif_CS' if args['dataset']['run_mode'] == 'train_cs' else 'FlatMotif_Baseline'
            load_model_name = args['training']['checkpoint_path'] + load_model_name + str(args['training']['start_epoch'] - 1) + '_0' + '.pth'
        if rank == 0:
            print('Loading pretrained model from %s...' % load_model_name)
        relation_classifier.load_state_dict(torch.load(load_model_name, map_location=map_location))

    if rank == 0:
        total_params = sum(p.numel() for p in relation_classifier.parameters())
        print(f"Total number of parameters in the model: {total_params}")

    optimizer = optim.SGD([{'params': relation_classifier.parameters(), 'initial_lr': args['training']['learning_rate']}],
                          lr=args['training']['learning_rate'], momentum=0.9, weight_decay=args['training']['weight_decay'])
    relation_classifier.train()

    original_lr = optimizer.param_groups[0]["lr"]
    relation_count = get_num_each_class_reordered(args)
    class_weight = 1 - relation_count / torch.sum(relation_count)

    if args['models']['hierarchical_pred']:
        criterion_relationship_1 = torch.nn.NLLLoss(weight=class_weight[:args['models']['num_geometric']].to(rank))  # log softmax already applied
        criterion_relationship_2 = torch.nn.NLLLoss(weight=class_weight[args['models']['num_geometric']:args['models']['num_geometric']+args['models']['num_possessive']].to(rank))
        criterion_relationship_3 = torch.nn.NLLLoss(weight=class_weight[args['models']['num_geometric']+args['models']['num_possessive']:].to(rank))
        criterion_super_relationship = torch.nn.NLLLoss()
        criterion_relationship = [criterion_relationship_1, criterion_relationship_2, criterion_relationship_3, criterion_super_relationship]
    else:
        criterion_relationship = torch.nn.CrossEntropyLoss(weight=class_weight.to(rank))
    criterion_contrast = SupConLossHierar()
    criterion_connectivity = torch.nn.BCEWithLogitsLoss()

    running_losses, running_loss_connectivity, running_loss_relationship, running_loss_contrast, running_loss_pseudo_consistency, running_loss_commonsense, \
        connectivity_recall, connectivity_precision, num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    recall_top3, recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs, wmap_rel, wmap_phrase = None, None, None, None, None, None, None, None

    Recall = Evaluator(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])
    Recall_top3 = None
    if args['dataset']['dataset'] == 'vg':
        Recall_top3 = Evaluator_Top3(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])

    commonsense_yes_triplets = torch.load('triplets/commonsense_yes_triplets.pt') if args['training']['run_mode'] == 'train_cs' else None
    commonsense_no_triplets = torch.load('triplets/commonsense_no_triplets.pt') if args['training']['run_mode'] == 'train_cs' else None

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
                images, images_aug, image_depth, categories, super_categories, bbox, relationships, subj_or_obj, _ = data
            except:
                continue
            batch_size = len(images)

            with torch.no_grad():
                image_feature = process_image_features(args, images, detr, rank)
                image_feature_aug = process_image_features(args, images_aug, detr, rank)
                del images, images_aug

            categories = [category.to(rank) for category in categories]  # [batch_size][curr_num_obj, 1]
            if super_categories[0] is not None:
                super_categories = [[sc.to(rank) for sc in super_category] for super_category in super_categories]  # [batch_size][curr_num_obj, [1 or more]]
            image_depth = torch.stack([depth.to(rank) for depth in image_depth])
            bbox = [box.to(rank) for box in bbox]  # [batch_size][curr_num_obj, 4]
            optimizer.param_groups[0]["lr"] = original_lr

            masks = []
            for i in range(len(bbox)):
                mask = torch.zeros(bbox[i].shape[0], args['models']['feature_size'], args['models']['feature_size'], dtype=torch.bool).to(rank)
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
            losses, loss_connectivity, loss_relationship, loss_contrast, loss_commonsense = 0.0, 0.0, 0.0, 0.0, 0.0

            num_graph_iter = torch.as_tensor([len(mask) for mask in masks])
            for graph_iter in range(max(num_graph_iter)):
                keep_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1).to(rank)
                optimizer.param_groups[0]["lr"] = original_lr * lr_decay * math.sqrt(len(keep_in_batch) / len(num_graph_iter))  # dynamic batch size needs dynamic learning rate

                curr_graph_masks = torch.stack([torch.unsqueeze(masks[i][graph_iter], dim=0) for i in keep_in_batch])
                h_graph = torch.cat((image_feature[keep_in_batch] * curr_graph_masks, image_depth[keep_in_batch] * curr_graph_masks), dim=1)  # (bs, 256, 64, 64), (bs, 1, 64, 64)
                h_graph_aug = torch.cat((image_feature_aug[keep_in_batch] * curr_graph_masks, image_depth[keep_in_batch] * curr_graph_masks), dim=1)
                cat_graph = torch.tensor([torch.unsqueeze(categories[i][graph_iter], dim=0) for i in keep_in_batch]).to(rank)
                spcat_graph = [super_categories[i][graph_iter] for i in keep_in_batch] if super_categories[0] is not None else None
                bbox_graph = torch.stack([bbox[i][graph_iter] for i in keep_in_batch]).to(rank)

                for edge_iter in range(graph_iter):
                    curr_edge_masks = torch.stack([torch.unsqueeze(masks[i][edge_iter], dim=0) for i in keep_in_batch])  # seg mask of every prev obj
                    h_edge = torch.cat((image_feature[keep_in_batch] * curr_edge_masks, image_depth[keep_in_batch] * curr_edge_masks), dim=1)
                    h_edge_aug = torch.cat((image_feature_aug[keep_in_batch] * curr_edge_masks, image_depth[keep_in_batch] * curr_edge_masks), dim=1)
                    cat_edge = torch.tensor([torch.unsqueeze(categories[i][edge_iter], dim=0) for i in keep_in_batch]).to(rank)
                    spcat_edge = [super_categories[i][edge_iter] for i in keep_in_batch] if super_categories[0] is not None else None
                    bbox_edge = torch.stack([bbox[i][edge_iter] for i in keep_in_batch]).to(rank)
                    iou_mask = torch.ones(len(keep_in_batch), dtype=torch.bool).to(rank)

                    """
                    FIRST DIRECTION
                    """
                    curr_loss_relationship, curr_loss_connectivity, curr_loss_commonsense, curr_num_not_connected, curr_num_connected, curr_num_connected_pred, \
                    curr_connectivity_precision, curr_connectivity_recall, hidden_cat_accumulated, hidden_cat_labels_accumulated = \
                        train_one_direction(relation_classifier, args, h_graph, h_edge, cat_graph, cat_edge, spcat_graph, spcat_edge, bbox_graph, bbox_edge, h_graph_aug, h_edge_aug, iou_mask, rank, graph_iter, edge_iter,
                                            keep_in_batch, Recall, Recall_top3, criterion_relationship, criterion_connectivity, relations_target, direction_target, batch_count,
                                            hidden_cat_accumulated, hidden_cat_labels_accumulated, commonsense_yes_triplets, commonsense_no_triplets, len(train_loader), first_direction=True)

                    loss_relationship += curr_loss_relationship
                    loss_connectivity += curr_loss_connectivity
                    loss_commonsense += curr_loss_commonsense
                    num_not_connected += curr_num_not_connected
                    num_connected += curr_num_connected
                    num_connected_pred += curr_num_connected_pred
                    connectivity_precision += curr_connectivity_precision
                    connectivity_recall += curr_connectivity_recall

                    losses += loss_relationship \
                              + args['training']['lambda_connectivity'] * loss_connectivity \
                              + args['training']['lambda_commonsense'] * loss_commonsense
                    running_loss_connectivity += args['training']['lambda_connectivity'] * loss_connectivity
                    running_loss_relationship += loss_relationship
                    running_loss_commonsense += args['training']['lambda_commonsense'] * loss_commonsense

                    """
                    SECOND DIRECTION
                    """
                    curr_loss_relationship, curr_loss_connectivity, curr_loss_commonsense, curr_num_not_connected, curr_num_connected, curr_num_connected_pred, \
                    curr_connectivity_precision, curr_connectivity_recall, hidden_cat_accumulated, hidden_cat_labels_accumulated = \
                        train_one_direction(relation_classifier, args, h_edge, h_graph, cat_edge, cat_graph, spcat_edge, spcat_graph, bbox_edge, bbox_graph, h_edge_aug, h_graph_aug, iou_mask, rank, graph_iter, edge_iter,
                                            keep_in_batch, Recall, Recall_top3, criterion_relationship, criterion_connectivity, relations_target, direction_target, batch_count,
                                            hidden_cat_accumulated, hidden_cat_labels_accumulated, commonsense_yes_triplets, commonsense_no_triplets, len(train_loader), first_direction=False)

                    loss_relationship += curr_loss_relationship
                    loss_connectivity += curr_loss_connectivity
                    loss_commonsense += curr_loss_commonsense
                    num_not_connected += curr_num_not_connected
                    num_connected += curr_num_connected
                    num_connected_pred += curr_num_connected_pred
                    connectivity_precision += curr_connectivity_precision
                    connectivity_recall += curr_connectivity_recall

                    losses += loss_relationship \
                              + args['training']['lambda_connectivity'] * loss_connectivity \
                              + args['training']['lambda_commonsense'] * loss_commonsense
                    running_loss_connectivity += args['training']['lambda_connectivity'] * loss_connectivity
                    running_loss_relationship += loss_relationship
                    running_loss_commonsense += args['training']['lambda_commonsense'] * loss_commonsense

            if not all(len(sublist) == 0 for sublist in hidden_cat_accumulated):
                # concatenate all hidden_cat and hidden_cat_labels along the 0th dimension
                hidden_cat_accumulated = [torch.stack(sublist) for sublist in hidden_cat_accumulated if len(sublist) > 0]
                hidden_cat_labels_accumulated = [torch.stack(sublist) for sublist in hidden_cat_labels_accumulated if len(sublist) > 0]

                hidden_cat_all = torch.cat(hidden_cat_accumulated, dim=0)
                hidden_cat_labels_all = torch.cat(hidden_cat_labels_accumulated, dim=0)

                temp = criterion_contrast(rank, hidden_cat_all, hidden_cat_labels_all)
                loss_contrast += 0.0 if torch.isnan(temp) else args['training']['lambda_contrast'] * temp

            running_loss_contrast += args['training']['lambda_contrast'] * loss_contrast
            losses += args['training']['lambda_contrast'] * loss_contrast
            running_losses += losses

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if rank == 0:
                global_step = batch_count + len(train_loader) * epoch
                writer.add_scalar('train/running_loss_relationship', running_loss_relationship, global_step)
                writer.add_scalar('train/running_loss_connectivity', running_loss_connectivity, global_step)
                writer.add_scalar('train/running_loss_contrast', running_loss_contrast, global_step)
                writer.add_scalar('train/running_loss_commonsense', running_loss_commonsense, global_step)
                writer.add_scalar('train/running_losses', running_losses, global_step)

            """
            EVALUATE AND PRINT CURRENT TRAINING RESULTS
            """
            if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == len(train_loader)):
                recall_top3, mean_recall_top3 = None, None
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
                record_train_results(args, record, rank, epoch, batch_count, optimizer.param_groups[0]['lr'], recall_top3, recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs,
                                     running_losses, running_loss_relationship, running_loss_contrast, running_loss_connectivity, running_loss_commonsense,
                                     connectivity_recall, num_connected, num_not_connected, connectivity_precision, num_connected_pred, wmap_rel, wmap_phrase)
                dist.monitored_barrier()

            running_losses, running_loss_connectivity, running_loss_relationship, running_loss_contrast, running_loss_commonsense, \
                connectivity_precision, num_connected, num_not_connected = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # if args['models']['hierarchical_pred']:
        #     save_model_name = 'HierMotif_CS' if args['dataset']['run_mode'] == 'train_cs' else 'HierMotif_Baseline'
        #     save_model_name = args['training']['checkpoint_path'] + save_model_name + str(epoch) + '_' + str(rank) + '.pth'
        # else:
        #     save_model_name = 'FlatMotif_CS' if args['dataset']['run_mode'] == 'train_cs' else 'FlatMotif_Baseline'
        #     save_model_name = args['training']['checkpoint_path'] + save_model_name + str(epoch) + '_' + str(rank) + '.pth'
        # if rank == 0:
        #     print('Saving model to %s...' % save_model_name)
        # torch.save(relation_classifier.state_dict(), save_model_name)
        dist.monitored_barrier()

        testing(args, detr, relation_classifier, test_loader, test_record, epoch, rank, writer)

    dist.destroy_process_group()  # clean up
    if rank == 0:
        writer.close()
    print('FINISHED TRAINING\n')


def testing(args, detr, relation_classifier, test_loader, test_record, epoch, rank, writer):
    detr.eval()
    relation_classifier.eval()

    connectivity_recall, connectivity_precision, num_connected, num_not_connected, num_connected_pred = 0.0, 0.0, 0.0, 0.0, 0.0
    recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs, wmap_rel, wmap_phrase = None, None, None, None, None, None, None
    Recall = Evaluator(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])
    Recall_top3 = None
    if args['dataset']['dataset'] == 'vg':
        Recall_top3 = Evaluator_Top3(args=args, num_classes=args['models']['num_relations'], iou_thresh=0.5, top_k=[20, 50, 100])

    print('Start Testing PC...')
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            """
            PREPARE INPUT DATA
            """
            try:
                images, _, image_depth, categories, super_categories, bbox, relationships, subj_or_obj, _ = data
            except:
                continue

            image_feature = process_image_features(args, images, detr, rank)

            categories = [category.to(rank) for category in categories]  # [batch_size][curr_num_obj, 1]
            if super_categories[0] is not None:
                super_categories = [[sc.to(rank) for sc in super_category] for super_category in super_categories]  # [batch_size][curr_num_obj, [1 or more]]
            image_depth = torch.stack([depth.to(rank) for depth in image_depth])
            bbox = [box.to(rank) for box in bbox]  # [batch_size][curr_num_obj, 4]

            masks = []
            for i in range(len(bbox)):
                mask = torch.zeros(bbox[i].shape[0], args['models']['feature_size'], args['models']['feature_size'], dtype=torch.bool).to(rank)
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
                keep_in_batch = torch.nonzero(num_graph_iter > graph_iter).view(-1).to(rank)

                curr_graph_masks = torch.stack([torch.unsqueeze(masks[i][graph_iter], dim=0) for i in keep_in_batch])
                h_graph = torch.cat((image_feature[keep_in_batch] * curr_graph_masks, image_depth[keep_in_batch] * curr_graph_masks), dim=1)  # (bs, 256, 64, 64), (bs, 1, 64, 64)
                cat_graph = torch.tensor([torch.unsqueeze(categories[i][graph_iter], dim=0) for i in keep_in_batch]).to(rank)
                spcat_graph = [super_categories[i][graph_iter] for i in keep_in_batch] if super_categories[0] is not None else None
                bbox_graph = torch.stack([bbox[i][graph_iter] for i in keep_in_batch]).to(rank)

                for edge_iter in range(graph_iter):
                    curr_edge_masks = torch.stack([torch.unsqueeze(masks[i][edge_iter], dim=0) for i in keep_in_batch])  # seg mask of every prev obj
                    h_edge = torch.cat((image_feature[keep_in_batch] * curr_edge_masks, image_depth[keep_in_batch] * curr_edge_masks), dim=1)
                    cat_edge = torch.tensor([torch.unsqueeze(categories[i][edge_iter], dim=0) for i in keep_in_batch]).to(rank)
                    spcat_edge = [super_categories[i][edge_iter] for i in keep_in_batch] if super_categories[0] is not None else None
                    bbox_edge = torch.stack([bbox[i][edge_iter] for i in keep_in_batch]).to(rank)

                    # filter out subject-object pairs whose iou=0
                    joint_intersect = torch.logical_or(curr_graph_masks, curr_edge_masks)
                    joint_union = torch.logical_and(curr_graph_masks, curr_edge_masks)
                    joint_iou = (torch.sum(torch.sum(joint_intersect, dim=-1), dim=-1) / torch.sum(torch.sum(joint_union, dim=-1), dim=-1)).flatten()
                    joint_iou[torch.isinf(joint_iou)] = 0
                    iou_mask = joint_iou > 0
                    if torch.sum(iou_mask) == 0:
                        continue
                    # iou_mask = torch.ones(len(keep_in_batch), dtype=torch.bool).to(rank)

                    """
                    FIRST DIRECTION
                    """
                    curr_num_not_connected, curr_num_connected, curr_num_connected_pred, curr_connectivity_precision, curr_connectivity_recall = \
                        evaluate_one_direction(relation_classifier, args, h_graph, h_edge, cat_graph, cat_edge, spcat_graph, spcat_edge, bbox_graph, bbox_edge, iou_mask, rank, graph_iter, edge_iter, keep_in_batch,
                                               Recall, Recall_top3, relations_target, direction_target, batch_count, len(test_loader), first_direction=True)

                    num_not_connected += curr_num_not_connected
                    num_connected += curr_num_connected
                    num_connected_pred += curr_num_connected_pred
                    connectivity_precision += curr_connectivity_precision
                    connectivity_recall += curr_connectivity_recall

                    """
                    SECOND DIRECTION
                    """
                    curr_num_not_connected, curr_num_connected, curr_num_connected_pred, curr_connectivity_precision, curr_connectivity_recall = \
                        evaluate_one_direction(relation_classifier, args, h_edge, h_graph, cat_edge, cat_graph, spcat_edge, spcat_graph, bbox_edge, bbox_graph, iou_mask, rank, graph_iter, edge_iter, keep_in_batch,
                                               Recall, Recall_top3, relations_target, direction_target, batch_count, len(test_loader), first_direction=False)

                    num_not_connected += curr_num_not_connected
                    num_connected += curr_num_connected
                    num_connected_pred += curr_num_connected_pred
                    connectivity_precision += curr_connectivity_precision
                    connectivity_recall += curr_connectivity_recall

            """
            EVALUATE AND PRINT CURRENT RESULTS
            """
            if (batch_count % args['training']['eval_freq_test'] == 0) or (batch_count + 1 == len(test_loader)):
                recall_top3, mean_recall_top3 = None, None
                if args['dataset']['dataset'] == 'vg':
                    recall, _, mean_recall, recall_zs, _, mean_recall_zs = Recall.compute(per_class=True)
                    if rank == 0:
                        global_step = batch_count + len(test_loader) * epoch
                        writer.add_scalar('test/Recall@20', recall[0], global_step)
                        writer.add_scalar('test/Recall@50', recall[1], global_step)
                        writer.add_scalar('test/Recall@100', recall[2], global_step)
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
