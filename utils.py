import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import json
import torchmetrics
import torch.nn as nn
from torch import Tensor
from typing import Optional, List
import torchvision
import math
from collections import Counter, OrderedDict
import re
import random


def collate_fn(batch):
    """
    This function solves the problem when some data samples in a batch are None.
    :param batch: the current batch in dataloader
    :return: a new batch with all non-None data samples
    """
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))


def super_relation_processing(args, connected, curr_relations_target):
    # Clone the target relations for modification without affecting the original data
    super_relation_target = curr_relations_target[connected].clone()
    super_relation_target[super_relation_target < args['models']['num_geometric']] = 0
    super_relation_target[torch.logical_and(super_relation_target >= args['models']['num_geometric'], super_relation_target < args['models']['num_geometric'] + args['models']['num_possessive'])] = 1
    super_relation_target[super_relation_target >= args['models']['num_geometric'] + args['models']['num_possessive']] = 2

    return super_relation_target


def resize_boxes(boxes, original_size, new_size):
    """
    This function resizes an object bounding box.
    :param boxes: original bounding box
    :param original_size: original image size
    :param new_size: target image size
    :return: the resized bounding box
    """
    ratios = [s / s_orig for s, s_orig in zip(new_size, original_size)]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes[0], boxes[1], boxes[2], boxes[3]

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height

    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def iou(bbox_target, bbox_pred):
    """
    This function calculates the IOU score between two bounding boxes.
    :param bbox_target: target bounding box
    :param bbox_pred: predicted bounding box
    :return: the IOU score
    """
    mask_pred = torch.zeros(32, 32)
    mask_pred[int(bbox_pred[0]):int(bbox_pred[1]), int(bbox_pred[2]):int(bbox_pred[3])] = 1
    mask_target = torch.zeros(32, 32)
    mask_target[int(bbox_target[0]):int(bbox_target[1]), int(bbox_target[2]):int(bbox_target[3])] = 1
    intersect = torch.sum(torch.logical_and(mask_target, mask_pred))
    union = torch.sum(torch.logical_or(mask_target, mask_pred))
    if union == 0:
        return 0
    else:
        return float(intersect) / float(union)


def find_union_bounding_box(bbox1, bbox2):
    # bbox expects format x_min, x_max, y_min, y_max
    [x1min, x1max, y1min, y1max] = bbox1
    [x2min, x2max, y2min, y2max] = bbox2
    xmin = min(x1min, x2min)
    xmax = max(x1max, x2max)
    ymin = min(y1min, y2min)
    ymax = max(y1max, y2max)
    return xmin, xmax, ymin, ymax


def build_detr101(args):
    """
    This function builds the DETR-101 object detection backbone.
    It loads the model from source, change key names in the model state dict if needed,
    and loads state dict from a pretrained checkpoint
    :param args: input arguments in config.yaml file
    :return: the pretrained model
    """
    with open(args['models']['detr101_key_before'], 'r') as f:
        name_before = f.readlines()
        name_before = [line[:-1] for line in name_before]
    with open(args['models']['detr101_key_after'], 'r') as f:
        name_after = f.readlines()
        name_after = [line[:-1] for line in name_after]

    if args['dataset']['dataset'] == 'vg':
        model_path = args['models']['detr101_pretrained_vg']
    else:
        model_path = args['models']['detr101_pretrained_oiv6']
    model_param = torch.load(model_path)

    keys = [key for key in model_param['model'] if key in name_before]
    for idx, key in enumerate(keys):
        model_param['model'][name_after[idx]] = model_param['model'].pop(key)
        # print(idx, key, ' -> ', name_after[idx])

    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101', pretrained=False)
    if args['dataset']['dataset'] == 'vg':
        model.class_embed = nn.Linear(256, 151)
    else:
        model.class_embed = nn.Linear(256, 602)
    model.load_state_dict(model_param['model'], strict=False) # every param except "criterion.empty_weight"
    return model


# https://github.com/yrcong/RelTR/blob/main/util/misc.py
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

# https://github.com/yrcong/RelTR/blob/main/util/misc.py
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

# https://github.com/yrcong/RelTR/blob/main/util/misc.py
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def remove_ddp_module_in_weights(saved_state_dict):
    # Handle key renaming for matching
    renamed_state_dict = {}
    for k, v in saved_state_dict.items():
        # Remove 'module.' prefix if it exists
        k = k.replace('module.', '')
        renamed_state_dict[k] = v
    return renamed_state_dict


def process_super_class(s1, s2, num_super_classes, rank):
    sc1 = F.one_hot(torch.tensor([s[0] for s in s1]), num_classes=num_super_classes)
    for i in range(1, 4):  # at most 4 diff super class for each subclass instance
        idx = torch.nonzero(torch.tensor([len(s) == i + 1 for s in s1])).view(-1)
        if len(idx) > 0:
            sc1[idx] += F.one_hot(torch.tensor([s[i] for s in [s1[j] for j in idx]]), num_classes=num_super_classes)
    sc2 = F.one_hot(torch.tensor([s[0] for s in s2]), num_classes=num_super_classes)
    for i in range(1, 4):
        idx = torch.nonzero(torch.tensor([len(s) == i + 1 for s in s2])).view(-1)
        if len(idx) > 0:
            sc2[idx] += F.one_hot(torch.tensor([s[i] for s in [s2[j] for j in idx]]), num_classes=num_super_classes)

    sc1, sc2 = sc1.to(rank), sc2.to(rank)
    return sc1, sc2


def match_bbox(bbox_semi, bbox_raw, eval_mode):
    """
    Returns the index of the bounding box from bbox_raw that most closely matches the pseudo bbox.
    """
    if eval_mode == 'pc':
        for idx, bbox in enumerate(bbox_raw):
            if torch.sum(torch.abs(bbox - torch.as_tensor(bbox_semi))) == 0:
                return idx
        return None
    else:
        ious = calculate_iou_for_all(bbox_semi, bbox_raw)
        return torch.argmax(ious).item()


def calculate_iou_for_all(box1, boxes):
    """
    Calculate the Intersection over Union (IoU) of a bounding box with a set of bounding boxes.
    """
    x1 = torch.max(box1[0], boxes[:, 0])
    y1 = torch.max(box1[1], boxes[:, 1])
    x2 = torch.min(box1[2], boxes[:, 2])
    y2 = torch.min(box1[3], boxes[:, 3])

    inter_area = torch.clamp(x2 - x1 + 1, 0) * torch.clamp(y2 - y1 + 1, 0)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    union_area = box1_area + boxes_area - inter_area

    return inter_area / union_area


def get_num_each_class():  # number of training data in total for each relationship class
    return torch.tensor([712432, 277943, 251756, 146339, 136099, 96589, 66425, 47342, 42722,
                         41363, 22596, 18643, 15457, 14185, 13715, 10191, 9903, 9894, 9317, 9145, 8856,
                         5213, 4688, 4613, 3810, 3806, 3739, 3624, 3490, 3477, 3411, 3288, 3095, 3092,
                         3083, 2945, 2721, 2517, 2380, 2312, 2253, 2241, 2065, 1996, 1973, 1925, 1914,
                         1869, 1853, 1740])


def get_num_each_class_reordered(args):  # number of training data in total for each relationship class
    if args['dataset']['dataset'] == 'vg':
        return torch.tensor([47342, 1996, 3092, 3624, 3477, 9903, 41363, 3411, 251756,
                             13715, 96589, 712432, 1914, 9317, 22596, 3288, 9145, 2945,
                             277943, 2312, 146339, 2065, 2517, 136099, 15457, 66425, 10191,
                             5213, 2312, 3806, 4688, 1973, 1853, 9894, 42722, 3739,
                             3083, 1869, 2253, 3095, 2721, 3810, 8856, 2241, 18643,
                             14185, 1925, 1740, 4613, 3490])
    else:
        return torch.tensor([150983, 7665, 841, 455, 9402, 52561, 145480, 157, 175, 77, 27, 4827, 1146, 198, 77, 1,
                             12, 4, 43, 702, 8, 1111, 51, 43, 367, 10, 462, 11, 2094, 114])

def get_weight_oiv6():
    num = torch.tensor([1974, 120, 27, 2, 284, 571, 2059, 8, 26, 2, 0, 163, 25, 30, 2, 0, 0, 1, 0, 17, 0, 29, 14, 4, 3, 0, 6, 0, 67, 5]) + 1
    # freq = num / torch.sum(num)
    # print(1 / freq)
    return num

def get_distri_over_classes():  # distribution of training data over all relationship class
    return torch.tensor([0.3482, 0.1358, 0.1230, 0.0715, 0.0665, 0.0472, 0.0325, 0.0231, 0.0209,
                         0.0202, 0.0110, 0.0091, 0.0076, 0.0069, 0.0067, 0.0050, 0.0048, 0.0048,
                         0.0046, 0.0045, 0.0043, 0.0025, 0.0023, 0.0023, 0.0019, 0.0019, 0.0018,
                         0.0018, 0.0017, 0.0017, 0.0017, 0.0016, 0.0015, 0.0015, 0.0015, 0.0014,
                         0.0013, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011, 0.0010, 0.0010, 0.0010,
                         0.0009, 0.0009, 0.0009, 0.0009, 0.0009])


def get_accumu_over_classes():  # accumulation of training data distributions over all relationship class
    return torch.tensor([0.3482, 0.4840, 0.6070, 0.6785, 0.7450, 0.7922, 0.8247, 0.8478,
                         0.8687, 0.8889, 0.8999, 0.9090, 0.9166, 0.9235, 0.9302, 0.9352, 0.9400,
                         0.9448, 0.9494, 0.9539, 0.9582, 0.9607, 0.9630, 0.9653, 0.9672, 0.9691,
                         0.9709, 0.9727, 0.9744, 0.9761, 0.9778, 0.9794, 0.9809, 0.9824, 0.9839,
                         0.9853, 0.9866, 0.9878, 0.9890, 0.9901, 0.9912, 0.9923, 0.9933, 0.9943,
                         0.9953, 0.9962, 0.9971, 0.9980, 0.9989, 0.9998])


def match_target_sgd(rank, relationships, subj_or_obj, categories_target, bbox_target):
    """
    this function returns the target direction and relationship for scene graph detection, i.e., with predicted bbox and clf
    the predicted bbox and target bbox might be two different sets of bbox
    so we can not use the original sets of target direction and relationships
    """
    cat_subject_target = []
    cat_object_target = []
    bbox_subject_target = []
    bbox_object_target = []
    relation_target = []     # the target relation for target sets of triplets, not for predicted sets
    for image_idx in range(len(relationships)):
        curr_cat_subject = None
        curr_cat_object = None
        curr_bbox_subject = None
        curr_bbox_object = None
        curr_relation_target = None

        for graph_iter in range(len(relationships[image_idx])):
            for edge_iter in range(graph_iter):
                if subj_or_obj[image_idx][graph_iter-1][edge_iter] == 1:
                    if curr_cat_subject is None:
                        curr_cat_subject = torch.tensor([categories_target[image_idx][graph_iter]]).to(rank)
                        curr_cat_object = torch.tensor([categories_target[image_idx][edge_iter]]).to(rank)
                        curr_bbox_subject = bbox_target[image_idx][graph_iter]
                        curr_bbox_object = bbox_target[image_idx][edge_iter]
                        curr_relation_target = torch.tensor([relationships[image_idx][graph_iter-1][edge_iter]]).to(rank)
                    else:
                        curr_cat_subject = torch.hstack((curr_cat_subject, categories_target[image_idx][graph_iter]))
                        curr_cat_object = torch.hstack((curr_cat_object, categories_target[image_idx][edge_iter]))
                        curr_bbox_subject = torch.vstack((curr_bbox_subject, bbox_target[image_idx][graph_iter]))
                        curr_bbox_object = torch.vstack((curr_bbox_object, bbox_target[image_idx][edge_iter]))
                        curr_relation_target = torch.hstack((curr_relation_target, relationships[image_idx][graph_iter-1][edge_iter].to(rank)))

                elif subj_or_obj[image_idx][graph_iter-1][edge_iter] == 0:
                    if curr_cat_subject is None:
                        curr_cat_subject = torch.tensor([categories_target[image_idx][edge_iter]]).to(rank)
                        curr_cat_object = torch.tensor([categories_target[image_idx][graph_iter]]).to(rank)
                        curr_bbox_subject = bbox_target[image_idx][edge_iter]
                        curr_bbox_object = bbox_target[image_idx][graph_iter]
                        curr_relation_target = torch.tensor([relationships[image_idx][graph_iter-1][edge_iter]]).to(rank)
                    else:
                        curr_cat_subject = torch.hstack((curr_cat_subject, categories_target[image_idx][edge_iter]))
                        curr_cat_object = torch.hstack((curr_cat_object, categories_target[image_idx][graph_iter]))
                        curr_bbox_subject = torch.vstack((curr_bbox_subject, bbox_target[image_idx][edge_iter]))
                        curr_bbox_object = torch.vstack((curr_bbox_object, bbox_target[image_idx][graph_iter]))
                        curr_relation_target = torch.hstack((curr_relation_target, relationships[image_idx][graph_iter-1][edge_iter].to(rank)))

        cat_subject_target.append(curr_cat_subject)
        cat_object_target.append(curr_cat_object)
        if curr_relation_target is not None:
            bbox_subject_target.append(curr_bbox_subject.view(-1, 4))
            bbox_object_target.append(curr_bbox_object.view(-1, 4))
        else:
            bbox_subject_target.append(curr_bbox_subject)
            bbox_object_target.append(curr_bbox_object)
        relation_target.append(curr_relation_target)

    return cat_subject_target, cat_object_target, bbox_subject_target, bbox_object_target, relation_target


def compare_object_cat(pred_cat, target_cat):
    # man, person, woman, people, boy, girl, lady, child, kid, men  # tree, plant  # plane, airplane
    equiv = [[1, 5, 11, 23, 38, 44, 121, 124, 148, 149], [0, 50], [92, 137]]
    # vehicle -> car, bus, motorcycle, truck, vehicle
    # animal -> zebra, sheep, horse, giraffe, elephant, dog, cow, cat, bird, bear, animal
    # food -> vegetable, pizza, orange, fruit, banana, food
    unsymm_equiv = {123: [14, 63, 95, 87, 123], 108: [89, 102, 67, 72, 71, 81, 96, 105, 90, 111, 108], 60: [145, 106, 142, 144, 77, 60]}

    if pred_cat == target_cat:
        return True
    for group in equiv:
        if pred_cat in group and target_cat in group:
            return True
    for key in unsymm_equiv:
        if pred_cat == key and target_cat in unsymm_equiv[key]:
            return True
        elif target_cat == key and pred_cat in unsymm_equiv[key]:
            return True
    return False


def match_object_categories(categories_pred, cat_pred_confidence, bbox_pred, bbox_target):
    """
    This function matches the predicted object category for each ground-truth bounding box.
    For each ground-truth bounding box, the function finds the predicted bounding box with the largest IOU
    and regards its predicted object category as the predicted object category of the ground-truth bounding box
    :param categories_pred: a tensor of size N, where N is the number of predicted objects
    :param cat_pred_confidence: a tensor of size N, where N is the number of predicted objects
    :param bbox_pred: a tensor of size Nx4, where N is the number of predicted objects
    :param bbox_target: a tensor of size Mx4, where M is the number of ground-truth objects
    """
    categories_pred_matched = []
    categories_pred_conf_matched = []
    bbox_target_matched = bbox_target.copy()
    if len(bbox_target) != len(bbox_pred):   # batch size
        return None, None, None

    for i in range(len(bbox_target)):
        repeat_count = 0
        assert len(categories_pred[i]) == len(bbox_pred[i])
        curr_categories_pred_matched = []
        curr_categories_pred_conf_matched = []

        for k, curr_bbox_target in enumerate(bbox_target[i]):
            all_ious = []
            for j, curr_bbox_pred in enumerate(bbox_pred[i]):
                all_ious.append(iou(curr_bbox_target, curr_bbox_pred))
            if len(all_ious) < 2:
                return None, None, None
            top_ious = torch.topk(torch.tensor(all_ious), 2)

            # if top two come from the same repeated bounding box
            if top_ious[0][0] == top_ious[0][1]:
                curr_categories_pred_matched.append(categories_pred[i][top_ious[1][0]])
                curr_categories_pred_matched.append(categories_pred[i][top_ious[1][1]])
                curr_categories_pred_conf_matched.append(cat_pred_confidence[i][top_ious[1][0]] * top_ious[0][0])
                curr_categories_pred_conf_matched.append(cat_pred_confidence[i][top_ious[1][1]] * top_ious[0][1])
                # repeat the curr_bbox_target
                bbox_target_matched[i] = torch.cat([bbox_target_matched[i][:k+repeat_count], bbox_target_matched[i][k+repeat_count].view(1, 4),
                                                    bbox_target_matched[i][k+repeat_count:]])
                repeat_count += 1
            else:
                curr_categories_pred_matched.append(categories_pred[i][top_ious[1][0]])
                curr_categories_pred_conf_matched.append(cat_pred_confidence[i][top_ious[1][0]] * top_ious[0][0])

        categories_pred_matched.append(curr_categories_pred_matched)
        categories_pred_conf_matched.append(curr_categories_pred_conf_matched)
    return categories_pred_matched, categories_pred_conf_matched, bbox_target_matched


def record_train_results(args, record, rank, epoch, batch_count, lr, recall_top3, recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs,
                         running_losses, running_loss_relationship, running_loss_contrast, running_loss_connectivity, running_loss_commonsense,
                         connectivity_recall, num_connected, num_not_connected, connectivity_precision, num_connected_pred, wmap_rel, wmap_phrase):

    if args['dataset']['dataset'] == 'vg':
        if args['models']['hierarchical_pred']:
            print('TRAIN, rank %d, epoch %d, batch %d, lr: %.7f, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, loss: %.4f, %.4f, %.4f, %.4f.'
                  % (rank, epoch, batch_count, lr, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2],
                     running_loss_relationship / (args['training']['print_freq'] * args['training']['batch_size']),
                     running_loss_contrast / (args['training']['print_freq'] * args['training']['batch_size']),
                     running_loss_connectivity / (args['training']['print_freq'] * args['training']['batch_size']),
                     running_loss_commonsense / (args['training']['print_freq'] * args['training']['batch_size'])))

            record.append({'rank': rank, 'epoch': epoch, 'batch': batch_count, 'lr': lr,
                           'recall_relationship': [recall[0], recall[1], recall[2]],
                           'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                           'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                           'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                           'zero_shot_recall': [recall_zs[0], recall_zs[1], recall_zs[2]],
                           'mean_zero_shot_recall': [mean_recall_zs[0].item(), mean_recall_zs[1].item(), mean_recall_zs[2].item()],
                           'relationship_loss': running_loss_relationship.item() / (args['training']['print_freq'] * args['training']['batch_size'])})
        else:
            print('TRAIN, rank %d, epoch %d, batch %d, lr: %.7f, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, loss: %.4f, %.4f, conn: %.4f, %.4f.'
                  % (rank, epoch, batch_count, lr, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2],
                     running_loss_relationship / (args['training']['print_freq'] * args['training']['batch_size']),
                     running_loss_connectivity / (args['training']['print_freq'] * args['training']['batch_size']),
                     connectivity_recall / (num_connected + 1e-5), connectivity_precision / (num_connected_pred + 1e-5)))

            record.append({'rank': rank, 'epoch': epoch, 'batch': batch_count, 'lr': lr,
                           'recall_relationship': [recall[0], recall[1], recall[2]],
                           'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                           'zero_shot_recall': [recall_zs[0], recall_zs[1], recall_zs[2]],
                           'mean_zero_shot_recall': [mean_recall_zs[0].item(), mean_recall_zs[1].item(), mean_recall_zs[2].item()],
                           'connectivity_recall': connectivity_recall.item() / (num_connected + 1e-5), 'connectivity_precision': connectivity_precision.item() / (num_connected_pred + 1e-5),
                           'total_losses': running_losses / (args['training']['print_freq'] * args['training']['batch_size']),
                           'relationship_loss': running_loss_relationship.item() / (args['training']['print_freq'] * args['training']['batch_size']),
                           'connectivity_loss': running_loss_connectivity.item() / (args['training']['print_freq'] * args['training']['batch_size']),
                           'num_connected': num_connected, 'num_not_connected': num_not_connected})
    else:
        print('TRAIN, rank %d, epoch %d, batch %d, lr: %.7f, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, '
              'wmap_rel: %.4f, wmap_phrase: %.4f, loss: %.4f, %.4f, conn: %.4f, %.4f.'
              % (rank, epoch, batch_count, lr, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2], wmap_rel, wmap_phrase,
                 running_loss_relationship / (args['training']['print_freq'] * args['training']['batch_size']),
                 running_loss_connectivity / (args['training']['print_freq'] * args['training']['batch_size']),
                 connectivity_recall / (num_connected + 1e-5), connectivity_precision / (num_connected_pred + 1e-5)))

        record.append({'rank': rank, 'epoch': epoch, 'batch': batch_count, 'lr': lr,
                       'recall_relationship': [recall[0], recall[1], recall[2]],
                       'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                       'wmap_rel': wmap_rel.item(), 'wmap_phrase': wmap_phrase.item(),
                       'connectivity_recall': connectivity_recall.item() / (num_connected + 1e-5), 'connectivity_precision': connectivity_precision.item() / (num_connected_pred + 1e-5),
                       'total_losses': running_losses / (args['training']['print_freq'] * args['training']['batch_size']),
                       'relationship_loss': running_loss_relationship.item() / (args['training']['print_freq'] * args['training']['batch_size']),
                       'connectivity_loss': running_loss_connectivity.item() / (args['training']['print_freq'] * args['training']['batch_size']),
                       'num_connected': num_connected, 'num_not_connected': num_not_connected})
    with open(args['training']['result_path'] + 'train_results_' + str(rank) + '.json', 'w') as f:
        json.dump(record, f)


def record_test_results(args, test_record, rank, epoch, recall_top3, recall, mean_recall_top3, mean_recall, recall_zs, mean_recall_zs,
                        connectivity_recall, num_connected, num_not_connected, connectivity_precision, num_connected_pred, wmap_rel, wmap_phrase):

    if args['dataset']['dataset'] == 'vg':
        if args['models']['hierarchical_pred']:
            print('TEST, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f'
                  % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2]))

            test_record.append({'rank': rank, 'epoch': epoch, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                                'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                                'num_connected': num_connected, 'num_not_connected': num_not_connected})
        else:
            print('TEST, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, conn: %.4f, %.4f.'
                  % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2],
                     connectivity_recall / (num_connected + 1e-5), connectivity_precision / (num_connected_pred + 1e-5)))

        test_record.append({'rank': rank, 'epoch': epoch, 'recall_relationship': [recall[0], recall[1], recall[2]],
                            'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                            'num_connected': num_connected, 'num_not_connected': num_not_connected})
    else:
        print('TEST, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, wmap_rel: %.4f, wmap_phrase: %.4f, conn: %.4f, %.4f.'
              % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2], wmap_rel, wmap_phrase,
                 connectivity_recall / (num_connected + 1e-5), connectivity_precision / (num_connected_pred + 1e-5)))

        test_record.append({'rank': rank, 'epoch': epoch, 'recall_relationship': [recall[0], recall[1], recall[2]],
                            'wmap_rel': wmap_rel.item(), 'wmap_phrase': wmap_phrase.item(),
                            'connectivity_recall': connectivity_recall.item() / (num_connected + 1e-5),
                            'connectivity_precision': connectivity_precision.item() / (num_connected_pred + 1e-5),
                            'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                            'num_connected': num_connected, 'num_not_connected': num_not_connected})

        with open(args['training']['result_path'] + 'test_results_' + str(rank) + '.json', 'w') as f:  # append current logs
            json.dump(test_record, f)
