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


def collate_fn(batch):
    """
    This function solves the problem when some data samples in a batch are None.
    :param batch: the current batch in dataloader
    :return: a new batch with all non-None data samples
    """
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))


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

    model_path = args['models']['detr101_pretrained']
    model_param = torch.load(model_path)

    keys = [key for key in model_param['model'] if key in name_before]
    for idx, key in enumerate(keys):
        model_param['model'][name_after[idx]] = model_param['model'].pop(key)
        # print(idx, key, ' -> ', name_after[idx])

    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101', pretrained=False)
    model.class_embed = nn.Linear(256, 151)
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


def get_num_each_class():  # number of training data in total for each relationship class
    return torch.tensor([712432, 277943, 251756, 146339, 136099, 96589, 66425, 47342, 42722,
                         41363, 22596, 18643, 15457, 14185, 13715, 10191, 9903, 9894, 9317, 9145, 8856,
                         5213, 4688, 4613, 3810, 3806, 3739, 3624, 3490, 3477, 3411, 3288, 3095, 3092,
                         3083, 2945, 2721, 2517, 2380, 2312, 2253, 2241, 2065, 1996, 1973, 1925, 1914,
                         1869, 1853, 1740])


def get_num_each_class_reordered():  # number of training data in total for each relationship class
    return torch.tensor([47342, 1996, 3092, 3624, 3477, 9903, 41363, 3411, 251756,
                         13715, 96589, 712432, 1914, 9317, 22596, 3288, 9145, 2945,
                         277943, 2312, 146339, 2065, 2517, 136099, 15457, 66425, 10191,
                         5213, 2312, 3806, 4688, 1973, 1853, 9894, 42722, 3739,
                         3083, 1869, 2253, 3095, 2721, 3810, 8856, 2241, 18643,
                         14185, 1925, 1740, 4613, 3490])


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
