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
import openai
import math
from collections import Counter
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


def calculate_losses_on_relationships(args, relation, super_relation, connected, curr_relations_target, pseudo_label_mask, criterion_relationship, lambda_pseudo=1):
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
            loss_relationship += criterion_relationship(relation[connected], curr_relations_target[connected])

    return loss_relationship


def super_relation_processing(args, connected, curr_relations_target):
    # Clone the target relations for modification without affecting the original data
    super_relation_target = curr_relations_target[connected].clone()
    super_relation_target[super_relation_target < args['models']['num_geometric']] = 0
    super_relation_target[torch.logical_and(super_relation_target >= args['models']['num_geometric'], super_relation_target < args['models']['num_geometric'] + args['models']['num_possessive'])] = 1
    super_relation_target[super_relation_target >= args['models']['num_geometric'] + args['models']['num_possessive']] = 2

    return super_relation_target


def calculate_semi_supervised_loss(predictions, targets, pseudo_label_mask, criterion, lambda_pseudo):
    # Separate pseudo and true labels based on mask
    pseudo_labels = pseudo_label_mask.bool()
    true_labels = ~pseudo_labels

    # Initialize losses
    loss_pseudo = loss_true = torch.tensor(0.0).to(predictions.device)

    # Calculate pseudo loss if there are pseudo labels
    if pseudo_labels.any():
        loss_pseudo = lambda_pseudo * criterion(predictions[pseudo_labels], targets[pseudo_labels])

    # Calculate true loss if there are true labels
    if true_labels.any():
        loss_true = criterion(predictions[true_labels], targets[true_labels])

    return loss_pseudo, loss_true


def query_openai_gpt(predicted_edges, cache=None, model='gpt-3.5-turbo'):
    # load your secret OpenAI API key
    # you can register yours at https://platform.openai.com/account/api-keys and save it as openai_api_key.txt
    # do not share your API key with others, or expose it in the browser or other client-side code
    openai.api_key_path = 'openai_key_mc.txt' #'openai_api_key.txt'
    random_val = random.random()    # without randomness, the model will always return the same answer for the same prompt

    responses = []
    for predicted_edge in predicted_edges:
        # first check cache
        if cache is not None and predicted_edge in cache and random_val < 0.9:
            cache.move_to_end(predicted_edge)
            responses.append(cache[predicted_edge])

        else:
            prompt_template = "Considering common sense and typical real-world scenarios, does the relation '{}' make logical sense? Show your reasoning and answer Yes or No. " #Briefly show your reasoning."
            # prompt_template = "Based on the commonsense, is '{}' a physically valid relation or not? Briefly show your reasoning, but make sure your last word must be either 'Yes' or 'No'."
            # for i, prompt_template in enumerate([
            #     "Based on the commonsense, is '{}' a physically valid relation or not? Briefly show your reasoning, but make sure your last word must be either 'Yes' or 'No'."
            #     # "Does the relationship '{}' violate the commonsense or not? Explain your reasoning, but end your answer with the word 'Yes' or 'No'.",
            #     # "Consider a scene where objects are described in terms of their relationships. Is it reasonable to say '{}' in such a scene? Briefly show your reasoning, but your last word in the answer must be either 'Yes' or 'No'."
            # ]):
            messages = [{"role": "user", "content": prompt_template.format(predicted_edge)}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
            )

            response = response.choices[0].message["content"]
            # answer = response.split(" ")[-1]
            if re.search(r'Yes', response):
                # print('predicted_edge', predicted_edge, ' [YES] response', response)
                cache[predicted_edge] = 1  # update cache
                responses.append(1)
            elif re.search(r'No', response):
                # print('predicted_edge', predicted_edge, ' [NO] response', response)
                cache[predicted_edge] = -1  # update cache
                responses.append(-1)
            else:
                # print('predicted_edge', predicted_edge, ' [INVALID] response', response)
                responses.append(-1)

    return responses


def batch_query_openai_gpt_instruct(predicted_edges, cache=None, batch_size=6):
    total_edges = len(predicted_edges)
    all_responses = []

    for i in range(0, total_edges, batch_size):
        batched_edges = predicted_edges[i: i + batch_size]
        responses = _batch_query_openai_gpt_instruct(batched_edges)
        all_responses.extend(responses)

    return all_responses


def _batch_query_openai_gpt_instruct(predicted_edges, model='gpt-3.5-turbo-instruct'):
    openai.api_key_path = 'openai_key_mc.txt' #'openai_api_key.txt'
    responses = torch.ones(len(predicted_edges)) * -1

    # random_val = random.random()
    # # first check cache
    # if cache is not None and edge in cache and random_val < 0.9:
    #     responses[ind] = cache[edge]
    #     continue

    prompts = []

    # Prepare multiple variations of each prompt
    prompt_variations = [
        "Considering common sense and typical real-world scenarios, does the relation '{}' make logical sense? Answer with Yes or No and briefly provide your reasoning.",
        "Given the general knowledge and understanding of the world, is the relation '{}' logical? Provide a Yes or No response.", #and briefly explain your choice.",
        "Would you say the relation '{}' violates typical logic and is not plausible? Yes or No." # Provide a brief explanation and answer with Yes or No."
    ]

    # For each predicted edge, create multiple prompts
    for edge in predicted_edges:
        for variation in prompt_variations:
            prompts.append(variation.format(edge))

    # Call OpenAI with the batch of prompts
    completions = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0,
        max_tokens=100
    )

    # Gather responses and decide based on majority
    for i, edge in enumerate(predicted_edges):
        yes_votes = 0
        no_votes = 0
        for j in range(len(prompt_variations)):
            completion_text = completions.choices[i * len(prompt_variations) + j].text

            if j == 2:  # For the third question, we reverse the logic
                if re.search(r'Yes', completion_text):
                    no_votes += 1
                elif re.search(r'No', completion_text):
                    yes_votes += 1
                else:
                    no_votes += 1
            else:
                if re.search(r'Yes', completion_text):
                    yes_votes += 1
                elif re.search(r'No', completion_text):
                    no_votes += 1
                else:
                    no_votes += 1

        if yes_votes > no_votes:
            # print(f'predicted_edge {edge} [MAJORITY YES] {yes_votes} Yes votes vs {no_votes} No votes')
            responses[i] = 1
        else:
            # print(f'predicted_edge {edge} [MAJORITY NO] {no_votes} No votes vs {yes_votes} Yes votes')
            responses[i] = -1

    return responses

# def query_openai_gpt(predicted_edge, model='gpt-3.5-turbo'):
#     # load your secret OpenAI API key
#     # you can register yours at https://platform.openai.com/account/api-keys and save it as openai_api_key.txt
#     # do not share your API key with others, or expose it in the browser or other client-side code
#     openai.api_key_path = 'openai_api_key.txt'
#
#     # we will query the GPT model with more than one different prompts to ensure robustness
#     answers = []
#
#     prompt = "In scene graph generation, the model shall predict relationships between a given subject and object pair. Based on the commonsense, is the relationship '"
#     prompt += predicted_edge
#     prompt += "' reasonable or not? Show your reasoning, but make sure your last word must be either 'Yes' or 'No'."
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#     )
#     answer = response.choices[0].message["content"].split(" ")[-1]
#     # print('Prompt', prompt, 'Response: ', answer)
#     if re.search(r'Yes', answer):
#         answers.append('Yes.')
#     elif re.search(r'No', answer):
#         answers.append('No.')
#     else:
#         answers.append(answer)
#
#     prompt = "Does the relationship '"
#     prompt += predicted_edge
#     prompt += "' physically make sense based on the commonsense? Show your reasoning, but end your answer with the word 'Yes' or 'No'."
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#     )
#     answer = response.choices[0].message["content"].split(" ")[-1]
#     # print('Prompt', prompt, 'Response: ', answer)
#     if re.search(r'Yes', answer):
#         answers.append('Yes.')
#     elif re.search(r'No', answer):
#         answers.append('No.')
#     else:
#         answers.append(answer)
#
#     prompt = "Consider a scene where objects are described in terms of their relationships. Is it reasonable to say '"
#     prompt += predicted_edge
#     prompt += "' in such a scene? Show your reasoning, but your last word in the answer must be either 'Yes' or 'No'."
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#     )
#     answer = response.choices[0].message["content"].split(" ")[-1]
#     # print('Prompt', prompt, 'Response: ', answer)
#     if re.search(r'Yes', answer):
#         answers.append('Yes.')
#     elif re.search(r'No', answer):
#         answers.append('No.')
#     else:
#         answers.append(answer)
#
#     # find the majority vote
#     answer_counts = Counter(answers)
#     majority_vote = answer_counts.most_common(1)[0][0]
#     print('predicted_edge', predicted_edge, 'answers', answers, 'majority_vote', majority_vote)
#     if majority_vote == 'Yes.':
#         return 1.0
#     elif majority_vote == 'No.':
#         return float('-inf')
#     else:   # ignore any invalid response
#         return None


def sigmoid_rampup(current, rampup_length):
    """Sigmoid ramp-up. current is the current epoch, rampup_length is the length of the ramp-up."""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


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
                         running_losses, running_loss_relationship, running_loss_contrast, running_loss_connectivity, running_loss_pseudo_consistency,
                         connectivity_recall, num_connected, num_not_connected, connectivity_precision, num_connected_pred, wmap_rel, wmap_phrase):

    if args['dataset']['dataset'] == 'vg':
        if args['models']['hierarchical_pred']:
            print('TRAIN, rank %d, epoch %d, batch %d, lr: %.7f, R@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), mR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), '
                  'zsR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), loss: %.4f, %.4f, %.4f, %.4f.'
                  % (rank, epoch, batch_count, lr, recall_top3[0], recall_top3[1], recall_top3[2], recall[0], recall[1], recall[2],
                     mean_recall_top3[0], mean_recall_top3[1], mean_recall_top3[2], mean_recall[0], mean_recall[1], mean_recall[2],
                     recall_zs[0], recall_zs[1], recall_zs[2], mean_recall_zs[0], mean_recall_zs[1], mean_recall_zs[1],
                     running_loss_relationship / (args['training']['print_freq'] * args['training']['batch_size']),
                     running_loss_contrast / (args['training']['print_freq'] * args['training']['batch_size']),
                     running_loss_connectivity / (args['training']['print_freq'] * args['training']['batch_size']),
                     running_loss_pseudo_consistency / (args['training']['print_freq'] * args['training']['batch_size'])))

            record.append({'rank': rank, 'epoch': epoch, 'batch': batch_count, 'lr': lr,
                           'recall_relationship': [recall[0], recall[1], recall[2]],
                           'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                           'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                           'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                           'zero_shot_recall': [recall_zs[0], recall_zs[1], recall_zs[2]],
                           'mean_zero_shot_recall': [mean_recall_zs[0].item(), mean_recall_zs[1].item(), mean_recall_zs[2].item()],
                           'relationship_loss': running_loss_relationship.item() / (args['training']['print_freq'] * args['training']['batch_size'])})
        else:
            print('TRAIN, rank %d, epoch %d, batch %d, lr: %.7f, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, '
                  'zsR@k: %.4f, %.4f, %.4f, zs-mR@k: %.4f, %.4f, %.4f,loss: %.4f, %.4f, conn: %.4f, %.4f.'
                  % (rank, epoch, batch_count, lr, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2],
                     recall_zs[0], recall_zs[1], recall_zs[2], mean_recall_zs[0], mean_recall_zs[1], mean_recall_zs[2],
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
                        connectivity_recall, num_connected, num_not_connected, connectivity_precision, num_connected_pred, wmap_rel, wmap_phrase, global_refine=False):

    if args['dataset']['dataset'] == 'vg':
        if args['models']['hierarchical_pred']:
            if global_refine:
                if recall_zs is None:
                    print('TEST, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f.'
                          % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2]))
                else:
                    print('TEST, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, zsR@k: %.4f, %.4f, %.4f, zs-mR@k: %.4f, %.4f, %.4f.'
                          % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2],
                             recall_zs[0], recall_zs[1], recall_zs[2], mean_recall_zs[0], mean_recall_zs[1], mean_recall_zs[2]))

                test_record.append({'rank': rank, 'epoch': epoch, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                    'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                    'num_connected': num_connected, 'num_not_connected': num_not_connected})
            else:
                if recall_top3 is None:
                    if recall_zs is None:
                        print('TEST, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f.'
                              % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2]))

                        test_record.append({'rank': rank, 'epoch': epoch, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                            'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                            'num_connected': num_connected, 'num_not_connected': num_not_connected})
                    else:
                        print('TEST, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, '
                              'zsR@k: %.4f, %.4f, %.4f, zs-mR@k: %.4f, %.4f, %.4f.'
                              % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2],
                                 recall_zs[0], recall_zs[1], recall_zs[2], mean_recall_zs[0], mean_recall_zs[1], mean_recall_zs[2]))

                        test_record.append({'rank': rank, 'epoch': epoch, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                            'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                            'num_connected': num_connected, 'num_not_connected': num_not_connected})
                else:
                    print('TEST, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), mR@k: %.4f, %.4f, %.4f (%.4f, %.4f, %.4f), '
                          'zsR@k: %.4f, %.4f, %.4f, zs-mR@k: %.4f, %.4f, %.4f.'
                          % (rank, epoch, recall_top3[0], recall_top3[1], recall_top3[2], recall[0], recall[1], recall[2],
                             mean_recall_top3[0], mean_recall_top3[1], mean_recall_top3[2], mean_recall[0], mean_recall[1], mean_recall[2],
                             recall_zs[0], recall_zs[1], recall_zs[2], mean_recall_zs[0], mean_recall_zs[1], mean_recall_zs[2]))

                    test_record.append({'rank': rank, 'epoch': epoch, 'recall_relationship': [recall[0], recall[1], recall[2]],
                                        'recall_relationship_top3': [recall_top3[0], recall_top3[1], recall_top3[2]],
                                        'mean_recall': [mean_recall[0].item(), mean_recall[1].item(), mean_recall[2].item()],
                                        'mean_recall_top3': [mean_recall_top3[0].item(), mean_recall_top3[1].item(), mean_recall_top3[2].item()],
                                        'num_connected': num_connected, 'num_not_connected': num_not_connected})
        else:
            if global_refine:
                if recall_zs is None:
                    print('GRAP, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, conn: %.4f, %.4f'
                          % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2],
                             connectivity_recall / (num_connected + 1e-5), connectivity_precision / (num_connected_pred + 1e-5)))
                else:
                    print('GRAP, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, zsR@k: %.4f, %.4f, %.4f, zs-mR@k: %.4f, %.4f, %.4f, conn: %.4f, %.4f'
                          % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2],
                             recall_zs[0], recall_zs[1], recall_zs[2], mean_recall_zs[0], mean_recall_zs[1], mean_recall_zs[2],
                             connectivity_recall / (num_connected + 1e-5), connectivity_precision / (num_connected_pred + 1e-5)))
            else:
                if recall_zs is None:
                    print('TEST, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f.'
                          % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2]))
                else:
                    print('TEST, rank: %d, epoch: %d, R@k: %.4f, %.4f, %.4f, mR@k: %.4f, %.4f, %.4f, zsR@k: %.4f, %.4f, %.4f, zs-mR@k: %.4f, %.4f, %.4f, conn: %.4f, %.4f.'
                          % (rank, epoch, recall[0], recall[1], recall[2], mean_recall[0], mean_recall[1], mean_recall[2],
                             recall_zs[0], recall_zs[1], recall_zs[2], mean_recall_zs[0], mean_recall_zs[1], mean_recall_zs[2],
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