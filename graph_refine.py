import torch
import os
from PIL import Image
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from collections import deque
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from itertools import repeat
import datetime
import yaml
import re
from torch.utils.tensorboard import SummaryWriter
import shutil

from evaluate import *
from utils import *
from dataset_utils import relation_by_super_class_int2str
from model import *
from sup_contrast.losses import SupConLossGraph


# define some global lists
rel_id2txt = relation_by_super_class_int2str()
all_labels = list(relation_by_super_class_int2str().values())
# load hyperparameters
with open('config.yaml', 'r') as file:
    args = yaml.safe_load(file)
all_labels_geometric = all_labels[:args['models']['num_geometric']]
all_labels_possessive = all_labels[args['models']['num_geometric']:args['models']['num_geometric'] + args['models']['num_possessive']]
all_labels_semantic = all_labels[-args['models']['num_semantic']:]


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


class ImageGraph:
    def __init__(self, args, rank, tokenizer, processor, clip_model, targets=None):
        self.args = args
        self.rank = rank
        self.tokenizer = tokenizer
        self.processor = processor
        self.clip_model = clip_model
        self.targets = targets
        # node to neighbors mapping
        self.adj_list = {}
        # edge to nodes mapping
        self.edge_node_map = {}
        # store all nodes and their degree
        self.nodes = []
        self.edges = []
        self.edge_embeddings = {}   # key: edge, value: concatenated clip text and vision embedding of the edge
        self.rel_embeddings = {}    # key: edge, value: clip text embedding of the relationship on the edge
        self.confidence = []    # only record new confidence after refinement

    def prepare_clip_embeds(self, subject_bbox, object_bbox, string, image):
        with torch.no_grad():
            _, relation, _ = extract_words_from_edge(string, all_labels)
            inputs = self.tokenizer([string], padding=False, return_tensors="pt").to(self.rank)
            txt_embed = self.clip_model.module.get_text_features(**inputs)
            txt_embed = F.normalize(txt_embed, dim=1, p=2)
            inputs = self.tokenizer([relation], padding=False, return_tensors="pt").to(self.rank)
            relation_embed = self.clip_model.module.get_text_features(**inputs)

            image = image.to(self.rank)
            mask_sub = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.bool).to(self.rank)
            mask_obj = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.bool).to(self.rank)
            mask_sub[:, subject_bbox[2]:subject_bbox[3], subject_bbox[0]:subject_bbox[1]] = 1
            mask_obj[:, object_bbox[2]:object_bbox[3], object_bbox[0]:object_bbox[1]] = 1
            img_sub = image * mask_sub
            img_obj = image * mask_obj
            img_edge = torch.cat((img_sub.unsqueeze(dim=0), img_obj.unsqueeze(dim=0)), dim=0)

            inputs = self.processor(images=img_edge, return_tensors="pt").to(self.rank)
            img_embed = self.clip_model.module.get_image_features(**inputs)
            img_embed = F.normalize(img_embed, dim=1, p=2)
            img_embed = img_embed.view(1, -1)

        edge_embed = torch.cat([txt_embed, img_embed], dim=1)
        return edge_embed, relation_embed

    def add_edge(self, subject_bbox, object_bbox, relation_id, string, image, training=True, verbose=False):
        subject_bbox, object_bbox = tuple(subject_bbox), tuple(object_bbox)
        edge = (subject_bbox, relation_id, object_bbox, string)
        edge_wo_string = (subject_bbox, relation_id, object_bbox)

        if edge not in self.edges:
            self.edges.append(edge)
            self.confidence.append(-1)

            # add clip text and image embeddings of this edge
            edge_embed, relation_embed = self.prepare_clip_embeds(subject_bbox, object_bbox, string, image)
            self.edge_embeddings[edge] = edge_embed
            self.rel_embeddings[edge] = relation_embed

        if subject_bbox not in self.nodes:
            self.nodes.append(subject_bbox)
        if object_bbox not in self.nodes:
            self.nodes.append(object_bbox)

        if verbose:
            print('subject_bbox', subject_bbox)
            print('object_bbox', object_bbox)
            print('edge', edge, '\n')

        # check if the node is already present, otherwise initialize with an empty list
        if subject_bbox not in self.adj_list:
            self.adj_list[subject_bbox] = []
        if object_bbox not in self.adj_list:
            self.adj_list[object_bbox] = []

        # in training, store ground truth neighbors if a target is matched
        if training:
            matched_target_edge = find_matched_target(self.args, edge, self.targets)
            # if matched_target_edge is not None:
            self.adj_list[subject_bbox].append(matched_target_edge)
            self.adj_list[object_bbox].append(matched_target_edge)

            # if matched_target_edge is different from the current edge
            if matched_target_edge not in self.edge_embeddings:
                matched_subject_bbox, matched_object_bbox, matched_string = matched_target_edge[0], matched_target_edge[2], matched_target_edge[3]
                matched_edge_embed, matched_relation_embed = self.prepare_clip_embeds(matched_subject_bbox, matched_object_bbox, matched_string, image)
                self.edge_embeddings[matched_target_edge] = matched_edge_embed
                self.rel_embeddings[matched_target_edge] = matched_relation_embed

        else:
            self.adj_list[subject_bbox].append(edge)
            self.adj_list[object_bbox].append(edge)

        self.edge_node_map[edge_wo_string] = (subject_bbox, object_bbox)

    def get_edge_neighbors(self, edge, hops=1):
        # find all the edges belonging to the 1-hop neighbor of the current edge
        curr_pair = self.edge_node_map[edge]
        subject_node, object_node = curr_pair[0], curr_pair[1]

        # find all edges connecting to the current subject and object node
        neighbor_edges = self.adj_list[subject_node] + self.adj_list[object_node]

        # remove the current edge from the set
        for neighbor_edge in neighbor_edges:
            if neighbor_edge[:-1] == edge:
                neighbor_edges.remove(neighbor_edge)

        if hops == 1:
            return set(neighbor_edges)

        elif hops == 2:
            # copy all hop1 edges
            hop2_neighbor_edges = [hop1_edge for hop1_edge in neighbor_edges]

            for hop2_edge in neighbor_edges:
                curr_pair = self.edge_node_map[hop2_edge[:-1]]
                subject_node, object_node = curr_pair[0], curr_pair[1]
                hop2_neighbor_edges += self.adj_list[subject_node] + self.adj_list[object_node]
                # don't have to remove curr hop2_edge because it is already in neighbor_edges and a set operation is enough

            # remove the current edge from the set by any chance
            for hop2_neighbor_edge in hop2_neighbor_edges:
                if hop2_neighbor_edge[:-1] == edge:
                    hop2_neighbor_edges.remove(hop2_neighbor_edge)

            return set(hop2_neighbor_edges)

        else:
            assert hops == 1 or hops == 2, "Not implemented"

    def get_node_degrees(self):
        degrees = {node: len(self.adj_list[node]) for node in self.adj_list}
        return degrees


def colored_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


def save_png(image, save_name="image.png"):
    image = image.mul(255).cpu().byte().numpy()  # convert to 8-bit integer values
    image = Image.fromarray(image.transpose(1, 2, 0))  # transpose dimensions for RGB order
    image.save(save_name)


def print_layers_in_optimizer(optimizer, attention_layer, relationship_refiner):
    # Collect all parameters into a list
    all_params = []
    for param_group in optimizer.param_groups:
        params = param_group['params']
        all_params.extend(params)

    # Create a dictionary to map parameters to layer names
    param_to_layer = {}
    for name, param in attention_layer.module.named_parameters():
        param_to_layer[param] = f'attention_layer.{name}'
    for name, param in relationship_refiner.module.named_parameters():
        param_to_layer[param] = f'relationship_refiner.{name}'

    # Extract and print the parameters along with layer names
    print("Parameters to be backpropagated:")
    for param in all_params:
        layer_name = param_to_layer[param]
        print(f"Layer: {layer_name}, Size: {param.size()}, Requires Grad: {param.requires_grad}")


def extract_updated_edges(graph, rank):
    # initialize a torch tensor for updated relations
    relation_pred = torch.tensor([graph.edges[i][1] for i in range(len(graph.edges))]).to(rank)
    confidence = torch.tensor([graph.confidence[i] for i in range(len(graph.confidence))]).to(rank)
    return relation_pred, confidence


def extract_words_from_edge(phrase, all_relation_labels):
    # create a regular expression pattern to match the relations
    pattern = r'\b(' + '|'.join(map(re.escape, all_relation_labels)) + r')\b'
    phrase = re.split(pattern, phrase)
    subject, relation, object = phrase[0], phrase[1], phrase[2]
    return subject, relation, object


def crop_image(image, edge, args, crop=True):
    # crop out the subject and object from the image
    width, height = image.shape[1], image.shape[2]
    subject_bbox = torch.tensor(edge[0]) / args['models']['feature_size']
    object_bbox = torch.tensor(edge[2]) / args['models']['feature_size']
    subject_bbox[:2] *= height
    subject_bbox[2:] *= width
    object_bbox[:2] *= height
    object_bbox[2:] *= width
    # print('image', image.shape, 'subject_bbox', subject_bbox, 'object_bbox', object_bbox)

    # create the union bounding box
    union_bbox = torch.zeros(image.shape[1:], dtype=torch.bool)
    union_bbox[int(subject_bbox[2]):int(subject_bbox[3]), int(subject_bbox[0]):int(subject_bbox[1])] = 1
    union_bbox[int(object_bbox[2]):int(object_bbox[3]), int(object_bbox[0]):int(object_bbox[1])] = 1

    if crop:
        # find the minimum rectangular bounding box around the union bounding box
        nonzero_indices = torch.nonzero(union_bbox)
        min_row = nonzero_indices[:, 0].min()
        max_row = nonzero_indices[:, 0].max()
        min_col = nonzero_indices[:, 1].min()
        max_col = nonzero_indices[:, 1].max()

        # crop the image using the minimum rectangular bounding box
        cropped_image = image[:, min_row:max_row + 1, min_col:max_col + 1]

        # print('Cropped Image:', cropped_image.shape)
        return cropped_image
    else:
        return image * union_bbox


def find_matched_target(args, edge, targets):
    subject_bbox, object_bbox = edge[0], edge[2]
    current_subject, _, current_object = extract_words_from_edge(edge[-1], all_labels)

    for target in targets:
        target_subject_bbox = target[0]
        target_object_bbox = target[2]

        if args['training']['eval_mode'] == 'pc':
            condition = target_subject_bbox == subject_bbox and target_object_bbox == object_bbox
        else:
            condition = iou(target_subject_bbox, subject_bbox) >= 0.5 and iou(target_object_bbox, object_bbox) >= 0.5

        if condition:
            target_subject, _, target_object = extract_words_from_edge(target[-1], all_labels)

            if target_subject == current_subject and target_object == current_object:
                return target

    # return None
    return edge  # return the original edge if no target matched


def prepare_target_txt_embeds(clip_model, tokenizer, rank):
    # extract current subject and object from the edge
    queries = [f"{label}" for label in all_labels]

    # collect text_embeds for all possible labels
    inputs = tokenizer(queries, padding=True, return_tensors="pt").to(rank)
    with torch.no_grad():
        all_possible_embeds = clip_model.module.get_text_features(**inputs)  # size [num_edges * 50, hidden_embed]

    return all_possible_embeds


def eval_refined_output(clip_model, tokenizer, predicted_txt_embeds, current_edges, rank, args, verbose=False):
    # pre-compute common arguments
    num_geom, num_poss, num_sem = args['models']['num_geometric'], args['models']['num_possessive'], args['models']['num_semantic']

    # extract current subject and object from the edge
    queries = []
    num_candidate_labels = []
    for current_edge in current_edges:
        if args['models']['hierarchical_pred']:
            relation_id = current_edge[1]
            if relation_id < num_geom:
                # subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels_geometric)
                queries.extend([f"{label}" for label in all_labels_geometric])
                num_candidate_labels.append(num_geom)
            elif num_geom <= relation_id < num_geom + num_poss:
                # subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels_possessive)
                queries.extend([f"{label}" for label in all_labels_possessive])
                num_candidate_labels.append(num_poss)
            else:
                # subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels_semantic)
                queries.extend([f"{label}" for label in all_labels_semantic])
                num_candidate_labels.append(num_sem)
        else:
            # subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels)
            queries.extend([f"{label}" for label in all_labels])

    predicted_txt_embeds = predicted_txt_embeds.unsqueeze(dim=1)

    # collect text_embeds for all possible labels
    inputs = tokenizer(queries, padding=True, return_tensors="pt").to(rank)
    with torch.no_grad():
        all_possible_embeds = clip_model.module.get_text_features(**inputs)     # size [num_edges * 50, hidden_embed]
    all_possible_embeds = F.normalize(all_possible_embeds, dim=1, p=2)

    if args['models']['hierarchical_pred']:
        # split out each data sample in the batch
        all_possible_embeds = torch.split(all_possible_embeds, [num for num in num_candidate_labels])

        num_candidate_labels = torch.tensor(num_candidate_labels).to(rank)
        max_vals = -torch.ones(len(current_edges), dtype=torch.float32).to(rank)
        max_indices = -torch.ones(len(current_edges), dtype=torch.int64).to(rank)

        mask_geom = num_candidate_labels == num_geom
        mask_poss = num_candidate_labels == num_poss
        mask_sem = num_candidate_labels == num_sem

        # get integer indices for each label type
        ids_geometric = torch.nonzero(mask_geom).flatten()
        ids_possessive = torch.nonzero(mask_poss).flatten()
        ids_semantic = torch.nonzero(mask_sem).flatten()
        all_possible_embeds_geom = torch.stack([all_possible_embeds[i] for i in ids_geometric]) if torch.sum(ids_geometric) > 0 else None
        all_possible_embeds_poss = torch.stack([all_possible_embeds[i] for i in ids_possessive]) if torch.sum(ids_possessive) > 0 else None
        all_possible_embeds_sem = torch.stack([all_possible_embeds[i] for i in ids_semantic]) if torch.sum(ids_semantic) > 0 else None

        # calculate Cosine Similarities
        CosSim = nn.CosineSimilarity(dim=2)
        for label_type, masked_ids, embeds in [("geometric", mask_geom, all_possible_embeds_geom),
                                               ("possessive", mask_poss, all_possible_embeds_poss),
                                               ("semantic", mask_sem, all_possible_embeds_sem)]:
            if embeds is not None:
                cos_sims = CosSim(predicted_txt_embeds[masked_ids], embeds)
                cos_sims = F.softmax(cos_sims, dim=1)
                max_vals_cur, max_indices_cur = cos_sims.max(dim=1)

                max_vals[masked_ids] = max_vals_cur
                max_indices[masked_ids] = max_indices_cur

        if verbose:
            print('predicted_txt_embeds', predicted_txt_embeds.shape, 'all_possible_embeds_geom', all_possible_embeds_geom.shape,
                  'all_possible_embeds_poss', all_possible_embeds_poss.shape, 'all_possible_embeds_sem', all_possible_embeds_sem.shape)
    else:
        all_possible_embeds = torch.split(all_possible_embeds, [len(all_labels) for _ in range(len(current_edges))])    # size [[50, hidden_embed] * num_edges]
        all_possible_embeds = torch.stack(all_possible_embeds)    # size [num_edges, 50, hidden_embed]

        # compute cosine similarity between predicted embedding and all query embeddings
        CosSim = nn.CosineSimilarity(dim=2)  # Set dim=2 to compute cosine similarity across the embedding dimension
        cos_sims = CosSim(predicted_txt_embeds, all_possible_embeds)
        cos_sims = F.softmax(cos_sims, dim=1)
        max_vals, max_indices = cos_sims.max(dim=1)

        if verbose:
            print('predicted_txt_embeds', predicted_txt_embeds.shape, 'all_possible_embeds', all_possible_embeds.shape)

    updated_edges = []
    for idx, current_edge in enumerate(current_edges):
        if args['models']['hierarchical_pred']:
            if num_candidate_labels[idx] == num_geom:
                predicted_rel = all_labels_geometric[max_indices[idx]]
                predicted_rel_id = max_indices[idx].item()
            elif num_candidate_labels[idx] == num_poss:
                predicted_rel = all_labels_possessive[max_indices[idx]]
                predicted_rel_id = max_indices[idx].item() + num_geom
            else:
                predicted_rel = all_labels_semantic[max_indices[idx]]
                predicted_rel_id = max_indices[idx].item() + num_geom + num_poss
        else:
            predicted_rel = all_labels[max_indices[idx]]
            predicted_rel_id = max_indices[idx].item()

        subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels)
        updated_phrase = subject + predicted_rel + object
        updated_edge = (current_edge[0], predicted_rel_id, current_edge[2], updated_phrase)
        updated_edges.append(updated_edge)

        # show the results
        if verbose:
            light_blue_code = 94
            light_pink_code = 95
            text_blue_colored = colored_text(predicted_rel, light_blue_code)
            text_pink_colored = colored_text(relation, light_pink_code)
            print(f"Predicted label: '{text_blue_colored}' with confidence {max_vals[idx]}, Old label: '{text_pink_colored}'")

            dark_blue_code = 34
            text_blue_colored = colored_text(updated_edge, dark_blue_code)
            print('Updated_edge', text_blue_colored, '\n')

    # print('max_vals', max_vals, 'max_indices', max_indices)
    return updated_edges, max_vals


def find_negative_targets(current_subject, current_object, target_label, target_txt_embed, clip_model, tokenizer, rank, args):
    # find the label with the highest similarity to the current positive target
    queries = [f"a photo of a {current_subject} {label} {current_object}" for label in all_labels if label != target_label]

    # Collect text embeddings for all possible labels
    inputs = tokenizer(queries, padding=True, return_tensors="pt").to(rank)
    with torch.no_grad():
        candidate_txt_embeds = clip_model.module.get_text_features(**inputs)

    # # select top negative_txt_embeds with closest cosine distances but not the target
    # CosSim = nn.CosineSimilarity(dim=1)
    # cos_sim = CosSim(target_txt_embed, candidate_txt_embeds)
    # top_similarities, top_indices = cos_sim.topk(args['models']['num_negatives'], largest=False)
    # # print('top_similarities', top_similarities)
    #
    # negative_txt_embeds = candidate_txt_embeds[top_indices]

    # return negative_txt_embeds
    return candidate_txt_embeds


def prepare_training(attention_layer, current_edge_embeds, neighbor_edge_embeds, all_relation_embeds, rank, verbose=False):
    """
    - tgt::math: `(T, E)`, for unbatched input,: math: `(T, N, E)` if `batch_first = False ` by default or `(N, T, E)` if `batch_first = True.
    - tgt_mask: `(T, T)`
    - memory_mask: `(T, S)`.
    - tgt_key_padding_mask: `(T)`, for unbatched input otherwise: `(N, T)`.
    - memory_key_padding_mask: `(S)`, for unbatched input otherwise: `(N, S)`.
    """
    # ---------------------------------------------------------------------------- #
    query = torch.stack(current_edge_embeds).permute(1, 0, 2)  # size [seq_len, batch_size, hidden_dim]
    memory = [torch.stack(embeds).squeeze(dim=1) for embeds in neighbor_edge_embeds]
    init_pred = torch.stack(all_relation_embeds).permute(1, 0, 2)
    if verbose:
        print('query', query.shape, 'init_pred', init_pred.shape, 'memory', len(memory), memory[0].shape)

    # generate memory_key_padding_mask to handle paddings in the memories
    seq_len = [len(embed) for embed in neighbor_edge_embeds]
    memory_key_padding_mask = torch.zeros(len(seq_len), max(seq_len), dtype=torch.bool).to(rank)
    for i, length in enumerate(seq_len):
        memory_key_padding_mask[i, length:] = 1  # a True value indicates that the corresponding key value will be ignored

    # pad all memories in the batch
    memory = pad_sequence(memory, batch_first=False, padding_value=0.0)  # size [seq_len, batch_size, hidden_dim]
    if verbose:
        print('memory', memory.shape)

    predicted_txt_embeds = attention_layer(query.to(rank), memory.to(rank), init_pred.to(rank), memory_key_padding_mask=memory_key_padding_mask)
    if verbose:
        print('predicted_txt_embeds', predicted_txt_embeds.shape)

    return predicted_txt_embeds.squeeze(dim=0)


def batch_training(clip_model, tokenizer, attention_layer,
                   optimizer, scheduler, criterion, contrast_loss, running_loss_cos, running_loss_con, running_loss_counter,
                   graphs, target_triplets, data_len, batch_count, rank, args, writer=None, training=True, verbose=False):

    all_current_edges = []
    all_neighbor_edges = []
    all_current_edge_embeds = []
    all_neighbor_edge_embeds = []
    all_relation_embeds = []
    all_target_txt_embeds = []
    all_negative_target_txt_embeds = []
    all_targets = []
    target_mask = []    # mask of predicate that has matched with a target
    edges_per_image = []

    for graph_idx, (graph, targets) in enumerate(zip(graphs, target_triplets)):
        edges_per_image.append(len(graph.edges))

        for edge_idx, current_edge in enumerate(graph.edges):
            subject_node, object_node = current_edge[0], current_edge[2]
            current_subject, _, current_object = extract_words_from_edge(current_edge[-1], all_labels)

            if training:
                # find corresponding target edge for the current edge
                current_target = None
                for target in targets:
                    target_subject_bbox = target[0]
                    target_object_bbox = target[2]

                    # when the current prediction is about the current target
                    if args['training']['eval_mode'] == 'pc':
                        condition = target_subject_bbox == subject_node and target_object_bbox == object_node
                    else:
                        condition = iou(target_subject_bbox, subject_node) >= 0.5 and iou(target_object_bbox, object_node) >= 0.5
                    if condition:
                        target_subject, target_relation, target_object = extract_words_from_edge(target[-1], all_labels)
                        if target_subject == current_subject and target_object == current_object:
                            phrase = [f"{target_relation}"]
                            inputs = tokenizer(phrase, padding=True, return_tensors="pt").to(rank)
                            with torch.no_grad():
                                target_txt_embed = clip_model.module.get_text_features(**inputs)
                                target_txt_embed = F.normalize(target_txt_embed, dim=1, p=2)

                            all_target_txt_embeds.append(target_txt_embed)
                            target_mask.append(edge_idx)
                            current_target = target
                            all_targets.append(target)
                            # all_negative_target_txt_embeds.append(find_negative_targets(current_subject, current_object, target_relation, target_txt_embed,
                            #                                                             clip_model, tokenizer, rank, args))
                            break

            # find neighbor edges for the current edge
            subject_neighbor_edges = graph.adj_list[subject_node][:15]  # maximum 15 neighbors each edge for more efficient training
            object_neighbor_edges = graph.adj_list[object_node][:15]
            if not training or current_target is None:
                if current_edge in subject_neighbor_edges:
                    subject_neighbor_edges.remove(current_edge)  # do not include the current edge redundantly
                if current_edge in object_neighbor_edges:
                    object_neighbor_edges.remove(current_edge)
            else:
                if current_target in subject_neighbor_edges:
                    subject_neighbor_edges.remove(current_target)   # current target should not be used to train the current edge
                if current_target in object_neighbor_edges:
                    object_neighbor_edges.remove(current_target)

            neighbor_edges = [current_edge] + subject_neighbor_edges + object_neighbor_edges
            neighbor_edge_embeds = [graphs[graph_idx].edge_embeddings[edge] for edge in neighbor_edges]

            all_current_edges.append(current_edge)
            all_current_edge_embeds.append(graphs[graph_idx].edge_embeddings[current_edge])
            all_relation_embeds.append(graphs[graph_idx].rel_embeddings[current_edge])
            all_neighbor_edges.append(neighbor_edges)
            all_neighbor_edge_embeds.append(neighbor_edge_embeds)

    # forward pass
    if len(all_targets) > 0 or (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == data_len):
        predicted_txt_embeds = prepare_training(attention_layer, all_current_edge_embeds, all_neighbor_edge_embeds, all_relation_embeds, rank, verbose=verbose)

    target_mask = torch.tensor(target_mask).to(rank)
    if training and len(all_targets) > 0:
        all_target_txt_embeds = torch.stack(all_target_txt_embeds).squeeze(dim=1)
        # print('all_target_txt_embeds', torch.min(all_target_txt_embeds), torch.mean(all_target_txt_embeds), torch.max(all_target_txt_embeds))
        # print('predicted_txt_embeds', torch.min(predicted_txt_embeds[target_mask]), torch.mean(predicted_txt_embeds[target_mask]), torch.max(predicted_txt_embeds[target_mask]))
        # all_negative_target_txt_embeds = torch.stack(all_negative_target_txt_embeds)  # size [batch_size, num_rel-1, hidden_embed]
        # print('all_negative_target_txt_embeds', all_negative_target_txt_embeds.shape)

        input1_positive = F.normalize(predicted_txt_embeds[target_mask], dim=1, p=2)
        input2_positive = all_target_txt_embeds
        labels_positive = torch.ones(len(all_targets)).to(rank)
        # print('input1_positive', input1_positive.shape, 'input2_positive', input2_positive.shape)
        # print('before', input1_positive[:, 0], all_negative_target_txt_embeds[0, :, 0])

        # input1_negative = predicted_txt_embeds[target_mask].repeat_interleave(args['models']['num_negatives'], dim=0)
        # input2_negative = all_negative_target_txt_embeds.reshape(-1, input1_negative.shape[1])
        # # print('input1_negative', input1_negative.shape, 'input2_negative', input2_negative.shape)
        # # print('after', input1_negative[:, 0], all_negative_target_txt_embeds[:49, 0])
        # labels_negative = torch.full((len(all_targets) * args['models']['num_negatives'],), -1).to(rank)
        #
        # input1 = torch.cat([input1_positive, input1_negative], dim=0)
        # input2 = torch.cat([input2_positive, input2_negative], dim=0)
        # labels = torch.cat([labels_positive, labels_negative], dim=0)

        optimizer.zero_grad()

        cos_loss = criterion(input1_positive, input2_positive, labels_positive)
        # cos_loss = criterion(input1_positive, input2_positive)
        # con_loss = 0.1 * contrast_loss(input1_positive, [tar[1] for tar in all_targets], rank)
        con_loss = contrast_loss(input1_positive, [tar[1] for tar in all_targets], rank)
        loss = cos_loss + con_loss

        running_loss_cos += cos_loss.item()
        running_loss_con += con_loss.item()
        running_loss_counter += 1
        loss.backward()

        optimizer.step()

        if writer is not None:  # rank == 0
            global_step = batch_count
            writer.add_scalar('train/running_loss_cos', running_loss_cos, global_step)
            writer.add_scalar('train/running_loss_con', running_loss_con, global_step)
            writer.add_scalar('train/running_loss', running_loss_con + running_loss_cos, global_step)

    scheduler.step()

    if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == data_len):
        # updated_edges, confidences = eval_refined_output(all_possible_embeds, predicted_txt_embeds, all_current_edges, rank, args, verbose=verbose)
        updated_edges, confidences = eval_refined_output(clip_model, tokenizer, predicted_txt_embeds, all_current_edges, rank, args, verbose=verbose)

        # saved_name = 'results/visualization_results/init_predicates_' + str(batch_count) + '.pt'
        # print('saved_name', saved_name)
        # torch.save(graphs[0].edges, saved_name)

        counter = 0
        for idx, graph in enumerate(graphs):
            # old_graph = graph.edges.copy()
            # print('old graphs.edges', graph.edges[:5])

            graph.edges = updated_edges[counter:counter + edges_per_image[idx]]
            graph.confidence = confidences[counter:counter + edges_per_image[idx]]
            counter += edges_per_image[idx]

            # # print('new graphs.edges', graph.edges[:5])
            # if graph.edges != old_graph:
            #     print('!!!!!!!!!!!new graph != old graph')
            #     print('old_graph', old_graph[:5])
            #     print('new_graph', graph.edges[:5])

        # saved_name = 'results/visualization_results/refined_predicates_' + str(batch_count) + '.pt'
        # print('saved_name', saved_name)
        # torch.save(graphs[0].edges, saved_name)


    if training and ((batch_count % args['training']['print_freq'] == 0) or (batch_count + 1 == data_len)):
        print(f'Rank {rank} batch {batch_count}, graphRefineLoss {running_loss_cos / (running_loss_counter + 1e-5)}, {running_loss_con / (running_loss_counter + 1e-5)}, '
              f'lr {scheduler.get_last_lr()[0]}')

    return graphs, attention_layer, running_loss_cos, running_loss_con, running_loss_counter


def process_batch_sgg_results(clip_model, processor, tokenizer, attention_layer,
                              optimizer, scheduler, criterion, contrast_loss, running_loss_cos, running_loss_con, running_loss_counter,
                              rank, batch_count, args, sgg_results, top_k, data_len, writer=None, training=True, multiple_eval_iter=False, prev_graphs=None, verbose=False):
    if not multiple_eval_iter:
        top_k_predictions = sgg_results['top_k_predictions']
        if verbose:
            print('top_k_predictions', top_k_predictions[0])
        top_k_image_graphs = sgg_results['top_k_image_graphs']
    images = sgg_results['images']
    target_triplets = sgg_results['target_triplets']
    Recall = sgg_results['Recall']

    if not multiple_eval_iter:
        graphs = []
        for batch_idx, (curr_strings, curr_triplet, curr_image) in enumerate(zip(top_k_predictions, top_k_image_graphs, images)):
            graph = ImageGraph(args, rank, tokenizer, processor, clip_model, targets=target_triplets[batch_idx])

            for string, triplet in zip(curr_strings, curr_triplet):
                subject_bbox, relation_id, object_bbox = triplet[0], triplet[1], triplet[2]
                graph.add_edge(subject_bbox, object_bbox, relation_id, string, curr_image, training=training, verbose=verbose)
            graphs.append(graph)
    else:
        graphs = prev_graphs

    updated_graphs, attention_layer, running_loss_cos, running_loss_con, running_loss_counter = \
            batch_training(clip_model, tokenizer, attention_layer,
                           optimizer, scheduler, criterion, contrast_loss, running_loss_cos, running_loss_con, running_loss_counter,
                           graphs, target_triplets, data_len, batch_count, rank, args, writer=writer, training=training, verbose=verbose)

    if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == data_len):
        for batch_idx in range(len(images)):
            relation_pred, confidence = extract_updated_edges(updated_graphs[batch_idx], rank)
            Recall.global_refine(relation_pred, confidence, batch_idx, top_k, rank)

    if training:
        return attention_layer, running_loss_cos, running_loss_con, running_loss_counter
    else:
        return updated_graphs


def query_clip(gpu, args, train_dataset, test_dataset):
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

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=test_sampler)
    print("Finished loading the datasets...")

    # initialize CLIP
    clip_model = DDP(CLIPModel.from_pretrained("openai/clip-vit-base-patch32")).to(rank)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    attention_layer = DDP(EdgeAttentionModel(d_model=clip_model.module.text_embed_dim)).to(rank)
    attention_layer.train()

    # all_possible_embeds = prepare_target_txt_embeds(clip_model, tokenizer, rank)

    optimizer = optim.Adam([
        {'params': attention_layer.module.parameters(), 'initial_lr': args['training']['refine_learning_rate']},
    ], lr=args['training']['refine_learning_rate'], weight_decay=args['training']['weight_decay'])
    # criterion = nn.MSELoss()
    criterion = nn.CosineEmbeddingLoss(margin=0.8)
    contrast_loss = SupConLossGraph(clip_model, tokenizer, all_labels_geometric, all_labels_possessive, all_labels_semantic, rank)

    num_training_steps = 1 * len(train_loader)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps)

    if not args['training']['run_mode'] == 'clip_eval':
        # receive current SGG predictions from a baseline model
        if args['training']['eval_mode'] == 'pc':
            sgg_results = eval_pc(rank, args, train_loader, topk_global_refine=args['training']['topk_global_refine'], epochs=1)
        elif args['training']['eval_mode'] == 'sgc':
            sgg_results = eval_sgc(rank, args, train_loader, topk_global_refine=args['training']['topk_global_refine'], epochs=1)
        elif args['training']['eval_mode'] == 'sgd':
            sgg_results = eval_sgd(rank, args, train_loader, topk_global_refine=args['training']['topk_global_refine'], epochs=1)
        else:
            raise NotImplementedError

        # iterate through the generator to receive results
        running_loss_cos = 0.0
        running_loss_con = 0.0
        running_loss_counter = 0

        for batch_sgg_results in sgg_results:
            batch_count = batch_sgg_results['batch_count']

            # process_batch_sgg_results(clip_model, processor, tokenizer, multimodal_transformer_encoder, optimizer, criterion,
            #                           rank, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(train_loader), verbose=args['training']['verbose_global_refine'])

            attention_layer, running_loss_cos, running_loss_con, running_loss_counter = \
                    process_batch_sgg_results(clip_model, processor, tokenizer, attention_layer,
                                              optimizer, scheduler, criterion, contrast_loss, running_loss_cos, running_loss_con, running_loss_counter,
                                              rank, batch_count, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(train_loader),
                                              writer=writer, verbose=args['training']['verbose_global_refine'])

            if batch_count + 1 == num_training_steps:
                if args['models']['hierarchical_pred']:
                    torch.save(attention_layer.state_dict(), args['training']['checkpoint_path'] + 'AttentionLayerHierar' + '_ckpt' + str(batch_count) + '_' + str(rank) + '.pth')
                else:
                    torch.save(attention_layer.state_dict(), args['training']['checkpoint_path'] + 'AttentionLayer' + '_ckpt' + str(batch_count) + '_' + str(rank) + '.pth')
                dist.monitored_barrier(timeout=datetime.timedelta(seconds=3600))

        if args['models']['hierarchical_pred']:
            torch.save(attention_layer.state_dict(), args['training']['checkpoint_path'] + 'AttentionLayerHierar' + '_' + str(rank) + '.pth')
        else:
            torch.save(attention_layer.state_dict(), args['training']['checkpoint_path'] + 'AttentionLayerNoSkip' + '_' + str(rank) + '.pth')
        dist.monitored_barrier(timeout=datetime.timedelta(seconds=3600))

    if rank == 0:
        writer.close()

    # evaluate on test datasettr
    map_location = {'cuda:%d' % rank: 'cuda:%d' % rank}
    if args['models']['hierarchical_pred']:
        attention_layer.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'AttentionLayerHierar' + '_' + str(rank) + '.pth', map_location=map_location))
    else:
        attention_layer.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'AttentionLayerNoSkip' + '_' + str(rank) + '.pth', map_location=map_location))
    attention_layer.eval()

    with torch.no_grad():
        if args['training']['eval_mode'] == 'pc':
            test_sgg_results = eval_pc(rank, args, test_loader, topk_global_refine=args['training']['topk_global_refine'])
        elif args['training']['eval_mode'] == 'sgc':
            test_sgg_results = eval_sgc(rank, args, test_loader, topk_global_refine=args['training']['topk_global_refine'])
        elif args['training']['eval_mode'] == 'sgd':
            test_sgg_results = eval_sgd(rank, args, test_loader, topk_global_refine=args['training']['topk_global_refine'])
        else:
            raise NotImplementedError

        # iterate through the generator to receive results
        for batch_sgg_results in test_sgg_results:
            batch_count = batch_sgg_results['batch_count']

            # use attention_layer trained just now
            for eval_iter in range(1):
                if eval_iter == 0:
                    updated_graphs = process_batch_sgg_results(clip_model, processor, tokenizer, attention_layer,
                                                               optimizer, scheduler, criterion, contrast_loss, 0.0, 0.0, 0,
                                                               rank, batch_count, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(test_loader),
                                                               training=False, multiple_eval_iter=False, verbose=args['training']['verbose_global_refine'])
                else:
                    updated_graphs = process_batch_sgg_results(clip_model, processor, tokenizer, attention_layer,
                                                               optimizer, scheduler, criterion, contrast_loss, 0.0, 0.0, 0,
                                                               rank, batch_count, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(test_loader),
                                                               training=False, multiple_eval_iter=True, prev_graphs=updated_graphs, verbose=args['training']['verbose_global_refine'])

    dist.destroy_process_group()  # clean up




# import torch
# import os
# from PIL import Image
# import torch.multiprocessing as mp
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# import transformers
# from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
# from collections import deque
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# import torch.nn.functional as F
# from tqdm import tqdm
# from torch.nn.utils.rnn import pad_sequence
# from itertools import repeat
# import datetime
# import yaml
# import re
#
# from evaluate import *
# from utils import *
# from dataset_utils import relation_by_super_class_int2str
# from model import *
# from sup_contrast.losses import SupConLossGraph
#
#
# # define some global lists
# rel_id2txt = relation_by_super_class_int2str()
# all_labels = list(relation_by_super_class_int2str().values())
# # load hyperparameters
# with open('config.yaml', 'r') as file:
#     args = yaml.safe_load(file)
# all_labels_geometric = all_labels[:args['models']['num_geometric']]
# all_labels_possessive = all_labels[args['models']['num_geometric']:args['models']['num_geometric'] + args['models']['num_possessive']]
# all_labels_semantic = all_labels[-args['models']['num_semantic']:]
#
#
# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12356'
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)
#
#
# class ImageGraph:
#     def __init__(self, args, targets=None):
#         self.args = args
#         self.targets = targets
#         # node to neighbors mapping
#         self.adj_list = {}
#         # edge to nodes mapping
#         self.edge_node_map = {}
#         # store all nodes and their degree
#         self.nodes = []
#         self.edges = []
#         self.confidence = []    # only record new confidence after refinement
#
#     def add_edge(self, subject_bbox, object_bbox, relation_id, string, training=True, verbose=False):
#         subject_bbox, object_bbox = tuple(subject_bbox), tuple(object_bbox)
#         edge = (subject_bbox, relation_id, object_bbox, string)
#         edge_wo_string = (subject_bbox, relation_id, object_bbox)
#
#         if edge not in self.edges:
#             self.edges.append(edge)
#             self.confidence.append(-1)
#         if subject_bbox not in self.nodes:
#             self.nodes.append(subject_bbox)
#         if object_bbox not in self.nodes:
#             self.nodes.append(object_bbox)
#
#         if verbose:
#             print('subject_bbox', subject_bbox)
#             print('object_bbox', object_bbox)
#             print('edge', edge, '\n')
#
#         # check if the node is already present, otherwise initialize with an empty list
#         if subject_bbox not in self.adj_list:
#             self.adj_list[subject_bbox] = []
#         if object_bbox not in self.adj_list:
#             self.adj_list[object_bbox] = []
#
#         # in training, store ground truth neighbors if a target is matched
#         if training:
#             matched_target_edge = find_matched_target(self.args, edge, self.targets)
#             # if matched_target_edge is not None:
#             self.adj_list[subject_bbox].append(matched_target_edge)
#             self.adj_list[object_bbox].append(matched_target_edge)
#         else:
#             self.adj_list[subject_bbox].append(edge)
#             self.adj_list[object_bbox].append(edge)
#
#         self.edge_node_map[edge_wo_string] = (subject_bbox, object_bbox)
#
#     def get_edge_neighbors(self, edge, hops=1):
#         # find all the edges belonging to the 1-hop neighbor of the current edge
#         curr_pair = self.edge_node_map[edge]
#         subject_node, object_node = curr_pair[0], curr_pair[1]
#
#         # find all edges connecting to the current subject and object node
#         neighbor_edges = self.adj_list[subject_node] + self.adj_list[object_node]
#
#         # remove the current edge from the set
#         for neighbor_edge in neighbor_edges:
#             if neighbor_edge[:-1] == edge:
#                 neighbor_edges.remove(neighbor_edge)
#
#         if hops == 1:
#             return set(neighbor_edges)
#
#         elif hops == 2:
#             # copy all hop1 edges
#             hop2_neighbor_edges = [hop1_edge for hop1_edge in neighbor_edges]
#
#             for hop2_edge in neighbor_edges:
#                 curr_pair = self.edge_node_map[hop2_edge[:-1]]
#                 subject_node, object_node = curr_pair[0], curr_pair[1]
#                 hop2_neighbor_edges += self.adj_list[subject_node] + self.adj_list[object_node]
#                 # don't have to remove curr hop2_edge because it is already in neighbor_edges and a set operation is enough
#
#             # remove the current edge from the set by any chance
#             for hop2_neighbor_edge in hop2_neighbor_edges:
#                 if hop2_neighbor_edge[:-1] == edge:
#                     hop2_neighbor_edges.remove(hop2_neighbor_edge)
#
#             return set(hop2_neighbor_edges)
#
#         else:
#             assert hops == 1 or hops == 2, "Not implemented"
#
#     def get_node_degrees(self):
#         degrees = {node: len(self.adj_list[node]) for node in self.adj_list}
#         return degrees
#
#
# def colored_text(text, color_code):
#     return f"\033[{color_code}m{text}\033[0m"
#
#
# def save_png(image, save_name="image.png"):
#     image = image.mul(255).cpu().byte().numpy()  # convert to 8-bit integer values
#     image = Image.fromarray(image.transpose(1, 2, 0))  # transpose dimensions for RGB order
#     image.save(save_name)
#
#
# def print_layers_in_optimizer(optimizer, attention_layer, relationship_refiner):
#     # Collect all parameters into a list
#     all_params = []
#     for param_group in optimizer.param_groups:
#         params = param_group['params']
#         all_params.extend(params)
#
#     # Create a dictionary to map parameters to layer names
#     param_to_layer = {}
#     for name, param in attention_layer.module.named_parameters():
#         param_to_layer[param] = f'attention_layer.{name}'
#     for name, param in relationship_refiner.module.named_parameters():
#         param_to_layer[param] = f'relationship_refiner.{name}'
#
#     # Extract and print the parameters along with layer names
#     print("Parameters to be backpropagated:")
#     for param in all_params:
#         layer_name = param_to_layer[param]
#         print(f"Layer: {layer_name}, Size: {param.size()}, Requires Grad: {param.requires_grad}")
#
#
# def extract_updated_edges(graph, rank):
#     # initialize a torch tensor for updated relations
#     relation_pred = torch.tensor([graph.edges[i][1] for i in range(len(graph.edges))]).to(rank)
#     confidence = torch.tensor([graph.confidence[i] for i in range(len(graph.confidence))]).to(rank)
#     return relation_pred, confidence
#
#
# def extract_words_from_edge(phrase, all_relation_labels):
#     # create a regular expression pattern to match the relations
#     pattern = r'\b(' + '|'.join(map(re.escape, all_relation_labels)) + r')\b'
#     phrase = re.split(pattern, phrase)
#     subject, relation, object = phrase[0], phrase[1], phrase[2]
#     return subject, relation, object
#
#
# def crop_image(image, edge, args, crop=True):
#     # crop out the subject and object from the image
#     width, height = image.shape[1], image.shape[2]
#     subject_bbox = torch.tensor(edge[0]) / args['models']['feature_size']
#     object_bbox = torch.tensor(edge[2]) / args['models']['feature_size']
#     subject_bbox[:2] *= height
#     subject_bbox[2:] *= width
#     object_bbox[:2] *= height
#     object_bbox[2:] *= width
#     # print('image', image.shape, 'subject_bbox', subject_bbox, 'object_bbox', object_bbox)
#
#     # create the union bounding box
#     union_bbox = torch.zeros(image.shape[1:], dtype=torch.bool)
#     union_bbox[int(subject_bbox[2]):int(subject_bbox[3]), int(subject_bbox[0]):int(subject_bbox[1])] = 1
#     union_bbox[int(object_bbox[2]):int(object_bbox[3]), int(object_bbox[0]):int(object_bbox[1])] = 1
#
#     if crop:
#         # find the minimum rectangular bounding box around the union bounding box
#         nonzero_indices = torch.nonzero(union_bbox)
#         min_row = nonzero_indices[:, 0].min()
#         max_row = nonzero_indices[:, 0].max()
#         min_col = nonzero_indices[:, 1].min()
#         max_col = nonzero_indices[:, 1].max()
#
#         # crop the image using the minimum rectangular bounding box
#         cropped_image = image[:, min_row:max_row + 1, min_col:max_col + 1]
#
#         # print('Cropped Image:', cropped_image.shape)
#         return cropped_image
#     else:
#         return image * union_bbox
#
#
# def find_matched_target(args, edge, targets):
#     subject_bbox, object_bbox = edge[0], edge[2]
#     current_subject, _, current_object = extract_words_from_edge(edge[-1], all_labels)
#
#     for target in targets:
#         target_subject_bbox = target[0]
#         target_object_bbox = target[2]
#
#         if args['training']['eval_mode'] == 'pc':
#             condition = target_subject_bbox == subject_bbox and target_object_bbox == object_bbox
#         else:
#             condition = iou(target_subject_bbox, subject_bbox) >= 0.5 and iou(target_object_bbox, object_bbox) >= 0.5
#
#         if condition:
#             target_subject, _, target_object = extract_words_from_edge(target[-1], all_labels)
#
#             if target_subject == current_subject and target_object == current_object:
#                 return target
#
#     # return None
#     return edge  # return the original edge if no target matched
#
#
# def clip_zero_shot(clip_model, processor, image, edge, rank, args, based_on_hierar=True):
#     # prepare text labels from the relation dictionary
#     labels_geometric = all_labels[:args['models']['num_geometric']]
#     labels_possessive = all_labels[args['models']['num_geometric']:args['models']['num_geometric']+args['models']['num_possessive']]
#     labels_semantic = all_labels[-args['models']['num_semantic']:]
#
#     # extract current subject and object from the edge
#     phrase = edge[-1]
#     subject, relation, object = extract_words_from_edge(phrase, all_labels)
#
#     if based_on_hierar:
#         # assume the relation super-category has a high accuracy
#         if relation in labels_geometric:
#             queries = [f"a photo of a {subject} {label} {object}" for label in labels_geometric]
#         elif relation in labels_possessive:
#             queries = [f"a photo of a {subject} {label} {object}" for label in labels_possessive]
#         else:
#             queries = [f"a photo of a {subject} {label} {object}" for label in labels_semantic]
#     else:
#         queries = [f"a photo of a {subject} {label} {object}" for label in all_labels]
#
#     # crop out the subject and object from the image
#     cropped_image = crop_image(image, edge, args)
#     save_png(cropped_image, "cropped_image.png")
#
#     # inference CLIP
#     inputs = processor(text=queries, images=image, return_tensors="pt", padding=True).to(rank)
#     outputs = clip_model(**inputs)
#     logits_per_image = outputs.logits_per_image  # image-text similarity score
#     probs = logits_per_image.softmax(dim=1)  # label probabilities
#
#     # get top predicted label
#     top_label_idx = probs.argmax().item()
#     top_label_str = relation_by_super_class_int2str()[top_label_idx]
#
#     # show the results
#     light_blue_code = 94
#     light_pink_code = 95
#     text_blue_colored = colored_text(top_label_str, light_blue_code)
#     text_pink_colored = colored_text(relation, light_pink_code)
#     print(f"Top predicted label from zero-shot CLIP: {text_blue_colored} (probability: {probs[0, top_label_idx]:.4f}), Target label: {text_pink_colored}\n")
#
#
# def prepare_training(clip_model, attention_layer, relationship_refiner, tokenizer, processor, images,
#                      current_edges, neighbor_edges, edges_per_image, rank, verbose=False):
#     # ---------------------------------------------------------------------------- #
#     # collect global image embeddings
#     inputs = processor(images=images, return_tensors="pt").to(rank)
#     with torch.no_grad():
#         image_embeds = clip_model.module.get_image_features(**inputs)
#     if verbose:
#         print('edges_per_image', edges_per_image, 'image_embeds', image_embeds.shape)
#     image_embeds_glob = image_embeds.repeat_interleave(torch.as_tensor(edges_per_image).to(rank), dim=0)
#     # image_embeds_glob = torch.cat([image_embeds[i].repeat(num, 1) for i, num in enumerate(edges_per_image)], dim=0)
#
#     # collect subject and object image embeddings
#     images = [img.to(rank) for img, count in zip(images, edges_per_image) for _ in repeat(None, count)]
#     assert len(images) == len(current_edges)
#
#     mask_sub_list = []
#     mask_obj_list = []
#     for img, edge in zip(images, current_edges):
#         mask_sub = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.bool).to(rank)
#         mask_obj = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.bool).to(rank)
#         mask_sub[:, edge[0][2]:edge[0][3], edge[0][0]:edge[0][1]] = 1
#         mask_obj[:, edge[2][2]:edge[2][3], edge[2][0]:edge[2][1]] = 1
#         mask_sub_list.append(mask_sub)
#         mask_obj_list.append(mask_obj)
#
#     images_sub = [img * mask_sub for img, mask_sub in zip(images, mask_sub_list)]
#     images_obj = [img * mask_obj for img, mask_obj in zip(images, mask_obj_list)]
#     del images
#
#     # images_sub = [img[:, edge[0][2]:edge[0][3], edge[0][0]:edge[0][1]] for (img, edge) in zip(images, current_edges)]
#     # images_obj = [img[:, edge[2][2]:edge[2][3], edge[2][0]:edge[2][1]] for (img, edge) in zip(images, current_edges)]
#
#     # collect image embedding
#     inputs = processor(images=images_sub + images_obj, return_tensors="pt").to(rank)
#     with torch.no_grad():
#         image_embeds = clip_model.module.get_image_features(**inputs)
#     image_embeds_sub = image_embeds[:len(images_sub), :]
#     image_embeds_obj = image_embeds[len(images_sub):, :]
#     if verbose:
#         print('edges_per_image', edges_per_image, 'image_embeds_obj', image_embeds_sub.shape, 'image_embeds_obj', image_embeds_obj.shape)
#     # ---------------------------------------------------------------------------- #
#
#     # accumulate all neighbor edges
#     num_neighbor_edges = [len(edge) for edge in neighbor_edges]
#     inputs = tokenizer([f"{edge[-1]}" for curr_neighbor_edges in neighbor_edges for edge in curr_neighbor_edges],
#                        padding=True, return_tensors="pt").to(rank)
#     with torch.no_grad():
#         neighbor_text_embeds = clip_model.module.get_text_features(**inputs)
#     neighbor_text_embeds = torch.split(neighbor_text_embeds, num_neighbor_edges)
#     # print('neighbor_text_embeds', len(neighbor_text_embeds), [embed.shape for embed in neighbor_text_embeds])
#
#     # pad the sequences to make them the same length (max_seq_len)
#     seq_len = [len(embed) for embed in neighbor_text_embeds]
#     key_padding_mask = torch.zeros(len(seq_len), max(seq_len), dtype=torch.bool).to(rank)
#     for i, length in enumerate(seq_len):
#         key_padding_mask[i, length:] = 1  # a True value indicates that the corresponding key value will be ignored
#
#     neighbor_text_embeds = pad_sequence(neighbor_text_embeds, batch_first=False, padding_value=0.0)  # size [seq_len, batch_size, hidden_dim]
#
#     # feed neighbor_text_embeds to a self-attention layer to get learnable weights
#     # current_text_embeds = neighbor_text_embeds[0, :, :]  # [batch_size, hidden_dim]
#     # print('neighbor_text_embeds padded', neighbor_text_embeds.shape)
#     neighbor_text_embeds = attention_layer(neighbor_text_embeds.to(rank).detach(), key_padding_mask=key_padding_mask)
#
#     # fuse all neighbor_text_embeds after the attention layer
#     neighbor_text_embeds = neighbor_text_embeds.permute(1, 0, 2)  # size [batch_size, seq_len, hidden_dim]
#     neighbor_text_embeds = neighbor_text_embeds * key_padding_mask.unsqueeze(-1)
#     # print('neighbor_text_embeds 2', neighbor_text_embeds.shape)
#     neighbor_text_embeds = neighbor_text_embeds.sum(dim=1)
#     # print('neighbor_text_embeds 3', neighbor_text_embeds.shape)
#
#     if verbose:
#         print('neighbor_text_embeds', neighbor_text_embeds.shape)
#     # ---------------------------------------------------------------------------- #
#
#     # extract current subject and object to condition the relation prediction
#     queries_sub = []
#     queries_obj = []
#     curr_rel = []
#     for edge in current_edges:
#         subject, relation, object = extract_words_from_edge(edge[-1], all_labels)
#         queries_sub.append(f"{subject}")
#         queries_obj.append(f"{object}")
#         curr_rel.append(f"{relation}")
#     inputs_sub = tokenizer(queries_sub, padding=True, return_tensors="pt").to(rank)
#     inputs_obj = tokenizer(queries_obj, padding=True, return_tensors="pt").to(rank)
#     inputs_rel = tokenizer(curr_rel, padding=True, return_tensors="pt").to(rank)
#
#     with torch.no_grad():
#         sub_txt_embed = clip_model.module.get_text_features(**inputs_sub)
#         obj_txt_embed = clip_model.module.get_text_features(**inputs_obj)
#         curr_rel_txt_embed = clip_model.module.get_text_features(**inputs_rel)
#
#     if verbose:
#         print('sub_txt_embed', sub_txt_embed.shape)
#         print('obj_txt_embed', obj_txt_embed.shape)
#         print('curr_rel_txt_embed', curr_rel_txt_embed.shape)
#     # ---------------------------------------------------------------------------- #
#
#     # forward to the learnable layers
#     predicted_txt_embed = relationship_refiner(image_embeds_glob.detach(), image_embeds_sub.detach(), image_embeds_obj.detach(), curr_rel_txt_embed.detach(),
#                                                sub_txt_embed.detach(), obj_txt_embed.detach(), neighbor_text_embeds)
#     if verbose:
#         print('predicted_txt_embed', predicted_txt_embed.shape)
#
#     return predicted_txt_embed
#
#
# def prepare_training_multimodal_transformer(clip_model, multimodal_transformer_encoder, tokenizer, processor, image, current_edge,
#                                             neighbor_edges, rank, batch=True, verbose=False):
#     # collect image embedding
#     inputs = processor(images=image, return_tensors="pt").to(rank)
#     with torch.no_grad():
#         image_embed = clip_model.module.get_image_features(**inputs)
#
#     # if batch:   # process all edges in a graph in parallel, but they all belong to the same image
#     #     image_embed = image_embed.repeat(len(current_edge), 1)
#     if verbose:
#         print('image_embed', image_embed.shape)
#
#     # extract current subject and object to condition the relation prediction
#     if batch:
#         queries = []
#         for edge in current_edge:
#             edge = edge[-1]
#             subject, relation, object = extract_words_from_edge(edge, all_labels)
#             queries.append(f"{subject} and {object}")
#         inputs = tokenizer(queries, padding=True, return_tensors="pt").to(rank)
#     else:
#         current_edge = current_edge[-1]
#         subject, relation, object = extract_words_from_edge(current_edge, all_labels)
#         query = [f"{subject} and {object}"]
#         inputs = tokenizer(query, padding=False, return_tensors="pt").to(rank)
#
#     with torch.no_grad():
#         query_txt_embed = clip_model.module.get_text_features(**inputs)
#     if verbose:
#         print('query_txt_embed', query_txt_embed.shape)
#
#     # accumulate all neighbor edges
#     neighbor_text_embeds = []
#     if verbose:
#         print('current neighbor edges', neighbor_edges)
#     # TODO use ground-truth neighbor edges when possible in training
#
#     # collect all neighbors of the current edge
#     if batch:
#         for curr_neighbor_edges in neighbor_edges:
#             inputs = tokenizer([f"{edge[-1]}" for edge in curr_neighbor_edges],
#                                padding=True, return_tensors="pt").to(rank)
#             with torch.no_grad():
#                 text_embed = clip_model.module.get_text_features(**inputs)
#                 text_embed = torch.cat((image_embed, query_txt_embed, text_embed), dim=0)
#             neighbor_text_embeds.append(text_embed)
#
#         # pad the sequences to make them the same length (max_seq_len)
#         seq_len = [len(embed) for embed in neighbor_text_embeds]
#         key_padding_mask = torch.zeros(len(seq_len), max(seq_len), dtype=torch.bool).to(rank)
#         for i, length in enumerate(seq_len):
#             key_padding_mask[i, length:] = 1    # a True value indicates that the corresponding key value will be ignored
#
#         neighbor_text_embeds = pad_sequence(neighbor_text_embeds, batch_first=False, padding_value=0.0)  # size [seq_len, batch_size, hidden_dim]
#     else:
#         for edge in neighbor_edges:
#             inputs = tokenizer([f"{edge[-1]}"], padding=False, return_tensors="pt").to(rank)
#             with torch.no_grad():
#                 text_embed = clip_model.module.get_text_features(**inputs)
#                 text_embed = torch.cat((image_embed, query_txt_embed, text_embed), dim=0)
#             neighbor_text_embeds.append(text_embed)
#
#         neighbor_text_embeds = torch.stack(neighbor_text_embeds)    # size [seq_len, batch_size, hidden_dim]
#         key_padding_mask = None
#
#     # feed neighbor_text_embeds to a self-attention layer to get learnable weights
#     neighbor_text_embeds = multimodal_transformer_encoder(neighbor_text_embeds.to(rank).detach(), key_padding_mask=key_padding_mask)
#
#     # fuse all neighbor_text_embeds after the attention layer
#     if batch:
#         neighbor_text_embeds = neighbor_text_embeds.permute(1, 0, 2)  # size [batch_size, seq_len, hidden_dim]
#         neighbor_text_embeds = neighbor_text_embeds * key_padding_mask.unsqueeze(-1)
#         neighbor_text_embeds = neighbor_text_embeds.sum(dim=1)
#     else:
#         neighbor_text_embeds = torch.sum(neighbor_text_embeds, dim=0)
#
#     if verbose:
#         print('neighbor_text_embeds', neighbor_text_embeds.shape)
#
#     return neighbor_text_embeds
#
#
# def prepare_target_txt_embeds(clip_model, tokenizer, rank):
#     # pre-compute common arguments
#     num_geom, num_poss, num_sem = args['models']['num_geometric'], args['models']['num_possessive'], args['models']['num_semantic']
#
#     # extract current subject and object from the edge
#     queries = []
#     if args['models']['hierarchical_pred']:
#         relation_id = current_edge[1]
#         if relation_id < num_geom:
#             queries.extend([f"{label}" for label in all_labels_geometric])
#         elif num_geom <= relation_id < num_geom + num_poss:
#             queries.extend([f"{label}" for label in all_labels_possessive])
#         else:
#             queries.extend([f"{label}" for label in all_labels_semantic])
#     else:
#         queries.extend([f"{label}" for label in all_labels])
#
#     # collect text_embeds for all possible labels
#     inputs = tokenizer(queries, padding=True, return_tensors="pt").to(rank)
#     with torch.no_grad():
#         all_possible_embeds = clip_model.module.get_text_features(**inputs)  # size [num_edges * 50, hidden_embed]
#
#     return all_possible_embeds
#
#
# # def eval_refined_output(all_possible_embeds, predicted_txt_embeds, current_edges, rank, args, verbose=False):
# #     # pre-compute common arguments
# #     num_geom, num_poss, num_sem = args['models']['num_geometric'], args['models']['num_possessive'], args['models']['num_semantic']
# #
# #     predicted_txt_embeds = predicted_txt_embeds.unsqueeze(dim=1)
# #
# #     super_categories = []
# #     for current_edge in current_edges:
# #         relation_id = current_edge[1]
# #         if relation_id < num_geom:
# #             super_categories.append(0)
# #         elif num_geom <= relation_id < num_geom + num_poss:
# #             super_categories.append(1)
# #         else:
# #             super_categories.append(2)
# #     super_categories = torch.tensor(super_categories).to(rank)
# #
# #     if args['models']['hierarchical_pred']:
# #         all_possible_embeds_geom = all_possible_embeds[:num_geom, :]
# #         all_possible_embeds_poss = all_possible_embeds[num_geom:num_geom+num_poss, :]
# #         all_possible_embeds_sem = all_possible_embeds[-num_sem:, :]
# #         max_vals = -torch.ones(len(current_edges), dtype=torch.float32).to(rank)
# #         max_indices = -torch.ones(len(current_edges), dtype=torch.int64).to(rank)
# #
# #         mask_geom = super_categories == 0
# #         mask_poss = super_categories == 1
# #         mask_sem = super_categories == 2
# #
# #         # calculate cosine similarities
# #         CosSim = nn.CosineSimilarity(dim=2)
# #         for label_type, masked_ids, embeds in [("geometric", mask_geom, all_possible_embeds_geom),
# #                                                ("possessive", mask_poss, all_possible_embeds_poss),
# #                                                ("semantic", mask_sem, all_possible_embeds_sem)]:
# #             if embeds is not None:
# #                 cos_sims = CosSim(predicted_txt_embeds[masked_ids], embeds)
# #                 cos_sims = F.softmax(cos_sims, dim=1)
# #                 max_vals_cur, max_indices_cur = cos_sims.max(dim=1)
# #
# #                 max_vals[masked_ids] = max_vals_cur
# #                 max_indices[masked_ids] = max_indices_cur
# #         if verbose:
# #             print('predicted_txt_embeds', predicted_txt_embeds.shape, 'all_possible_embeds_geom', all_possible_embeds_geom.shape,
# #                   'all_possible_embeds_poss', all_possible_embeds_poss.shape, 'all_possible_embeds_sem', all_possible_embeds_sem.shape)
# #     else:
# #         # compute cosine similarity between predicted embedding and all query embeddings
# #         CosSim = nn.CosineSimilarity(dim=2)  # Set dim=2 to compute cosine similarity across the embedding dimension
# #         cos_sims = CosSim(predicted_txt_embeds, all_possible_embeds)
# #         cos_sims = F.softmax(cos_sims, dim=1)
# #         max_vals, max_indices = cos_sims.max(dim=1)
# #
# #         if verbose:
# #             print('predicted_txt_embeds', predicted_txt_embeds.shape, 'all_possible_embeds', all_possible_embeds.shape)
# #
# #     updated_edges = []
# #     for idx, current_edge in enumerate(current_edges):
# #         if args['models']['hierarchical_pred']:
# #             if super_categories[idx] == num_geom:
# #                 predicted_rel = all_labels_geometric[max_indices[idx]]
# #                 predicted_rel_id = max_indices[idx].item()
# #             elif super_categories[idx] == num_poss:
# #                 predicted_rel = all_labels_possessive[max_indices[idx]]
# #                 predicted_rel_id = max_indices[idx].item() + num_geom
# #             else:
# #                 predicted_rel = all_labels_semantic[max_indices[idx]]
# #                 predicted_rel_id = max_indices[idx].item() + num_geom + num_poss
# #         else:
# #             predicted_rel = all_labels[max_indices[idx]]
# #             predicted_rel_id = max_indices[idx].item()
# #
# #         subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels)
# #         updated_phrase = subject + predicted_rel + object
# #         updated_edge = (current_edge[0], predicted_rel_id, current_edge[2], updated_phrase)
# #         updated_edges.append(updated_edge)
# #
# #         # show the results
# #         if verbose:
# #             light_blue_code = 94
# #             light_pink_code = 95
# #             text_blue_colored = colored_text(predicted_rel, light_blue_code)
# #             text_pink_colored = colored_text(relation, light_pink_code)
# #             print(f"Predicted label: '{text_blue_colored}' with confidence {max_vals[idx]}, Old label: '{text_pink_colored}'")
# #
# #             dark_blue_code = 34
# #             text_blue_colored = colored_text(updated_edge, dark_blue_code)
# #             print('Updated_edge', text_blue_colored, '\n')
# #
# #     # print('max_vals', max_vals, 'max_indices', max_indices)
# #     return updated_edges, max_vals
#
# def eval_refined_output(clip_model, tokenizer, predicted_txt_embeds, current_edges, rank, args, verbose=False):
#     # print('predicted_txt_embeds', torch.min(predicted_txt_embeds), torch.max(predicted_txt_embeds), torch.mean(predicted_txt_embeds))
#
#     # pre-compute common arguments
#     num_geom, num_poss, num_sem = args['models']['num_geometric'], args['models']['num_possessive'], args['models']['num_semantic']
#
#     # extract current subject and object from the edge
#     queries = []
#     num_candidate_labels = []
#     for current_edge in current_edges:
#         if args['models']['hierarchical_pred']:
#             relation_id = current_edge[1]
#             if relation_id < num_geom:
#                 # subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels_geometric)
#                 queries.extend([f"{label}" for label in all_labels_geometric])
#                 num_candidate_labels.append(num_geom)
#             elif num_geom <= relation_id < num_geom + num_poss:
#                 # subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels_possessive)
#                 queries.extend([f"{label}" for label in all_labels_possessive])
#                 num_candidate_labels.append(num_poss)
#             else:
#                 # subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels_semantic)
#                 queries.extend([f"{label}" for label in all_labels_semantic])
#                 num_candidate_labels.append(num_sem)
#         else:
#             # subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels)
#             queries.extend([f"{label}" for label in all_labels])
#
#     predicted_txt_embeds = predicted_txt_embeds.unsqueeze(dim=1)
#
#     # collect text_embeds for all possible labels
#     inputs = tokenizer(queries, padding=True, return_tensors="pt").to(rank)
#     with torch.no_grad():
#         all_possible_embeds = clip_model.module.get_text_features(**inputs)     # size [num_edges * 50, hidden_embed]
#     all_possible_embeds = F.normalize(all_possible_embeds, dim=1, p=2)
#
#     if args['models']['hierarchical_pred']:
#         # split out each data sample in the batch
#         all_possible_embeds = torch.split(all_possible_embeds, [num for num in num_candidate_labels])
#
#         num_candidate_labels = torch.tensor(num_candidate_labels).to(rank)
#         max_vals = -torch.ones(len(current_edges), dtype=torch.float32).to(rank)
#         max_indices = -torch.ones(len(current_edges), dtype=torch.int64).to(rank)
#
#         mask_geom = num_candidate_labels == num_geom
#         mask_poss = num_candidate_labels == num_poss
#         mask_sem = num_candidate_labels == num_sem
#
#         # get integer indices for each label type
#         ids_geometric = torch.nonzero(mask_geom).flatten()
#         ids_possessive = torch.nonzero(mask_poss).flatten()
#         ids_semantic = torch.nonzero(mask_sem).flatten()
#         all_possible_embeds_geom = torch.stack([all_possible_embeds[i] for i in ids_geometric]) if torch.sum(ids_geometric) > 0 else None
#         all_possible_embeds_poss = torch.stack([all_possible_embeds[i] for i in ids_possessive]) if torch.sum(ids_possessive) > 0 else None
#         all_possible_embeds_sem = torch.stack([all_possible_embeds[i] for i in ids_semantic]) if torch.sum(ids_semantic) > 0 else None
#
#         # calculate Cosine Similarities
#         CosSim = nn.CosineSimilarity(dim=2)
#         for label_type, masked_ids, embeds in [("geometric", mask_geom, all_possible_embeds_geom),
#                                                ("possessive", mask_poss, all_possible_embeds_poss),
#                                                ("semantic", mask_sem, all_possible_embeds_sem)]:
#             if embeds is not None:
#                 cos_sims = CosSim(predicted_txt_embeds[masked_ids], embeds)
#                 cos_sims = F.softmax(cos_sims, dim=1)
#                 max_vals_cur, max_indices_cur = cos_sims.max(dim=1)
#
#                 max_vals[masked_ids] = max_vals_cur
#                 max_indices[masked_ids] = max_indices_cur
#
#         if verbose:
#             print('predicted_txt_embeds', predicted_txt_embeds.shape, 'all_possible_embeds_geom', all_possible_embeds_geom.shape,
#                   'all_possible_embeds_poss', all_possible_embeds_poss.shape, 'all_possible_embeds_sem', all_possible_embeds_sem.shape)
#     else:
#         all_possible_embeds = torch.split(all_possible_embeds, [len(all_labels) for _ in range(len(current_edges))])    # size [[50, hidden_embed] * num_edges]
#         all_possible_embeds = torch.stack(all_possible_embeds)    # size [num_edges, 50, hidden_embed]
#
#         # compute cosine similarity between predicted embedding and all query embeddings
#         CosSim = nn.CosineSimilarity(dim=2)  # Set dim=2 to compute cosine similarity across the embedding dimension
#         cos_sims = CosSim(predicted_txt_embeds, all_possible_embeds)
#         cos_sims = F.softmax(cos_sims, dim=1)
#         max_vals, max_indices = cos_sims.max(dim=1)
#
#         if verbose:
#             print('predicted_txt_embeds', predicted_txt_embeds.shape, 'all_possible_embeds', all_possible_embeds.shape)
#
#     updated_edges = []
#     for idx, current_edge in enumerate(current_edges):
#         if args['models']['hierarchical_pred']:
#             if num_candidate_labels[idx] == num_geom:
#                 predicted_rel = all_labels_geometric[max_indices[idx]]
#                 predicted_rel_id = max_indices[idx].item()
#             elif num_candidate_labels[idx] == num_poss:
#                 predicted_rel = all_labels_possessive[max_indices[idx]]
#                 predicted_rel_id = max_indices[idx].item() + num_geom
#             else:
#                 predicted_rel = all_labels_semantic[max_indices[idx]]
#                 predicted_rel_id = max_indices[idx].item() + num_geom + num_poss
#         else:
#             predicted_rel = all_labels[max_indices[idx]]
#             predicted_rel_id = max_indices[idx].item()
#
#         subject, relation, object = extract_words_from_edge(current_edge[-1], all_labels)
#         updated_phrase = subject + predicted_rel + object
#         updated_edge = (current_edge[0], predicted_rel_id, current_edge[2], updated_phrase)
#         updated_edges.append(updated_edge)
#
#         # show the results
#         if verbose:
#             light_blue_code = 94
#             light_pink_code = 95
#             text_blue_colored = colored_text(predicted_rel, light_blue_code)
#             text_pink_colored = colored_text(relation, light_pink_code)
#             print(f"Predicted label: '{text_blue_colored}' with confidence {max_vals[idx]}, Old label: '{text_pink_colored}'")
#
#             dark_blue_code = 34
#             text_blue_colored = colored_text(updated_edge, dark_blue_code)
#             print('Updated_edge', text_blue_colored, '\n')
#
#     # print('max_vals', max_vals, 'max_indices', max_indices)
#     return updated_edges, max_vals
#
#
# def bfs_explore(clip_model, processor, tokenizer, attention_layer, relationship_refiner,
#                 image, graph, target_triplets, batch_count, data_len, rank, args, verbose=False):
#
#     # get the node with the highest degree
#     node_degrees = graph.get_node_degrees()
#     if verbose:
#         print('node_degrees', node_degrees, '\n')
#     start_node = max(node_degrees, key=node_degrees.get)
#
#     # initialize queue and visited set for BFS
#     queue = deque([(start_node, 0)])  # the second element in the tuple is used to keep track of levels
#     visited_nodes = set()
#     visited_edges = set()
#
#     while True:
#         while queue:
#             # dequeue the next node to visit
#             current_node, level = queue.popleft()
#
#             # if the node hasn't been visited yet
#             if current_node not in visited_nodes:
#                 if verbose:
#                     deep_green_code = 32
#                     text_green_colored = colored_text(current_node, deep_green_code)
#                     print(f"Visiting node: {text_green_colored} at level {level}")
#
#                 # mark the node as visited
#                 visited_nodes.add(current_node)
#
#                 # get all the neighboring edges for the current node
#                 neighbor_edges = graph.adj_list[current_node]
#
#                 # create a mapping from neighbor_node to neighbor_edge
#                 neighbor_to_edge_map = {edge[2] if edge[2] != current_node else edge[0]: edge for edge in neighbor_edges}
#
#                 # extract neighbor nodes and sort them by their degree
#                 neighbor_nodes = [edge[2] if edge[2] != current_node else edge[0] for edge in neighbor_edges]  # the neighbor node could be either the subject or the object
#                 neighbor_nodes = sorted(neighbor_nodes, key=lambda x: node_degrees.get(x, 0), reverse=True)
#
#                 # add neighbors to the queue for future exploration
#                 for neighbor_node in neighbor_nodes:
#                     current_edge = neighbor_to_edge_map[neighbor_node]
#                     if current_edge not in visited_edges:
#                         if verbose:
#                             light_green_code = 92
#                             text_green_colored = colored_text(current_edge, light_green_code)
#                             print(f"Visiting edge: {text_green_colored}")
#
#                         # mark the edge as visited
#                         visited_edges.add(current_edge)
#
#                         if args['training']['run_mode'] == 'clip_zs':
#                             # query CLIP on the current neighbor edge in zero shot
#                             clip_zero_shot(clip_model, processor, image, current_edge, rank, args)
#                         else:
#                             subject_neighbor_edges = [edge for edge in neighbor_edges if edge != current_edge]
#                             object_neighbor_edges = [edge for edge in graph.adj_list[neighbor_node] if edge != current_edge]
#                             neighbor_edges = [current_edge] + subject_neighbor_edges + object_neighbor_edges
#
#                             predicted_txt_embed = prepare_training(clip_model, attention_layer, relationship_refiner, tokenizer, processor, image, current_edge,
#                                                                    neighbor_edges, rank, batch=False, verbose=verbose)
#                             updated_edge, confidence = eval_refined_output(clip_model, tokenizer, predicted_txt_embed, current_edge, rank, args, verbose=verbose)
#
#                             # find the index to update in a more efficient way
#                             index_to_update = next((index for index, stored_edge in enumerate(graph.edges) if stored_edge == current_edge), None)
#                             if index_to_update is not None:
#                                 graph.edges[index_to_update] = updated_edge
#                                 graph.confidence[index_to_update] = confidence
#
#                             # # train the model to predict relations from neighbors and image features
#                             # subject_neighbor_edges = list(neighbor_edges)   # use the list constructor to create a new list with the elements of the original list
#                             # object_neighbor_edges = list(graph.adj_list[neighbor_node])
#                             # subject_neighbor_edges.remove(current_edge)    # do not include the current edge redundantly
#                             # object_neighbor_edges.remove(current_edge)
#                             #
#                             # neighbor_edges = [current_edge] + subject_neighbor_edges + object_neighbor_edges
#                             #
#                             # # forward pass
#                             # predicted_txt_embed = prepare_training(clip_model, attention_layer, relationship_refiner, tokenizer, processor, image, current_edge,
#                             #                                        neighbor_edges, rank, batch=False, verbose=verbose)
#                             #
#                             # updated_edge, confidence = eval_refined_output(clip_model, tokenizer, predicted_txt_embed, current_edge, rank, verbose=verbose)
#                             #
#                             # # # prepare learning target
#                             # # curr_subject_bbox = current_edge[0]
#                             # # curr_object_bbox = current_edge[2]
#                             # # for target in target_triplets:
#                             # #     target_subject_bbox = target[0]
#                             # #     target_object_bbox = target[2]
#                             # #
#                             # #     # when the current prediction is about the current target
#                             # #     if args['training']['eval_mode'] == 'pc':
#                             # #         condition = target_subject_bbox == curr_subject_bbox and target_object_bbox == curr_object_bbox
#                             # #     else:
#                             # #         condition = iou(target_subject_bbox, curr_subject_bbox) >= 0.5 and iou(target_object_bbox, curr_object_bbox) >= 0.5
#                             # #     if condition:
#                             # #         target_subject, target_relation, target_object = extract_words_from_edge(target[-1].split(), all_labels)
#                             # #         target = [f"a photo of a {target_subject} {target_relation} {target_object}"]
#                             # #         inputs = tokenizer(target, padding=False, return_tensors="pt").to(rank)
#                             # #         with torch.no_grad():
#                             # #             target_txt_embed = clip_model.module.get_text_features(**inputs)
#                             # #
#                             # #         # # back-propagate
#                             # #         # grad_acc_counter += 1
#                             # #         # loss = criterion(predicted_txt_embed, target_txt_embed)
#                             # #         # loss.backward()
#                             # #         #
#                             # #         # if grad_acc_counter % accumulation_steps == 0:  # Only step optimizer every `accumulation_steps`
#                             # #         #     optimizer.step()
#                             # #         #     optimizer.zero_grad()  # Ensure we clear gradients after an update
#                             # #
#                             # #         # break the target matching loop
#                             # #         break
#                             #
#                             # # update self.edges in the graph using predicted_txt_embed
#                             # index_to_update = None
#                             # for index, stored_edge in enumerate(graph.edges):
#                             #     if stored_edge == current_edge:
#                             #         index_to_update = index
#                             #         break
#                             # if index_to_update is not None:
#                             #     graph.edges[index_to_update] = updated_edge
#                             #     graph.confidence[index_to_update] = confidence
#
#                         queue.append((neighbor_node, level + 1))
#
#         if verbose:
#             print("Finished BFS for current connected component.\n")
#
#         # check if there are any unvisited nodes
#         unvisited_nodes = set(node_degrees.keys()) - visited_nodes
#         if not unvisited_nodes:
#             break  # all nodes have been visited, exit the loop
#
#         # start a new BFS from the unvisited node with the highest degree
#         new_start_node = max(unvisited_nodes, key=lambda x: node_degrees.get(x, 0))
#         if verbose:
#             print(f"Starting new BFS from node: {new_start_node}")
#         queue.append((new_start_node, 0))
#
#     return graph
#
#
# def find_negative_targets(current_subject, current_object, target_label, target_txt_embed, clip_model, tokenizer, rank, args):
#     # find the label with the highest similarity to the current positive target
#     queries = [f"a photo of a {current_subject} {label} {current_object}" for label in all_labels if label != target_label]
#
#     # Collect text embeddings for all possible labels
#     inputs = tokenizer(queries, padding=True, return_tensors="pt").to(rank)
#     with torch.no_grad():
#         candidate_txt_embeds = clip_model.module.get_text_features(**inputs)
#
#     # # select top negative_txt_embeds with closest cosine distances but not the target
#     # CosSim = nn.CosineSimilarity(dim=1)
#     # cos_sim = CosSim(target_txt_embed, candidate_txt_embeds)
#     # top_similarities, top_indices = cos_sim.topk(args['models']['num_negatives'], largest=False)
#     # # print('top_similarities', top_similarities)
#     #
#     # negative_txt_embeds = candidate_txt_embeds[top_indices]
#
#     # return negative_txt_embeds
#     return candidate_txt_embeds
#
#
# def batch_training(clip_model, processor, tokenizer, attention_layer, relationship_refiner, all_possible_embeds,
#                    optimizer, scheduler, criterion, contrast_loss, running_loss_cos, running_loss_con, running_loss_counter,
#                    images, graphs, target_triplets, data_len, batch_count, rank, args, training=True, verbose=False):
#
#     all_current_edges = []
#     all_neighbor_edges = []
#     all_target_txt_embeds = []
#     all_negative_target_txt_embeds = []
#     all_targets = []
#     target_mask = []    # mask of predicate that has matched with a target
#     edges_per_image = []
#
#     for graph_idx, (graph, targets) in enumerate(zip(graphs, target_triplets)):
#         edges_per_image.append(len(graph.edges))
#
#         for edge_idx, current_edge in enumerate(graph.edges):
#             subject_node, object_node = current_edge[0], current_edge[2]
#             current_subject, _, current_object = extract_words_from_edge(current_edge[-1], all_labels)
#
#             if training:
#                 # find corresponding target edge for the current edge
#                 current_target = None
#                 for target in targets:
#                     target_subject_bbox = target[0]
#                     target_object_bbox = target[2]
#
#                     # when the current prediction is about the current target
#                     if args['training']['eval_mode'] == 'pc':
#                         condition = target_subject_bbox == subject_node and target_object_bbox == object_node
#                     else:
#                         condition = iou(target_subject_bbox, subject_node) >= 0.5 and iou(target_object_bbox, object_node) >= 0.5
#                     if condition:
#                         target_subject, target_relation, target_object = extract_words_from_edge(target[-1], all_labels)
#                         if target_subject == current_subject and target_object == current_object:
#                             phrase = [f"{target_relation}"]
#                             inputs = tokenizer(phrase, padding=True, return_tensors="pt").to(rank)
#                             with torch.no_grad():
#                                 target_txt_embed = clip_model.module.get_text_features(**inputs)
#                                 target_txt_embed = F.normalize(target_txt_embed, dim=1, p=2)
#
#                             all_target_txt_embeds.append(target_txt_embed)
#                             target_mask.append(edge_idx)
#                             current_target = target
#                             all_targets.append(target)
#                             # all_negative_target_txt_embeds.append(find_negative_targets(current_subject, current_object, target_relation, target_txt_embed,
#                             #                                                             clip_model, tokenizer, rank, args))
#                             break
#
#             # find neighbor edges for the current edge
#             subject_neighbor_edges = graph.adj_list[subject_node][:20]  # maximum 20 neighbors each edge for more efficient training
#             object_neighbor_edges = graph.adj_list[object_node][:20]
#             if not training or current_target is None:
#                 if current_edge in subject_neighbor_edges:
#                     subject_neighbor_edges.remove(current_edge)  # do not include the current edge redundantly
#                 if current_edge in object_neighbor_edges:
#                     object_neighbor_edges.remove(current_edge)
#             else:
#                 if current_target in subject_neighbor_edges:
#                     subject_neighbor_edges.remove(current_target)   # current target should not be used to train the current edge
#                 if current_target in object_neighbor_edges:
#                     object_neighbor_edges.remove(current_target)
#
#             neighbor_edges = [current_edge] + subject_neighbor_edges + object_neighbor_edges
#
#             all_current_edges.append(current_edge)
#             all_neighbor_edges.append(neighbor_edges)
#
#     # forward pass
#     if len(all_targets) > 0 or (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == data_len):
#         predicted_txt_embeds = prepare_training(clip_model, attention_layer, relationship_refiner, tokenizer, processor, images,
#                                                 all_current_edges, all_neighbor_edges, edges_per_image, rank, verbose=verbose)
#
#     target_mask = torch.tensor(target_mask).to(rank)
#     if training and len(all_targets) > 0:
#         all_target_txt_embeds = torch.stack(all_target_txt_embeds).squeeze(dim=1)
#         # all_negative_target_txt_embeds = torch.stack(all_negative_target_txt_embeds)  # size [batch_size, num_rel-1, hidden_embed]
#         # print('all_negative_target_txt_embeds', all_negative_target_txt_embeds.shape)
#
#         input1_positive = predicted_txt_embeds[target_mask]
#         input2_positive = all_target_txt_embeds
#         labels_positive = torch.ones(len(all_targets)).to(rank)
#         # print('input1_positive', input1_positive.shape, 'input2_positive', input2_positive.shape)
#         # print('before', input1_positive[:, 0], all_negative_target_txt_embeds[0, :, 0])
#
#         # input1_negative = predicted_txt_embeds[target_mask].repeat_interleave(args['models']['num_negatives'], dim=0)
#         # input2_negative = all_negative_target_txt_embeds.reshape(-1, input1_negative.shape[1])
#         # # print('input1_negative', input1_negative.shape, 'input2_negative', input2_negative.shape)
#         # # print('after', input1_negative[:, 0], all_negative_target_txt_embeds[:49, 0])
#         # labels_negative = torch.full((len(all_targets) * args['models']['num_negatives'],), -1).to(rank)
#         #
#         # input1 = torch.cat([input1_positive, input1_negative], dim=0)
#         # input2 = torch.cat([input2_positive, input2_negative], dim=0)
#         # labels = torch.cat([labels_positive, labels_negative], dim=0)
#
#         optimizer.zero_grad()
#
#         # cos_loss = criterion(input1_positive, input2_positive, labels_positive)
#         # cos_loss = criterion(input1_positive, input2_positive)
#         # con_loss = 0.1 * contrast_loss(input1_positive, [tar[1] for tar in all_targets], rank)
#         cos_loss = 0
#         con_loss = contrast_loss(input1_positive, [tar[1] for tar in all_targets], rank)
#         loss = con_loss
#
#         # running_loss_cos += cos_loss.item()
#         running_loss_con += con_loss.item()
#         running_loss_counter += 1
#         loss.backward()
#
#         optimizer.step()
#     scheduler.step()
#
#     if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == data_len):
#         # updated_edges, confidences = eval_refined_output(all_possible_embeds, predicted_txt_embeds, all_current_edges, rank, args, verbose=verbose)
#         updated_edges, confidences = eval_refined_output(clip_model, tokenizer, predicted_txt_embeds, all_current_edges, rank, args, verbose=verbose)
#
#         # saved_name = 'results/visualization_results/init_predicates_' + str(batch_count) + '.pt'
#         # print('saved_name', saved_name)
#         # torch.save(graphs[0].edges, saved_name)
#
#         counter = 0
#         for idx, graph in enumerate(graphs):
#             # old_graph = graph.edges.copy()
#             # print('old graphs.edges', graph.edges[:5])
#
#             graph.edges = updated_edges[counter:counter + edges_per_image[idx]]
#             graph.confidence = confidences[counter:counter + edges_per_image[idx]]
#             counter += edges_per_image[idx]
#
#             # # print('new graphs.edges', graph.edges[:5])
#             # if graph.edges != old_graph:
#             #     print('!!!!!!!!!!!new graph != old graph')
#             #     print('old_graph', old_graph[:5])
#             #     print('new_graph', graph.edges[:5])
#
#         # saved_name = 'results/visualization_results/refined_predicates_' + str(batch_count) + '.pt'
#         # print('saved_name', saved_name)
#         # torch.save(graphs[0].edges, saved_name)
#
#
#     if training and ((batch_count % args['training']['print_freq'] == 0) or (batch_count + 1 == data_len)):
#         print(f'Rank {rank} batch {batch_count}, graphRefineLoss {running_loss_cos / (running_loss_counter + 1e-5)}, {running_loss_con / (running_loss_counter + 1e-5)}, '
#               f'lr {scheduler.get_last_lr()[0]}')
#
#     return graphs, attention_layer, relationship_refiner, running_loss_cos, running_loss_con, running_loss_counter,
#
#
# def process_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner,
#                         rank, batch_count, args, sgg_results, top_k, data_len, verbose=False):
#     top_k_predictions = sgg_results['top_k_predictions']
#     if verbose:
#         print('top_k_predictions', top_k_predictions[0])
#     top_k_image_graphs = sgg_results['top_k_image_graphs']
#     images = sgg_results['images']
#     target_triplets = sgg_results['target_triplets']
#     Recall = sgg_results['Recall']
#
#     for batch_idx, (curr_strings, curr_image, curr_target_triplet) in enumerate(zip(top_k_predictions, top_k_image_graphs, target_triplets)):
#         graph = ImageGraph(args)
#         print('batch_idx', batch_idx, '/', len(top_k_predictions))
#
#         for string, triplet in zip(curr_strings, curr_image):
#             subject_bbox, relation_id, object_bbox = triplet[0], triplet[1], triplet[2]
#             graph.add_edge(subject_bbox, object_bbox, relation_id, string, training=False, verbose=verbose)
#
#         if verbose:
#             print('batch_idx', batch_idx, '/', len(top_k_predictions))
#             dark_orange_code = 33
#             text_orange_colored = colored_text(curr_target_triplet, dark_orange_code)
#             print(f"curr_target_triplet edge: {text_orange_colored}")
#             save_png(images[batch_idx], "curr_image.png")
#
#         updated_graph = bfs_explore(clip_model, processor, tokenizer, attention_layer, relationship_refiner,
#                                     images[batch_idx], graph, curr_target_triplet, batch_count, data_len, rank, args, verbose=verbose)
#         relation_pred, confidence = extract_updated_edges(updated_graph, rank)
#         Recall.global_refine(relation_pred, confidence, batch_idx, top_k, rank)
#
#
# def process_batch_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner, all_possible_embeds,
#                               optimizer, scheduler, criterion, contrast_loss, running_loss_cos, running_loss_con, running_loss_counter,
#                               rank, batch_count, args, sgg_results, top_k, data_len, training=True, multiple_eval_iter=False, prev_graphs=None, verbose=False):
# # def process_batch_sgg_results(clip_model, processor, tokenizer, multimodal_transformer_encoder, optimizer, criterion,
# #                               rank, args, sgg_results, top_k, data_len, verbose=False):
#     if not multiple_eval_iter:
#         top_k_predictions = sgg_results['top_k_predictions']
#         if verbose:
#             print('top_k_predictions', top_k_predictions[0])
#         top_k_image_graphs = sgg_results['top_k_image_graphs']
#     images = sgg_results['images']
#     target_triplets = sgg_results['target_triplets']
#     Recall = sgg_results['Recall']
#
#     if not multiple_eval_iter:
#         graphs = []
#         for batch_idx, (curr_strings, curr_image) in enumerate(zip(top_k_predictions, top_k_image_graphs)):
#             graph = ImageGraph(args, targets=target_triplets[batch_idx])
#
#             for string, triplet in zip(curr_strings, curr_image):
#                 subject_bbox, relation_id, object_bbox = triplet[0], triplet[1], triplet[2]
#                 graph.add_edge(subject_bbox, object_bbox, relation_id, string, training=training, verbose=verbose)
#
#             graphs.append(graph)
#     else:
#         graphs = prev_graphs
#
#     updated_graphs, attention_layer, relationship_refiner, running_loss_cos, running_loss_con, running_loss_counter = \
#             batch_training(clip_model, processor, tokenizer, attention_layer, relationship_refiner, all_possible_embeds,
#                            optimizer, scheduler, criterion, contrast_loss, running_loss_cos, running_loss_con, running_loss_counter,
#                            images, graphs, target_triplets, data_len, batch_count, rank, args, training=training, verbose=verbose)
#     # updated_graphs, multimodal_transformer_encoder = batch_training(clip_model, processor, tokenizer, multimodal_transformer_encoder, optimizer, criterion,
#     #                                                                        images, graphs, target_triplets, data_len, rank, args, verbose=verbose)
#
#     # print('len(images)', len(images), 'updated_graphs', len(updated_graphs))
#     if (batch_count % args['training']['eval_freq'] == 0) or (batch_count + 1 == data_len):
#         for batch_idx in range(len(images)):
#             relation_pred, confidence = extract_updated_edges(updated_graphs[batch_idx], rank)
#             Recall.global_refine(relation_pred, confidence, batch_idx, top_k, rank)
#
#     if training:
#         return attention_layer, relationship_refiner, running_loss_cos, running_loss_con, running_loss_counter
#     else:
#         return updated_graphs
#
#
# def query_clip(gpu, args, train_dataset, test_dataset):
#     rank = gpu
#     world_size = torch.cuda.device_count()
#     setup(rank, world_size)
#     print('rank', rank, 'torch.distributed.is_initialized', torch.distributed.is_initialized())
#
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=train_sampler)
#     test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=test_sampler)
#     print("Finished loading the datasets...")
#
#     # initialize CLIP
#     clip_model = DDP(CLIPModel.from_pretrained("openai/clip-vit-base-patch32")).to(rank)
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#
#     attention_layer = DDP(SimpleSelfAttention(hidden_dim=clip_model.module.text_embed_dim)).to(rank)
#     attention_layer.train()
#     relationship_refiner = DDP(RelationshipRefiner(hidden_dim=clip_model.module.text_embed_dim)).to(rank)
#     relationship_refiner.train()
#     # attention_layer = DDP(MultimodalTransformerEncoder(hidden_dim=clip_model.module.text_embed_dim)).to(rank)
#     # attention_layer.train()
#
#     all_possible_embeds = prepare_target_txt_embeds(clip_model, tokenizer, rank)
#
#     optimizer = optim.Adam([
#         {'params': attention_layer.module.parameters(), 'initial_lr': args['training']['refine_learning_rate']},
#         {'params': relationship_refiner.module.parameters(), 'initial_lr': args['training']['refine_learning_rate']}
#     ], lr=args['training']['refine_learning_rate'], weight_decay=args['training']['weight_decay'])
#     # optimizer = optim.Adam([
#     #     {'params': multimodal_transformer_encoder.module.parameters(), 'initial_lr': args['training']['refine_learning_rate']},
#     # ], lr=args['training']['refine_learning_rate'], weight_decay=args['training']['weight_decay'])
#     # criterion = nn.MSELoss()
#     criterion = nn.CosineEmbeddingLoss(margin=0.8)
#     contrast_loss = SupConLossGraph(clip_model, tokenizer, all_labels_geometric, all_labels_possessive, all_labels_semantic, rank)
#
#     # optimizer = optim.Adam(attention_layer.module.parameters(), lr=args['training']['refine_learning_rate'], weight_decay=args['training']['weight_decay'])
#     # optimizer_refiner = optim.SGD(relationship_refiner.module.parameters(), lr=args['training']['refine_learning_rate'], weight_decay=args['training']['weight_decay'], momentum=0.9)
#     num_training_steps = 1 * len(train_loader)
#     # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps)
#     scheduler = StepLR(optimizer, step_size=2 * num_training_steps, gamma=0.1)
#
#     # relation_count = get_num_each_class_reordered(args)
#     # class_weight = 1 - relation_count / torch.sum(relation_count)
#     # criterion = torch.nn.CrossEntropyLoss(weight=class_weight.to(rank))
#
#     if not args['training']['run_mode'] == 'clip_eval':
#         # receive current SGG predictions from a baseline model
#         if args['training']['eval_mode'] == 'pc':
#             sgg_results = eval_pc(rank, args, train_loader, topk_global_refine=args['training']['topk_global_refine'], epochs=3)
#         elif args['training']['eval_mode'] == 'sgc':
#             sgg_results = eval_sgc(rank, args, train_loader, topk_global_refine=args['training']['topk_global_refine'], epochs=3)
#         elif args['training']['eval_mode'] == 'sgd':
#             sgg_results = eval_sgd(rank, args, train_loader, topk_global_refine=args['training']['topk_global_refine'], epochs=3)
#         else:
#             raise NotImplementedError
#
#         # iterate through the generator to receive results
#         running_loss_cos = 0.0
#         running_loss_con = 0.0
#         running_loss_counter = 0
#
#         for batch_sgg_results in sgg_results:
#             batch_count = batch_sgg_results['batch_count']
#
#             # process_batch_sgg_results(clip_model, processor, tokenizer, multimodal_transformer_encoder, optimizer, criterion,
#             #                           rank, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(train_loader), verbose=args['training']['verbose_global_refine'])
#
#             attention_layer, relationship_refiner, running_loss_cos, running_loss_con, running_loss_counter = \
#                     process_batch_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner, all_possible_embeds,
#                                               optimizer, scheduler, criterion, contrast_loss, running_loss_cos, running_loss_con, running_loss_counter,
#                                               rank, batch_count, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(train_loader),
#                                               verbose=args['training']['verbose_global_refine'])
#             # process_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner, optimizer, criterion,
#             #                     rank, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(train_loader), verbose=args['training']['verbose_global_refine'])
#
#             if batch_count + 1 == num_training_steps:
#                 if args['models']['hierarchical_pred']:
#                     torch.save(attention_layer.state_dict(), args['training']['checkpoint_path'] + 'AttentionLayerHierarNoSkip' + '_ckpt' + str(batch_count) + '_' + str(rank) + '.pth')
#                     torch.save(relationship_refiner.state_dict(), args['training']['checkpoint_path'] + 'RelationshipRefinerHierarNoSkip' + '_ckpt' + str(batch_count) + '_' + str(rank) + '.pth')
#                 else:
#                     torch.save(attention_layer.state_dict(), args['training']['checkpoint_path'] + 'AttentionLayerNoSkip' + '_ckpt' + str(batch_count) + '_' + str(rank) + '.pth')
#                     torch.save(relationship_refiner.state_dict(), args['training']['checkpoint_path'] + 'RelationshipRefinerNoSkip' + '_ckpt' + str(batch_count) + '_' + str(rank) + '.pth')
#                     # torch.save(multimodal_transformer_encoder.state_dict(), args['training']['checkpoint_path'] + 'MultimodalTransformerEncoder' + '_' + str(rank) + '.pth')
#                 dist.monitored_barrier(timeout=datetime.timedelta(seconds=3600))
#
#         if args['models']['hierarchical_pred']:
#             torch.save(attention_layer.state_dict(), args['training']['checkpoint_path'] + 'AttentionLayerHierarNoSkip' + '_' + str(rank) + '.pth')
#             torch.save(relationship_refiner.state_dict(), args['training']['checkpoint_path'] + 'RelationshipRefinerHierarNoSkip' + '_' + str(rank) + '.pth')
#         else:
#             torch.save(attention_layer.state_dict(), args['training']['checkpoint_path'] + 'AttentionLayerNoSkip' + '_' + str(rank) + '.pth')
#             torch.save(relationship_refiner.state_dict(), args['training']['checkpoint_path'] + 'RelationshipRefinerNoSkip' + '_' + str(rank) + '.pth')
#             # torch.save(multimodal_transformer_encoder.state_dict(), args['training']['checkpoint_path'] + 'MultimodalTransformerEncoder' + '_' + str(rank) + '.pth')
#         dist.monitored_barrier(timeout=datetime.timedelta(seconds=3600))
#
#     # evaluate on test datasettr
#     map_location = {'cuda:%d' % rank: 'cuda:%d' % rank}
#     if args['models']['hierarchical_pred']:
#         attention_layer.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'AttentionLayerHierarNoSkip' + '_' + str(rank) + '.pth', map_location=map_location))
#         relationship_refiner.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'RelationshipRefinerHierarNoSkip' + '_' + str(rank) + '.pth', map_location=map_location))
#     else:
#         attention_layer.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'AttentionLayerNoSkip' + '_' + str(rank) + '.pth', map_location=map_location))
#         relationship_refiner.load_state_dict(torch.load(args['training']['checkpoint_path'] + 'RelationshipRefinerNoSkip' + '_' + str(rank) + '.pth', map_location=map_location))
#     attention_layer.eval()
#     relationship_refiner.eval()
#     # updated_graphs = torch.load(args['training']['checkpoint_path'] + 'updated_graphs_0.pt', map_location=map_location)
#
#     with torch.no_grad():
#         if args['training']['eval_mode'] == 'pc':
#             test_sgg_results = eval_pc(rank, args, test_loader, topk_global_refine=args['training']['topk_global_refine'])
#         elif args['training']['eval_mode'] == 'sgc':
#             test_sgg_results = eval_sgc(rank, args, test_loader, topk_global_refine=args['training']['topk_global_refine'])
#         elif args['training']['eval_mode'] == 'sgd':
#             test_sgg_results = eval_sgd(rank, args, test_loader, topk_global_refine=args['training']['topk_global_refine'])
#         else:
#             raise NotImplementedError
#
#         # iterate through the generator to receive results
#         for batch_sgg_results in test_sgg_results:
#             batch_count = batch_sgg_results['batch_count']
#
#             # use attention_layer and relationship_refiner trained just now
#             for eval_iter in range(1):
#                 # if (batch_count % args['training']['print_freq_test'] == 0) or (batch_count + 1 == data_len):
#                 #     print('eval_iter', eval_iter)
#                 if eval_iter == 0:
#                     updated_graphs = process_batch_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner, all_possible_embeds,
#                                                                optimizer, scheduler, criterion, contrast_loss, 0.0, 0.0, 0,
#                                                                rank, batch_count, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(test_loader),
#                                                                training=False, multiple_eval_iter=False, verbose=args['training']['verbose_global_refine'])
#                     # process_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner,
#                     #                     rank, batch_count, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(test_loader),
#                     #                     verbose=args['training']['verbose_global_refine'])
#                 else:
#                     updated_graphs = process_batch_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner, all_possible_embeds,
#                                                                optimizer, scheduler, criterion, contrast_loss, 0.0, 0.0, 0,
#                                                                rank, batch_count, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(test_loader),
#                                                                training=False, multiple_eval_iter=True, prev_graphs=updated_graphs, verbose=args['training']['verbose_global_refine'])
#
#                 # torch.save(updated_graphs, args['training']['checkpoint_path'] + 'updated_graphs_' + str(eval_iter) + '_' + str(rank) + '.pt')
#
#     dist.destroy_process_group()  # clean up
#
