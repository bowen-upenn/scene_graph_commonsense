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
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from evaluate import inference, eval_pc
from utils import *
from dataset_utils import relation_by_super_class_int2str
from model import SimpleSelfAttention, RelationshipRefiner


all_labels = list(relation_by_super_class_int2str().values())


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


class ImageGraph:
    def __init__(self, targets=None):
        self.targets = targets
        # node to neighbors mapping
        self.adj_list = {}
        # edge to nodes mapping
        self.edge_node_map = {}
        # store all nodes and their degree
        self.nodes = []
        self.edges = []
        self.confidence = []    # only record new confidence after refinement

    def add_edge(self, subject_bbox, object_bbox, relation_id, string, training=True, verbose=False):
        subject_bbox, object_bbox = tuple(subject_bbox), tuple(object_bbox)
        edge = (subject_bbox, relation_id, object_bbox, string)
        edge_wo_string = (subject_bbox, relation_id, object_bbox)

        if edge not in self.edges:
            self.edges.append(edge)
            self.confidence.append(-1)
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
            edge = find_matched_target(edge, self.targets)
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
    # iterate through the phrase to extract the parts
    for i in range(len(phrase)):
        if phrase[i] in all_relation_labels:
            relation = phrase[i]
            subject = " ".join(phrase[:i])
            object = " ".join(phrase[i + 1:])
            break  # exit loop once the relation is found

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


def find_matched_target(edge, targets):
    subject_node, object_node = edge[0], edge[2]
    current_subject, _, current_object = extract_words_from_edge(edge[-1].split(), all_labels)

    for target in targets:
        target_subject_bbox = target[0]
        target_object_bbox = target[1]

        if target_subject_bbox == subject_node and target_object_bbox == object_node:
            target_subject, target_relation, target_object = extract_words_from_edge(target[-1].split(), all_labels)

            if target_subject == current_subject and target_object == current_object:
                return target

    return edge  # return the original edge if no target matched


def clip_zero_shot(clip_model, processor, image, edge, rank, args, based_on_hierar=True):
    # prepare text labels from the relation dictionary
    labels_geometric = all_labels[:args['models']['num_geometric']]
    labels_possessive = all_labels[args['models']['num_geometric']:args['models']['num_geometric']+args['models']['num_possessive']]
    labels_semantic = all_labels[-args['models']['num_semantic']:]

    # extract current subject and object from the edge
    phrase = edge[-1].split()
    subject, relation, object = extract_words_from_edge(phrase, all_labels)

    if based_on_hierar:
        # assume the relation super-category has a high accuracy
        if relation in labels_geometric:
            queries = [f"a photo of a {subject} {label} {object}" for label in labels_geometric]
        elif relation in labels_possessive:
            queries = [f"a photo of a {subject} {label} {object}" for label in labels_possessive]
        else:
            queries = [f"a photo of a {subject} {label} {object}" for label in labels_semantic]
    else:
        queries = [f"a photo of a {subject} {label} {object}" for label in all_labels]

    # crop out the subject and object from the image
    cropped_image = crop_image(image, edge, args)
    save_png(cropped_image, "cropped_image.png")

    # inference CLIP
    inputs = processor(text=queries, images=image, return_tensors="pt", padding=True).to(rank)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # label probabilities

    # get top predicted label
    top_label_idx = probs.argmax().item()
    top_label_str = relation_by_super_class_int2str()[top_label_idx]

    # show the results
    light_blue_code = 94
    light_pink_code = 95
    text_blue_colored = colored_text(top_label_str, light_blue_code)
    text_pink_colored = colored_text(relation, light_pink_code)
    print(f"Top predicted label from zero-shot CLIP: {text_blue_colored} (probability: {probs[0, top_label_idx]:.4f}), Target label: {text_pink_colored}\n")


def prepare_training(clip_model, attention_layer, relationship_refiner, tokenizer, processor, image, current_edge,
                     neighbor_edges, rank, batch=True, verbose=False):
    # collect image embedding
    inputs = processor(images=image, return_tensors="pt").to(rank)
    with torch.no_grad():
        image_embed = clip_model.module.get_image_features(**inputs)

    if batch:   # process all edges in a graph in parallel, but they all belong to the same image
        image_embed = image_embed.repeat(len(current_edge), 1)
    if verbose:
        print('image_embed', image_embed.shape)

    # accumulate all neighbor edges
    neighbor_text_embeds = []
    if verbose:
        print('current neighbor edges', neighbor_edges)
    # TODO use ground-truth neighbor edges when possible in training

    # collect all neighbors of the current edge
    if batch:
        for curr_neighbor_edges in neighbor_edges:
            inputs = tokenizer([f"a photo of a {edge[-1]}" for edge in curr_neighbor_edges],
                               padding=True, return_tensors="pt").to(rank)
            with torch.no_grad():
                text_embed = clip_model.module.get_text_features(**inputs)
            neighbor_text_embeds.append(text_embed)

        # pad the sequences to make them the same length (max_seq_len)
        seq_len = [len(embed) for embed in neighbor_text_embeds]
        key_padding_mask = torch.zeros(len(seq_len), max(seq_len), dtype=torch.bool).to(rank)
        for i, length in enumerate(seq_len):
            key_padding_mask[i, length:] = 1    # a True value indicates that the corresponding key value will be ignored

        neighbor_text_embeds = pad_sequence(neighbor_text_embeds, batch_first=False, padding_value=0.0)  # size [seq_len, batch_size, hidden_dim]
    else:
        for edge in neighbor_edges:
            inputs = tokenizer([f"a photo of a {edge[-1]}"], padding=False, return_tensors="pt").to(rank)
            with torch.no_grad():
                text_embed = clip_model.module.get_text_features(**inputs)
            neighbor_text_embeds.append(text_embed)

        neighbor_text_embeds = torch.stack(neighbor_text_embeds)    # size [seq_len, batch_size, hidden_dim]
        key_padding_mask = None

    # feed neighbor_text_embeds to a self-attention layer to get learnable weights
    neighbor_text_embeds = attention_layer(neighbor_text_embeds.to(rank).detach(), key_padding_mask=key_padding_mask)

    # fuse all neighbor_text_embeds after the attention layer
    if batch:
        neighbor_text_embeds = neighbor_text_embeds.permute(1, 0, 2)  # size [batch_size, seq_len, hidden_dim]
        neighbor_text_embeds = neighbor_text_embeds * key_padding_mask.unsqueeze(-1)
        neighbor_text_embeds = neighbor_text_embeds.sum(dim=1)
    else:
        neighbor_text_embeds = torch.sum(neighbor_text_embeds, dim=0)

    if verbose:
        print('neighbor_text_embeds', neighbor_text_embeds.shape)

    # extract current subject and object to condition the relation prediction
    if batch:
        queries = []
        for edge in current_edge:
            edge = edge[-1].split()
            subject, relation, object = extract_words_from_edge(edge, all_labels)
            queries.append(f"a photo of a {subject} and {object}")
        inputs = tokenizer(queries, padding=True, return_tensors="pt").to(rank)
    else:
        current_edge = current_edge[-1].split()
        subject, relation, object = extract_words_from_edge(current_edge, all_labels)
        query = [f"a photo of a {subject} and {object}"]
        inputs = tokenizer(query, padding=False, return_tensors="pt").to(rank)

    with torch.no_grad():
        query_txt_embed = clip_model.module.get_text_features(**inputs)
    if verbose:
        print('query_txt_embed', query_txt_embed.shape)

    # forward to the learnable layers
    predicted_txt_embed = relationship_refiner(image_embed.detach(), neighbor_text_embeds, query_txt_embed.detach())
    if verbose:
        print('predicted_txt_embed', predicted_txt_embed.shape)
    return predicted_txt_embed


def eval_refined_output(clip_model, tokenizer, predicted_txt_embed, current_edge, rank, verbose=False):
    # extract current subject and object from the edge
    phrase = current_edge[-1].split()
    subject, relation, object = extract_words_from_edge(phrase, all_labels)
    queries = [f"a photo of a {subject} {label} {object}" for label in all_labels]

    # collect text_embeds for all possible labels
    inputs = tokenizer(queries, padding=True, return_tensors="pt").to(rank)
    with torch.no_grad():
        all_possible_embeds = clip_model.module.get_text_features(**inputs)
    all_possible_embeds = F.normalize(all_possible_embeds, p=2, dim=-1)

    predicted_txt_embed = F.normalize(predicted_txt_embed, p=2, dim=-1)  # normalize the predicted embedding

    # compute cosine similarity between predicted embedding and all query embeddings
    CosSim = nn.CosineSimilarity(dim=1)
    cos_sim = CosSim(predicted_txt_embed, all_possible_embeds)
    cos_sim = F.softmax(cos_sim, dim=0)  # remove batch dimension

    # find the query with the highest similarity
    max_val, max_index = cos_sim.max(dim=0)
    most_probable_query = all_labels[max_index]

    # show the results
    if verbose:
        light_blue_code = 94
        light_pink_code = 95
        text_blue_colored = colored_text(most_probable_query, light_blue_code)
        text_pink_colored = colored_text(relation, light_pink_code)
        print(f"Predicted label: '{text_blue_colored}' with confidence {max_val}, Target label: '{text_pink_colored}'")

    # return the updated edge
    updated_phrase = subject + ' ' + most_probable_query + ' ' + object
    updated_edge = (current_edge[0], max_index.item(), current_edge[2], updated_phrase)
    if verbose:
        dark_blue_code = 34
        text_blue_colored = colored_text(updated_edge, dark_blue_code)
        print('updated_edge', text_blue_colored, '\n')

    return updated_edge, max_val


def bfs_explore(clip_model, processor, tokenizer, attention_layer, relationship_refiner, optimizer, criterion,
                image, graph, target_triplets, batch_idx, data_len, rank, args, verbose=False):
    if args['training']['run_mode'] == 'clip_train':
        # the gradient accumulation process begins by zeroing out gradients at the start of each epoch.
        grad_acc_counter = 0
        accumulation_steps = 16
        optimizer.zero_grad()

    # get the node with the highest degree
    node_degrees = graph.get_node_degrees()
    if verbose:
        print('node_degrees', node_degrees, '\n')
    start_node = max(node_degrees, key=node_degrees.get)

    # initialize queue and visited set for BFS
    queue = deque([(start_node, 0)])  # the second element in the tuple is used to keep track of levels
    visited_nodes = set()
    visited_edges = set()

    while True:
        while queue:
            # dequeue the next node to visit
            current_node, level = queue.popleft()

            # if the node hasn't been visited yet
            if current_node not in visited_nodes:
                if verbose:
                    deep_green_code = 32
                    text_green_colored = colored_text(current_node, deep_green_code)
                    print(f"Visiting node: {text_green_colored} at level {level}")

                # mark the node as visited
                visited_nodes.add(current_node)

                # get all the neighboring edges for the current node
                neighbor_edges = graph.adj_list[current_node]

                # create a mapping from neighbor_node to neighbor_edge
                neighbor_to_edge_map = {edge[2] if edge[2] != current_node else edge[0]: edge for edge in neighbor_edges}

                # extract neighbor nodes and sort them by their degree
                neighbor_nodes = [edge[2] if edge[2] != current_node else edge[0] for edge in neighbor_edges]  # the neighbor node could be either the subject or the object
                neighbor_nodes = sorted(neighbor_nodes, key=lambda x: node_degrees.get(x, 0), reverse=True)

                # add neighbors to the queue for future exploration
                for neighbor_node in neighbor_nodes:
                    current_edge = neighbor_to_edge_map[neighbor_node]
                    if current_edge not in visited_edges:
                        if verbose:
                            light_green_code = 92
                            text_green_colored = colored_text(current_edge, light_green_code)
                            print(f"Visiting edge: {text_green_colored}")

                        # mark the edge as visited
                        visited_edges.add(current_edge)

                        if args['training']['run_mode'] == 'clip_zs':
                            # query CLIP on the current neighbor edge in zero shot
                            clip_zero_shot(clip_model, processor, image, current_edge, rank, args)
                        else:
                            # train the model to predict relations from neighbors and image features
                            subject_neighbor_edges = list(neighbor_edges)   # use the list constructor to create a new list with the elements of the original list
                            object_neighbor_edges = list(graph.adj_list[neighbor_node])
                            subject_neighbor_edges.remove(current_edge)    # do not include the current edge redundantly
                            object_neighbor_edges.remove(current_edge)

                            neighbor_edges = [current_edge] + subject_neighbor_edges + object_neighbor_edges

                            # forward pass
                            predicted_txt_embed = prepare_training(clip_model, attention_layer, relationship_refiner, tokenizer, processor, image, current_edge,
                                                                   neighbor_edges, rank, batch=False, verbose=verbose)

                            updated_edge, confidence = eval_refined_output(clip_model, tokenizer, predicted_txt_embed, current_edge, rank, verbose=verbose)

                            # prepare learning target
                            curr_subject_bbox = current_edge[0]
                            curr_object_bbox = current_edge[2]
                            for target in target_triplets:
                                target_subject_bbox = target[0]
                                target_object_bbox = target[1]

                                # when the current prediction is about the current target
                                if args['training']['eval_mode'] == 'pc':
                                    condition = target_subject_bbox == curr_subject_bbox and target_object_bbox == curr_object_bbox
                                else:
                                    condition = iou(target_subject_bbox, curr_subject_bbox) >= 0.5 and iou(target_object_bbox, curr_object_bbox) >= 0.5
                                if condition:
                                    target_subject, target_relation, target_object = extract_words_from_edge(target[-1].split(), all_labels)
                                    target = [f"a photo of a {target_subject} {target_relation} {target_object}"]
                                    inputs = tokenizer(target, padding=False, return_tensors="pt").to(rank)
                                    with torch.no_grad():
                                        target_txt_embed = clip_model.module.get_text_features(**inputs)

                                    # back-propagate
                                    grad_acc_counter += 1
                                    loss = criterion(predicted_txt_embed, target_txt_embed)
                                    loss.backward()

                                    if grad_acc_counter % accumulation_steps == 0:  # Only step optimizer every `accumulation_steps`
                                        optimizer.step()
                                        optimizer.zero_grad()  # Ensure we clear gradients after an update

                                    if verbose:
                                        if (batch_idx % args['training']['eval_freq'] == 0) or (batch_idx + 1 == data_len):
                                            print(f'Rank {rank} epoch {batch_idx}, graphRefineLoss {loss.item()}')
                                    # break the target matching loop
                                    break

                            # update self.edges in the graph using predicted_txt_embed
                            index_to_update = None
                            for index, stored_edge in enumerate(graph.edges):
                                if stored_edge == current_edge:
                                    index_to_update = index
                                    break
                            if index_to_update is not None:
                                graph.edges[index_to_update] = updated_edge
                                graph.confidence[index_to_update] = confidence

                        queue.append((neighbor_node, level + 1))

        if verbose:
            print("Finished BFS for current connected component.\n")

        # check if there are any unvisited nodes
        unvisited_nodes = set(node_degrees.keys()) - visited_nodes
        if not unvisited_nodes:
            break  # all nodes have been visited, exit the loop

        # start a new BFS from the unvisited node with the highest degree
        new_start_node = max(unvisited_nodes, key=lambda x: node_degrees.get(x, 0))
        if verbose:
            print(f"Starting new BFS from node: {new_start_node}")
        queue.append((new_start_node, 0))

    return graph, attention_layer, relationship_refiner


def batch_training(clip_model, processor, tokenizer, attention_layer, relationship_refiner, optimizer, criterion,
                   images, graphs, target_triplets, data_len, rank, args, verbose=False):

    # for all graphs in the batch, iterate all its edges being added
    for graph, targets, image in zip(graphs, target_triplets, images):
        all_current_edges = []
        all_neighbor_edges = []
        all_target_txt_embeds = []
        target_mask = []    # mask of predicate that has matched with a target

        for edge_idx, current_edge in enumerate(graph.edges):
            subject_node, object_node = current_edge[0], current_edge[2]
            current_subject, _, current_object = extract_words_from_edge(current_edge[-1].split(), all_labels)

            # find corresponding target edge for the current edge
            target_txt_embed = None
            for target in targets:
                target_subject_bbox = target[0]
                target_object_bbox = target[1]

                # when the current prediction is about the current target
                if iou(target_subject_bbox, subject_node) >= 0.5 and iou(target_object_bbox, object_node) >= 0.5:
                    target_subject, target_relation, target_object = extract_words_from_edge(target[-1].split(), all_labels)

                    if target_subject == current_subject and target_object == current_object:
                        phrase = [f"a photo of a {target_subject} {target_relation} {target_object}"]
                        inputs = tokenizer(phrase, padding=True, return_tensors="pt").to(rank)
                        with torch.no_grad():
                            target_txt_embed = clip_model.module.get_text_features(**inputs)
                        all_target_txt_embeds.append(target_txt_embed)
                        target_mask.append(edge_idx)
                    break

            # find neighbor edges for the current edge
            subject_neighbor_edges = graph.adj_list[subject_node]
            object_neighbor_edges = graph.adj_list[object_node]
            if target_txt_embed is None:
                subject_neighbor_edges.remove(current_edge)  # do not include the current edge redundantly
                object_neighbor_edges.remove(current_edge)
            else:
                subject_neighbor_edges.remove(target)
                object_neighbor_edges.remove(target)

            neighbor_edges = [current_edge] + subject_neighbor_edges + object_neighbor_edges

            all_current_edges.append(current_edge)
            all_neighbor_edges.append(neighbor_edges)

        target_mask = torch.tensor(target_mask).to(rank)

        # forward pass
        predicted_txt_embeds = prepare_training(clip_model, attention_layer, relationship_refiner, tokenizer, processor, image,
                                               all_current_edges, all_neighbor_edges, rank, batch=True, verbose=verbose)

        if len(all_target_txt_embeds) > 0:
            all_target_txt_embeds = torch.stack(all_target_txt_embeds).squeeze(dim=1)

            loss = criterion(predicted_txt_embeds[target_mask], all_target_txt_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                if (batch_idx % args['training']['eval_freq'] == 0) or (batch_idx + 1 == data_len):
                    print(f'Rank {rank} epoch {batch_idx}, graphRefineLoss {loss.item()}')

        # for each graph in the batch, update self.edges using corresponding predicted_txt_embed
        # even predicate outside the mask may still improve their reasonable yet not annotated predictions
        for edge_idx in range(len(graph.edges)):
            updated_edges, confidences = eval_refined_output(clip_model, tokenizer, predicted_txt_embeds[edge_idx], all_current_edges[edge_idx], rank, verbose=verbose)

            graph.edges[edge_idx] = updated_edges
            graph.confidence[edge_idx] = confidences

    return graphs, attention_layer, relationship_refiner


def process_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner, optimizer, criterion,
                        rank, args, sgg_results, top_k, data_len, verbose=False):
    top_k_predictions = sgg_results['top_k_predictions']
    if verbose:
        print('top_k_predictions', top_k_predictions[0])
    top_k_image_graphs = sgg_results['top_k_image_graphs']
    images = sgg_results['images']
    target_triplets = sgg_results['target_triplets']
    Recall = sgg_results['Recall']

    for batch_idx, (curr_strings, curr_image, curr_target_triplet) in enumerate(zip(top_k_predictions, top_k_image_graphs, target_triplets)):
        graph = ImageGraph()

        for string, triplet in zip(curr_strings, curr_image):
            subject_bbox, relation_id, object_bbox = triplet[0], triplet[1], triplet[2]
            graph.add_edge(subject_bbox, object_bbox, relation_id, string, verbose=verbose)

        if verbose:
            print('batch_idx', batch_idx, '/', len(top_k_predictions))
            dark_orange_code = 33
            text_orange_colored = colored_text(curr_target_triplet, dark_orange_code)
            print(f"curr_target_triplet edge: {text_orange_colored}")
            save_png(images[batch_idx], "curr_image.png")

        updated_graph, attention_layer, relationship_refiner = bfs_explore(clip_model, processor, tokenizer, attention_layer, relationship_refiner, optimizer, criterion,
                                                                           images[batch_idx], graph, curr_target_triplet, batch_idx, data_len, rank, args, verbose=verbose)
        relation_pred, confidence = extract_updated_edges(updated_graph, rank)
        Recall.global_refine(relation_pred, confidence, batch_idx, top_k, rank)

    torch.save(attention_layer.state_dict(), args['training']['checkpoint_path'] + 'AttentionLayer' + '_' + str(rank) + '.pth')
    torch.save(relationship_refiner.state_dict(), args['training']['checkpoint_path'] + 'RelationshipRefiner' + '_' + str(rank) + '.pth')


def process_batch_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner, optimizer, criterion,
                              rank, args, sgg_results, top_k, data_len, verbose=False):
    top_k_predictions = sgg_results['top_k_predictions']
    if verbose:
        print('top_k_predictions', top_k_predictions[0])
    top_k_image_graphs = sgg_results['top_k_image_graphs']
    images = sgg_results['images']
    target_triplets = sgg_results['target_triplets']
    Recall = sgg_results['Recall']

    graphs = []
    for batch_idx, (curr_strings, curr_image) in enumerate(zip(top_k_predictions, top_k_image_graphs)):
        graph = ImageGraph(targets=target_triplets[batch_idx])

        for string, triplet in zip(curr_strings, curr_image):
            subject_bbox, relation_id, object_bbox = triplet[0], triplet[1], triplet[2]
            graph.add_edge(subject_bbox, object_bbox, relation_id, string, training=True, verbose=verbose)

        graphs.append(graph)

    updated_graphs, attention_layer, relationship_refiner = batch_training(clip_model, processor, tokenizer, attention_layer, relationship_refiner, optimizer, criterion,
                                                                           images, graphs, target_triplets, data_len, rank, args, verbose=verbose)

    for batch_idx in range(len(top_k_predictions)):
        relation_pred, confidence = extract_updated_edges(updated_graphs[batch_idx], rank)
        Recall.global_refine(relation_pred, confidence, batch_idx, top_k, rank)

    torch.save(attention_layer.state_dict(), args['training']['checkpoint_path'] + 'AttentionLayer' + '_' + str(rank) + '.pth')
    torch.save(relationship_refiner.state_dict(), args['training']['checkpoint_path'] + 'RelationshipRefiner' + '_' + str(rank) + '.pth')


def query_clip(gpu, args, train_dataset, test_dataset):
    rank = gpu
    world_size = torch.cuda.device_count()
    setup(rank, world_size)
    print('rank', rank, 'torch.distributed.is_initialized', torch.distributed.is_initialized())

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=test_sampler)
    print("Finished loading the datasets...")

    # initialize CLIP
    clip_model = DDP(CLIPModel.from_pretrained("openai/clip-vit-base-patch32")).to(rank)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    attention_layer = DDP(SimpleSelfAttention(hidden_dim=clip_model.module.text_embed_dim)).to(rank)
    attention_layer.train()
    relationship_refiner = DDP(RelationshipRefiner(hidden_dim=clip_model.module.text_embed_dim)).to(rank)
    relationship_refiner.train()

    optimizer = optim.Adam([
        {'params': attention_layer.module.parameters(), 'initial_lr': args['training']['refine_learning_rate']},
        {'params': relationship_refiner.module.parameters(), 'initial_lr': args['training']['refine_learning_rate']}
    ], lr=args['training']['refine_learning_rate'], weight_decay=args['training']['weight_decay'])
    criterion = nn.MSELoss()

    # receive current SGG predictions from a baseline model
    sgg_results = eval_pc(rank, args, train_loader, topk_global_refine=args['training']['topk_global_refine'])

    # iterate through the generator to receive results
    for batch_idx, batch_sgg_results in enumerate(sgg_results):
        if args['training']['verbose_global_refine']:
            print('batch_idx', batch_idx)

        process_batch_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner, optimizer, criterion,
                                  rank, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(train_loader), verbose=args['training']['verbose_global_refine'])
        # process_sgg_results(clip_model, processor, tokenizer, attention_layer, relationship_refiner, optimizer, criterion,
        #                     rank, args, batch_sgg_results, top_k=args['training']['topk_global_refine'], data_len=len(train_loader), verbose=args['training']['verbose_global_refine'])

    dist.destroy_process_group()  # clean up

