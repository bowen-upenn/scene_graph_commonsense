import torch
import os
from PIL import Image
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import CLIPProcessor, CLIPModel
from collections import deque

from evaluate import inference, eval_pc
from utils import *
from dataset_utils import relation_by_super_class_int2str


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


class ImageGraph:
    def __init__(self):
        # node to neighbors mapping
        self.adj_list = {}
        # edge to nodes mapping
        self.edge_node_map = {}
        # store all nodes and their degree
        self.nodes = []
        self.edges = []

    def add_edge(self, subject_bbox, object_bbox, relation_id, string):
        subject_bbox, object_bbox = tuple(subject_bbox), tuple(object_bbox)
        edge = (subject_bbox, relation_id, object_bbox, string)
        edge_wo_string = (subject_bbox, relation_id, object_bbox)

        if edge not in self.edges:
            self.edges.append(edge_wo_string)
        if subject_bbox not in self.nodes:
            self.nodes.append(subject_bbox)
        if object_bbox not in self.nodes:
            self.nodes.append(object_bbox)

        print('subject_bbox', subject_bbox)
        print('object_bbox', object_bbox)
        print('edge', edge, '\n')

        # Check if the node is already present, otherwise initialize with an empty list
        if subject_bbox not in self.adj_list:
            self.adj_list[subject_bbox] = []
        if object_bbox not in self.adj_list:
            self.adj_list[object_bbox] = []

        # change list as immutable tuple as the dict key
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


def extract_words_from_edge(phrase, all_relation_labels):
    # iterate through the phrase to extract the parts
    for i in range(len(phrase)):
        if phrase[i] in all_relation_labels:
            relation = phrase[i]
            subject = " ".join(phrase[:i])
            object = " ".join(phrase[i + 1:])
            break  # exit loop once the relation is found

    return subject, relation, object


def forward_clip(model, processor, image, edge):
    # prepare text labels from the relation dictionary
    labels = list(relation_by_super_class_int2str().values())

    # extract current subject and object from the edge
    phrase = edge[-1].split()
    subject, relation, object = extract_words_from_edge(phrase, labels)

    queries = [f"a photo of a {subject} {label} {object}" for label in labels]

    # inference CLIP
    inputs = processor(text=queries, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # label probabilities

    # get top predicted label
    top_label_idx = probs.argmax().item()
    top_label_str = relation_by_super_class_int2str()[top_label_idx]

    light_blue_code = 94
    light_pink_code = 95
    text_blue_colored = colored_text(top_label_str, light_blue_code)
    text_pink_colored = colored_text(relation, light_pink_code)

    print(f"Top predicted label from zero-shot CLIP: {text_blue_colored} (probability: {probs[0, top_label_idx]:.4f}), Target label: {text_pink_colored}\n")


def bfs_explore(image, graph):
    # initialize CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # get the node with the highest degree
    node_degrees = graph.get_node_degrees()
    print('node_degrees', node_degrees)
    start_node = max(node_degrees, key=node_degrees.get)

    # initialize queue and visited set for BFS
    queue = deque([(start_node, 0)])  # the second element in the tuple is used to keep track of levels
    visited = set()

    while True:
        while queue:
            # dequeue the next node to visit
            current_node, level = queue.popleft()

            # if the node hasn't been visited yet
            if current_node not in visited:
                print(f"Visiting node: {current_node} at level {level}")

                # mark the node as visited
                visited.add(current_node)

                # get all the neighboring edges for the current node
                neighbor_edges = graph.adj_list[current_node]

                # create a mapping from neighbor_node to neighbor_edge
                neighbor_to_edge_map = {edge[2] if edge[2] != current_node else edge[0]: edge for edge in neighbor_edges}

                # extract neighbor nodes and sort them by their degree
                neighbor_nodes = [edge[2] if edge[2] != current_node else edge[0] for edge in neighbor_edges]  # the neighbor node could be either the subject or the object
                neighbor_nodes = sorted(neighbor_nodes, key=lambda x: node_degrees.get(x, 0), reverse=True)

                # add neighbors to the queue for future exploration
                for neighbor_node in neighbor_nodes:
                    if neighbor_node not in visited:
                        neighbor_edge = neighbor_to_edge_map[neighbor_node]
                        print(f"Edge for next neighbor: {neighbor_edge}")

                        # query CLIP on the current neighbor edge
                        forward_clip(model, processor, image, neighbor_edge)

                        queue.append((neighbor_node, level + 1))

        print("Finished BFS for current connected component.\n")

        # check if there are any unvisited nodes
        unvisited_nodes = set(node_degrees.keys()) - visited
        if not unvisited_nodes:
            break  # all nodes have been visited, exit the loop

        # start a new BFS from the unvisited node with the highest degree
        new_start_node = max(unvisited_nodes, key=lambda x: node_degrees.get(x, 0))
        print(f"Starting new BFS from node: {new_start_node}")
        queue.append((new_start_node, 0))


def process_sgg_results(rank, sgg_results):
    top_k_predictions = sgg_results['top_k_predictions']
    print('top_k_predictions', top_k_predictions[0])
    top_k_image_graphs = sgg_results['top_k_image_graphs']
    images = sgg_results['images']

    image_graphs = []
    for batch_idx, (curr_strings, curr_image) in enumerate(zip(top_k_predictions, top_k_image_graphs)):
        graph = ImageGraph()

        for string, triplet in zip(curr_strings, curr_image):
            subject_bbox, relation_id, object_bbox = triplet[0], triplet[1], triplet[2]
            graph.add_edge(subject_bbox, object_bbox, relation_id, string)

        bfs_explore(images[batch_idx], graph)

        image_graphs.append(graph)

        # image = images[0].mul(255).cpu().byte().numpy()  # convert to 8-bit integer values
        # image = Image.fromarray(image.transpose(1, 2, 0))  # transpose dimensions for RGB order
        # image.save("image.png")
        #
        # print('adj_list', graph.adj_list, '\n')
        #
        # curr_edge = (( 8., 23., 10., 20.), 11., ( 4., 31.,  4., 29.))
        # neighbors = graph.get_edge_neighbors(curr_edge, hops=1)
        # print("1-hop neighbors:", neighbors, '\n')
        #
        # neighbors_2_hop = graph.get_edge_neighbors(curr_edge, hops=2)
        # print("2-hop neighbors:", neighbors_2_hop)
        #
        # degrees = graph.get_node_degrees()
        # print('degrees', degrees)

        break

    return image_graphs


def query_clip(gpu, args, test_dataset):
    rank = gpu
    world_size = torch.cuda.device_count()
    setup(rank, world_size)
    print('rank', rank, 'torch.distributed.is_initialized', torch.distributed.is_initialized())

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=test_sampler)
    print("Finished loading the datasets...")

    # receive current SGG predictions from a baseline model
    sgg_results = eval_pc(rank, args, test_loader, return_sgg_results=True, top_k=5)

    # iterate through the generator to receive results
    for batch_idx, batch_sgg_results in enumerate(sgg_results):
        print('batch_idx', batch_idx)
        image_graphs = process_sgg_results(rank, batch_sgg_results)

    dist.destroy_process_group()  # clean up

