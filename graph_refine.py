import torch
import os
from PIL import Image
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluate import inference, eval_pc
from utils import *


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

    def add_edge(self, subject_bbox, object_bbox, relation_id, string):
        subject_bbox, object_bbox = tuple(subject_bbox), tuple(object_bbox)
        edge = (subject_bbox, relation_id, object_bbox, string)
        edge_wo_string = (subject_bbox, relation_id, object_bbox)

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
            return

    def get_nodes_from_edge(self, edge):
        return self.edge_node_map.get(edge, (None, None))


def process_sgg_results(rank, sgg_results):
    top_k_predictions = sgg_results['top_k_predictions']
    print('top_k_predictions', top_k_predictions[0])
    top_k_image_graphs = sgg_results['top_k_image_graphs']
    images = sgg_results['images']

    image_graphs = []
    for curr_strings, curr_image in zip(top_k_predictions, top_k_image_graphs):
        graph = ImageGraph()

        for string, triplet in zip(curr_strings, curr_image):
            subject_bbox, relation_id, object_bbox = triplet[0], triplet[1], triplet[2]
            # print('subject_bbox, relation_id, object_bbox', subject_bbox, relation_id, object_bbox)
            graph.add_edge(subject_bbox, object_bbox, relation_id, string)

        image_graphs.append(graph)

        image = images[0].mul(255).cpu().byte().numpy()  # convert to 8-bit integer values
        image = Image.fromarray(image.transpose(1, 2, 0))  # transpose dimensions for RGB order
        image.save("image.png")

        print('adj_list', graph.adj_list)
        print('edge_node_map', graph.edge_node_map, '\n')

        curr_edge = (( 8., 23., 10., 20.), 11., ( 4., 31.,  4., 29.))
        neighbors = graph.get_edge_neighbors(curr_edge, hops=1)
        print("1-hop neighbors:", neighbors, '\n')

        neighbors_2_hop = graph.get_edge_neighbors(curr_edge, hops=2)
        print("2-hop neighbors:", neighbors_2_hop)

        # another_node = ( 4., 31.,  4., 29.)
        # nodes_from_edge = graph.get_nodes_from_edge((curr_node, 11, another_node))
        # print("Nodes connected by edge:", nodes_from_edge)

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

