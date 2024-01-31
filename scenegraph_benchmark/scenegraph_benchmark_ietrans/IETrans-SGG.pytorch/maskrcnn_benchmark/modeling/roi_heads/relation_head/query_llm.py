import torch
from tqdm import tqdm
import openai
# from openai import OpenAI
import math
from collections import OrderedDict
import re
import random
import json


class EdgeCache:
    def __init__(self, max_cache_size):
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size
        self.access_frequency = {}

    def get(self, key):
        return self.cache.get(key, None)

    def put(self, key, value):
        if key in self.cache:
            # Move to end to show it was recently accessed
            self.cache.move_to_end(key)
            # Increase access frequency
            self.access_frequency[key] += 1
        else:
            if len(self.cache) >= self.max_cache_size:
                self._purge_least_frequent()
            self.cache[key] = value
            self.access_frequency[key] = 1

    def _purge_least_frequent(self):
        # Find the least frequently accessed item
        least_frequent_key = min(self.access_frequency, key=self.access_frequency.get)
        # Remove the least frequently accessed item
        if least_frequent_key in self.cache:
            del self.cache[least_frequent_key]
        if least_frequent_key in self.access_frequency:
            del self.access_frequency[least_frequent_key]

    def cache_info(self):
        return len(self.cache), self.max_cache_size


class CommonsenseValidator:
    def __init__(self, top_k=10, max_cache_size=10000):
        self.cache_hits = 0
        self.top_k = top_k
        self.total_cache_queries = 0
        self.cache = EdgeCache(max_cache_size=max_cache_size)
        self.cache.put("", -1)  # Update cache access frequency

        # read object and relation labels from the dataset
        data_path = '/raid0/docker-raid/bwjiang/datasets/vg/50/VG-SGG-dicts-with-attri.json'
        with open(data_path, 'r') as file:
            data = json.load(file)
        self.idx_to_predicate = data['idx_to_predicate']    # 50
        self.idx_to_object = data['idx_to_label']   # 150
        print('idx_to_predicate', self.idx_to_predicate)
        print('idx_to_object', self.idx_to_object)


    def query(self, rel_pair_idx, rel_labels):
        """
        :param rel_pair_idx: subject and object labels, sorted by triplet confidence in descending order. torch tensor of size batch_size, 2
        :param rel_labels: relation labels, sorted by triplet confidence in descending order. torch tensor of size batch_size
        :return: all_responses: +1 or -1 for each queried edge. torch tensor of size batch_size
        """
        # convert ids to strings for each triplet
        batched_edges = []
        batch_size = rel_pair_idx.shape[0]
        for i in range(batch_size):
            if rel_pair_idx[i][0].item() == 0 or rel_pair_idx[i][1].item() == 0:
                edge = ""
            else:
                edge = self.idx_to_object[str(rel_pair_idx[i][0].item())] + ' ' + self.idx_to_predicate[str(rel_labels[i].item())] \
                       + ' ' + self.idx_to_object[str(rel_pair_idx[i][1].item())]
            batched_edges.append(edge)

        # query
        all_responses = self.batch_query_openai_gpt(batched_edges)
        all_responses = torch.tensor(all_responses)

        return all_responses


    def batch_query_openai_gpt(self, predicted_edges, batch_size=4):
        total_edges = len(predicted_edges)
        all_responses = []

        for i in range(0, total_edges, batch_size): # We can not use a large batch size because of the OpenAI API limit
            batched_edges = predicted_edges[i: i + batch_size]
            batched_edges_to_query = []

            for edge in batched_edges:
                cached_response = self.cache.get(edge)
                if cached_response is not None and random.random() < 0.9:
                    all_responses.append(cached_response)
                    self.cache_hits += 1
                    self.cache.put(edge, cached_response)  # Update cache access frequency
                else:
                    batched_edges_to_query.append(edge)

            if batched_edges_to_query:
                responses = self._batch_query_openai_gpt_3p5_instruct(batched_edges_to_query)

                for edge, response in zip(batched_edges_to_query, responses):
                    self.cache.put(edge, response)
                    all_responses.append(response)

        return all_responses


    def _batch_query_openai_gpt_3p5_instruct(self, predicted_edges, verbose=False):
        """ This function queries OpenAI GPT-3.5-turbo-instruct model with a batch of subject-relation-object triplets whether each triplet is commonsense-aligned or violated.
        We support GPT3.5 for plug-and-play fashion in the Scene-Graph-Benchmark codebase, and support both GPT3.5 and GPT4V in our standalone codebase.
        """
        openai.api_key_path = '/raid0/docker-raid/bwjiang/scene_graph/openai_key.txt'  # Path to your OpenAI API key
        responses = torch.ones(len(predicted_edges)) * -1

        prompts = []

        # Prepare multiple variations of each prompt
        prompt_variations = [
            "Is the relation '{}' generally make sense or a trivially true fact? Answer with 'Yes' or 'No' and justify your answer. A trivially true relation is still a 'Yes'.",
            "Could there be either a {} or a {}s? Yes or No and justify your answer.",
            "Regardless of whether it is basic or redundant, is the relation '{}' incorrect and is a mis-classification in scene graph generation? Show your reasoning and answer 'Yes' or 'No'.",
            "Is the relation {} impossible in real world? Answer 'Yes' or 'No' and explain your answer."
        ]

        # For each predicted edge, create multiple prompts
        for edge in predicted_edges:
            for i, variation in enumerate(prompt_variations):
                if i == 1:
                    prompts.append(variation.format(edge, edge))
                else:
                    prompts.append(variation.format(edge))

        # Call OpenAI with the batch of prompts
        completions = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
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
                if verbose:
                    print(completion_text)
                # completion_text = completions.choices[i * len(prompt_variations) + j].message

                if j == 2 or j == 3:  # For the last two questions, we reverse the logic
                    if re.search(r'Yes', completion_text):
                        no_votes += 1
                    elif re.search(r'No', completion_text):
                        yes_votes += 1
                    else:
                        no_votes += 1
                else:
                    if re.search(r'Yes', completion_text):
                        if j == 0:
                            yes_votes += 2
                        else:
                            yes_votes += 1
                    else:
                        if j == 0:
                            no_votes += 2
                        else:
                            no_votes += 1

            if yes_votes > no_votes:
                if verbose:
                    print(f'predicted_edge {edge} [MAJORITY YES] {yes_votes} Yes votes vs {no_votes} No votes')
                responses[i] = 1
            else:
                if verbose:
                    print(f'predicted_edge {edge} [MAJORITY NO] {no_votes} No votes vs {yes_votes} Yes votes')
                responses[i] = -1

        return responses
