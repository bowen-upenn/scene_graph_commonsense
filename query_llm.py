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
# from openai import OpenAI
import math
from collections import Counter, OrderedDict
import re
import random


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


def batch_query_openai_gpt(predicted_edges, edge_cache, batch_size=4, cache_hits=0):
    total_edges = len(predicted_edges)
    all_responses = []

    for i in range(0, total_edges, batch_size):
        batched_edges = predicted_edges[i: i + batch_size]
        batched_edges_to_query = []

        for edge in batched_edges:
            cached_response = edge_cache.get(edge)
            if cached_response is not None and random.random() < 0.9:
                all_responses.append(cached_response)
                cache_hits += 1
                edge_cache.put(edge, cached_response)  # Update cache access frequency
            else:
                batched_edges_to_query.append(edge)

        if batched_edges_to_query:
            responses = _batch_query_openai_gpt_instruct(batched_edges_to_query)
            for edge, response in zip(batched_edges_to_query, responses):
                edge_cache.put(edge, response)
                all_responses.append(response)

    return all_responses, cache_hits


def _batch_query_openai_gpt_instruct(predicted_edges, verbose=False):
    openai.api_key_path = 'openai_key.txt'
    responses = torch.ones(len(predicted_edges)) * -1

    prompts = []

    # Prepare multiple variations of each prompt
    prompt_variations = [
        "Is the relation '{}' generally make sense or a trivially true fact? Answer with 'Yes' or 'No' and justify your answer. A trivially true relation is still a 'Yes'.",
        "Is the relation '{}' generally make sense or a trivially true fact? Answer with 'Yes' or 'No' and justify your answer. A trivially true relation is still a 'Yes'.",
        "Could there be either a {} or a {}s? Yes or No and justify your answer.",
        "Regardless of whether it is basic or redundant, is the relation '{}' incorrect and is a mis-classification in scene graph generation? Show your reasoning and answer 'Yes' or 'No'.",
        "Is the relation {} impossible in real world? Answer 'Yes' or 'No' and explain your answer."
    ]

    # For each predicted edge, create multiple prompts
    for edge in predicted_edges:
        for i, variation in enumerate(prompt_variations):
            if i == 2:
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

            if j > 2:  # For the last two questions, we reverse the logic
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
            if verbose:
                print(f'predicted_edge {edge} [MAJORITY YES] {yes_votes} Yes votes vs {no_votes} No votes')
            responses[i] = 1
        else:
            if verbose:
                print(f'predicted_edge {edge} [MAJORITY NO] {no_votes} No votes vs {yes_votes} Yes votes')
            responses[i] = -1

    return responses

