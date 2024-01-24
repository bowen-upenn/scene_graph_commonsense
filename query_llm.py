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
import cv2
import base64
import requests


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


def batch_query_openai_gpt(predicted_edges, edge_cache, batch_size=4, cache_hits=0,
                           annot_name=None, sub_bbox=None, obj_bbox=None, image_cache=None, image_dir=None):
    # input arguments annot_name, sub_bbox, obj_bbox, image_cache, and image_dir are only required if using GPT-4V as the commonsense validator
    use_vision = annot_name is not None
    if use_vision:
        batch_size = 1

    total_edges = len(predicted_edges)
    all_responses = []

    for i in range(0, total_edges, batch_size):
        batched_edges = predicted_edges[i: i + batch_size]
        batched_edges_to_query = []

        for edge in batched_edges:
            if use_vision:  # do not use edge cache
                batched_edges_to_query.append(edge)
            else:
                cached_response = edge_cache.get(edge)
                if cached_response is not None and random.random() < 0.9:
                    all_responses.append(cached_response)
                    cache_hits += 1
                    edge_cache.put(edge, cached_response)  # Update cache access frequency
                else:
                    batched_edges_to_query.append(edge)

        if batched_edges_to_query:
            if use_vision:
                responses = _query_openai_gpt_4v(batched_edges_to_query, annot_name, sub_bbox[i], obj_bbox[i], image_cache, image_dir)
            else:
                responses = _batch_query_openai_gpt_3p5_instruct(batched_edges_to_query)

            for edge, response in zip(batched_edges_to_query, responses):
                if not use_vision:
                    edge_cache.put(edge, response)
                all_responses.append(response)

    return all_responses, cache_hits


def _batch_query_openai_gpt_3p5_instruct(predicted_edges, verbose=False):
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


class ImageCache:
    def __init__(self, image_size, feature_size):
        self.cache = {}
        self.image_size = image_size
        self.feature_size = feature_size

    def get_image(self, image_path, bbox=None):
        if image_path not in self.cache:
            image = cv2.imread(image_path)
            image = cv2.resize(image, dsize=(self.image_size, self.image_size))

            if bbox is not None:
                x1, x2, y1, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                image = image[y1:y2, x1:x2]

            # Convert image to bytes for base64 encoding
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = np.array(buffer).tobytes()

            self.cache[image_path] = base64.b64encode(image_bytes).decode('utf-8')
        return self.cache[image_path]


def get_union_bbox(sub_bbox, obj_bbox):
    # Calculate the smallest bounding box that contains both subject and object
    x1 = min(sub_bbox[0], obj_bbox[0])
    y1 = min(sub_bbox[1], obj_bbox[1])
    x2 = max(sub_bbox[2], obj_bbox[2])
    y2 = max(sub_bbox[3], obj_bbox[3])
    return [x1, y1, x2, y2]


def _query_openai_gpt_4v(predicted_edges, annot_name, sub_bbox, obj_bbox, image_cache, image_dir, verbose=True):
    with open("openai_key.txt", "r") as api_key_file:
        api_key = api_key_file.read()

    responses = torch.ones(len(predicted_edges)) * -1

    # GPT-4V does not support batch inference at this moment, but we keep the same structure as in _batch_query_openai_gpt_instruct for code simplicity
    assert len(predicted_edges) == 1

    for i, edge in enumerate(predicted_edges):
        # Construct the path to the image and annotations
        image_path = os.path.join(image_dir, annot_name[:-16] + '.jpg')

        # Load and process image and annotations if they exist
        if os.path.exists(image_path):
            image_path = os.path.join(image_dir, annot_name[:-16] + '.jpg')
            print('annot_name', annot_name, 'image_path', image_path)

            sub_bbox *= image_cache.feature_size
            obj_bbox *= image_cache.feature_size
            union_bbox = get_union_bbox(sub_bbox, obj_bbox)
            base64_image = image_cache.get_image(image_path, bbox=union_bbox)

            # Form the prompt including the image.
            # Due to the strong performance of the vision model, we omit multiple queries and majority vote to reduce costs
            prompt = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Does the image contain a relation '{}'? Let us think about it step by step.".format(edge)},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 300
            }

            # Send request to OpenAI API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=prompt)
            response_json = response.json()

            # Process the response
            # Check if the response is valid and contains the expected data
            if 'choices' in response_json and len(response_json['choices']) > 0:
                completion_text = response_json['choices'][0].get('message', {}).get('content', '')

                # Parse the response for 'Yes' or 'No'
                if re.search(r'\bYes\b', completion_text, re.IGNORECASE):
                    responses[i] = 1
                else:
                    responses[i] = -1  # 'No' or default to -1 if neither 'Yes' nor 'No' is found

                if verbose:
                    print(f'Edge: {edge}, Response: {completion_text}, Vote: {responses[i]}')
        else:
            responses[i] = -1  # or any other indicator for missing data

    return responses

