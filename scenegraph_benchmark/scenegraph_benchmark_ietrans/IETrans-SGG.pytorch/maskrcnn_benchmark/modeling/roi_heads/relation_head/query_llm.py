import torch
from tqdm import tqdm
import math
from collections import OrderedDict
import re
import random
import json
import os
import ast
import torchvision.transforms as transforms
import transformers
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration
import cv2
import base64

import openai
from openai import OpenAI
# Meta Llama API from Replicate
import replicate

# Define the ANSI escape sequence for purple
PURPLE = '\033[95m'
ENDC = '\033[0m'  # Reset to default color


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
    def __init__(self, top_k=20, max_cache_size=10000, llm_model='meta-llama-3-8b-instruct', use_api=True): #'gpt-3.5-turbo-instruct'
        self.cache_hits = 0
        self.top_k = top_k
        self.total_cache_queries = 0
        self.cache = EdgeCache(max_cache_size=max_cache_size)
        self.cache.put("", -1)  # Update cache access frequency

        # read object and relation labels from the dataset
        data_path = '/tmp/datasets/vg/50/VG-SGG-dicts-with-attri.json'
        with open(data_path, 'r') as file:
            data = json.load(file)
        self.idx_to_predicate = data['idx_to_predicate']    # 50
        self.idx_to_object = data['idx_to_label']   # 150

        self.llm_model = llm_model
        self.use_api = use_api

        if self.llm_model == 'meta-llama-3-8b-instruct':
            if self.use_api:
                with open("llama_key.txt", "r") as llama_key_file:
                    llama_key = llama_key_file.read()
                os.environ['REPLICATE_API_TOKEN'] = llama_key
            else:
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

                self.llama_tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id  # Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    # low_cpu_mem_usage=True,
                )
                # model.parallelize()  # This distributes the model across all available GPUs

                self.llama_terminators = [
                    self.llama_tokenizer.eos_token_id,
                    self.llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

        elif self.llm_model == 'gpt-3.5-turbo':
            with open("openai_key.txt", "r") as api_key_file:
                self.api_key = api_key_file.read()

        elif self.llm_model == 'llava-v1.6-vicuna-7b':
            self.top_k = 10
            if self.use_api:
                with open("llama_key.txt", "r") as replicate_key_file:
                    replicate_key = replicate_key_file.read()
                os.environ['REPLICATE_API_TOKEN'] = replicate_key
            else:
                self.llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf") #AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")
                self.llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf") #AutoProcessor.from_pretrained("liuhaotian/llava-v1.5-7b")

        else:
            raise ValueError('llm_model not recognized')

        with open('/tmp/datasets/vg_scene_graph_annot/all_gt_triplets_in_training.json', 'r') as f:
            self.all_gt_triplets_in_training = json.load(f)


    def process_image(self, image):
        # Determine the maximum size
        max_size = 256

        # Calculate the scale to maintain aspect ratio
        height, width = image.shape[1], image.shape[2]
        scale = min(max_size / height, max_size / width)

        # Calculate new dimensions
        new_height = int(height * scale)
        new_width = int(width * scale)

        # Resize transform
        resize_transform = transforms.Resize((new_height, new_width))

        # Apply the transform
        image = resize_transform(image.unsqueeze(0))[0, 0, :, :]  # Unsqueeze to add batch dimension

        if self.use_api:
            _, buffer = cv2.imencode('.jpg', image.cpu().numpy())
            image_data = base64.b64encode(buffer).decode('utf-8')
            image = f"data:application/octet-stream;base64,{image_data}"
        return image


    def query(self, rel_pair_idx, rel_labels, image=None, boxlist=None):
        """
        :param rel_pair_idx: subject and object labels, sorted by triplet confidence in descending order. torch tensor of size batch_size, 2
        :param rel_labels: relation labels, sorted by triplet confidence in descending order. torch tensor of size batch_size
        :return: all_responses: +1 or -1 for each queried edge. torch tensor of size batch_size
        """
        # convert ids to strings for each triplet
        batched_edges = []
        batch_size = rel_pair_idx.shape[0]
        skip_validation_indices = []
        for i in range(batch_size):
            if rel_pair_idx[i][0].item() == 0 or rel_pair_idx[i][1].item() == 0:
                edge = ""
            else:
                edge = self.idx_to_object[str(rel_pair_idx[i][0].item())] + ' ' + self.idx_to_predicate[str(rel_labels[i].item())] \
                       + ' ' + self.idx_to_object[str(rel_pair_idx[i][1].item())]

            if edge not in self.all_gt_triplets_in_training:
                batched_edges.append(edge)
            else:
                skip_validation_indices.append(i)

        # query
        if self.llm_model == 'gpt-3.5-turbo-instruct':
            all_responses = self.batch_query_openai_gpt_instruct(batched_edges)
        elif self.llm_model == 'gpt-3.5-turbo':
            all_responses = self.batch_query_openai_gpt(batched_edges)
            # all_responses = []
            # for edge in batched_edges:
            #     all_responses.append(self.query_openai_gpt(edge))
        elif self.llm_model == 'meta-llama-3-8b-instruct':
            all_responses = []
            for edge in batched_edges:
                all_responses.append(self.query_llama(edge))
        elif self.llm_model == 'llava-v1.6-vicuna-7b':
            image = self.process_image(image)
            all_responses = []
            for edge in batched_edges:
                all_responses.append(self.query_llava(edge, image))
        else:
            raise ValueError('llm_model not recognized')

        try:
            # fill in the skipped indices
            for i in range(batch_size):
                if i in skip_validation_indices:
                    all_responses.insert(i, 1)
        except:
            all_responses = [1] * batch_size

        all_responses = torch.tensor(all_responses)

        return all_responses


    def query_llama(self, edge, verbose=False):
        # Prepare multiple variations of each prompt
        prompt_variations = [
            "Is the relation '{}' generally make sense or a trivially true fact? Answer only 'Yes' or 'No'.",
            "Could there be either a {} or a {}s? Answer only Yes or No.",
            "Regardless of whether it is basic or redundant, is the relation '{}' impossible in the physical world? Answer only 'Yes' or 'No'."
            # "Is the relation {} impossible in real world? Answer only 'Yes' or 'No'."
        ]

        # For each predicted edge, create multiple prompts
        yes_votes = 0
        no_votes = 0
        for j, variation in enumerate(prompt_variations):
            if j == 1:
                prompt = variation.format(edge, edge)
            else:
                prompt = variation.format(edge)

            if self.use_api:
                response = ""
                for event in replicate.stream(
                        "meta/meta-llama-3-8b-instruct",
                        input={
                            "prompt": prompt,
                            "max_length": 128,
                            "max_new_tokens": 64
                        },
                ):
                    response += str(event)
            else:
                messages = [
                    {"role": "user", "content": prompt},
                ]

                input_ids = self.llama_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.llama_model.device)

                attention_mask = input_ids.ne(self.llama_tokenizer.pad_token_id).int()

                outputs = self.llama_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    eos_token_id=self.llama_terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=self.llama_tokenizer.eos_token_id
                )
                response = outputs[0][input_ids.shape[-1]:]
                response = self.llama_tokenizer.decode(response, skip_special_tokens=True)

            # print(f"{PURPLE}prompt{ENDC}", prompt, f"{PURPLE}response{ENDC}", response)

            # if j == 2 or j == 3:  # For the last two questions, we reverse the logic
            if j == 2:
                if re.search(r'Yes', response):
                    no_votes += 1
                elif re.search(r'No', response):
                    yes_votes += 1
                else:
                    no_votes += 1
            else:
                if re.search(r'Yes', response):
                    # if j == 0:
                    #     yes_votes += 2
                    # else:
                    yes_votes += 1
                else:
                    # if j == 0:
                    #     no_votes += 2
                    # else:
                    no_votes += 1

        # if yes_votes > no_votes:
        if yes_votes >= 1:
            if verbose:
                print(f'predicted_edge {edge} [MAJORITY YES] {yes_votes} Yes votes vs {no_votes} No votes')
            score = 1
        else:
            if verbose:
                print(f'predicted_edge {edge} [MAJORITY NO] {no_votes} No votes vs {yes_votes} Yes votes')
            score = -1
        return score


    def query_llava(self, edge, image, verbose=False):
        if self.use_api:
            try:
                input = {
                    "image": image,
                    "prompt": "In this image, can you find a '{}'? Answer only 'Yes' or 'No'.".format(edge)
                }

                response = replicate.run(
                    "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
                    input=input
                )
                response = "".join(response)
            except:
                return 1

        else:
            prompt = "USER: <image>\nIn this image, can you find a '{}'? Answer only 'Yes' or 'No'. ASSISTANT:".format(edge)
            inputs = self.llava_processor(text=prompt, images=image, return_tensors="pt")

            # Generate
            generate_ids = self.llava_model.generate(**inputs, max_new_tokens=64)
            response = self.llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if verbose:
            print(f"predicted_edge {edge} response {response}")

        if re.search(r'Yes', response):
            score = 1
        else:
            score = -1

        return score


    def query_openai_gpt(self, edge, verbose=False):
        # Prepare multiple variations of each prompt
        prompt_variations = [
            "Is the relation '{}' generally make sense or a trivially true fact? Answer only 'Yes' or 'No'.",
            "Could there be either a {} or a {}s? Answer only Yes or No.",
            "Regardless of whether it is basic or redundant, is the relation '{}' impossible in the physical world? Answer only 'Yes' or 'No'."
        ]

        # For each predicted edge, create multiple prompts
        yes_votes = 0
        no_votes = 0
        for j, variation in enumerate(prompt_variations):
            if j == 1:
                prompt = variation.format(edge, edge)
            else:
                prompt = variation.format(edge)

            message = [
                {"role": "user", "content": prompt},
            ]

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=message,
                max_tokens=128
            )
            response = response.choices[0].message.content

            if j == 2:  # For the last two questions, we reverse the logic
                if re.search(r'Yes', response):
                    no_votes += 1
                elif re.search(r'No', response):
                    yes_votes += 1
                else:
                    no_votes += 1
            else:
                if re.search(r'Yes', response):
                    # if j == 0:
                    #     yes_votes += 2
                    # else:
                    yes_votes += 1
                else:
                    # if j == 0:
                    #     no_votes += 2
                    # else:
                    no_votes += 1

        # if yes_votes > no_votes:
        if yes_votes >= 1:
            if verbose:
                print(f'predicted_edge {edge} [MAJORITY YES] {yes_votes} Yes votes vs {no_votes} No votes')
            score = 1
        else:
            if verbose:
                print(f'predicted_edge {edge} [MAJORITY NO] {no_votes} No votes vs {yes_votes} Yes votes')
            score = -1
        return score


    def batch_query_openai_gpt(self, predicted_edges, verbose=False):
        # Prepare multiple variations of each prompt
        message = [
            # {"role": "system", "content": "Your task is to filter out invalid triplets that violate common sense and are impossible in the physical world. "
            #                               "Given the following list, check if each one generally make sense, is a trivial fact, or is possible in the physical world. "
            #                               "List each triplet, and provide your answer with a single 'Yes' or 'No' for it."},
            {"role": "system", "content": "Your task is to filter out invalid triplets. "
                                          "Given the following list, check each triplet one by one and explain. "
                                          "If it violate the common sense or is impossible in the physical world, say 'No'. Otherwise, say 'Yes'."},
            {"role": "user", "content": str(predicted_edges)},
        ]

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=message,
            max_tokens=128
        )
        response = response.choices[0].message.content
        if verbose:
            print('predicted_edges', predicted_edges)
            print('before', response)

        message = [
            {"role": "assistant", "content": response},
            {"role": "user", "content": "Summarize your answers as a Python list of 'Yes' and 'No' only, such as ['Yes', 'Yes', 'No', ..., 'Yes']. "
                                        "The length must be equal to the number of triplets mentioned. Do not include any other words."},
        ]

        try:
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=message,
                max_tokens=128
            )
            response = response.choices[0].message.content
            if verbose:
                print('After', response)

        # find the substring that represents the list in case the LLM outputs more words
            start_index = response.find("[")
            end_index = response.find("]") + 1
            response = response[start_index:end_index]

            response = ast.literal_eval(response)
            response = [1 if item == 'Yes' else -1 for item in response]
            if verbose:
                print('response', response)
        except:
            response = [1] * len(predicted_edges)

        if len(response) != len(predicted_edges):
            response = [1] * len(predicted_edges)

        return response


    def batch_query_openai_gpt_instruct(self, predicted_edges, batch_size=4):
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
        openai.api_key_path = '/pool/bwjiang/scene_graph/openai_key.txt'  # Path to your OpenAI API key
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

            # if yes_votes > no_votes:
            if yes_votes >= 2:
                if verbose:
                    print(f'predicted_edge {edge} [MAJORITY YES] {yes_votes} Yes votes vs {no_votes} No votes')
                responses[i] = 1
            else:
                # print(f'predicted_edge {edge} [MAJORITY NO] {no_votes} No votes vs {yes_votes} Yes votes')
                if verbose:
                    print(f'predicted_edge {edge} [MAJORITY NO] {no_votes} No votes vs {yes_votes} Yes votes')
                responses[i] = -1
        return responses
