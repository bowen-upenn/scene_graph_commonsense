import numpy as np
import torch
import torchvision
import yaml
import os
import cv2
from transformers import AutoProcessor
import torch.multiprocessing as mp
import transformers
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

from evaluate import inference


def image_captioning(device, world_size, args, test_dataset):
    assert world_size == 1

    # load prediction results from scene graph
    sgg_results = inference(device, args, test_dataset, top_k=5, file_idx=0) #2
    image_path = sgg_results['image_path'][0]
    top_k_predictions = sgg_results['top_k_predictions'][0]

    # load the pretrained GIT visual question answering model
    # processor = AutoProcessor.from_pretrained("microsoft/git-large-textvqa")
    # model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-textvqa").to(device)
    processor = AutoProcessor.from_pretrained("microsoft/git-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-large").to(device)

    # prepare GIT model inputs
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # convert the relationships to a formatted string
    prompt = ', '.join(top_k_predictions)
    prompt = f"an image of {prompt}, and"
    print('prompt:', prompt)

    input_ids = processor(text=prompt, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    generated_ids_w_prompt = model.generate(pixel_values=pixel_values.to(device), input_ids=input_ids.to(device), max_length=200)
    print('image caption with prompt:', processor.batch_decode(generated_ids_w_prompt, skip_special_tokens=True))

    generated_ids_wo_prompt = model.generate(pixel_values=pixel_values.to(device), max_length=200)
    print('image caption without prompt:', processor.batch_decode(generated_ids_wo_prompt, skip_special_tokens=True))

    # save the PIL image as a PNG file
    image = Image.fromarray(image, 'RGB')
    image.save('image.png')
