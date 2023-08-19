import numpy as np
import torch
import torchvision
import yaml
import os
from transformers import AutoProcessor
import torch.multiprocessing as mp

from evaluate import inference


def image_captioning(device, world_size, args, test_dataset):
    assert world_size == 1

    results = inference(device, args, test_dataset, file_idx=0)
    print('results', results)

    # load the pretrained GIT model
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)

    for idx, data in enumerate(train_loader):
        batch, negative_attributes = data

        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        # print('decoded input', processor.decode(input_ids[0]))

        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)

        loss = outputs.loss
        print("Loss:", loss.item())

        generated_ids = model.module.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print('generated_caption', generated_caption)
        break