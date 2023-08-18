import os
import numpy as np
import torch
import json
from PIL import Image
import string
import tqdm
import torchvision
from torchvision import transforms
from collections import Counter
from utils import *
from dataset_utils import *
import cv2
import random
from dataset_utils import TwoCropTransform


class PrepareVisualGenomeDataset(torch.utils.data.Dataset):
    def __init__(self, annotations):
        with open(annotations) as f:
            self.annotations = json.load(f)

    def __getitem__(self, idx):
        return None

    def __len__(self):
        return len(self.annotations['images'])


class VisualGenomeDataset(torch.utils.data.Dataset):
    def __init__(self, args, device, annotations):
        self.args = args
        self.device = device
        self.image_dir = self.args['dataset']['image_dir']
        self.annot_dir = self.args['dataset']['annot_dir']
        self.subset_indices = None
        with open(annotations) as f:
            self.annotations = json.load(f)
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=600, max_size=1000, antialias=True)])
        self.image_transform_to_tensor = transforms.ToTensor()
        self.image_transform_s = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((self.args['models']['image_size'], self.args['models']['image_size']), antialias=True)])
        self.image_transform_s_jitter = transforms.Compose([transforms.ToTensor(),
                                                            transforms.RandomApply([
                                                                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                                            ], p=0.8),
                                                            transforms.Resize((self.args['models']['image_size'], self.args['models']['image_size']), antialias=True)])
        self.image_transform_contrastive = TwoCropTransform(self.image_transform_s, self.image_transform_s_jitter)
        # self.image_norm = transforms.Compose([transforms.Normalize((103.530, 116.280, 123.675), (1.0, 1.0, 1.0))])
        self.image_norm = transforms.Compose([transforms.Normalize((102.9801, 115.9465, 122.7717), (1.0, 1.0, 1.0))])

    def __getitem__(self, idx):
        """
        Dataloader Outputs:
            image: an image in the Visual Genome dataset (to predict bounding boxes and labels in DETR-101)
            image_s: an image in the Visual Genome dataset resized to a square shape (to generate a uniform-sized image features)
            image_depth: the estimated image depth map
            categories: categories of all objects in the image
            super_categories: super-categories of all objects in the image
            masks: squared masks of all objects in the image
            bbox: bounding boxes of all objects in the image
            relationships: all target relationships in the image
            subj_or_obj: the edge directions of all target relationships in the image
        """
        annot_name = self.annotations['images'][idx]['file_name'][:-4] + '_annotations.pkl'
        annot_path = os.path.join(self.annot_dir, annot_name)
        try:
            curr_annot = torch.load(annot_path)
        except:
            return None

        image_path = os.path.join(self.image_dir, self.annotations['images'][idx]['file_name'])
        images = cv2.imread(image_path)

        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        images = 255 * self.image_transform_contrastive(images)
        images, images_aug = images[0], images[1]
        images = self.image_norm(images)  # squared size that unifies the size of feature maps
        images_aug = self.image_norm(images_aug)

        if self.args['training']['run_mode'] == 'eval' and self.args['training']['eval_mode'] != 'pc':
            del images_aug
            image2 = Image.open(image_path).convert('RGB')  # keep original shape ratio, not reshaped to square
            image2 = 255 * self.image_transform(image2)[[2, 1, 0]]  # BGR
            image2 = self.image_norm(image2)

        if self.args['models']['use_depth']:
            image_depth = curr_annot['image_depth']
        else:
            image_depth = torch.zeros(1, self.args['models']['feature_size'], self.args['models']['feature_size'])    # ablation no depth map
        categories = curr_annot['categories']
        super_categories = curr_annot['super_categories']
        # total in train: 60548, >20: 2651, >30: 209, >40: 23, >50: 4. Don't let rarely long data dominate the computation power.
        if categories.shape[0] <= 1 or categories.shape[0] > 20:
            return None
        bbox = curr_annot['bbox']   # x_min, x_max, y_min, y_max

        subj_or_obj = curr_annot['subj_or_obj']
        relationships = curr_annot['relationships']
        relationships_reordered = []
        rel_reorder_dict = relation_class_freq2scat()
        for rel in relationships:
            rel[rel == 12] = 4      # wearing <- wears
            relationships_reordered.append(rel_reorder_dict[rel])
        relationships = relationships_reordered

        if self.args['training']['run_mode'] == 'eval' and self.args['training']['eval_mode'] != 'pc':
            return images, image2, image_depth, categories, super_categories, bbox, relationships, subj_or_obj
        else:
            return images, images_aug, image_depth, categories, super_categories, bbox, relationships, subj_or_obj

    def __len__(self):
        return len(self.annotations['images'])


class VisualGenomeDatasetEfficient(torch.utils.data.Dataset):
    def __init__(self, args, device, annotations):
        self.args = args
        self.device = device
        self.image_dir = self.args['dataset']['image_dir']
        self.annot_dir = self.args['dataset']['annot_dir']
        self.subset_indices = None
        with open(annotations) as f:
            self.annotations = json.load(f)
        self.image_transform_to_tensor = transforms.ToTensor()
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((self.args['models']['image_size'], self.args['models']['image_size']))])

        self.image_transform_jitter = transforms.Compose([transforms.ToTensor(),
                                                   transforms.RandomApply([
                                                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                                   ], p=0.8),
                                                   transforms.Resize((self.args['models']['image_size'], self.args['models']['image_size']))])
        self.image_norm = transforms.Compose([transforms.Normalize((102.9801, 115.9465, 122.7717), (1.0, 1.0, 1.0))])

    def __getitem__(self, idx):
        """
        Dataloader Outputs:
            image: an image in the Visual Genome dataset (to predict bounding boxes and labels in DETR-101)
            image_s: an image in the Visual Genome dataset resized to a square shape (to generate a uniform-sized image features)
            image_depth: the estimated image depth map
            categories: categories of all objects in the image
            super_categories: super-categories of all objects in the image
            masks: squared masks of all objects in the image
            bbox: bounding boxes of all objects in the image
            relationships: all target relationships in the image
            subj_or_obj: the edge directions of all target relationships in the image
        """
        annot_name = self.annotations['images'][idx]['file_name'][:-4] + '_annotations.pkl'
        annot_path = os.path.join(self.annot_dir, annot_name)
        try:
            curr_annot = torch.load(annot_path)
        except:
            return None

        image_path = os.path.join(self.image_dir, self.annotations['images'][idx]['file_name'])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = 255 * self.image_transform(image)
        image = self.image_norm(image)  # original size that produce better bounding boxes

        if self.args['models']['use_depth']:
            image_depth = curr_annot['image_depth']
        else:
            image_depth = torch.zeros(1, self.args['models']['feature_size'], self.args['models']['feature_size'])    # ablation no depth map

        categories = curr_annot['categories']
        super_categories = curr_annot['super_categories']
        masks = curr_annot['masks']
        # total in train: 60548, >20: 2651, >30: 209, >40: 23, >50: 4. Don't let rarely long data dominate the computation power.
        if masks.shape[0] <= 1 or masks.shape[0] > 30: # 25
            return None
        bbox = curr_annot['bbox']   # x_min, x_max, y_min, y_max

        subj_or_obj = curr_annot['subj_or_obj']
        relationships = curr_annot['relationships']
        relationships_reordered = []
        rel_reorder_dict = relation_class_freq2scat()
        for rel in relationships:
            rel[rel == 12] = 4      # wearing <- wears
            relationships_reordered.append(rel_reorder_dict[rel])
        relationships = relationships_reordered

        triplets = []
        for i in range(len(categories)):
            for j in range(i):
                if subj_or_obj[i-1][j] == 1:
                    triplets.append([categories[i], super_categories[i], bbox[i], masks[i], relationships[i-1][j],
                                     categories[j], super_categories[j], bbox[j], masks[j]])
                elif subj_or_obj[i-1][j] == 0:
                    triplets.append([categories[j], super_categories[j], bbox[j], masks[j], relationships[i-1][j],
                                     categories[i], super_categories[i], bbox[i], masks[i]])
                else:
                    continue    # no relationship in the annotation

        if len(triplets) == 0:
            return None
        # print("triplets", triplets, "\n")

        return image, image_depth, triplets

    def __len__(self):
        return len(self.annotations['images'])


class PrepareOpenImageV6Dataset(torch.utils.data.Dataset):
    def __init__(self, args, annotations):
        self.image_dir = "../datasets/open_image_v6/images/"
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((args['models']['image_size'], args['models']['image_size']))])
        with open(annotations) as f:
            self.annotations = json.load(f)

    def __getitem__(self, idx):
        rel = self.annotations[idx]['rel']
        image_id = self.annotations[idx]['img_fn'] + '.jpg'
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        return rel, image, self.annotations[idx]['img_fn']

    def __len__(self):
        return len(self.annotations)


class OpenImageV6Dataset(torch.utils.data.Dataset):
    def __init__(self, args, device, annotations):
        self.args = args
        self.device = device
        self.image_dir = "../datasets/open_image_v6/images/"
        self.depth_dir = "../datasets/open_image_v6/image_depths/"
        with open(annotations) as f:
            self.annotations = json.load(f)
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=600, max_size=1000)])
        self.image_transform_s = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((self.args['models']['image_size'], self.args['models']['image_size']))])
        self.image_norm = transforms.Compose([transforms.Normalize((103.530, 116.280, 123.675), (1.0, 1.0, 1.0))])
        self.rel_super_dict = oiv6_reorder_by_super()

    def __getitem__(self, idx):
        # print('idx', idx, self.annotations[idx])
        image_id = self.annotations[idx]['img_fn']
        image_path = os.path.join(self.image_dir, image_id + '.jpg')

        image = Image.open(image_path).convert('RGB')
        h_img, w_img = self.annotations[idx]['img_size'][1], self.annotations[idx]['img_size'][0]

        image = 255 * self.image_transform(image)[[2, 1, 0]]  # BGR
        image = self.image_norm(image)  # original size that produce better bounding boxes
        image_s = Image.open(image_path).convert('RGB')
        image_s = 255 * self.image_transform_s(image_s)[[2, 1, 0]]  # BGR
        image_s = self.image_norm(image_s)  # squared size that unifies the size of feature maps

        if self.args['models']['use_depth']:
            image_depth = torch.load(self.depth_dir + image_id + '_depth.pt')
        else:
            image_depth = torch.zeros(1, self.args['models']['feature_size'], self.args['models']['feature_size'])

        categories = torch.tensor(self.annotations[idx]['det_labels'])
        if len(categories) <= 1 or len(categories) > 20: # 25
            return None

        bbox = []
        raw_bbox = self.annotations[idx]['bbox']    # x_min, y_min, x_max, y_max
        masks = torch.zeros(len(raw_bbox), self.args['models']['feature_size'], self.args['models']['feature_size'], dtype=torch.uint8)
        for i, b in enumerate(raw_bbox):
            box = resize_boxes(b, (h_img, w_img), (self.args['models']['feature_size'], self.args['models']['feature_size']))
            masks[i, box[0]:box[2], box[1]:box[3]] = 1
            bbox.append([box[0], box[2], box[1], box[3]])  # x_min, x_max, y_min, y_max
        bbox = torch.as_tensor(bbox)

        raw_relations = self.annotations[idx]['rel']
        relationships = []
        subj_or_obj = []
        for i in range(1, len(categories)):
            relationships.append(-1 * torch.ones(i, dtype=torch.int64))
            subj_or_obj.append(-1 * torch.ones(i, dtype=torch.float32))

        for triplet in raw_relations:
            # if curr is the subject
            if triplet[0] > triplet[1]:
                relationships[triplet[0]-1][triplet[1]] = self.rel_super_dict[triplet[2]]
                subj_or_obj[triplet[0]-1][triplet[1]] = 1
            # if curr is the object
            elif triplet[0] < triplet[1]:
                relationships[triplet[1]-1][triplet[0]] = self.rel_super_dict[triplet[2]]
                subj_or_obj[triplet[1]-1][triplet[0]] = 0

        return image, image_s, image_depth, categories, None, masks, bbox, relationships, subj_or_obj

    def __len__(self):
        return len(self.annotations)
