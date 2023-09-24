# reference:
# https://github.com/ronghanghu/seg_every_thing
# https://github.com/danfeiX/scene-graph-TF-release

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import yaml
import numpy as np
import h5py
from collections import Counter
from dataset import *
from dataset_utils import *


# load hyper-parameters
try:
    with open ('config.yaml', 'r') as file:
        args = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config file')

N_class = args['models']['num_classes']  # keep the top 150 classes
N_relation = args['models']['num_relations']  # keep the top 50 relations
raw_data_dir = args['dataset']['raw_annot_dir']
output_dir = raw_data_dir

# Ensure an identical train-test split with the pretrained detr backbone and other works
# using VG-SGG-with-attri.h5 from https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md
roi_h5 = h5py.File('/tmp/datasets/vg/VG-SGG-with-attri.h5', 'r')
data_split = roi_h5['split'][:]
all_training_idx = np.where(data_split == 0)[0]
all_testing_idx = np.where(data_split == 2)[0]
assert len(all_training_idx) == 75651 and len(all_testing_idx) == 32422

# ---------------------------------------------------------------------------- #
# Load raw VG annotations and collect top-frequent synsets
# ---------------------------------------------------------------------------- #

with open(raw_data_dir + 'image_data.json') as f:
    raw_img_data = json.load(f)
with open(raw_data_dir + 'objects.json') as f:
    raw_obj_data = json.load(f)
with open(raw_data_dir + 'relationships.json') as f:
    raw_relation_data = json.load(f)

# ---------------------------------------------------------------------------- #
# Clean raw dataset
# ---------------------------------------------------------------------------- #

# same pre-processing as in https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/vg_to_roidb.py
sync_objects(raw_obj_data, raw_relation_data)
obj_rel_cross_check(raw_obj_data, raw_relation_data)

obj_alias_dict, obj_vocab_list = make_alias_dict(args['dataset']['object_alias'])
pred_alias_dict, pred_vocab_list = make_alias_dict(args['dataset']['predicate_alias'])
preprocess_object_labels(raw_obj_data, alias_dict=obj_alias_dict)
preprocess_predicates(raw_relation_data, alias_dict=pred_alias_dict)

images_id2area = {img['image_id']: img['width'] * img['height'] for img in raw_img_data}
filter_object_boxes(raw_obj_data, images_id2area)

merge_duplicate_boxes(raw_obj_data)

# ---------------------------------------------------------------------------- #
# collect top-frequent synsets
# ---------------------------------------------------------------------------- #

# collect top frequent object synsets
obj_list = make_list(args['dataset']['object_list'])
all_synsets = [
    name for img in raw_obj_data
    for obj in img['objects'] for name in obj['names'] if name in obj_list]
synset_counter = Counter(all_synsets)
top_synsets_obj = [
    synset for synset, _ in synset_counter.most_common(N_class)]
print('top_synsets_obj', top_synsets_obj)

# collect top frequent relationship synsets
all_synsets = [
    synset for img in raw_relation_data
    for obj in img['relationships'] for synset in obj['predicate']]
synset_counter = Counter(all_synsets)
top_synsets_relation = [
    synset for synset, _ in synset_counter.most_common(N_relation)]
print('top_synsets_relation', top_synsets_relation)

# ---------------------------------------------------------------------------- #
# build "image"
# ---------------------------------------------------------------------------- #

corrupted_ims = [1592, 1722, 4616, 4617]
images = [
    {'id': img['image_id'],
     'width': img['width'],
     'height': img['height'],
     'file_name': img['url'].replace('https://cs.stanford.edu/people/rak248/', ''),
     'coco_id': img['coco_id']}
    for img in raw_img_data if img['image_id'] not in corrupted_ims]

# ---------------------------------------------------------------------------- #
# build raw "categories"
# ---------------------------------------------------------------------------- #

categories = [
    {'id': n, 'name': synset} for n, synset in enumerate(top_synsets_obj)]
synset2cid = {c['name']: c['id'] for c in categories}

sub2super_dict = preprocess_super_class(synset2cid, args['dataset']['object_types'])
torch.save(sub2super_dict, args['dataset']['sub2super_cat_dict'])

relationships = [
    {'id': n, 'name': synset} for n, synset in enumerate(top_synsets_relation)]
synset2rid = {c['name']: c['id'] for c in relationships}
# print('synset2rid', synset2rid)

# ---------------------------------------------------------------------------- #
# build "instances"
# ---------------------------------------------------------------------------- #

instances = []
skip_count_1, skip_count_2, skip_count_3, skip_count_4 = 0, 0, 0, 0
ave_num_obj = []
for idx, img in enumerate(raw_obj_data):
    if img['image_id'] in corrupted_ims:
        continue
    image_area = images_id2area[img['image_id']]

    num_objs = 0
    for obj in img['objects']:
        synsets = obj['names']
        if len(synsets) == 0:
            skip_count_1 += 1
        elif len(synsets) > 1:
            skip_count_2 += 1
        elif synsets[0] not in synset2cid:
            skip_count_3 += 1
        else:
            bbox = [obj['x'], obj['y'], obj['x']+obj['w'], obj['y']+obj['h']]
            area = obj['w'] * obj['h']
            if area <= image_area * args['dataset']['area_frac_thresh']:
                skip_count_4 += 1
                continue
            cid = synset2cid[synsets[0]]
            scid = sub2super_dict[cid]
            ann = {'id': obj['object_id'],
                   'image_id': img['image_id'],
                   'category_id': cid,
                   'super_category_id': scid,
                   'bbox': bbox,
                   'area': area}
            instances.append(ann)
            num_objs += 1

    ave_num_obj.append(num_objs)
ave_num_obj = np.mean(ave_num_obj)

print('average num of obj per image:', ave_num_obj, 'skip_count:', skip_count_1, skip_count_2, skip_count_3, skip_count_4)
print('Done building instance annotations.')

# ---------------------------------------------------------------------------- #
# build "annotations" for relationships
# ---------------------------------------------------------------------------- #

annotations = []
num_relation_per_pair = []
skip_count_1, skip_count_2, skip_count_3 = 0, 0, 0
for img in raw_relation_data:
    for pair in img['relationships']:
        synsets_relation = pair['predicate']
        synsets_obj1 = pair['subject']['names']
        synsets_obj2 = pair['object']['names']

        if len(synsets_relation) == 0 or len(synsets_obj1) == 0 or len(synsets_obj2) == 0:
            skip_count_1 += 1
        elif len(synsets_obj1) > 1 or len(synsets_obj2) > 1:
            skip_count_2 += 1
        elif (synsets_relation[0] not in synset2rid) or (synsets_obj1[0] not in synset2cid) or (synsets_obj2[0] not in synset2cid):
            skip_count_3 += 1
        else:
            rid = synset2rid[synsets_relation[0]]
            oid1 = pair['subject']['object_id']
            oid2 = pair['object']['object_id']
            cid1 = synset2cid[synsets_obj1[0]]
            cid2 = synset2cid[synsets_obj2[0]]
            scid1 = sub2super_dict[cid1]
            scid2 = sub2super_dict[cid2]

            ann = {'image_id': img['image_id'],
                   'relation_id': rid,
                   'subject_id': oid1,
                   'object_id': oid2,
                   'category1': cid1,
                   'category2': cid2,
                   'super_category1': scid1,
                   'super_category2': scid2}
            annotations.append(ann)
print('Done building relationship annotations.', len(annotations), 'skip_count:', skip_count_1, skip_count_2, skip_count_3)

# ---------------------------------------------------------------------------- #
# Split into train and test
# ---------------------------------------------------------------------------- #

# Save the dataset splits
images_train = [images[i] for i in all_training_idx]
images_test = [images[i] for i in all_testing_idx]
print('len(images_train)', len(images_train), 'len(images_test)', len(images_test))
assert len(images_train) == 75651 and len(images_test) == 32422

annotations_train = [annotations[i] for i in all_training_idx]
annotations_test = [annotations[i] for i in all_testing_idx]
instances_train = [instances[i] for i in all_training_idx]
instances_test = [instances[i] for i in all_testing_idx]

# ---------------------------------------------------------------------------- #
# Save to json file
# ---------------------------------------------------------------------------- #

dataset_vg3k_train = {
    'images': images_train,
    'annotations': annotations_train,
    'categories': categories,
    'instances': instances_train,
    'relationships': relationships}
dataset_vg3k_test = {
    'images': images_test,
    'annotations': annotations_test,
    'categories': categories,
    'instances': instances_test,
    'relationships': relationships}

with open(output_dir + 'instances_vg_train.json', 'w') as f:
    json.dump(dataset_vg3k_train, f)
with open(output_dir + 'instances_vg_test.json', 'w') as f:
    json.dump(dataset_vg3k_test, f)
print('Done saving training and testing datasets.')


# top_synsets_obj ['tree', 'man', 'window', 'shirt', 'building', 'person', 'sign', 'leg', 'head', 'pole', 'table', 'woman', 'hair',
# 'hand', 'car', 'door', 'leaf', 'light', 'pant', 'fence', 'ear', 'shoe', 'chair', 'people', 'plate', 'arm', 'glass', 'jacket', 'street',
# 'sidewalk', 'snow', 'tail', 'face', 'wheel', 'handle', 'flower', 'hat', 'rock', 'boy', 'tile', 'short', 'bag', 'roof', 'letter',
# 'girl', 'umbrella', 'helmet', 'bottle', 'branch', 'tire', 'plant', 'train', 'track', 'nose', 'boat', 'post', 'bench', 'shelf', 'wave',
# 'box', 'food', 'pillow', 'jean', 'bus', 'bowl', 'eye', 'trunk', 'horse', 'clock', 'counter', 'neck', 'elephant', 'giraffe', 'mountain',
# 'board', 'house', 'cabinet', 'banana', 'paper', 'hill', 'logo', 'dog', 'wing', 'book', 'bike', 'coat', 'seat', 'truck', 'glove', 'zebra',
# 'bird', 'cup', 'plane', 'cap', 'lamp', 'motorcycle', 'cow', 'skateboard', 'wire', 'surfboard', 'beach', 'mouth', 'sheep', 'kite', 'sink',
# 'cat', 'pizza', 'bed', 'animal', 'ski', 'curtain', 'bear', 'sock', 'player', 'flag', 'finger', 'windshield', 'towel', 'desk', 'number',
# 'railing', 'lady', 'stand', 'vehicle', 'child', 'boot', 'tower', 'basket', 'laptop', 'engine', 'vase', 'toilet', 'drawer', 'racket',
# 'tie', 'pot', 'paw', 'airplane', 'fork', 'screen', 'room', 'guy', 'orange', 'phone', 'fruit', 'vegetable', 'sneaker', 'skier', 'kid', 'men']
#
# top_synsets_relation ['on', 'has', 'in', 'of', 'wearing', 'near', 'with', 'above', 'holding', 'behind', 'under', 'sitting on',
# 'wears', 'standing on', 'in front of', 'attached to', 'at', 'hanging from', 'over', 'for', 'riding', 'carrying', 'eating',
# 'walking on', 'playing', 'covering', 'laying on', 'along', 'watching', 'and', 'between', 'belonging to', 'painted on', 'against',
# 'looking at', 'from', 'parked on', 'to', 'made of', 'covered in', 'mounted on', 'says', 'part of', 'across', 'flying in', 'using',
# 'on back of', 'lying on', 'growing on', 'walking in']
