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
from tqdm import tqdm
from collections import Counter, OrderedDict
from dataset import *

# load hyper-parameters
try:
    with open ('config.yaml', 'r') as file:
        args = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config file')

N_class = args['models']['num_classes']  # keep the top 150 classes
raw_data_dir = args['dataset']['raw_annot_dir']
output_dir = raw_data_dir

# ---------------------------------------------------------------------------- #
# Load raw VG annotations and collect top-frequent synsets
# ---------------------------------------------------------------------------- #

with open(raw_data_dir + 'image_data.json') as f:
    raw_img_data = json.load(f)
with open(raw_data_dir + 'objects.json') as f:
    raw_obj_data = json.load(f)
assert len(raw_img_data) == len(raw_obj_data)

# ---------------------------------------------------------------------------- #
# Clean raw dataset
# ---------------------------------------------------------------------------- #

obj_alias_dict, obj_vocab_list = make_alias_dict(args['dataset']['object_alias'])
preprocess_object_labels(raw_obj_data, alias_dict=obj_alias_dict)

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

# ---------------------------------------------------------------------------- #
# build "image"
# ---------------------------------------------------------------------------- #

corrupted_ims = [1592, 1722, 4616, 4617]
images = OrderedDict()
for img in tqdm(raw_img_data):
    if img['image_id'] not in corrupted_ims:
        image = cv2.imread(img['url'].replace('https://cs.stanford.edu/people/rak248/', '/tmp/datasets/vg/images/'))

        images[img['image_id']] = {'id': img['image_id'],
                                   'width': image.shape[1],     # actual image size may differ from the provided annotation
                                   'height': image.shape[0],
                                   'file_name': img['url'].replace('https://cs.stanford.edu/people/rak248/', '/tmp/datasets/vg/images/'),
                                   'coco_id': img['coco_id']}

# ---------------------------------------------------------------------------- #
# build raw "categories"
# ---------------------------------------------------------------------------- #

categories = [
    {'id': n, 'name': synset} for n, synset in enumerate(top_synsets_obj)]
synset2cid = {c['name']: c['id'] for c in categories}

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
    annotations = []
    for obj in img['objects']:
        synsets = obj['names']
        if len(synsets) == 0:
            skip_count_1 += 1
        elif len(synsets) > 1:
            skip_count_2 += 1
        elif synsets[0] not in synset2cid:
            skip_count_3 += 1
        else:
            bbox = [obj['x'], obj['y'], obj['w'], obj['h']]
            area = obj['w'] * obj['h']
            if area <= image_area * args['dataset']['area_frac_thresh']:
                skip_count_4 += 1
                continue
            cid = synset2cid[synsets[0]]
            ann = {'category_id': cid,
                   'bbox': bbox,
                   'bbox_mode': 1}      # XYWH_ABS
            annotations.append(ann)
            num_objs += 1

    instance = {'file_name': images[img['image_id']]['file_name'],
                'height': images[img['image_id']]['height'],
                'width': images[img['image_id']]['width'],
                'image_id': img['image_id'],
                'annotations': annotations}

    ave_num_obj.append(num_objs)
    instances.append(instance)

ave_num_obj = np.mean(ave_num_obj)

print('average num of obj per image:', ave_num_obj, 'skip_count:', skip_count_1, skip_count_2, skip_count_3, skip_count_4)
print('Done building instance annotations.')

# ---------------------------------------------------------------------------- #
# Split into train and test
# ---------------------------------------------------------------------------- #

instances_traintest = [img for img in instances]
total_num = len(instances_traintest)
test_begin_idx = int(args['dataset']['train_test_split'] * total_num)

instances_test = instances_traintest[test_begin_idx:]
instances_train = instances_traintest[:test_begin_idx]

print('number of total, train, test images: {}, {}, {}'.format(
    total_num, len(instances_train), len(instances_test)))

# ---------------------------------------------------------------------------- #
# Save to json file
# ---------------------------------------------------------------------------- #

with open(output_dir + 'instances_vg_train_coco.json', 'w') as f:
    json.dump(instances_train, f)
with open(output_dir + 'instances_vg_test_coco.json', 'w') as f:
    json.dump(instances_test, f)
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
