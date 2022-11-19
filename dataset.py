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
                                                   transforms.Resize(size=600, max_size=1000)])
        self.image_transform_s = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((self.args['models']['image_size'], self.args['models']['image_size']))])
        self.image_norm = transforms.Compose([transforms.Normalize((103.530, 116.280, 123.675), (1.0, 1.0, 1.0))])

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
        image = Image.open(image_path).convert('RGB')
        image = 255 * self.image_transform(image)[[2, 1, 0]]    # BGR
        image = self.image_norm(image)  # original size that produce better bounding boxes

        image_s = Image.open(image_path).convert('RGB')
        image_s = 255 * self.image_transform_s(image_s)[[2, 1, 0]]  # BGR
        image_s = self.image_norm(image_s)  # squared size that unifies the size of feature maps

        image_depth = curr_annot['image_depth']
        categories = curr_annot['categories']
        super_categories = curr_annot['super_categories']
        masks = curr_annot['masks']
        # total in train: 60548, >20: 2651, >30: 209, >40: 23, >50: 4. Don't let rarely long data dominate the computation power.
        if masks.shape[0] <= 1 or masks.shape[0] > 30: # 25
            return None
        bbox = curr_annot['bbox'] #/ 32

        subj_or_obj = curr_annot['subj_or_obj']
        relationships = curr_annot['relationships']
        relationships_reordered = []
        rel_reorder_dict = relation_class_freq2scat()
        for rel in relationships:
            rel[rel == 12] = 4      # wearing <- wears
            relationships_reordered.append(rel_reorder_dict[rel])
        relationships = relationships_reordered

        return image, image_s, image_depth, categories, super_categories, masks, bbox, relationships, subj_or_obj

    def __len__(self):
        return len(self.annotations['images'])


def prepare_data_offline(args, data_loader, device, annot, image_transform, depth_estimator, start=0):
    """
    This function organizes all information that VisualGenomeDataset __getitem__ function needs
    to provide images, depth maps, ground-truth object categories and relationships.
    An offline pre-process speeds avoids dealing with data preparations during the actual training process.
    """
    with open(annot) as f:
        annotations = json.load(f)
        annotations['annotations'] = np.array(annotations['annotations'])
        annotations['instances'] = np.array(annotations['instances'])

    # processed_annotations = {}
    curr_instance_start = 0
    curr_relations_start = 0
    for idx, _ in enumerate(tqdm(data_loader)):     # dataloader is only a placeholder who returns null every iter, need to gather all data right in this func
        idx += start
        '''
        load instances
        '''
        # find all instances belonging to the current image
        curr_instance = []
        flag = False
        gap = 0
        for i in range(curr_instance_start, len(annotations['instances'])):
            curr_instance_temp = annotations['instances'][i]['image_id']
            if curr_instance_temp == annotations['images'][idx]['id']:
                curr_instance.append(curr_instance_temp)
                flag = True
            elif flag:
                break
            else:
                gap += 1
        curr_instance = curr_instance_start + gap + np.nonzero(curr_instance)[0]  # indices of instances in curr image in annotations['instances']
        if i != len(annotations['instances']) - 1:
            curr_instance_start = i
        num_instances = len(annotations['instances'][curr_instance])
        # print('curr_instance', idx, curr_instance_start, curr_instance)

        '''
        load relationships
        '''
        # find all relationships belonging to the current image
        curr_relations = []
        flag = False
        gap = 0
        for i in range(curr_relations_start, len(annotations['annotations'])):
            curr_relation_temp = annotations['annotations'][i]['image_id']
            if curr_relation_temp == annotations['images'][idx]['id']:
                curr_relations.append(curr_relation_temp)
                flag = True
            elif flag:
                break
            else:
                gap += 1
        curr_relations = curr_relations_start + gap + np.nonzero(curr_relations)[0]  # indices of instances in curr image in annotations['instances']
        if i != len(annotations['annotations']) - 1:
            curr_relations_start = i
        num_relations = len(annotations['annotations'][curr_relations])
        # print('curr_relations', idx, curr_relations_start, curr_relations)

        if num_instances == 0 or num_relations == 0:
            continue

        '''
        load image depth map
        '''
        image_path = os.path.join(args['dataset']['image_dir'], annotations['images'][idx]['file_name'])
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        h_img, w_img = image.shape[:2]
        image = image_transform(image).view(1, 3, args['models']['image_size'], args['models']['image_size']).to(device)

        depth_estimator = depth_estimator.cuda()
        depth_estimator.eval()

        with torch.no_grad():
            image_depth = depth_estimator(image)  # size (1, 256, 256)

        h_fea, w_fea = args['models']['feature_size'], args['models']['feature_size']

        image_depth = torchvision.transforms.functional.resize(image_depth, args['models']['feature_size'])  # size (1, 64, 64)
        image_depth = image_depth / (torch.max(image_depth) - torch.min(image_depth))
        image_depth = image_depth.cpu()

        '''
        load bbox and categories of each instance in order
        '''
        areas = []
        for inst in curr_instance:
            areas.append(annotations['instances'][inst]['area'])
        area_sorted, sorted_idices = torch.sort(torch.as_tensor(areas), descending=True)

        masks = []
        bbox = []
        bbox_origin = []
        categories = []
        super_categories = []
        for area_index in sorted_idices:
            inst = curr_instance[area_index]

            box = annotations['instances'][inst]['bbox']
            bbox_origin.append([box[0], box[2], box[1], box[3]])
            box = resize_boxes(box, (h_img, w_img), (h_fea, w_fea))
            bbox.append([box[0], box[2], box[1], box[3]])

            mask = torch.zeros(h_fea, w_fea, dtype=torch.uint8)
            mask[box[0]:box[2], box[1]:box[3]] = 1
            masks.append(mask)

            category = annotations['instances'][inst]['category_id']
            categories.append(category)
            super_category = annotations['instances'][inst]['super_category_id']
            super_categories.append(torch.as_tensor(super_category, dtype=torch.int64))

        categories = torch.as_tensor(categories, dtype=torch.int64)
        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        bbox_origin = torch.as_tensor(bbox_origin, dtype=torch.float32)
        masks = torch.stack(masks)

        '''
        load relationships in order
        '''
        curr_instance_id = [annotations['instances'][curr_instance[area_index]]['id'] for area_index in sorted_idices]  # object_id of instances in curr image

        # prepare all relationships in order, iterate through all instances one by one, find its relationship to each of the previous instance
        relation_id = [annotations['annotations'][curr_relations][i]['relation_id'] for i in range(num_relations)]
        subject_id = [annotations['annotations'][curr_relations][i]['subject_id'] for i in range(num_relations)]
        object_id = [annotations['annotations'][curr_relations][i]['object_id'] for i in range(num_relations)]

        relationships = []
        subj_or_obj = []

        for i, inst_id in enumerate(curr_instance_id):
            relationships_temp = []
            subj_or_obj_temp = []  # 1 if curr is subject and prev is object, 0 if prev is subject and curr is object

            for j, prev_inst_id in enumerate(curr_instance_id[:i]):
                match1 = [(subject_id[k] == inst_id) and (object_id[k] == prev_inst_id) for k in range(num_relations)]
                match_idx1 = np.nonzero(match1)[0]
                match2 = [(object_id[k] == inst_id) and (subject_id[k] == prev_inst_id) for k in range(num_relations)]
                match_idx2 = np.nonzero(match2)[0]

                # there is only one relation per pair in the VG dataset
                if len(match_idx1) > 0:  # if curr is subject
                    relationships_temp.append(relation_id[match_idx1[0]])
                    subj_or_obj_temp.append(1)

                elif len(match_idx2) > 0:  # if curr is object
                    relationships_temp.append(relation_id[match_idx2[0]])
                    subj_or_obj_temp.append(0)

                else:  # len(match_idx1) == 0 and len(match_idx2) == 0, no relationship
                    relationships_temp.append(-1)
                    subj_or_obj_temp.append(-1)

            if len(relationships_temp) > 0:
                relationships.append(torch.as_tensor(relationships_temp, dtype=torch.int64))
                subj_or_obj.append(torch.as_tensor(subj_or_obj_temp, dtype=torch.float32))

        data_annot = {
            'image_depth': image_depth,
            'curr_instance': curr_instance,
            'num_relations': num_relations,
            'categories': categories,
            'super_categories': super_categories,
            'masks': masks,
            'bbox': bbox,
            'bbox_origin': bbox_origin,
            'relationships': relationships,
            'subj_or_obj': subj_or_obj}

        file_name_temp = annotations['images'][idx]['file_name'][:-4] + '_annotations.pkl'
        file_name = os.path.join(args['dataset']['annot_dir'], file_name_temp)
        torch.save(data_annot, file_name)


# all functions below are built from the open-source code:
# https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/vg_to_roidb.py
def merge_duplicate_boxes(raw_obj_data):
    def IoU(b1, b2):
        if b1[2] <= b2[0] or b1[3] <= b2[1] or b1[0] >= b2[2] or b1[1] >= b2[3]:
            return 0
        b1b2 = np.vstack([b1, b2])
        minc = np.min(b1b2, 0)
        maxc = np.max(b1b2, 0)
        union_area = (maxc[2] - minc[0]) * (maxc[3] - minc[1])
        int_area = (minc[2] - maxc[0]) * (minc[3] - maxc[1])
        return float(int_area) / float(union_area)

    def to_x1y1x2y2(obj):
        x1 = obj['x']
        y1 = obj['y']
        x2 = obj['x'] + obj['w']
        y2 = obj['y'] + obj['h']
        return np.array([x1, y1, x2, y2], dtype=np.int32)

    def inside(b1, b2):
        return b1[0] >= b2[0] and b1[1] >= b2[1] \
               and b1[2] <= b2[2] and b1[3] <= b2[3]

    def overlap(obj1, obj2):
        b1 = to_x1y1x2y2(obj1)
        b2 = to_x1y1x2y2(obj2)
        iou = IoU(b1, b2)
        if all(b1 == b2) or iou > 0.9:  # consider as the same box
            return 1
        elif (inside(b1, b2) or inside(b2, b1)) \
                and obj1['names'][0] == obj2['names'][0]:  # same object inside the other
            return 2
        elif iou > 0.6 and obj1['names'][0] == obj2['names'][0]:  # multiple overlapping same object
            return 3
        else:
            return 0  # no overlap

    num_merged = {1: 0, 2: 0, 3: 0}
    print('merging boxes..')
    for img in tqdm(raw_obj_data):
        # mark objects to be merged and save their ids
        objs = img['objects']
        num_obj = len(objs)
        for i in range(num_obj):
            if 'M_TYPE' in objs[i]:  # has been merged
                continue
            merged_objs = []  # circular refs, but fine
            for j in range(i + 1, num_obj):
                if 'M_TYPE' in objs[j]:  # has been merged
                    continue
                overlap_type = overlap(objs[i], objs[j])
                if overlap_type > 0:
                    objs[j]['M_TYPE'] = overlap_type
                    merged_objs.append(objs[j])
            objs[i]['mobjs'] = merged_objs

        # merge boxes
        filtered_objs = []
        merged_num_obj = 0
        for obj in objs:
            if 'M_TYPE' not in obj:
                ids = [obj['object_id']]
                dims = [to_x1y1x2y2(obj)]
                prominent_type = 1
                for mo in obj['mobjs']:
                    ids.append(mo['object_id'])
                    obj['names'].extend(mo['names'])
                    dims.append(to_x1y1x2y2(mo))
                    if mo['M_TYPE'] > prominent_type:
                        prominent_type = mo['M_TYPE']
                merged_num_obj += len(ids)
                obj['ids'] = ids
                mdims = np.zeros(4)
                if prominent_type > 1:  # use extreme
                    mdims[:2] = np.min(np.vstack(dims)[:, :2], 0)
                    mdims[2:] = np.max(np.vstack(dims)[:, 2:], 0)
                else:  # use mean
                    mdims = np.mean(np.vstack(dims), 0)
                obj['x'] = int(mdims[0])
                obj['y'] = int(mdims[1])
                obj['w'] = int(mdims[2] - mdims[0])
                obj['h'] = int(mdims[3] - mdims[1])

                num_merged[prominent_type] += len(obj['mobjs'])

                obj['mobjs'] = None
                obj['names'] = list(set(obj['names']))  # remove duplicates

                filtered_objs.append(obj)
            else:
                assert 'mobjs' not in obj

        img['objects'] = filtered_objs
        assert (merged_num_obj == num_obj)

    print('# merged boxes per merging type:')
    print(num_merged)


def sentence_preprocess(phrase):
    """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
    replacements = {
      '½': 'half',
      '—' : '-',
      '™': '',
      '¢': 'cent',
      'ç': 'c',
      'û': 'u',
      'é': 'e',
      '°': ' degree',
      'è': 'e',
      '…': '',
    }
    phrase = phrase.strip()
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    return phrase.lower().translate(str.maketrans('', '', string.punctuation))


def preprocess_object_labels(data, alias_dict={}):
    for img in data:
        for obj in img['objects']:
            obj['ids'] = [obj['object_id']]
            names = []
            for name in obj['names']:
                label = sentence_preprocess(name)
                if label in alias_dict:
                    label = alias_dict[label]
                names.append(label)
            obj['names'] = names


def preprocess_predicates(data, alias_dict={}):
    for img in data:
        for relation in img['relationships']:
            predicate = sentence_preprocess(relation['predicate'])
            if predicate in alias_dict:
                predicate = alias_dict[predicate]
            relation['predicate'] = [predicate]

            try:
                sub_name = sentence_preprocess(relation['subject']['name'])
            except:
                sub_name = sentence_preprocess(relation['subject']['names'][0])
            if sub_name in alias_dict:
                sub_name = alias_dict[sub_name]
            relation['subject']['names'] = [sub_name]

            try:
                obj_name = sentence_preprocess(relation['object']['name'])
            except:
                obj_name = sentence_preprocess(relation['object']['names'][0])
            if obj_name in alias_dict:
                obj_name = alias_dict[obj_name]
            relation['object']['names'] = [obj_name]


def make_alias_dict(dict_file):
    """create an alias dictionary from a file"""
    out_dict = {}
    vocab = []
    for line in open(dict_file, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
        vocab.append(alias_target)
    return out_dict, vocab


def make_list(list_file):
    """create a blacklist list from a file"""
    return [line.strip('\n').strip('\r') for line in open(list_file)]


def obj_rel_cross_check(raw_obj_data, raw_relation_data):
    """
    make sure all objects that are in relationship dataset
    are in object dataset
    """
    num_img = len(raw_obj_data)
    num_correct = 0
    total_rel = 0
    for i in range(num_img):
        assert(raw_obj_data[i]['image_id'] == raw_relation_data[i]['image_id'])
        objs = raw_obj_data[i]['objects']
        rels = raw_relation_data[i]['relationships']
        ids = [obj['object_id'] for obj in objs]
        for rel in rels:
            if rel['subject']['object_id'] in ids and rel['object']['object_id'] in ids:
                num_correct += 1
            total_rel += 1
    print('cross check: %i/%i relationship are correct' % (num_correct, total_rel))


def sync_objects(raw_obj_data, raw_relation_data):
    num_img = len(raw_obj_data)
    for i in range(num_img):
        assert(raw_obj_data[i]['image_id'] == raw_relation_data[i]['image_id'])
        objs = raw_obj_data[i]['objects']
        rels = raw_relation_data[i]['relationships']

        ids = [obj['object_id'] for obj in objs]
        for rel in rels:
            if rel['subject']['object_id'] not in ids:
                rel_obj = rel['subject']
                rel_obj['names'] = [rel_obj['name']]
                objs.append(rel_obj)
            if rel['object']['object_id'] not in ids:
                rel_obj = rel['object']
                rel_obj['names'] = [rel_obj['name']]
                objs.append(rel_obj)

        raw_obj_data[i]['objects'] = objs


def filter_object_boxes(raw_obj_data, images_id2area, area_frac_thresh=0.002):
    """
    filter boxes by a box area-image area ratio threshold
    """
    thresh_count = 0
    all_count = 0
    for i, img in enumerate(raw_obj_data):
        filtered_obj = []
        area = images_id2area[img['image_id']]
        for obj in img['objects']:
            if float(obj['h'] * obj['w']) > area * area_frac_thresh:
                filtered_obj.append(obj)
                thresh_count += 1
            all_count += 1
        img['objects'] = filtered_obj
    print('box threshod: keeping %i/%i boxes' % (thresh_count, all_count))


def extract_object_token(raw_obj_data, num_tokens, obj_list=[], verbose=True):
    """ Builds a set that contains the object names. Filters infrequent tokens. """
    token_counter = Counter()
    for img in raw_obj_data:
        for region in img['objects']:
            for name in region['names']:
                if not obj_list or name in obj_list:
                    token_counter.update([name])
    tokens = set()
    # pick top N tokens
    token_counter_return = {}
    for token, count in token_counter.most_common():
        tokens.add(token)
        token_counter_return[token] = count
        if len(tokens) == num_tokens:
            break
    if verbose:
        print(('Keeping %d / %d objects'
                  % (len(tokens), len(token_counter))))
    return tokens, token_counter_return


def object_super_class():
    return {'vehicle': 0, 'animal': 1, 'part': 2, 'person': 3, 'clothes': 4, 'food': 5, 'artifact': 6, 'location': 7, 'furniture': 8, 'flora': 9,
            'building': 10, 'table': 11, 'structure': 12, 'door': 13, 'perosn': 14, 'laptop': 15, 'phone': 16}


def object_super_class_int2str():
    return {0: 'vehicle', 1: 'animal', 2: 'part', 3: 'person', 4: 'clothes', 5: 'food', 6: 'artifact', 7: 'location', 8: 'furniture', 9: 'flora',
            10: 'building', 11: 'table', 12: 'structure', 13: 'door', 14: 'perosn', 15: 'laptop', 16: 'phone'}


def object_class_int2str():
    return {0: 'tree', 1: 'man', 2: 'window', 3: 'shirt', 4: 'building', 5: 'person', 6: 'sign', 7: 'leg', 8: 'head', 9: 'pole',
            10: 'table', 11: 'woman', 12: 'hair', 13: 'hand', 14: 'car', 15: 'door', 16: 'leaf', 17: 'light', 18: 'pant', 19: 'fence',
            20: 'ear', 21: 'shoe', 22: 'chair', 23: 'people', 24: 'plate', 25: 'arm', 26: 'glass', 27: 'jacket', 28: 'street', 29: 'sidewalk',
            30: 'snow', 31: 'tail', 32: 'face', 33: 'wheel', 34: 'handle', 35: 'flower', 36: 'hat', 37: 'rock', 38: 'boy', 39: 'tile',
            40: 'short', 41: 'bag', 42: 'roof', 43: 'letter', 44: 'girl', 45: 'umbrella', 46: 'helmet', 47: 'bottle', 48: 'branch', 49: 'tire',
            50: 'plant', 51: 'train', 52: 'track', 53: 'nose', 54: 'boat', 55: 'post', 56: 'bench', 57: 'shelf', 58: 'wave', 59: 'box',
            60: 'food', 61: 'pillow', 62: 'jean', 63: 'bus', 64: 'bowl', 65: 'eye', 66: 'trunk', 67: 'horse', 68: 'clock', 69: 'counter',
            70: 'neck', 71: 'elephant', 72: 'giraffe', 73: 'mountain', 74: 'board', 75: 'house', 76: 'cabinet', 77: 'banana', 78: 'paper', 79: 'hill',
            80: 'logo', 81: 'dog', 82: 'wing', 83: 'book', 84: 'bike', 85: 'coat', 86: 'seat', 87: 'truck', 88: 'glove', 89: 'zebra',
            90: 'bird', 91: 'cup', 92: 'plane', 93: 'cap', 94: 'lamp', 95: 'motorcycle', 96: 'cow', 97: 'skateboard', 98: 'wire', 99: 'surfboard',
            100: 'beach', 101: 'mouth', 102: 'sheep', 103: 'kite', 104: 'sink', 105: 'cat', 106: 'pizza', 107: 'bed', 108: 'animal', 109: 'ski',
            110: 'curtain', 111: 'bear', 112: 'sock', 113: 'player', 114: 'flag', 115: 'finger', 116: 'windshield', 117: 'towel', 118: 'desk', 119: 'number',
            120: 'railing', 121: 'lady', 122: 'stand', 123: 'vehicle', 124: 'child', 125: 'boot', 126: 'tower', 127: 'basket', 128: 'laptop', 129: 'engine',
            130: 'vase', 131: 'toilet', 132: 'drawer', 133: 'racket', 134: 'tie', 135: 'pot', 136: 'paw', 137: 'airplane', 138: 'fork', 139: 'screen',
            140: 'room', 141: 'guy', 142: 'orange', 143: 'phone', 144: 'fruit', 145: 'vegetable', 146: 'sneaker', 147: 'skier', 148: 'kid', 149: 'men'}


# In our dataset, we arrange object labels by their frequency.
# However, in the dataset used by the pretrained DETR-101 backbone, object labels are arranges by their alphabets
def object_class_alp2fre():
    return {0: 137, 1: 108, 2: 25, 3: 41, 4: 77, 5: 127, 6: 100, 7: 111, 8: 107, 9: 56, 10: 84, 11: 90, 12: 74, 13: 54, 14: 83, 15: 125, 16: 47, 17: 64, 18: 59, 19: 38,
            20: 48, 21: 4, 22: 63, 23: 76, 24: 93, 25: 14, 26: 105, 27: 22, 28: 124, 29: 68, 30: 85, 31: 69, 32: 96, 33: 91, 34: 110, 35: 118, 36: 81, 37: 15, 38: 132, 39: 20,
            40: 71, 41: 129, 42: 65, 43: 32, 44: 19, 45: 115, 46: 114, 47: 35, 48: 60, 49: 138, 50: 144, 51: 72, 52: 44, 53: 26, 54: 88, 55: 141, 56: 12, 57: 13, 58: 34, 59: 36,
            60: 8, 61: 46, 62: 79, 63: 67, 64: 75, 65: 27, 66: 62, 67: 148, 68: 103, 69: 121, 70: 94, 71: 128, 72: 16, 73: 7, 74: 43, 75: 17, 76: 80, 77: 1, 78: 149, 79: 95,
            80: 73, 81: 101, 82: 70, 83: 53, 84: 119, 85: 142, 86: 18, 87: 78, 88: 136, 89: 23, 90: 5, 91: 143, 92: 61, 93: 106, 94: 92, 95: 50, 96: 24, 97: 113, 98: 9, 99: 55,
            100: 135, 101: 133, 102: 120, 103: 37, 104: 42, 105: 140, 106: 139, 107: 86, 108: 102, 109: 57, 110: 3, 111: 21, 112: 40, 113: 29, 114: 6, 115: 104, 116: 97, 117: 109,
            118: 147, 119: 146, 120: 30, 121: 112, 122: 122, 123: 28, 124: 99, 125: 10, 126: 31, 127: 134, 128: 39, 129: 49, 130: 131, 131: 117, 132: 126, 133: 52, 134: 51, 135: 0,
            136: 87, 137: 66, 138: 45, 139: 130, 140: 145, 141: 123, 142: 58, 143: 33, 144: 2, 145: 116, 146: 82, 147: 98, 148: 11, 149: 89, 150: 150}


def relation_class_by_freq():
    return {0: 'on', 1: 'has', 2: 'in', 3: 'of', 4: 'wearing', 5: 'near', 6: 'with', 7: 'above', 8: 'holding', 9: 'behind',
            10: 'under', 11: 'sitting on', 12: 'wears', 13: 'standing on', 14: 'in front of', 15: 'attached to', 16: 'at', 17: 'hanging from', 18: 'over', 19: 'for',
            20: 'riding', 21: 'carrying', 22: 'eating', 23: 'walking on', 24: 'playing', 25: 'covering', 26: 'laying on', 27: 'along', 28: 'watching', 29: 'and',
            30: 'between', 31: 'belonging to', 32: 'painted on', 33: 'against', 34: 'looking at', 35: 'from', 36: 'parked on', 37: 'to', 38: 'made of', 39: 'covered in',
            40: 'mounted on', 41: 'says', 42: 'part of', 43: 'across', 44: 'flying in', 45: 'using', 46: 'on back of', 47: 'lying on', 48: 'growing on', 49: 'walking in'}


def relation_by_super_class_int2str():
    return {0: 'above', 1: 'across', 2: 'against', 3: 'along', 4: 'and', 5: 'at', 6: 'behind', 7: 'between', 8: 'in', 9: 'in front of',
            10: 'near', 11: 'on', 12: 'on back of', 13: 'over', 14: 'under', 15: 'belonging to', 16: 'for', 17: 'from', 18: 'has', 19: 'made of',
            20: 'of', 21: 'part of', 22: 'to', 23: 'wearing', 24: 'wears', 25: 'with', 26: 'attached to', 27: 'carrying', 28: 'covered in', 29: 'covering',
            30: 'eating', 31: 'flying in', 32: 'growing on', 33: 'hanging from', 34: 'holding', 35: 'laying on', 36: 'looking at', 37: 'lying on', 38: 'mounted on', 39: 'painted on',
            40: 'parked on', 41: 'playing', 42: 'riding', 43: 'says', 44: 'sitting on', 45: 'standing on', 46: 'using', 47: 'walking in', 48: 'walking on', 49: 'watching'}


def relation_class_freq2scat():
    return torch.tensor([11, 18, 8, 20, 23, 10, 25, 0, 34, 6, 14, 44, 24, 45, 9, 26, 5, 33, 13, 16,
                         42, 27, 30, 48, 41, 29, 35, 3, 49, 4, 7, 15, 39, 2, 36, 17, 40, 22, 19, 28,
                         38, 43, 21, 1, 31, 46, 12, 37, 32, 47, -1])


def preprocess_super_class(synset2cid, super_class_dict):
    super_class_list = object_super_class()
    sub2super_dict = {}
    for line in open(super_class_dict, 'r'):
        line = line.strip('\n').strip('_').split(',')
        sub_class = synset2cid[line[0]]
        super_class = []
        for item in line[1:]:
            super_class.append(super_class_list[item])
        sub2super_dict[sub_class] = super_class
    return sub2super_dict
