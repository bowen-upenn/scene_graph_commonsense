import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math
import os
import concurrent.futures
from functools import partial

from utils import *
from query_llm import *
from dataset_utils import relation_by_super_class_int2str, object_class_int2str


class Evaluator:
    """
    The class evaluate the model performance on Recall@k and mean Recall@k evaluation metrics on predicate classification tasks.
    In our hierarchical relationship scheme, each edge has three predictions per direction under three disjoint super-categories.
    Therefore, each directed edge outputs three individual candidates to be ranked in the top k most confident predictions instead of one.
    """
    def __init__(self, args, num_classes, iou_thresh, top_k, max_cache_size=10000):
        self.args = args
        self.hierar = args['models']['hierarchical_pred']
        self.top_k = top_k
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.num_connected_target = 0.0
        self.motif_total = 0.0
        self.motif_correct = 0.0
        self.result_dict = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_per_class = {k: torch.tensor([0.0 for i in range(self.num_classes)]) for k in self.top_k}
        self.num_conn_target_per_class = torch.tensor([0.0 for i in range(self.num_classes)])
        self.feature_size = args['models']['feature_size']
        self.run_mode = args['training']['run_mode']

        if args['dataset']['dataset'] == 'vg':
            self.train_triplets = torch.load(args['dataset']['train_triplets'])
            self.test_triplets = torch.load(args['dataset']['test_triplets'])
            self.zero_shot_triplets = torch.load(args['dataset']['zero_shot_triplets'])
            self.result_dict_zs = {20: 0.0, 50: 0.0, 100: 0.0}
            self.result_per_class_zs = {k: torch.tensor([0.0 for i in range(self.num_classes)]) for k in self.top_k}
            self.num_connected_target_zs = 0.0
            self.num_conn_target_per_class_zs = torch.tensor([0.0 for i in range(self.num_classes)])
        elif args['dataset']['dataset'] == 'oiv6':
            self.result_per_class_ap = torch.tensor([0.0 for i in range(self.num_classes)])
            self.result_per_class_ap_union = torch.tensor([0.0 for i in range(self.num_classes)])
            self.num_conn_target_per_class_ap = torch.tensor([0.0 for i in range(self.num_classes)])

        self.which_in_batch = None
        self.which_in_batch_target = None
        # self.connected_pred = None
        self.confidence = None
        self.connectivity = None
        self.relation_pred = None
        self.relation_target = None

        self.subject_cat_pred = None
        self.object_cat_pred = None
        self.subject_cat_target = None
        self.object_cat_target = None

        self.subject_bbox_pred = None
        self.object_bbox_pred = None
        self.subject_bbox_target = None
        self.object_bbox_target = None

        self.annotation_paths = None
        self.cache_hits = 0
        self.total_cache_queries = 0
        self.max_cache_size = max_cache_size
        self.cache = EdgeCache(max_cache_size)
        self.commonsense_yes_triplets = torch.load('triplets/commonsense_yes_triplets.pt') if self.run_mode == 'train_cs' else None
        self.commonsense_no_triplets = torch.load('triplets/commonsense_no_triplets.pt') if self.run_mode == 'train_cs' else None
        self.dict_relation_names = relation_by_super_class_int2str()
        self.dict_object_names = object_class_int2str()


    def iou(self, bbox_target, bbox_pred):
        mask_pred = torch.zeros(self.feature_size, self.feature_size)
        mask_pred[int(bbox_pred[2]):int(bbox_pred[3]), int(bbox_pred[0]):int(bbox_pred[1])] = 1
        mask_target = torch.zeros(self.feature_size, self.feature_size)
        mask_target[int(bbox_target[2]):int(bbox_target[3]), int(bbox_target[0]):int(bbox_target[1])] = 1
        intersect = torch.sum(torch.logical_and(mask_target, mask_pred))
        union = torch.sum(torch.logical_or(mask_target, mask_pred))
        if union == 0:
            return 0
        else:
            return float(intersect) / float(union)


    def iou_union(self, bbox_pred1, bbox_pred2, bbox_target1, bbox_target2):
        mask_pred1 = torch.zeros(self.feature_size, self.feature_size)
        mask_pred1[int(bbox_pred1[2]):int(bbox_pred1[3]), int(bbox_pred1[0]):int(bbox_pred1[1])] = 1
        mask_pred2 = torch.zeros(self.feature_size, self.feature_size)
        mask_pred2[int(bbox_pred2[2]):int(bbox_pred2[3]), int(bbox_pred2[0]):int(bbox_pred2[1])] = 1
        mask_pred = torch.logical_or(mask_pred1, mask_pred2)

        mask_target1 = torch.zeros(self.feature_size, self.feature_size)
        mask_target1[int(bbox_target1[2]):int(bbox_target1[3]), int(bbox_target1[0]):int(bbox_target1[1])] = 1
        mask_target2 = torch.zeros(self.feature_size, self.feature_size)
        mask_target2[int(bbox_target2[2]):int(bbox_target2[3]), int(bbox_target2[0]):int(bbox_target2[1])] = 1
        mask_target = torch.logical_or(mask_target1, mask_target2)

        intersect = torch.sum(torch.logical_and(mask_target, mask_pred))
        union = torch.sum(torch.logical_or(mask_target, mask_pred))
        if union == 0:
            return 0
        else:
            return float(intersect) / float(union)


    def accumulate(self, which_in_batch, relation_pred, relation_target, super_relation_pred, connectivity,
                   subject_cat_pred, object_cat_pred, subject_cat_target, object_cat_target,
                   subject_bbox_pred, object_bbox_pred, subject_bbox_target, object_bbox_target, iou_mask,
                   predcls=True, cat_subject_confidence=None, cat_object_confidence=None, height=None, width=None):

        if self.relation_pred is None:
            if not self.hierar:     # flat relationship prediction
                self.which_in_batch = which_in_batch
                self.connectivity = connectivity

                self.confidence = torch.max(relation_pred, dim=1)[0] #+ connectivity
                if not predcls:
                    ins_pair_confidence = cat_subject_confidence + cat_object_confidence
                    self.confidence += ins_pair_confidence
                self.confidence[~iou_mask] = -math.inf

                self.relation_pred = torch.argmax(relation_pred, dim=1)
                self.subject_cat_pred = subject_cat_pred
                self.object_cat_pred = object_cat_pred
                self.subject_bbox_pred = subject_bbox_pred
                self.object_bbox_pred = object_bbox_pred

                if predcls:
                    self.which_in_batch_target = which_in_batch
                    self.relation_target = relation_target

                    self.subject_cat_target = subject_cat_target
                    self.object_cat_target = object_cat_target
                    self.subject_bbox_target = subject_bbox_target
                    self.object_bbox_target = object_bbox_target

                if self.run_mode == 'train_cs':
                    triplets = torch.hstack((self.subject_cat_pred.unsqueeze(1), self.relation_pred.unsqueeze(1), self.object_cat_pred.unsqueeze(1)))
                    is_in_no_dict = torch.tensor([tuple(triplets[i].cpu().tolist()) in self.commonsense_no_triplets for i in range(len(triplets))], device=self.confidence.device)
                    not_in_yes_dict = torch.tensor([tuple(triplets[i].cpu().tolist()) not in self.commonsense_yes_triplets for i in range(len(triplets))], device=self.confidence.device)
                    self.confidence[not_in_yes_dict] = -math.inf
                    self.confidence[is_in_no_dict] = -math.inf

            else:
                self.which_in_batch = which_in_batch.repeat(3)
                self.connectivity = connectivity.repeat(3)

                self.confidence = torch.hstack((torch.max(relation_pred[:, :self.args['models']['num_geometric']], dim=1)[0],
                                                torch.max(relation_pred[:, self.args['models']['num_geometric']:
                                                                           self.args['models']['num_geometric'] + self.args['models']['num_possessive']], dim=1)[0],
                                                torch.max(relation_pred[:, self.args['models']['num_geometric'] + self.args['models']['num_possessive']:], dim=1)[0]))
                if not predcls:
                    ins_pair_confidence = cat_subject_confidence + cat_object_confidence
                    self.confidence += ins_pair_confidence.repeat(3)
                iou_mask = iou_mask.repeat(3)
                self.confidence[~iou_mask] = -math.inf

                self.relation_pred = torch.hstack((torch.argmax(relation_pred[:, :self.args['models']['num_geometric']], dim=1),
                                                   torch.argmax(relation_pred[:, self.args['models']['num_geometric']:self.args['models']['num_geometric']+self.args['models']['num_possessive']], dim=1)
                                                   + self.args['models']['num_geometric'],
                                                   torch.argmax(relation_pred[:, self.args['models']['num_geometric']+self.args['models']['num_possessive']:], dim=1)
                                                   + self.args['models']['num_geometric'] + self.args['models']['num_possessive']))

                self.subject_cat_pred = subject_cat_pred.repeat(3)
                self.object_cat_pred = object_cat_pred.repeat(3)
                self.subject_bbox_pred = subject_bbox_pred.repeat(3, 1)
                self.object_bbox_pred = object_bbox_pred.repeat(3, 1)

                if predcls:
                    self.which_in_batch_target = which_in_batch
                    self.relation_target = relation_target
                    self.subject_cat_target = subject_cat_target
                    self.object_cat_target = object_cat_target
                    self.subject_bbox_target = subject_bbox_target
                    self.object_bbox_target = object_bbox_target

                if self.run_mode == 'train_cs':
                    triplets = torch.hstack((self.subject_cat_pred.unsqueeze(1), self.relation_pred.unsqueeze(1), self.object_cat_pred.unsqueeze(1)))
                    is_in_no_dict = torch.tensor([tuple(triplets[i].cpu().tolist()) in self.commonsense_no_triplets for i in range(len(triplets))], device=self.confidence.device)
                    not_in_yes_dict = torch.tensor([tuple(triplets[i].cpu().tolist()) not in self.commonsense_yes_triplets for i in range(len(triplets))], device=self.confidence.device)
                    self.confidence[is_in_no_dict] = -math.inf
                    self.confidence[not_in_yes_dict] = -math.inf

        else:
            if not self.hierar:     # flat relationship prediction
                self.which_in_batch = torch.hstack((self.which_in_batch, which_in_batch))
                confidence = torch.max(relation_pred, dim=1)[0] #+ connectivity
                if not predcls:
                    ins_pair_confidence = cat_subject_confidence + cat_object_confidence
                    confidence += ins_pair_confidence
                confidence[~iou_mask] = -math.inf

                relation_pred = torch.argmax(relation_pred, dim=1)
                self.relation_pred = torch.hstack((self.relation_pred, relation_pred))
                self.subject_cat_pred = torch.hstack((self.subject_cat_pred, subject_cat_pred))
                self.object_cat_pred = torch.hstack((self.object_cat_pred, object_cat_pred))
                self.subject_bbox_pred = torch.vstack((self.subject_bbox_pred, subject_bbox_pred))
                self.object_bbox_pred = torch.vstack((self.object_bbox_pred, object_bbox_pred))

                if predcls:
                    self.which_in_batch_target = torch.hstack((self.which_in_batch_target, which_in_batch))
                    self.relation_target = torch.hstack((self.relation_target, relation_target))
                    self.subject_cat_target = torch.hstack((self.subject_cat_target, subject_cat_target))
                    self.object_cat_target = torch.hstack((self.object_cat_target, object_cat_target))
                    self.subject_bbox_target = torch.vstack((self.subject_bbox_target, subject_bbox_target))
                    self.object_bbox_target = torch.vstack((self.object_bbox_target, object_bbox_target))

                if self.run_mode == 'train_cs':
                    triplets = torch.hstack((subject_cat_pred.unsqueeze(1), relation_pred.unsqueeze(1), object_cat_pred.unsqueeze(1)))
                    is_in_no_dict = torch.tensor([tuple(triplets[i].cpu().tolist()) in self.commonsense_no_triplets for i in range(len(triplets))], device=self.confidence.device)
                    not_in_yes_dict = torch.tensor([tuple(triplets[i].cpu().tolist()) not in self.commonsense_yes_triplets for i in range(len(triplets))], device=self.confidence.device)
                    confidence[is_in_no_dict] = -math.inf
                    confidence[not_in_yes_dict] = -math.inf

                self.confidence = torch.hstack((self.confidence, confidence))
                self.connectivity = torch.hstack((self.connectivity, connectivity))

            else:
                self.which_in_batch = torch.hstack((self.which_in_batch, which_in_batch.repeat(3)))

                relation_pred_candid = torch.hstack((torch.argmax(relation_pred[:, :self.args['models']['num_geometric']], dim=1),
                                                     torch.argmax(relation_pred[:, self.args['models']['num_geometric']:self.args['models']['num_geometric']+self.args['models']['num_possessive']], dim=1)
                                                     + self.args['models']['num_geometric'],
                                                     torch.argmax(relation_pred[:, self.args['models']['num_geometric']+self.args['models']['num_possessive']:], dim=1)
                                                     + self.args['models']['num_geometric'] + self.args['models']['num_possessive']))
                self.relation_pred = torch.hstack((self.relation_pred, relation_pred_candid))
                self.subject_cat_pred = torch.hstack((self.subject_cat_pred, subject_cat_pred.repeat(3)))
                self.object_cat_pred = torch.hstack((self.object_cat_pred, object_cat_pred.repeat(3)))
                self.subject_bbox_pred = torch.vstack((self.subject_bbox_pred, subject_bbox_pred.repeat(3, 1)))
                self.object_bbox_pred = torch.vstack((self.object_bbox_pred, object_bbox_pred.repeat(3, 1)))

                confidence = torch.hstack((torch.max(relation_pred[:, :self.args['models']['num_geometric']], dim=1)[0],
                                           torch.max(relation_pred[:, self.args['models']['num_geometric']:self.args['models']['num_geometric'] + self.args['models']['num_possessive']], dim=1)[0],
                                           torch.max(relation_pred[:, self.args['models']['num_geometric'] + self.args['models']['num_possessive']:], dim=1)[0]))
                if not predcls:
                    ins_pair_confidence = cat_subject_confidence + cat_object_confidence
                    confidence += ins_pair_confidence.repeat(3)
                iou_mask = iou_mask.repeat(3)
                confidence[~iou_mask] = -math.inf

                if predcls:
                    self.which_in_batch_target = torch.hstack((self.which_in_batch_target, which_in_batch))
                    self.relation_target = torch.hstack((self.relation_target, relation_target))
                    self.subject_cat_target = torch.hstack((self.subject_cat_target, subject_cat_target))
                    self.object_cat_target = torch.hstack((self.object_cat_target, object_cat_target))
                    self.subject_bbox_target = torch.vstack((self.subject_bbox_target, subject_bbox_target))
                    self.object_bbox_target = torch.vstack((self.object_bbox_target, object_bbox_target))

                if self.run_mode == 'train_cs':
                    triplets = torch.hstack((subject_cat_pred.repeat(3).unsqueeze(1), relation_pred_candid.unsqueeze(1), object_cat_pred.repeat(3).unsqueeze(1)))
                    is_in_no_dict = torch.tensor([tuple(triplets[i].cpu().tolist()) in self.commonsense_no_triplets for i in range(len(triplets))], device=self.confidence.device)
                    not_in_yes_dict = torch.tensor([tuple(triplets[i].cpu().tolist()) not in self.commonsense_yes_triplets for i in range(len(triplets))], device=self.confidence.device)
                    confidence[is_in_no_dict] = -math.inf
                    confidence[not_in_yes_dict] = -math.inf

                self.confidence = torch.hstack((self.confidence, confidence))
                self.connectivity = torch.hstack((self.connectivity, connectivity.repeat(3)))


    def accumulate_target(self, relation_target, subject_cat_target, object_cat_target, subject_bbox_target, object_bbox_target):
        self.relation_target = relation_target
        self.subject_cat_target = subject_cat_target
        self.object_cat_target = object_cat_target
        self.subject_bbox_target = subject_bbox_target
        self.object_bbox_target = object_bbox_target


    def compute(self, per_class=False, predcls=True):
        """
        A ground truth predicate is considered to match a hypothesized relationship iff the predicted relationship is correct,
        the subject and object labels match, and the bounding boxes associated with the subject and object both have IOU>0.5 with the ground-truth boxes.
        """

        """
        We calculate the recall scores for each image in a moving average fashion across the test dataset.
        Otherwise, uncomment the following two lines and select batch size = 1 in the config file to view the recall on each individual image.
        """

        recall_k_zs, recall_k_per_class_zs, mean_recall_k_zs = None, None, None
        self.confidence += self.connectivity

        for image in torch.unique(self.which_in_batch):  # image-wise
            curr_image = self.which_in_batch == image
            if self.which_in_batch_target is None:
                curr_image_target = image
                if self.relation_target[curr_image_target] is None:
                    continue
            else:
                curr_image_target = self.which_in_batch_target == image
            num_relation_pred = len(self.relation_pred[curr_image])
            curr_confidence = self.confidence[curr_image]
            sorted_inds = torch.argsort(curr_confidence, dim=0, descending=True)

            for i in range(len(self.relation_target[curr_image_target])):
                if self.relation_target[curr_image_target][i] == -1:  # if target is not connected
                    continue
                if self.args['dataset']['dataset'] == 'vg':
                    curr_triplet = str(self.subject_cat_target[curr_image_target][i].item()) + '_' + str(self.relation_target[curr_image_target][i].item()) \
                                   + '_' + str(self.object_cat_target[curr_image_target][i].item())

                # search in top k most confident predictions in each image
                num_target = torch.sum(self.relation_target[curr_image_target] != -1)
                this_k = min(self.top_k[-1], num_relation_pred)  # 100
                keep_inds = sorted_inds[:this_k]

                found = False   # found if any one of the three sub-models predict correctly
                for j in range(len(keep_inds)):     # for each target <subject, relation, object> triple, find any match in the top k confident predictions
                    if predcls:
                        label_condition = (self.subject_cat_target[curr_image_target][i] == self.subject_cat_pred[curr_image][keep_inds][j] and
                                           self.object_cat_target[curr_image_target][i] == self.object_cat_pred[curr_image][keep_inds][j])
                    else:
                        label_condition = (compare_object_cat(self.subject_cat_target[curr_image_target][i], self.subject_cat_pred[curr_image][keep_inds][j]) and
                                           compare_object_cat(self.object_cat_target[curr_image_target][i], self.object_cat_pred[curr_image][keep_inds][j]))
                    if label_condition:
                        sub_iou = self.iou(self.subject_bbox_target[curr_image_target][i], self.subject_bbox_pred[curr_image][keep_inds][j])
                        obj_iou = self.iou(self.object_bbox_target[curr_image_target][i], self.object_bbox_pred[curr_image][keep_inds][j])

                        if sub_iou >= self.iou_thresh and obj_iou >= self.iou_thresh:
                            if self.relation_target[curr_image_target][i] == self.relation_pred[curr_image][keep_inds][j]:
                                for k in self.top_k:
                                    if j >= k:
                                        continue
                                    self.result_dict[k] += 1.0
                                    if per_class:
                                        self.result_per_class[k][self.relation_target[curr_image_target][i]] += 1.0

                                    # if zero shot
                                    if self.args['dataset']['dataset'] == 'vg':
                                        if curr_triplet in self.zero_shot_triplets:
                                            assert curr_triplet not in self.train_triplets
                                            self.result_dict_zs[k] += 1.0
                                            if per_class:
                                                self.result_per_class_zs[k][self.relation_target[curr_image_target][i]] += 1.0
                                found = True
                            if found:
                                break

                self.num_connected_target += 1.0
                self.num_conn_target_per_class[self.relation_target[curr_image_target][i]] += 1.0
                # if zero shot
                if self.args['dataset']['dataset'] == 'vg':
                    if curr_triplet in self.zero_shot_triplets:
                        self.num_connected_target_zs += 1.0
                        self.num_conn_target_per_class_zs[self.relation_target[curr_image_target][i]] += 1.0

        recall_k = [self.result_dict[k] / max(self.num_connected_target, 1e-3) for k in self.top_k]
        recall_k_per_class = [self.result_per_class[k] / self.num_conn_target_per_class for k in self.top_k]
        mean_recall_k = [torch.nanmean(r) for r in recall_k_per_class]

        if self.args['dataset']['dataset'] == 'vg':
            recall_k_zs = [self.result_dict_zs[k] / max(self.num_connected_target_zs, 1e-3) for k in self.top_k]
            recall_k_per_class_zs = [self.result_per_class_zs[k] / self.num_conn_target_per_class_zs for k in self.top_k]
            mean_recall_k_zs = [torch.nanmean(r) for r in recall_k_per_class_zs]

        return recall_k, recall_k_per_class, mean_recall_k, recall_k_zs, recall_k_per_class_zs, mean_recall_k_zs


    def load_annotation_paths(self, annot_path):
        self.annotation_paths = None    # reset
        self.annotation_paths = annot_path


    def _get_related_top_k_predictions(self, image, top_k):
        curr_image = self.which_in_batch == image
        curr_confidence = self.confidence[curr_image]
        sorted_inds = torch.argsort(curr_confidence, dim=0, descending=True)

        curr_predictions = []
        curr_image_graph = []

        if self.which_in_batch_target is None:
            curr_image_target = image
            if self.relation_target[curr_image_target] is None:
                return
        else:
            curr_image_target = self.which_in_batch_target == image

        for i in range(0, len(self.subject_cat_target[curr_image_target])):
            if self.relation_target[curr_image_target][i] == -1:  # if target is not connected
                continue
            if len(curr_image_graph) >= 15:     # enforce efficiency
                break

            for j in range(min(top_k, len(sorted_inds))):
                ind = sorted_inds[j]

                subject_id_pred = self.subject_cat_pred[curr_image][ind].item()
                object_id_pred = self.object_cat_pred[curr_image][ind].item()

                # check if the predicted subject or object matches the target
                if (self.subject_cat_target[curr_image_target][i] == subject_id_pred and torch.sum(torch.abs(self.subject_bbox_target[curr_image_target][i] - self.subject_bbox_pred[curr_image][ind])) == 0) \
                        or (self.object_cat_target[curr_image_target][i] == object_id_pred and torch.sum(torch.abs(self.object_bbox_target[curr_image_target][i] - self.object_bbox_pred[curr_image][ind])) == 0):

                    relation_id = self.relation_pred[curr_image][ind].item()
                    string = self.dict_object_names[subject_id_pred] + ' ' + self.dict_relation_names[relation_id] + ' ' + self.dict_object_names[object_id_pred]
                    if string not in curr_predictions:
                        # filter the edge by commonsense
                        edge = [self.subject_bbox_pred[curr_image][ind].cpu().tolist(), relation_id, self.object_bbox_pred[curr_image][ind].cpu().tolist(), self.confidence[curr_image][ind].item(), j]
                        curr_image_graph.append(edge)
                        curr_predictions.append(string)

                if len(curr_image_graph) >= 10:  # enforce efficiency
                    break

        if len(curr_image_graph) > 0:
            if self.args['training']['common_sense']:
                responses, cache_hits = batch_query_openai_gpt(curr_predictions, self.cache, cache_hits=self.cache_hits)

                # calculate cache hit percentage
                self.cache_hits = cache_hits
                self.total_cache_queries += len(curr_predictions)

                valid_curr_image_graph = []
                invalid_curr_image_graph = []
                for i, response in enumerate(responses):
                    if response == 1:
                        valid_curr_image_graph.append(curr_image_graph[i])
                    else:
                        invalid_curr_image_graph.append(curr_image_graph[i])
            else:
                valid_curr_image_graph = curr_image_graph
                invalid_curr_image_graph = []

            annot_name = self.annotation_paths[image][:-16] + '_pseudo_annotations.pkl'
            annot_path = os.path.join(self.args['dataset']['annot_dir'], 'cs_aligned_top' + str(top_k), annot_name)
            torch.save(valid_curr_image_graph, annot_path)
            annot_path = os.path.join(self.args['dataset']['annot_dir'], 'cs_violated_top' + str(top_k), annot_name)
            torch.save(invalid_curr_image_graph, annot_path)

        return curr_predictions, curr_image_graph


    def get_related_top_k_predictions_parallel(self, top_k, save_to_annot=True):
        self.dict_relation_names = relation_by_super_class_int2str()
        self.dict_object_names = object_class_int2str()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda image: self._get_related_top_k_predictions(image, top_k), torch.unique(self.which_in_batch)))

        cache_hit_percentage = (self.cache_hits / self.total_cache_queries) * 100 if self.total_cache_queries > 0 else 0

        if save_to_annot:
            top_k_predictions = [item[0] for item in results]
            top_k_image_graphs = [item[1] for item in results]
            return top_k_predictions, top_k_image_graphs, cache_hit_percentage


    def save_visualization_results(self, annot_path, triplets, heights, widths, images, image_depth, bboxes, categories, batch_count, top_k):
        dict_relation_names = relation_by_super_class_int2str()
        dict_object_names = object_class_int2str()
        if self.which_in_batch is None:
            return

        for image in torch.unique(self.which_in_batch):  # image-wise
            curr_image = self.which_in_batch == image
            curr_confidence = self.confidence[curr_image]
            sorted_inds = torch.argsort(curr_confidence, dim=0, descending=True)

            # select the top k predictions
            this_k = min(top_k, len(self.relation_pred[curr_image]))
            keep_inds = sorted_inds[:this_k]

            curr_image_graph = []

            for ind in keep_inds:
                subject_id = self.subject_cat_pred[curr_image][ind].item()
                relation_id = self.relation_pred[curr_image][ind].item()
                object_id = self.object_cat_pred[curr_image][ind].item()

                subject_bbox = self.subject_bbox_pred[curr_image][ind].cpu() / self.feature_size
                object_bbox = self.object_bbox_pred[curr_image][ind].cpu() / self.feature_size
                height, width = heights[image], widths[image]
                subject_bbox[:2] *= height
                subject_bbox[2:] *= width
                object_bbox[:2] *= height
                object_bbox[2:] *= width
                subject_bbox = subject_bbox.ceil().int()
                object_bbox = object_bbox.ceil().int()

                edge = {'edge': dict_object_names[subject_id] + ' ' + dict_relation_names[relation_id] + ' ' + dict_object_names[object_id],
                        'subject_id': subject_id,
                        'relation_id': relation_id,
                        'object_id': object_id,
                        'bbox_sub': subject_bbox.tolist(),
                        'bbox_obj': object_bbox.tolist()}
                curr_image_graph.append(edge)

            vis_results = {'predicted_graph': curr_image_graph,
                           'image_path': annot_path[image],
                           'target_graph': triplets[image],
                           'bboxes': bboxes[image],
                           'categories': categories[image],
                           'image': images[image],
                           'image_depth': image_depth[image],
                           'height': heights[image],
                           'width': widths[image]}
            # print('vis_results', vis_results)

            annot_name = str(batch_count) + '_vis_results.pkl'
            annot_path = os.path.join('results/visualization_results/cs', annot_name)
            print('annot_path', annot_path)
            torch.save(vis_results, annot_path)


    def compute_precision(self):
        for image in torch.unique(self.which_in_batch):  # image-wise
            curr_image = self.which_in_batch == image
            num_relation_pred = len(self.relation_pred[curr_image])
            curr_confidence = self.confidence[curr_image]
            sorted_inds = torch.argsort(curr_confidence, dim=0, descending=True)
            this_k = min(20, num_relation_pred)  # 100
            keep_inds = sorted_inds[:this_k]

            for i in range(len(self.relation_pred[curr_image][keep_inds])):
                found = False  # found if any one of the three sub-models predict correctly
                found_union = False
                for j in range(len(self.relation_target[curr_image])):
                    if self.relation_target[curr_image][j] == -1:  # if target is not connected
                        continue

                    if (self.subject_cat_pred[curr_image][keep_inds][i] == self.subject_cat_target[curr_image][j]
                            and self.object_cat_pred[curr_image][keep_inds][i] == self.object_cat_target[curr_image][j]):

                        sub_iou = self.iou(self.subject_bbox_pred[curr_image][keep_inds][i], self.subject_bbox_target[curr_image][j])
                        obj_iou = self.iou(self.object_bbox_pred[curr_image][keep_inds][i], self.object_bbox_target[curr_image][j])
                        union_iou = self.iou_union(self.subject_bbox_pred[curr_image][keep_inds][i], self.object_bbox_pred[curr_image][keep_inds][i],
                                                   self.subject_bbox_target[curr_image][j], self.object_bbox_target[curr_image][j])

                        if self.relation_pred[curr_image][keep_inds][i] == self.relation_target[curr_image][j]:
                            if sub_iou >= self.iou_thresh and obj_iou >= self.iou_thresh and found == False:
                                self.result_per_class_ap[self.relation_pred[curr_image][keep_inds][i]] += 1.0
                                found = True
                            if union_iou >= self.iou_thresh and found_union == False:
                                self.result_per_class_ap_union[self.relation_pred[curr_image][keep_inds][i]] += 1.0
                                found_union = True

                        if found and found_union:
                            break

                self.num_conn_target_per_class_ap[self.relation_pred[curr_image][keep_inds][i]] += 1.0

        weight = get_weight_oiv6()
        precision_per_class = self.result_per_class_ap / self.num_conn_target_per_class_ap
        not_nan = torch.logical_not(torch.isnan(precision_per_class))
        weighted_mean_precision = torch.nansum(precision_per_class * weight) / torch.sum(weight[not_nan])

        precision_per_class_union = self.result_per_class_ap_union / self.num_conn_target_per_class_ap
        weighted_mean_precision_union = torch.nansum(precision_per_class_union * weight) / torch.sum(weight[not_nan])
        return weighted_mean_precision, weighted_mean_precision_union

    def clear_data(self):
        self.which_in_batch = None
        self.confidence = None
        self.connectivity = None
        self.relation_pred = None
        self.relation_target = None

        self.subject_cat_pred = None
        self.object_cat_pred = None
        self.subject_cat_target = None
        self.object_cat_target = None

        self.subject_bbox_pred = None
        self.object_bbox_pred = None
        self.subject_bbox_target = None
        self.object_bbox_target = None

    def clear_gpt_cache(self):
        self.cache = {}


class Evaluator_Top3:
    """
    The class evaluate the model performance on Recall@k^{*} and mean Recall@k^{*} evaluation metrics on predicate classification tasks.
    If any of the three super-category output heads correctly predicts the relationship, we score it as a match.
    Top3 represents three argmax predicate from three disjoint super-categories, instead of the top 3 predicates under a flat classification.
    """
    def __init__(self, args, num_classes, iou_thresh, top_k):
        self.args = args
        self.top_k = top_k
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.num_connected_target = 0.0
        self.motif_total = 0.0
        self.motif_correct = 0.0
        self.result_dict = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict_top1 = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_per_class = {k: torch.tensor([0.0 for i in range(self.num_classes)]) for k in self.top_k}
        self.result_per_class_top1 = {k: torch.tensor([0.0 for i in range(self.num_classes)]) for k in self.top_k}
        self.num_conn_target_per_class = torch.tensor([0.0 for i in range(self.num_classes)])
        self.feature_size = args['models']['feature_size']

        self.which_in_batch = None
        self.confidence = None
        self.connectivity = None
        self.relation_pred = None
        self.relation_target = None
        self.super_relation_pred = None

        self.subject_cat_pred = None
        self.object_cat_pred = None
        self.subject_cat_target = None
        self.object_cat_target = None

        self.subject_bbox_pred = None
        self.object_bbox_pred = None
        self.subject_bbox_target = None
        self.object_bbox_target = None

    def iou(self, bbox_target, bbox_pred):
        mask_pred = torch.zeros(self.feature_size, self.feature_size)
        mask_pred[int(bbox_pred[2]):int(bbox_pred[3]), int(bbox_pred[0]):int(bbox_pred[1])] = 1
        mask_target = torch.zeros(self.feature_size, self.feature_size)
        mask_target[int(bbox_target[2]):int(bbox_target[3]), int(bbox_target[0]):int(bbox_target[1])] = 1
        intersect = torch.sum(torch.logical_and(mask_target, mask_pred))
        union = torch.sum(torch.logical_or(mask_target, mask_pred))
        if union == 0:
            return 0
        else:
            return float(intersect) / float(union)

    def accumulate(self, which_in_batch, relation_pred, relation_target, super_relation_pred, connectivity,
                   subject_cat_pred, object_cat_pred, subject_cat_target, object_cat_target,
                   subject_bbox_pred, object_bbox_pred, subject_bbox_target, object_bbox_target, iou_mask):  # size (batch_size, num_relations_classes), (num_relations_classes)

        if self.relation_pred is None:
            self.which_in_batch = which_in_batch
            self.connectivity = connectivity
            self.confidence = torch.max(torch.vstack((torch.max(relation_pred[:, :self.args['models']['num_geometric']], dim=1)[0],
                                                      torch.max(relation_pred[:, self.args['models']['num_geometric']:self.args['models']['num_geometric']+self.args['models']['num_possessive']], dim=1)[0],
                                                      torch.max(relation_pred[:, self.args['models']['num_geometric']+self.args['models']['num_possessive']:], dim=1)[0])), dim=0)[0]  # in log space, [0] to take values
            self.confidence[~iou_mask] = -math.inf
            self.relation_pred = relation_pred
            self.relation_target = relation_target
            self.super_relation_pred = super_relation_pred

            self.subject_cat_pred = subject_cat_pred
            self.object_cat_pred = object_cat_pred
            self.subject_cat_target = subject_cat_target
            self.object_cat_target = object_cat_target

            self.subject_bbox_pred = subject_bbox_pred
            self.object_bbox_pred = object_bbox_pred
            self.subject_bbox_target = subject_bbox_target
            self.object_bbox_target = object_bbox_target
        else:
            self.which_in_batch = torch.hstack((self.which_in_batch, which_in_batch))
            self.connectivity = torch.hstack((self.connectivity, connectivity))

            confidence = torch.max(torch.vstack((torch.max(relation_pred[:, :self.args['models']['num_geometric']], dim=1)[0],
                                                 torch.max(relation_pred[:, self.args['models']['num_geometric']:self.args['models']['num_geometric']+self.args['models']['num_possessive']], dim=1)[0],
                                                 torch.max(relation_pred[:, self.args['models']['num_geometric']+self.args['models']['num_possessive']:], dim=1)[0])), dim=0)[0]  # in log space, [0] to take values
            confidence[~iou_mask] = -math.inf
            self.confidence = torch.hstack((self.confidence, confidence))

            self.relation_pred = torch.vstack((self.relation_pred, relation_pred))
            self.relation_target = torch.hstack((self.relation_target, relation_target))
            self.super_relation_pred = torch.vstack((self.super_relation_pred, super_relation_pred))

            self.subject_cat_pred = torch.hstack((self.subject_cat_pred, subject_cat_pred))
            self.object_cat_pred = torch.hstack((self.object_cat_pred, object_cat_pred))
            self.subject_cat_target = torch.hstack((self.subject_cat_target, subject_cat_target))
            self.object_cat_target = torch.hstack((self.object_cat_target, object_cat_target))

            self.subject_bbox_pred = torch.vstack((self.subject_bbox_pred, subject_bbox_pred))
            self.object_bbox_pred = torch.vstack((self.object_bbox_pred, object_bbox_pred))
            self.subject_bbox_target = torch.vstack((self.subject_bbox_target, subject_bbox_target))
            self.object_bbox_target = torch.vstack((self.object_bbox_target, object_bbox_target))

    def global_refine(self, refined_relation, connected_indices_accumulated):
        # print('self.relation_pred', self.relation_pred.shape, 'connected_indices_accumulated', connected_indices_accumulated.shape)
        # print('self.relation_pred[connected_indices_accumulated]', self.relation_pred[connected_indices_accumulated].shape, 'refined_relation', refined_relation.shape)
        self.relation_pred[connected_indices_accumulated, :] = refined_relation

        confidence = torch.max(torch.vstack((torch.max(refined_relation[:, :self.args['models']['num_geometric']], dim=1)[0],
                                             torch.max(refined_relation[:, self.args['models']['num_geometric']:self.args['models']['num_geometric'] + self.args['models']['num_possessive']], dim=1)[0],
                                             torch.max(refined_relation[:, self.args['models']['num_geometric'] + self.args['models']['num_possessive']:], dim=1)[0])), dim=0)[0]
        self.confidence[connected_indices_accumulated] = confidence

    def compute(self, per_class=False):
        """
        A ground truth predicate is considered to match a hypothesized relationship iff the predicted relationship is correct,
        the subject and object labels match, and the bounding boxes associated with the subject and object both have IOU>0.5 with the ground-truth boxes.
        """
        self.confidence += self.connectivity

        for image in torch.unique(self.which_in_batch):  # image-wise
            curr_image = self.which_in_batch == image
            num_relation_pred = len(self.relation_pred[curr_image])
            curr_confidence = self.confidence[curr_image]

            sorted_inds = torch.argsort(curr_confidence, dim=0, descending=True)

            for i in range(len(self.relation_target[curr_image])):
                if self.relation_target[curr_image][i] == -1:  # if target is not connected
                    continue

                # search in top k most confident predictions in each image
                num_target = torch.sum(self.relation_target[curr_image] != -1)
                this_k = min(self.top_k[-1], num_relation_pred)  # 100
                keep_inds = sorted_inds[:this_k]

                found = False   # found if any one of the three sub-models predict correctly
                found_top1 = False  # found if only the most confident one of the three sub-models predict correctly
                for j in range(len(keep_inds)):     # for each target <subject, relation, object> triple, find any match in the top k confident predictions
                    if (self.subject_cat_target[curr_image][i] == self.subject_cat_pred[curr_image][keep_inds][j]
                            and self.object_cat_target[curr_image][i] == self.object_cat_pred[curr_image][keep_inds][j]):

                        sub_iou = self.iou(self.subject_bbox_target[curr_image][i], self.subject_bbox_pred[curr_image][keep_inds][j])
                        obj_iou = self.iou(self.object_bbox_target[curr_image][i], self.object_bbox_pred[curr_image][keep_inds][j])

                        if sub_iou >= self.iou_thresh and obj_iou >= self.iou_thresh:
                            if not found:
                                relation_pred_1 = self.relation_pred[curr_image][keep_inds][j][:self.args['models']['num_geometric']]  # geometric
                                relation_pred_2 = self.relation_pred[curr_image][keep_inds][j][self.args['models']['num_geometric']:self.args['models']['num_geometric']
                                                                                                                                    + self.args['models']['num_possessive']]  # possessive
                                relation_pred_3 = self.relation_pred[curr_image][keep_inds][j][self.args['models']['num_geometric'] + self.args['models']['num_possessive']:]  # semantic
                                if self.relation_target[curr_image][i] == torch.argmax(relation_pred_1) \
                                        or self.relation_target[curr_image][i] == torch.argmax(relation_pred_2) + self.args['models']['num_geometric'] \
                                        or self.relation_target[curr_image][i] == torch.argmax(relation_pred_3) + self.args['models']['num_geometric'] + self.args['models']['num_possessive']:
                                    for k in self.top_k:
                                        if j >= max(k, num_target):
                                            continue
                                        self.result_dict[k] += 1.0
                                        if per_class:
                                            self.result_per_class[k][self.relation_target[curr_image][i]] += 1.0
                                    found = True

                            if not found_top1:
                                curr_super = torch.argmax(self.super_relation_pred[curr_image][keep_inds][j])
                                relation_preds = [torch.argmax(self.relation_pred[curr_image][keep_inds][j][:self.args['models']['num_geometric']]),
                                                  torch.argmax(self.relation_pred[curr_image][keep_inds][j][self.args['models']['num_geometric']:self.args['models']['num_geometric']
                                                                                                            + self.args['models']['num_possessive']]) + self.args['models']['num_geometric'],
                                                  torch.argmax(self.relation_pred[curr_image][keep_inds][j][self.args['models']['num_geometric'] + self.args['models']['num_possessive']:])
                                                                                                            + self.args['models']['num_geometric'] + self.args['models']['num_possessive']]
                                if self.relation_target[curr_image][i] == relation_preds[curr_super]:
                                    for k in self.top_k:
                                        if j >= max(k, num_target):
                                            continue
                                        self.result_dict_top1[k] += 1.0
                                        if per_class:
                                            self.result_per_class_top1[k][self.relation_target[curr_image][i]] += 1.0
                                    found_top1 = True

                            if found and found_top1:
                                break

                self.num_connected_target += 1.0
                self.num_conn_target_per_class[self.relation_target[curr_image][i]] += 1.0

        recall_k = [self.result_dict[k] / max(self.num_connected_target, 1e-3) for k in self.top_k]
        recall_k_per_class = [self.result_per_class[k] / self.num_conn_target_per_class for k in self.top_k]
        mean_recall_k = [torch.nanmean(r) for r in recall_k_per_class]
        # recall_k_top1 = [self.result_dict_top1[k] / self.num_connected_target for k in self.top_k]
        # mean_recall_k_top1 = [torch.nanmean(r) for r in recall_k_per_class_top1]
        return recall_k, recall_k_per_class, mean_recall_k

    def clear_data(self):
        self.which_in_batch = None
        self.confidence = None
        self.relation_pred = None
        self.connectivity = None
        self.relation_target = None

        self.subject_cat_pred = None
        self.object_cat_pred = None
        self.subject_cat_target = None
        self.object_cat_target = None

        self.subject_bbox_pred = None
        self.object_bbox_pred = None
        self.subject_bbox_target = None
        self.object_bbox_target = None
