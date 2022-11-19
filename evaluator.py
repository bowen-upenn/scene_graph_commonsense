import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math


class Evaluator_PC:
    def __init__(self, num_classes, iou_thresh, top_k):
        self.top_k = top_k
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.num_connected_target = 0.0
        self.motif_total = 0.0
        self.motif_correct = 0.0
        self.result_dict = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_per_class = {k: torch.tensor([0.0 for i in range(self.num_classes)]) for k in self.top_k}
        self.num_conn_target_per_class = torch.tensor([0.0 for i in range(self.num_classes)])

        self.which_in_batch = None
        self.connected_pred = None
        self.confidence = None
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

    def iou(self, bbox_target, bbox_pred):
        mask_pred = torch.zeros(32, 32)
        mask_pred[int(bbox_pred[2]):int(bbox_pred[3]), int(bbox_pred[0]):int(bbox_pred[1])] = 1
        mask_target = torch.zeros(32, 32)
        mask_target[int(bbox_target[2]):int(bbox_target[3]), int(bbox_target[0]):int(bbox_target[1])] = 1
        intersect = torch.sum(torch.logical_and(mask_target, mask_pred))
        union = torch.sum(torch.logical_or(mask_target, mask_pred))
        if union == 0:
            return 0
        else:
            return float(intersect) / float(union)

    def accumulate(self, which_in_batch, relation_pred, relation_target, super_relation_pred, connectivity,
                   subject_cat_pred, object_cat_pred, subject_cat_target, object_cat_target,
                   subject_bbox_pred, object_bbox_pred, subject_bbox_target, object_bbox_target):

        if self.relation_pred is None:
            if super_relation_pred is None:     # flat relationship prediction
                self.which_in_batch = which_in_batch
                self.confidence = connectivity + torch.max(relation_pred, dim=1)[0]

                self.relation_pred = torch.argmax(relation_pred, dim=1)
                self.relation_target = relation_target

                self.subject_cat_pred = subject_cat_pred
                self.object_cat_pred = object_cat_pred
                self.subject_cat_target = subject_cat_target
                self.object_cat_target = object_cat_target

                self.subject_bbox_pred = subject_bbox_pred
                self.object_bbox_pred = object_bbox_pred
                self.subject_bbox_target = subject_bbox_target
                self.object_bbox_target = object_bbox_target
            else:
                self.which_in_batch = which_in_batch.repeat(3)
                self.confidence = torch.hstack((torch.max(relation_pred[:, :15], dim=1)[0],
                                                torch.max(relation_pred[:, 15:26], dim=1)[0],
                                                torch.max(relation_pred[:, 26:], dim=1)[0]))
                self.confidence += connectivity.repeat(3)

                self.relation_pred = torch.hstack((torch.argmax(relation_pred[:, :15], dim=1),
                                                   torch.argmax(relation_pred[:, 15:26], dim=1) + 15,
                                                   torch.argmax(relation_pred[:, 26:], dim=1) + 26))
                self.relation_target = relation_target.repeat(3)

                self.subject_cat_pred = subject_cat_pred.repeat(3)
                self.object_cat_pred = object_cat_pred.repeat(3)
                self.subject_cat_target = subject_cat_target.repeat(3)
                self.object_cat_target = object_cat_target.repeat(3)

                self.subject_bbox_pred = subject_bbox_pred.repeat(3, 1)
                self.object_bbox_pred = object_bbox_pred.repeat(3, 1)
                self.subject_bbox_target = subject_bbox_target.repeat(3, 1)
                self.object_bbox_target = object_bbox_target.repeat(3, 1)
        else:
            if super_relation_pred is None:  # flat relationship prediction
                self.which_in_batch = torch.hstack((self.which_in_batch, which_in_batch))
                confidence = connectivity + torch.max(relation_pred, dim=1)[0]
                self.confidence = torch.hstack((self.confidence, confidence))

                self.relation_pred = torch.hstack((self.relation_pred, torch.argmax(relation_pred, dim=1)))
                self.relation_target = torch.hstack((self.relation_target, relation_target))

                self.subject_cat_pred = torch.hstack((self.subject_cat_pred, subject_cat_pred))
                self.object_cat_pred = torch.hstack((self.object_cat_pred, object_cat_pred))
                self.subject_cat_target = torch.hstack((self.subject_cat_target, subject_cat_target))
                self.object_cat_target = torch.hstack((self.object_cat_target, object_cat_target))

                self.subject_bbox_pred = torch.vstack((self.subject_bbox_pred, subject_bbox_pred))
                self.object_bbox_pred = torch.vstack((self.object_bbox_pred, object_bbox_pred))
                self.subject_bbox_target = torch.vstack((self.subject_bbox_target, subject_bbox_target))
                self.object_bbox_target = torch.vstack((self.object_bbox_target, object_bbox_target))
            else:
                self.which_in_batch = torch.hstack((self.which_in_batch, which_in_batch.repeat(3)))
                confidence = torch.hstack((torch.max(relation_pred[:, :15], dim=1)[0],
                                           torch.max(relation_pred[:, 15:26], dim=1)[0],
                                           torch.max(relation_pred[:, 26:], dim=1)[0]))
                confidence += connectivity.repeat(3)
                self.confidence = torch.hstack((self.confidence, confidence))

                relation_pred_candid = torch.hstack((torch.argmax(relation_pred[:, :15], dim=1),
                                                     torch.argmax(relation_pred[:, 15:26], dim=1) + 15,
                                                     torch.argmax(relation_pred[:, 26:], dim=1) + 26))
                self.relation_pred = torch.hstack((self.relation_pred, relation_pred_candid))
                self.relation_target = torch.hstack((self.relation_target, relation_target.repeat(3)))

                self.subject_cat_pred = torch.hstack((self.subject_cat_pred, subject_cat_pred.repeat(3)))
                self.object_cat_pred = torch.hstack((self.object_cat_pred, object_cat_pred.repeat(3)))
                self.subject_cat_target = torch.hstack((self.subject_cat_target, subject_cat_target.repeat(3)))
                self.object_cat_target = torch.hstack((self.object_cat_target, object_cat_target.repeat(3)))

                self.subject_bbox_pred = torch.vstack((self.subject_bbox_pred, subject_bbox_pred.repeat(3, 1)))
                self.object_bbox_pred = torch.vstack((self.object_bbox_pred, object_bbox_pred.repeat(3, 1)))
                self.subject_bbox_target = torch.vstack((self.subject_bbox_target, subject_bbox_target.repeat(3, 1)))
                self.object_bbox_target = torch.vstack((self.object_bbox_target, object_bbox_target.repeat(3, 1)))

    def compute(self, hierarchical_pred, per_class=False):
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
                for j in range(len(keep_inds)):     # for each target <subject, relation, object> triple, find any match in the top k confident predictions
                    if (self.subject_cat_target[curr_image][i] == self.subject_cat_pred[curr_image][keep_inds][j]
                            and self.object_cat_target[curr_image][i] == self.object_cat_pred[curr_image][keep_inds][j]):

                        sub_iou = self.iou(self.subject_bbox_target[curr_image][i], self.subject_bbox_pred[curr_image][keep_inds][j])
                        obj_iou = self.iou(self.object_bbox_target[curr_image][i], self.object_bbox_pred[curr_image][keep_inds][j])

                        if sub_iou >= self.iou_thresh and obj_iou >= self.iou_thresh:
                            if self.relation_target[curr_image][i] == self.relation_pred[curr_image][keep_inds][j]:
                                for k in self.top_k:
                                    if j >= k:
                                        continue
                                    self.result_dict[k] += 1.0
                                    if per_class:
                                        self.result_per_class[k][self.relation_target[curr_image][i]] += 1.0
                                found = True
                            if found:
                                break

                self.num_connected_target += 1.0
                self.num_conn_target_per_class[self.relation_target[curr_image][i]] += 1.0

        recall_k = [self.result_dict[k] / max(self.num_connected_target, 1e-3) for k in self.top_k]
        recall_k_per_class = [self.result_per_class[k] / self.num_conn_target_per_class for k in self.top_k]
        mean_recall_k = [torch.nanmean(r) for r in recall_k_per_class]
        return recall_k, recall_k_per_class, mean_recall_k # , recall_k_ng, mean_recall_k_ng

    def clear_data(self):
        self.which_in_batch = None
        self.confidence = None
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


class Evaluator_PC_Top3:
    def __init__(self, num_classes, iou_thresh, top_k):
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

        self.which_in_batch = None
        self.connected_pred = None

        self.confidence = None
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
        mask_pred = torch.zeros(32, 32)
        mask_pred[int(bbox_pred[2]):int(bbox_pred[3]), int(bbox_pred[0]):int(bbox_pred[1])] = 1
        mask_target = torch.zeros(32, 32)
        mask_target[int(bbox_target[2]):int(bbox_target[3]), int(bbox_target[0]):int(bbox_target[1])] = 1
        intersect = torch.sum(torch.logical_and(mask_target, mask_pred))
        union = torch.sum(torch.logical_or(mask_target, mask_pred))
        if union == 0:
            return 0
        else:
            return float(intersect) / float(union)

    def accumulate(self, which_in_batch, relation_pred, relation_target, super_relation_pred, connectivity,
                   subject_cat_pred, object_cat_pred, subject_cat_target, object_cat_target,
                   subject_bbox_pred, object_bbox_pred, subject_bbox_target, object_bbox_target):  # size (batch_size, num_relations_classes), (num_relations_classes)

        if self.relation_pred is None:
            self.which_in_batch = which_in_batch
            self.confidence = connectivity + torch.max(torch.vstack((torch.max(relation_pred[:, :15], dim=1)[0],
                                                                     torch.max(relation_pred[:, 15:26], dim=1)[0],
                                                                     torch.max(relation_pred[:, 26:], dim=1)[0])), dim=0)[0]  # in log space, [0] to take values
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

            confidence = connectivity + torch.max(torch.vstack((torch.max(relation_pred[:, :15], dim=1)[0],
                                                                torch.max(relation_pred[:, 15:26], dim=1)[0],
                                                                torch.max(relation_pred[:, 26:], dim=1)[0])), dim=0)[0]  # in log space, [0] to take values
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

    def compute(self, hierarchical_pred, per_class=False):
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
                            if hierarchical_pred and not found:
                                relation_pred_1 = self.relation_pred[curr_image][keep_inds][j][:15]  # geometric
                                relation_pred_2 = self.relation_pred[curr_image][keep_inds][j][15:26]  # possessive
                                relation_pred_3 = self.relation_pred[curr_image][keep_inds][j][26:]  # semantic
                                if self.relation_target[curr_image][i] == torch.argmax(relation_pred_1) \
                                        or self.relation_target[curr_image][i] == torch.argmax(relation_pred_2) + 15 \
                                        or self.relation_target[curr_image][i] == torch.argmax(relation_pred_3) + 26:
                                    for k in self.top_k:
                                        if j >= max(k, num_target):
                                            continue
                                        self.result_dict[k] += 1.0
                                        if per_class:
                                            self.result_per_class[k][self.relation_target[curr_image][i]] += 1.0
                                    found = True

                            if not found_top1:
                                curr_super = torch.argmax(self.super_relation_pred[curr_image][keep_inds][j])
                                relation_preds = [torch.argmax(self.relation_pred[curr_image][keep_inds][j][:15]),
                                                  torch.argmax(self.relation_pred[curr_image][keep_inds][j][15:26]) + 15,
                                                  torch.argmax(self.relation_pred[curr_image][keep_inds][j][26:]) + 26]
                                if self.relation_target[curr_image][i] == relation_preds[curr_super]:
                                    for k in self.top_k:
                                        if j >= max(k, num_target):
                                            continue
                                        self.result_dict_top1[k] += 1.0
                                        if per_class:
                                            self.result_per_class_top1[k][self.relation_target[curr_image][i]] += 1.0
                                    found_top1 = True

                            if (hierarchical_pred and found and found_top1) or (not hierarchical_pred and found_top1):
                                break

                self.num_connected_target += 1.0
                self.num_conn_target_per_class[self.relation_target[curr_image][i]] += 1.0

        recall_k = [self.result_dict[k] / max(self.num_connected_target, 1e-3) for k in self.top_k]
        recall_k_per_class = [self.result_per_class[k] / self.num_conn_target_per_class for k in self.top_k]
        mean_recall_k = [torch.nanmean(r) for r in recall_k_per_class]
        # recall_k_per_class_top1 = [self.result_per_class_top1[k] / self.num_conn_target_per_class for k in self.top_k]
        # mean_recall_k_top1 = [torch.nanmean(r) for r in recall_k_per_class_top1]
        return recall_k, recall_k_per_class, mean_recall_k

    def clear_data(self):
        self.which_in_batch = None
        self.confidence = None
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


class Evaluator_SGD:
    def __init__(self, num_classes, iou_thresh, top_k):
        self.top_k = top_k
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.num_connected_target = 0.0
        self.motif_total = 0.0
        self.motif_correct = 0.0
        self.result_dict = {20: 0.0, 50: 0.0, 100: 0.0}

        self.result_per_class = {k: torch.tensor([0.0 for i in range(self.num_classes)]) for k in self.top_k}
        self.num_conn_target_per_class = torch.tensor([0.0 for i in range(self.num_classes)])

        # man, person, woman, people, boy, girl, lady, child, kid, men  # tree, plant  # plane, airplane
        self.equiv = [[1, 5, 11, 23, 38, 44, 121, 124, 148, 149], [0, 50], [92, 137]]
        # vehicle -> car, bus, motorcycle, truck, vehicle
        # animal -> zebra, sheep, horse, giraffe, elephant, dog, cow, cat, bird, bear, animal
        # food -> vegetable, pizza, orange, fruit, banana, food
        self.unsymm_equiv = {123: [14, 63, 95, 87, 123], 108: [89, 102, 67, 72, 71, 81, 96, 105, 90, 111, 108], 60: [145, 106, 142, 144, 77, 60]}

        self.which_in_batch = None
        self.connected_pred = None
        self.confidence = None
        self.relation_pred = None
        self.relation_target = None

        self.subject_cat_pred = None
        self.object_cat_pred = None
        self.subject_cat_target = None
        self.object_cat_target = None

        self.cat_subject_confidence = None
        self.cat_object_confidence = None

        self.subject_bbox_pred = None
        self.object_bbox_pred = None
        self.subject_bbox_target = None
        self.object_bbox_target = None

    def iou(self, bbox_target, bbox_pred):
        mask_pred = torch.zeros(32, 32)
        mask_pred[int(bbox_pred[2]):int(bbox_pred[3]), int(bbox_pred[0]):int(bbox_pred[1])] = 1
        mask_target = torch.zeros(32, 32)
        mask_target[int(bbox_target[2]):int(bbox_target[3]), int(bbox_target[0]):int(bbox_target[1])] = 1
        intersect = torch.sum(torch.logical_and(mask_target, mask_pred))
        union = torch.sum(torch.logical_or(mask_target, mask_pred))
        if union == 0:
            return 0
        else:
            return float(intersect) / float(union)

    def accumulate_pred(self, which_in_batch, relation_pred, super_relation_pred, subject_cat_pred, object_cat_pred, subject_bbox_pred, object_bbox_pred,
                        cat_subject_confidence, cat_object_confidence, connectivity):
        if self.relation_pred is None:
            self.which_in_batch = which_in_batch.repeat(3)

            ins_pair_confidence = cat_subject_confidence + cat_object_confidence
            self.confidence = torch.hstack((torch.max(relation_pred[:, :15], dim=1)[0],
                                            torch.max(relation_pred[:, 15:26], dim=1)[0],
                                            torch.max(relation_pred[:, 26:], dim=1)[0]))

            self.confidence += connectivity.repeat(3) + ins_pair_confidence.repeat(3)
            self.relation_pred = torch.hstack((torch.argmax(relation_pred[:, :15], dim=1),
                                               torch.argmax(relation_pred[:, 15:26], dim=1) + 15,
                                               torch.argmax(relation_pred[:, 26:], dim=1) + 26))

            self.subject_cat_pred = subject_cat_pred.repeat(3)
            self.object_cat_pred = object_cat_pred.repeat(3)
            self.subject_bbox_pred = subject_bbox_pred.repeat(3, 1)
            self.object_bbox_pred = object_bbox_pred.repeat(3, 1)

        else:
            self.which_in_batch = torch.hstack((self.which_in_batch, which_in_batch.repeat(3)))

            ins_pair_confidence = cat_subject_confidence + cat_object_confidence
            confidence = torch.hstack((torch.max(relation_pred[:, :15], dim=1)[0],
                                       torch.max(relation_pred[:, 15:26], dim=1)[0],
                                       torch.max(relation_pred[:, 26:], dim=1)[0]))
            confidence += connectivity.repeat(3) + ins_pair_confidence.repeat(3)
            self.confidence = torch.hstack((self.confidence, confidence))

            relation_pred_candid = torch.hstack((torch.argmax(relation_pred[:, :15], dim=1),
                                                 torch.argmax(relation_pred[:, 15:26], dim=1) + 15,
                                                 torch.argmax(relation_pred[:, 26:], dim=1) + 26))
            self.relation_pred = torch.hstack((self.relation_pred, relation_pred_candid))

            self.subject_cat_pred = torch.hstack((self.subject_cat_pred, subject_cat_pred.repeat(3)))
            self.object_cat_pred = torch.hstack((self.object_cat_pred, object_cat_pred.repeat(3)))
            self.subject_bbox_pred = torch.vstack((self.subject_bbox_pred, subject_bbox_pred.repeat(3, 1)))
            self.object_bbox_pred = torch.vstack((self.object_bbox_pred, object_bbox_pred.repeat(3, 1)))

    def accumulate_target(self, relation_target, subject_cat_target, object_cat_target, subject_bbox_target, object_bbox_target):
        for i in range(len(relation_target)):
            if relation_target[i] is not None:
                relation_target[i] = relation_target[i].repeat(2)
                subject_cat_target[i] = subject_cat_target[i].repeat(2)
                object_cat_target[i] = object_cat_target[i].repeat(2)
                subject_bbox_target[i] = subject_bbox_target[i].repeat(2, 1)
                object_bbox_target[i] = object_bbox_target[i].repeat(2, 1)

        self.relation_target = relation_target
        self.subject_cat_target = subject_cat_target
        self.object_cat_target = object_cat_target
        self.subject_bbox_target = subject_bbox_target
        self.object_bbox_target = object_bbox_target

    def compare_object_cat(self, pred_cat, target_cat):
        if pred_cat == target_cat:
            return True
        for group in self.equiv:
            if pred_cat in group and target_cat in group:
                return True
        for key in self.unsymm_equiv:
            if pred_cat == key and target_cat in self.unsymm_equiv[key]:
                return True
            elif target_cat == key and pred_cat in self.unsymm_equiv[key]:
                return True
        return False

    """
    for each target <subject, relation, object> triplet, find among all top k predicted <subject, relation, object> triplets
    if there is any one with matched subject, relationship, and object categories, and iou>=0.5 for subject and object bounding boxes
    """
    def compute(self, hierarchical_pred, per_class=False):
        for curr_image in range(len(self.relation_target)):
            if self.relation_target[curr_image] is None:
                continue
            curr_image_pred = self.which_in_batch == curr_image

            for i in range(len(self.relation_target[curr_image])):
                if self.relation_target[curr_image][i] == -1:  # if target is not connected
                    continue
                # search in top k most confident predictions in each image
                num_target = torch.sum(self.relation_target[curr_image] != -1)
                num_relation_pred = len(self.relation_pred[curr_image_pred])

                # # As suggested by Neural Motifs, nearly all annotated relationships are between overlapping boxes
                curr_confidence = self.confidence[curr_image_pred]
                sorted_inds = torch.argsort(curr_confidence, dim=0, descending=True)
                this_k = min(self.top_k[-1], num_relation_pred)  # 100
                keep_inds = sorted_inds[:this_k]

                found = False   # found if any one of the three sub-models predict correctly
                for j in range(len(keep_inds)):     # for each target <subject, relation, object> triple, find any match in the top k confident predictions
                    if (self.compare_object_cat(self.subject_cat_target[curr_image][i], self.subject_cat_pred[curr_image_pred][keep_inds][j]) and
                        self.compare_object_cat(self.object_cat_target[curr_image][i], self.object_cat_pred[curr_image_pred][keep_inds][j])):

                        sub_iou = self.iou(self.subject_bbox_target[curr_image][i], self.subject_bbox_pred[curr_image_pred][keep_inds][j])
                        obj_iou = self.iou(self.object_bbox_target[curr_image][i], self.object_bbox_pred[curr_image_pred][keep_inds][j])
                        if sub_iou >= self.iou_thresh and obj_iou >= self.iou_thresh:
                            if self.relation_target[curr_image][i] == self.relation_pred[curr_image_pred][keep_inds][j]:
                                for k in self.top_k:
                                    if j >= k:
                                        continue
                                    self.result_dict[k] += 1.0
                                    if per_class:
                                        self.result_per_class[k][self.relation_target[curr_image][i]] += 1.0
                                found = True
                            if found:
                                break

                self.num_connected_target += 1.0
                self.num_conn_target_per_class[self.relation_target[curr_image][i]] += 1.0

        recall_k = [self.result_dict[k] / self.num_connected_target for k in self.top_k]
        recall_k_per_class = [self.result_per_class[k] / self.num_conn_target_per_class for k in self.top_k]
        mean_recall_k = [torch.nanmean(r) for r in recall_k_per_class]
        return recall_k, recall_k_per_class, mean_recall_k

    def clear_data(self):
        self.which_in_batch = None
        self.confidence = None
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


class Evaluator_SGD_Top3:
    def __init__(self, num_classes, iou_thresh, top_k):
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

        # man, person, woman, people, boy, girl, lady, child, kid, men  # tree, plant  # plane, airplane
        self.equiv = [[1, 5, 11, 23, 38, 44, 121, 124, 148, 149], [0, 50], [92, 137]]
        # vehicle -> car, bus, motorcycle, truck, vehicle
        # animal -> zebra, sheep, horse, giraffe, elephant, dog, cow, cat, bird, bear, animal
        # food -> vegetable, pizza, orange, fruit, banana, food
        self.unsymm_equiv = {123: [14, 63, 95, 87, 123], 108: [89, 102, 67, 72, 71, 81, 96, 105, 90, 111, 108], 60: [145, 106, 142, 144, 77, 60]}

        self.which_in_batch = None
        self.connected_pred = None
        self.confidence = None
        self.relation_pred = None
        self.relation_target = None
        self.super_relation_pred = None

        self.subject_cat_pred = None
        self.object_cat_pred = None
        self.subject_cat_target = None
        self.object_cat_target = None

        self.cat_subject_confidence = None
        self.cat_object_confidence = None

        self.subject_bbox_pred = None
        self.object_bbox_pred = None
        self.subject_bbox_target = None
        self.object_bbox_target = None

    def iou(self, bbox_target, bbox_pred):
        mask_pred = torch.zeros(32, 32)
        mask_pred[int(bbox_pred[2]):int(bbox_pred[3]), int(bbox_pred[0]):int(bbox_pred[1])] = 1
        mask_target = torch.zeros(32, 32)
        mask_target[int(bbox_target[2]):int(bbox_target[3]), int(bbox_target[0]):int(bbox_target[1])] = 1
        intersect = torch.sum(torch.logical_and(mask_target, mask_pred))
        union = torch.sum(torch.logical_or(mask_target, mask_pred))
        if union == 0:
            return 0
        else:
            return float(intersect) / float(union)

    def accumulate_pred(self, which_in_batch, relation_pred, super_relation_pred, subject_cat_pred, object_cat_pred, subject_bbox_pred, object_bbox_pred,
                        cat_subject_confidence, cat_object_confidence, connectivity):
        if self.relation_pred is None:
            self.which_in_batch = which_in_batch
            self.relation_pred = relation_pred
            self.super_relation_pred = super_relation_pred

            self.subject_cat_pred = subject_cat_pred
            self.object_cat_pred = object_cat_pred
            self.subject_bbox_pred = subject_bbox_pred
            self.object_bbox_pred = object_bbox_pred

            ins_pair_confidence = cat_subject_confidence + cat_object_confidence
            self.confidence = connectivity + ins_pair_confidence + torch.max(torch.vstack((torch.max(relation_pred[:, :15], dim=1)[0],
                                                                                           torch.max(relation_pred[:, 15:26], dim=1)[0],
                                                                                           torch.max(relation_pred[:, 26:], dim=1)[0])), dim=0)[0]  # values
        else:
            self.which_in_batch = torch.hstack((self.which_in_batch, which_in_batch))
            self.relation_pred = torch.vstack((self.relation_pred, relation_pred))
            self.super_relation_pred = torch.vstack((self.super_relation_pred, super_relation_pred))
            self.subject_cat_pred = torch.hstack((self.subject_cat_pred, subject_cat_pred))
            self.object_cat_pred = torch.hstack((self.object_cat_pred, object_cat_pred))
            self.subject_bbox_pred = torch.vstack((self.subject_bbox_pred, subject_bbox_pred))
            self.object_bbox_pred = torch.vstack((self.object_bbox_pred, object_bbox_pred))

            ins_pair_confidence = cat_subject_confidence + cat_object_confidence
            confidence = connectivity + ins_pair_confidence + torch.max(torch.vstack((torch.max(relation_pred[:, :15], dim=1)[0],
                                                                                      torch.max(relation_pred[:, 15:26], dim=1)[0],
                                                                                      torch.max(relation_pred[:, 26:], dim=1)[0])), dim=0)[0]  # values
            self.confidence = torch.hstack((self.confidence, confidence))

    def accumulate_target(self, relation_target, subject_cat_target, object_cat_target, subject_bbox_target, object_bbox_target):
        self.relation_target = relation_target
        self.subject_cat_target = subject_cat_target
        self.object_cat_target = object_cat_target
        self.subject_bbox_target = subject_bbox_target
        self.object_bbox_target = object_bbox_target

    def compare_object_cat(self, pred_cat, target_cat):
        if pred_cat == target_cat:
            return True
        for group in self.equiv:
            if pred_cat in group and target_cat in group:
                return True
        for key in self.unsymm_equiv:
            if pred_cat == key and target_cat in self.unsymm_equiv[key]:
                return True
            elif target_cat == key and pred_cat in self.unsymm_equiv[key]:
                return True
        return False

    """
    for each target <subject, relation, object> triplet, find among all top k predicted <subject, relation, object> triplets
    if there is any one with matched subject, relationship, and object categories, and iou>=0.5 for subject and object bounding boxes
    """
    def compute(self, hierarchical_pred, per_class=False):
        for curr_image in range(len(self.relation_target)):
            if self.relation_target[curr_image] is None:
                continue
            curr_image_pred = self.which_in_batch == curr_image

            for i in range(len(self.relation_target[curr_image])):
                if self.relation_target[curr_image][i] == -1:  # if target is not connected
                    continue
                # search in top k most confident predictions in each image
                num_target = torch.sum(self.relation_target[curr_image] != -1)
                num_relation_pred = len(self.relation_pred[curr_image_pred])

                # # As suggested by Neural Motifs, nearly all annotated relationships are between overlapping boxes
                curr_confidence = self.confidence[curr_image_pred]
                sorted_inds = torch.argsort(curr_confidence, dim=0, descending=True)
                this_k = min(self.top_k[-1], num_relation_pred)  # 100
                keep_inds = sorted_inds[:this_k]

                found = False   # found if any one of the three sub-models predict correctly
                found_top1 = False  # found if only the most confident one of the three sub-models predict correctly
                for j in range(len(keep_inds)):     # for each target <subject, relation, object> triple, find any match in the top k confident predictions
                    if (self.compare_object_cat(self.subject_cat_target[curr_image][i], self.subject_cat_pred[curr_image_pred][keep_inds][j]) and
                        self.compare_object_cat(self.object_cat_target[curr_image][i], self.object_cat_pred[curr_image_pred][keep_inds][j])):

                        sub_iou = self.iou(self.subject_bbox_target[curr_image][i], self.subject_bbox_pred[curr_image_pred][keep_inds][j])
                        obj_iou = self.iou(self.object_bbox_target[curr_image][i], self.object_bbox_pred[curr_image_pred][keep_inds][j])
                        if sub_iou >= self.iou_thresh and obj_iou >= self.iou_thresh:
                            for k in self.top_k:
                                if j >= max(k, num_target):     # in few cases, the number of targets is greater than k=20
                                    continue

                            if hierarchical_pred and not found:     # if already found, skip
                                relation_pred_1 = self.relation_pred[curr_image_pred][keep_inds][j][:15]  # geometric
                                relation_pred_2 = self.relation_pred[curr_image_pred][keep_inds][j][15:26]  # possessive
                                relation_pred_3 = self.relation_pred[curr_image_pred][keep_inds][j][26:]  # semantic
                                if self.relation_target[curr_image][i] == torch.argmax(relation_pred_1) \
                                        or self.relation_target[curr_image][i] == torch.argmax(relation_pred_2) + 15 \
                                        or self.relation_target[curr_image][i] == torch.argmax(relation_pred_3) + 26:
                                    for k in self.top_k:
                                        if j >= max(k, num_target):
                                            continue
                                        self.result_dict[k] += 1.0
                                        if per_class:
                                            self.result_per_class[k][self.relation_target[curr_image][i]] += 1.0
                                    found = True

                            if not found_top1:
                                curr_super = torch.argmax(self.super_relation_pred[curr_image_pred][keep_inds][j])
                                relation_preds = [torch.argmax(self.relation_pred[curr_image_pred][keep_inds][j][:15]),
                                                  torch.argmax(self.relation_pred[curr_image_pred][keep_inds][j][15:26]) + 15,
                                                  torch.argmax(self.relation_pred[curr_image_pred][keep_inds][j][26:]) + 26]
                                if self.relation_target[curr_image][i] == relation_preds[curr_super]:
                                    for k in self.top_k:
                                        if j >= max(k, num_target):
                                            continue
                                        self.result_dict_top1[k] += 1.0
                                        if per_class:
                                            self.result_per_class_top1[k][self.relation_target[curr_image][i]] += 1.0
                                    found_top1 = True

                            if (hierarchical_pred and found and found_top1) or (not hierarchical_pred and found_top1):
                                break

                self.num_connected_target += 1.0
                self.num_conn_target_per_class[self.relation_target[curr_image][i]] += 1.0

        recall_k = [self.result_dict[k] / self.num_connected_target for k in self.top_k]
        recall_k_per_class = [self.result_per_class[k] / self.num_conn_target_per_class for k in self.top_k]
        recall_k_per_class_top1 = [self.result_per_class_top1[k] / self.num_conn_target_per_class for k in self.top_k]
        mean_recall_k = [torch.nanmean(r) for r in recall_k_per_class]
        recall_k_top1 = [self.result_dict_top1[k] / self.num_connected_target for k in self.top_k]
        mean_recall_k_top1 = [torch.nanmean(r) for r in recall_k_per_class_top1]
        return recall_k, recall_k_per_class, mean_recall_k

    def clear_data(self):
        self.which_in_batch = None
        self.confidence = None
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