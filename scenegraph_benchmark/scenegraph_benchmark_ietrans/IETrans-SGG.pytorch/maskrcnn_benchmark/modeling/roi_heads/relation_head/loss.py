# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        loss_type="CrossEntropy",
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        rel_labels = torch.argmax(rel_labels, dim=1)  # convert one-hot vectors into integer labels
        loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss


class RelationHierarchicalLossComputation(object):
    def __init__(
            self,
            attri_on,
            num_attri_cat,
            max_num_attri,
            attribute_sampling,
            attribute_bgfg_ratio,
            use_label_smoothing,
            predicate_proportion,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5, ] + predicate_proportion)).cuda()

        # Assume NLL loss here.
        # TODO: is class_weight a pre-defined constant?
        # class_weight = 1 - relation_count / torch.sum(relation_count)
        self.criterion_loss = nn.CrossEntropyLoss()
        self.geo_criterion_loss = nn.NLLLoss()
        self.pos_criterion_loss = nn.NLLLoss()
        self.sem_criterion_loss = nn.NLLLoss()
        self.super_criterion_loss = nn.NLLLoss()
        # Hierarchical label
        self.geo_mapping = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 7, 22: 8, 23: 9, 29: 10,
            31: 11, 32: 12, 33: 13, 43: 14
        }

        self.pos_mapping = {
            9: 0, 16: 1, 17: 2, 20: 3, 27: 4, 30: 5, 36: 6, 42: 7, 48: 8, 49: 9, 50: 10
        }
        self.sem_mapping = {
            7: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 18: 6, 19: 7, 21: 8, 24: 9, 25: 10,
            26: 11, 28: 12, 34: 13, 35: 14, 37: 15, 38: 16, 39: 17, 40: 18, 41: 19,
            44: 20, 45: 21, 46: 22, 47: 23
        }
        self.geo_label_tensor = torch.tensor([x for x in self.geo_mapping.keys()])
        self.pos_label_tensor = torch.tensor([x for x in self.pos_mapping.keys()])
        self.sem_label_tensor = torch.tensor([x for x in self.sem_mapping.keys()])

    # Assume no refine obj, only relation prediction
    # relation_logits is [geo, pos, sem, super]
    def __call__(self, proposals, rel_labels, rel1_prob, rel2_prob, rel3_prob, super_rel_prob, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        refine_obj_logits = cat(refine_logits, dim=0)
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        rel_labels = cat(rel_labels, dim=0)  # (rel, 51)
        rel_labels = torch.argmax(rel_labels, dim=1)  # (rel, 1)
        rel1_prob = cat(rel1_prob, dim=0)  # (rel, 15)
        rel2_prob = cat(rel2_prob, dim=0)  # (rel, 11)
        rel3_prob = cat(rel3_prob, dim=0)  # (rel, 24)
        super_rel_prob = cat(super_rel_prob, dim=0)   # (rel, 4)
        cur_device = rel_labels.device

        # A mask to select labels within specific super category
        geo_label_tensor = self.geo_label_tensor.to(cur_device)
        pos_label_tensor = self.pos_label_tensor.to(cur_device)
        sem_label_tensor = self.sem_label_tensor.to(cur_device)
        # print(rel_labels.device)
        # print(self.geo_label_tensor.device)
        geo_label_mask = (rel_labels.unsqueeze(1) == geo_label_tensor).any(1)
        pos_label_mask = (rel_labels.unsqueeze(1) == pos_label_tensor).any(1)
        sem_label_mask = (rel_labels.unsqueeze(1) == sem_label_tensor).any(1)
        # Suppose 0 is geo, 1 is pos, 3 is sem
        # super_rel_label = pos_label_mask * 1 + sem_label_mask * 2
        # Suppose 0 is bg, 1 is geo, 2 is pos, 3 is sem
        super_rel_label = geo_label_mask + pos_label_mask * 2 + sem_label_mask * 3

        loss_relation = 0
        geo_labels = rel_labels[geo_label_mask]
        geo_labels = torch.tensor([self.geo_mapping[label.item()] for label in geo_labels]).to(cur_device)
        pos_labels = rel_labels[pos_label_mask]
        pos_labels = torch.tensor([self.pos_mapping[label.item()] for label in pos_labels]).to(cur_device)
        sem_labels = rel_labels[sem_label_mask]
        sem_labels = torch.tensor([self.sem_mapping[label.item()] for label in sem_labels]).to(cur_device)

        if geo_labels.shape[0] > 0:
            loss_relation += self.geo_criterion_loss(rel1_prob[geo_label_mask], geo_labels.long())
        if pos_labels.shape[0] > 0:
            loss_relation += self.pos_criterion_loss(rel2_prob[pos_label_mask], pos_labels.long())
        if sem_labels.shape[0] > 0:
            loss_relation += self.sem_criterion_loss(rel3_prob[sem_label_mask], sem_labels.long())
        if super_rel_label.shape[0] > 0:
            loss_relation += self.super_criterion_loss(super_rel_prob, super_rel_label.long())

        return loss_relation, loss_refine_obj


class WRelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        cfg,
        rel_loss='bce',
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.rel_loss = rel_loss
        self.obj_criterion_loss = nn.CrossEntropyLoss()
        if self.rel_loss == 'bce':
            self.rel_criterion_loss = nn.BCEWithLogitsLoss()
        elif self.rel_loss == 'ce':
            self.rel_criterion_loss = CEForSoftLabel()
        elif self.rel_loss == "ce_rwt":
            self.rel_criterion_loss = ReweightingCE()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, pos_weight=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        refine_obj_logits = refine_logits
        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        loss_relation = self.rel_criterion_loss(relation_logits, rel_labels)

        # self.obj_criterion_loss.to(fg_labels.device)
        loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, fg_labels.long())
        return loss_relation, loss_refine_obj



class CEForSoftLabel(nn.Module):
    """
    Given a soft label, choose the class with max class as the GT.
    converting label like [0.1, 0.3, 0.6] to [0, 0, 1] and apply CrossEntropy
    """
    def __init__(self, reduction="mean"):
        super(CEForSoftLabel, self).__init__()
        self.reduction=reduction

    def forward(self, input, target, pos_weight=None):
        final_target = torch.zeros_like(target)
        final_target[torch.arange(0, target.size(0)), target.argmax(1)] = 1.
        target = final_target
        x = F.log_softmax(input, 1)
        loss = torch.sum(- x * target, dim=1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class ReweightingCE(nn.Module):
    """
    Given a soft label, choose the class with max class as the GT.
    converting label like [0.1, 0.3, 0.6] to [0, 0, 1] and apply CrossEntropy
    """
    def __init__(self, reduction="mean"):
        super(ReweightingCE, self).__init__()
        self.reduction=reduction

    def forward(self, input, target):
        """
        Args:
            input: the prediction
            target: [N, N_classes]. For each slice [weight, 0, 0, 1, 0, ...]
                we need to extract weight.
        Returns:

        """
        final_target = torch.zeros_like(target)
        final_target[torch.arange(0, target.size(0)), target.argmax(1)] = 1.
        idxs = (target[:, 0] != 1).nonzero().squeeze()
        weights = torch.ones_like(target[:, 0])
        weights[idxs] = -target[:, 0][idxs]
        target = final_target
        x = F.log_softmax(input, 1)
        loss = torch.sum(- x * target, dim=1)*weights
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def make_weaksup_relation_loss_evaluator(cfg):
    loss_evaluator = WRelationLossComputation(
        cfg,
        cfg.WSUPERVISE.LOSS_TYPE,
    )

    return loss_evaluator


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def make_roi_relation_loss_evaluator(cfg):
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "MotifHierarchicalPredictor" or \
            cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "TransformerHierPredictor" or \
            cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "VCTreeHierPredictor":
        loss_evaluator = RelationHierarchicalLossComputation(
            cfg.MODEL.ATTRIBUTE_ON,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
            cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
            cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        )
    else:
        loss_evaluator = RelationLossComputation(
            cfg.MODEL.ATTRIBUTE_ON,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
            cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
            cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        )

    return loss_evaluator

