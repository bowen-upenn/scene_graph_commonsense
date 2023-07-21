# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer


inv_map = torch.tensor([0,  1,  2,  3,  4,  5,  6,  8, 10, 22, 23, 29, 31, 32, 33, 43,  9, 16,
        17, 20, 27, 30, 36, 42, 48, 49, 50,  7, 11, 12, 13, 14, 15, 18, 19, 21,
        24, 25, 26, 28, 34, 35, 37, 38, 39, 40, 41, 44, 45, 46, 47, 51])

class RelTR(nn.Module):
    """ RelTR: Relation Transformer for Scene Graph Generation """
    def __init__(self, backbone, transformer, num_classes, num_rel_classes, num_rel_prior, num_rel_geometric, num_rel_possessive,
                 num_rel_semantic, num_entities, num_triplets, aux_loss=False, hierar=False, matcher=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of entity classes
            num_entities: number of entity queries
            num_triplets: number of coupled subject/object queries
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_entities = num_entities
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.entity_embed = nn.Embedding(num_entities, hidden_dim*2)
        self.triplet_embed = nn.Embedding(num_triplets, hidden_dim*3)
        self.so_embed = nn.Embedding(2, hidden_dim) # subject and object encoding

        # entity prediction
        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # mask head
        self.so_mask_conv = nn.Sequential(torch.nn.Upsample(size=(28, 28)),
                                          nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=3, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(32))
        self.so_mask_fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 128))

        # subject/object label classfication and box regression
        self.sub_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        ## ADD-ON #################################################
        self.hierar = hierar
        if self.hierar:
            # bayesian predicate classification with relation hierarchy
            self.fc_rel = nn.Linear(hidden_dim * 2 + 128, hidden_dim)
            self.fc_rel_prior = nn.Linear(hidden_dim, num_rel_prior + 1)
            self.fc_rel_geo = nn.Linear(hidden_dim, num_rel_geometric)
            self.fc_rel_pos = nn.Linear(hidden_dim, num_rel_possessive)
            self.fc_rel_sem = nn.Linear(hidden_dim, num_rel_semantic)
            # self.rel_class_embed = nn.Sequential(
            #     nn.Linear(hidden_dim * 2 + 128, hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim, num_rel_classes + 1)
            # )
        else:
            # flat predicate classification
            self.rel_class_embed = MLP(hidden_dim * 2 + 128, hidden_dim, num_rel_classes + 1, 2)
        ##########################################################

    ## ADD-ON #################################################
    def bayesian_head(self, hidden):
        # hidden expects size [_, batch_size, num_queries, hidden_dim]
        hidden = F.relu(self.fc_rel(hidden))
        rel_prior = self.fc_rel_prior(hidden)

        rel_geo = self.fc_rel_geo(hidden)# * rel_prior[:, :, :, 0].unsqueeze(-1)  # geometric
        rel_pos = self.fc_rel_pos(hidden)# * rel_prior[:, :, :, 1].unsqueeze(-1)  # possessive
        rel_sem = self.fc_rel_sem(hidden)# * rel_prior[:, :, :, 2].unsqueeze(-1)  # semantic

        outputs_class_rel = torch.cat((rel_prior[:, :, :, -2].unsqueeze(-1), rel_geo, rel_pos, rel_sem,
                                       rel_prior[:, :, :, -1].unsqueeze(-1)), dim=-1)
        outputs_class_rel = outputs_class_rel[:, :, :, inv_map]

        # # hidden expects size [_, batch_size, num_queries, hidden_dim]
        # hidden = self.rel_dropout(F.relu(self.fc_rel(hidden)))
        # rel_prior = F.log_softmax(self.fc_rel_prior(hidden), dim=-1)
        #
        # rel_geo = self.fc_rel_geo(hidden)  # geometric
        # rel_geo = F.log_softmax(rel_geo, dim=-1) + rel_prior[:, :, :, 0].unsqueeze(-1)
        # rel_pos = self.fc_rel_pos(hidden)  # possessive
        # rel_pos = F.log_softmax(rel_pos, dim=-1) + rel_prior[:, :, :, 1].unsqueeze(-1)
        # rel_sem = self.fc_rel_sem(hidden)  # semantic
        # rel_sem = F.log_softmax(rel_sem, dim=-1) + rel_prior[:, :, :, 2].unsqueeze(-1)

        return rel_prior, outputs_class_rel #rel_prior, rel_geo, rel_pos, rel_sem
    ##########################################################

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the entity classification logits (including no-object) for all entity queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": the normalized entity boxes coordinates for all entity queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "sub_logits": the subject classification logits
               - "obj_logits": the object classification logits
               - "sub_boxes": the normalized subject boxes coordinates
               - "obj_boxes": the normalized object boxes coordinates
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, hs_t, so_masks, _ = self.transformer(self.input_proj(src), mask, self.entity_embed.weight,
                                                 self.triplet_embed.weight, pos[-1], self.so_embed.weight)
        so_masks = so_masks.detach()
        so_masks = self.so_mask_conv(so_masks.view(-1, 2, src.shape[-2],src.shape[-1])).view(hs_t.shape[0], hs_t.shape[1], hs_t.shape[2],-1)
        so_masks = self.so_mask_fc(so_masks)

        hs_sub, hs_obj = torch.split(hs_t, self.hidden_dim, dim=-1)

        outputs_class = self.entity_class_embed(hs)
        outputs_coord = self.entity_bbox_embed(hs).sigmoid()

        outputs_class_sub = self.sub_class_embed(hs_sub)
        outputs_coord_sub = self.sub_bbox_embed(hs_sub).sigmoid()

        outputs_class_obj = self.obj_class_embed(hs_obj)
        outputs_coord_obj = self.obj_bbox_embed(hs_obj).sigmoid()

        ## ADD-ON #################################################
        if self.hierar:
            # outputs_class_rel_prior, outputs_class_geo, outputs_class_pos, outputs_class_sem \
            #     = self.bayesian_head(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))
            outputs_class_rel_prior, outputs_class_rel = self.bayesian_head(torch.cat((hs_sub, hs_obj, so_masks), dim=-1).detach())
            # rel_prior id -2 is __back_ground__, and -1 is no_object
            # outputs_class_rel = torch.cat((outputs_class_rel_prior[:, :, :, -2].unsqueeze(-1), outputs_class_geo, outputs_class_pos, outputs_class_sem,
            #                                outputs_class_rel_prior[:, :, :, -1].unsqueeze(-1)), dim=-1)
        ###########################################################
        else:
            outputs_class_rel = self.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
               'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
               'rel_logits': outputs_class_rel[-1]}
        ## ADD-ON #################################################
        if self.hierar:
            out['rel_prior_logits'] = outputs_class_rel_prior[-1]
        ###########################################################

        if self.aux_loss:
            ## ADD-ON #################################################
            if self.hierar:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                        outputs_class_obj, outputs_coord_obj, outputs_class_rel, outputs_class_rel_prior)
            ###########################################################
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                        outputs_class_obj, outputs_coord_obj, outputs_class_rel)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                      outputs_class_obj, outputs_coord_obj, outputs_class_rel, outputs_class_rel_prior=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        ## ADD-ON #################################################
        if self.hierar:
            return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f,
                     'rel_logits': g, 'rel_prior_logits': h}
                    for a, b, c, d, e, f, g, h in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                                   outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1],
                                                   outputs_class_rel[:-1], outputs_class_rel_prior[:-1])]
        ###########################################################
        else:
            return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f,
                     'rel_logits': g}
                    for a, b, c, d, e, f, g in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                                   outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1],
                                                   outputs_class_rel[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for RelTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_rel_classes, matcher, weight_dict, eos_coef, losses,
                 num_rel_prior=4, num_rel_geo=15, num_rel_pos=11, num_rel_sem=24, hierar=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.num_rel_classes = 51 if num_classes == 151 else 31 # Using entity class numbers to adapt rel class numbers
        empty_weight_rel = torch.ones(num_rel_classes+1)
        empty_weight_rel[-1] = self.eos_coef
        self.register_buffer('empty_weight_rel', empty_weight_rel)

        ## ADD-ON #################################################
        self.num_rel_prior = num_rel_prior
        self.num_rel_geo = num_rel_geo
        self.num_rel_pos = num_rel_pos
        self.num_rel_sem = num_rel_sem
        self.hierar = hierar
        self.map = torch.tensor([ 0,  1,  2,  3,  4,  5,  6, 27,  7, 16,  8, 28, 29, 30, 31, 32, 17, 18,
                33, 34, 19, 35,  9, 10, 36, 37, 38, 20, 39, 11, 21, 12, 13, 14, 40, 41,
                22, 42, 43, 44, 45, 46, 23, 15, 47, 48, 49, 50, 24, 25, 26, 51])
        ###########################################################

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Entity/subject/object Classification loss
        """
        assert 'pred_logits' in outputs

        pred_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices[0])
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o

        sub_logits = outputs['sub_logits']
        obj_logits = outputs['obj_logits']

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices[1])])
        target_relo_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices[1])])

        target_sub_classes = torch.full(sub_logits.shape[:2], self.num_classes, dtype=torch.int64, device=sub_logits.device)
        target_obj_classes = torch.full(obj_logits.shape[:2], self.num_classes, dtype=torch.int64, device=obj_logits.device)

        target_sub_classes[rel_idx] = target_rels_classes_o
        target_obj_classes[rel_idx] = target_relo_classes_o

        target_classes = torch.cat((target_classes, target_sub_classes, target_obj_classes), dim=1)
        src_logits = torch.cat((pred_logits, sub_logits, obj_logits), dim=1)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')

        loss_weight = torch.cat((torch.ones(pred_logits.shape[:2]).to(pred_logits.device), indices[2]*0.5, indices[3]*0.5), dim=-1)
        losses = {'loss_ce': (loss_ce * loss_weight).sum()/self.empty_weight[target_classes].sum()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
            losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rels_classes_o)[0]
            losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_relo_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        ## ADD-ON #################################################
        if self.hierar:
            pred_logits = outputs['rel_prior_logits']
        ###########################################################
        else:
            pred_logits = outputs['rel_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_annotations"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the entity/subject/object bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices[0])
        pred_boxes = outputs['pred_boxes'][idx]
        target_entry_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices[0])], dim=0)

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 0]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        target_relo_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 1]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        rels_boxes = outputs['sub_boxes'][rel_idx]
        relo_boxes = outputs['obj_boxes'][rel_idx]

        src_boxes = torch.cat((pred_boxes, rels_boxes, relo_boxes), dim=0)
        target_boxes = torch.cat((target_entry_boxes, target_rels_boxes, target_relo_boxes), dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_relations(self, outputs, targets, indices, num_boxes, log=True):
        """Compute the predicate classification loss
        """
        assert 'rel_logits' in outputs

        src_logits = outputs['rel_logits']

        idx = self._get_src_permutation_idx(indices[1])
        target_classes_o = torch.cat([t["rel_annotations"][J, 2] for t, (_, J) in zip(targets, indices[1])])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o  # size [batch_size, num_queries]

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)

        if self.hierar:
            target_classes_prior = self.map[target_classes].to(target_classes.device)
            target_classes_prior[torch.logical_and(target_classes_prior > 0, target_classes_prior < self.num_rel_geo + 1)] = 0  # geometric
            target_classes_prior[torch.logical_and(target_classes_prior >= self.num_rel_geo + 1, target_classes_prior < self.num_rel_geo + self.num_rel_pos + 1)] = 1  # possessive
            target_classes_prior[torch.logical_and(target_classes_prior >= self.num_rel_geo + self.num_rel_pos + 1, target_classes_prior < self.num_rel_classes - 1)] = 2  # semantic
            target_classes_prior[target_classes == self.num_rel_classes - 1] = 3  # background relation
            target_classes_prior[target_classes == self.num_rel_classes] = 4  # no object

            if len(outputs['rel_prior_logits'][target_classes <= 2]) > 0:
                loss_ce += F.cross_entropy(outputs['rel_prior_logits'][target_classes <= 2][:, :3], target_classes_prior[target_classes <= 2])

        losses = {'loss_rel': loss_ce}

        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        # ## ADD-ON #################################################
        # loss_ce = 0.0
        # if self.hierar:
        #     target_classes_prior = target_classes.clone()
        #     target_classes_prior[target_classes < self.num_rel_geo] = 0  # geometric
        #     target_classes_prior[torch.logical_and(target_classes >= self.num_rel_geo, target_classes < self.num_rel_geo + self.num_rel_pos)] = 1  # possessive
        #     target_classes_prior[torch.logical_and(target_classes >= self.num_rel_geo + self.num_rel_pos, target_classes < self.num_rel_classes - 1)] = 2  # semantic
        #     target_classes_prior[target_classes == self.num_rel_classes - 1] = 3  # background relation
        #     target_classes_prior[target_classes == self.num_rel_classes] = 4  # no object
        #
        #     empty_weight_rel_prior = torch.ones(self.num_rel_prior + 1).to(target_classes.device)
        #     empty_weight_rel_prior[-1] = self.eos_coef  # weight for no object
        #     loss_ce += F.cross_entropy(outputs['rel_prior_logits'].transpose(1, 2), target_classes_prior, empty_weight_rel_prior)
        #     # print('outputs[rel_prior_logits].transpose(1, 2)', outputs['rel_prior_logits'].transpose(1, 2).shape, target_classes_prior.shape)
        #
        #     begin_pos = self.num_rel_geo
        #     begin_sem = self.num_rel_geo + self.num_rel_pos
        #     samples_geo = target_classes < self.num_rel_geo
        #     samples_pos = torch.logical_and(target_classes >= self.num_rel_geo, target_classes < self.num_rel_geo + self.num_rel_pos)
        #     samples_sem = torch.logical_and(target_classes >= self.num_rel_geo + self.num_rel_pos, target_classes < self.num_rel_classes - 1)
        #     rel_error = torch.zeros(3)
        #     # print('samples_geo', samples_geo.shape, 'samples_pos', samples_pos.shape, 'samples_sem', samples_sem.shape)
        #     if torch.sum(samples_geo) > 0:
        #         loss_ce += F.cross_entropy(src_logits[:, :, :begin_pos][samples_geo], target_classes[samples_geo])
        #         rel_error[0] = 100 - accuracy(src_logits[:, :, :begin_pos][samples_geo], target_classes[samples_geo])[0]
        #     if torch.sum(samples_pos) > 0:
        #         loss_ce += F.cross_entropy(src_logits[:, :, begin_pos:begin_sem][samples_pos], target_classes[samples_pos] - self.num_rel_geo)
        #         rel_error[1] = 100 - accuracy(src_logits[:, :, begin_pos:begin_sem][samples_pos], target_classes[samples_pos] - self.num_rel_geo)[0]
        #     if torch.sum(samples_sem) > 0:
        #         loss_ce += F.cross_entropy(src_logits[:, :, begin_sem:][samples_sem], target_classes[samples_sem] - self.num_rel_geo - self.num_rel_pos)
        #         rel_error[2] = 100 - accuracy(src_logits[:, :, begin_sem:][samples_sem], target_classes[samples_sem] - self.num_rel_geo - self.num_rel_pos)[0]
        #
        #     # print('src_logits[:, :, 1:begin_pos][samples_geo]', src_logits[:, :, 1:begin_pos][samples_geo].shape, target_classes[samples_geo].shape)
        #     # print('src_logits[:, :, begin_pos:begin_sem][samples_pos]', src_logits[:, :, begin_pos:begin_sem][samples_pos].shape, target_classes[samples_pos].shape)
        #     # print('src_logits[:, :, begin_sem:-1][samples_sem]', src_logits[:, :, begin_sem:-1][samples_sem].shape, target_classes[samples_sem].shape, '\n')
        #
        #     losses = {'loss_rel': loss_ce}
        #     if log:
        #         losses['rel_error'] = rel_error
        # ###########################################################
        # else:
        #     loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)
        #     losses = {'loss_rel': loss_ce}
        #
        #     if log:
        #         losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'relations': self.loss_relations
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"])+len(t["rel_annotations"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'relations':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    num_classes = 151 if args.dataset != 'oi' else 289 # some entity categories in OIV6 are deactivated.
    num_rel_classes = 51 if args.dataset != 'oi' else 31

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    matcher = build_matcher(args)
    model = RelTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_rel_classes = num_rel_classes,
        ## ADD-ON #################################################
        num_rel_prior=args.num_rel_prior,
        num_rel_geometric=args.num_rel_geometric,
        num_rel_possessive=args.num_rel_possessive,
        num_rel_semantic=args.num_rel_semantic,
        ###########################################################
        num_entities=args.num_entities,
        num_triplets=args.num_triplets,
        aux_loss=args.aux_loss,
        hierar=args.hierar,
        matcher=matcher)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_rel'] = args.rel_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', "relations"]

    ## ADD-ON #################################################
    criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, num_rel_prior=args.num_rel_prior, num_rel_geo=args.num_rel_geometric,
                             num_rel_pos=args.num_rel_possessive, num_rel_sem=args.num_rel_semantic, hierar=args.hierar)
    ###########################################################
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

