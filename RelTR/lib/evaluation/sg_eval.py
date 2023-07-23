"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import numpy as np
from functools import reduce
import math
from lib.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
np.set_printoptions(precision=3)
import torch
from util.misc import accuracy

inv_map = torch.tensor([0,  1,  2,  3,  4,  5,  6,  8, 10, 22, 23, 29, 31, 32, 33, 43,  9, 16,
        17, 20, 27, 30, 36, 42, 48, 49, 50,  7, 11, 12, 13, 14, 15, 18, 19, 21,
        24, 25, 26, 28, 34, 35, 37, 38, 39, 40, 41, 44, 45, 46, 47, 51])

class BasicSceneGraphEvaluator:
    def __init__(self, mode, matcher, multiple_preds=False):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: []}
        self.multiple_preds = multiple_preds
        self.matcher = matcher

    @classmethod
    def all_modes(cls, matcher, **kwargs):
        evaluators = {m: cls(mode=m, matcher=matcher, **kwargs) for m in ('sgdet', 'sgcls', 'predcls')}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ('preddet', 'phrdet')}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_scores, viz_dict=None, iou_thresh=0.5,
                                   ## ADD-ON #################################################
                                   hierar=False, num_rel_prior=4, num_rel_geometric=15,
                                   num_rel_possessive=11, num_rel_semantic=24):
                                   ###########################################################

        res = evaluate_from_dict(gt_entry, pred_scores, self.mode, self.result_dict,
                                 viz_dict=viz_dict, iou_thresh=iou_thresh, multiple_preds=self.multiple_preds,
                                 ## ADD-ON #################################################
                                 hierar=hierar, num_rel_prior=num_rel_prior, num_rel_geometric=num_rel_geometric,
                                 num_rel_possessive=num_rel_possessive, num_rel_semantic=num_rel_semantic, matcher=self.matcher)
                                 ###########################################################
        # self.print_stats()
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        output = {}
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))
            output['R@%i' % k] = np.mean(v)
        return output

def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, matcher, multiple_preds=False, viz_dict=None,
                       ## ADD-ON #################################################
                       hierar=True, has_rel_only=True, num_rel_prior=4, num_rel_geometric=15, num_rel_possessive=11, num_rel_semantic=24,
                       ###########################################################
                       **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict:
    :param viz_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry['gt_relations']

    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    rel_scores = pred_entry['rel_scores']

    sub_boxes = pred_entry['sub_boxes']
    obj_boxes = pred_entry['obj_boxes']
    sub_score = pred_entry['sub_scores']
    obj_score = pred_entry['obj_scores']
    sub_class = pred_entry['sub_classes']
    obj_class = pred_entry['obj_classes']

    ### ADD-ON #################################################
    if not hierar:
        rel_scores = rel_scores[:, 1:-1]
        rel_scores = rel_scores.softmax(-1)
        rel_scores = rel_scores.numpy()

        # label 0 is the __background__
        pred_rels = rel_scores.argmax(1) + 1
        predicate_scores = rel_scores.max(1)
    else:
        rel_scores_prior = pred_entry['rel_scores_prior']
        # rel_scores_prior = rel_scores_prior.softmax(-1).numpy()
        # rel_scores_prior = rel_scores_prior[:, :3].numpy()

        # # Retrieve the matching between the outputs of the last layer and the targets
        outputs, targets = pred_entry['outputs'], pred_entry['targets']
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        src_logits = outputs['rel_logits']
        indices = matcher(outputs_without_aux, targets)
        idx = get_src_permutation_idx(indices[1])
        target_classes_o = torch.cat([t["rel_annotations"][J, 2] for t, (_, J) in zip(targets, indices[1])])
        target_classes_o = inv_map[target_classes_o].to(rel_scores.device)
        target_classes = torch.full(src_logits.shape[:2], 51, dtype=torch.int64, device=rel_scores.device)
        target_classes[idx] = target_classes_o  # size [batch_size, num_queries]

        # target_classes_prior = target_classes.clone()
        # target_classes_prior[torch.logical_and(target_classes_prior > 0, target_classes_prior < 15 + 1)] = 0  # geometric
        # target_classes_prior[torch.logical_and(target_classes_prior >= 15 + 1, target_classes_prior < 15 + 11 + 1)] = 1  # possessive
        # target_classes_prior[torch.logical_and(target_classes_prior >= 15 + 11 + 1, target_classes_prior < 51)] = 2  # semantic
        # target_classes_prior[target_classes == 0] = 3  # background relation
        # target_classes_prior[target_classes == 51] = 4  # no object
        # print('rel_scores_prior', rel_scores_prior.shape)
        # print('from error output', outputs['rel_prior_logits'].shape, outputs['rel_prior_logits'][:, :, :3].shape, outputs['rel_prior_logits'][:, :, :3][target_classes_prior < 3].shape, 'target', target_classes_prior.shape, target_classes_prior[target_classes_prior < 3].shape)
        # print('pred', outputs['rel_prior_logits'][:, :, :3][target_classes_prior < 3].argmax(-1), 'tar', target_classes_prior[target_classes_prior < 3])
        # print('error has rel only', 100 - accuracy(outputs['rel_prior_logits'][target_classes_prior < 3].cpu(), target_classes_prior[target_classes_prior < 3])[0])
        # print('error full', 100 - accuracy(outputs['rel_prior_logits'][0].cpu(), target_classes_prior[0])[0])

        if has_rel_only:
            # has_rel = np.zeros(rel_scores.shape[0], dtype=np.uint8)
            # has_rel = target_classes[0] != 51
            has_rel = rel_scores.argmax(1) != 51
            # has_rel = np.logical_and(rel_scores_prior.argmax(1) != 3, rel_scores_prior.argmax(1) != 4)
        else:
            has_rel = np.ones(rel_scores_prior.shape[0], dtype=np.uint8)
        # rel_scores_prior_has_rel = rel_scores_prior[:, :3].softmax(-1).numpy()
        rel_scores_prior = rel_scores_prior[:, :3].softmax(-1).numpy()
        rel_scores = rel_scores[:, 1:-1]

        # splitting the tensor into three parts based on the specified ranges
        first_range = slice(0, num_rel_geometric)  # 0:15
        second_range = slice(num_rel_geometric, num_rel_geometric + num_rel_possessive)  # 15:26
        third_range = slice(num_rel_geometric + num_rel_possessive, rel_scores.shape[1])  # 26:50

        # apply softmax
        rel_scores[has_rel == True, first_range] = rel_scores[has_rel == True, first_range].softmax(-1)
        rel_scores[has_rel == True, second_range] = rel_scores[has_rel == True, second_range].softmax(-1)
        rel_scores[has_rel == True, third_range] = rel_scores[has_rel == True, third_range].softmax(-1)
        rel_scores[has_rel == False, :] = rel_scores[has_rel == False, :].softmax(-1)
        rel_scores = rel_scores.numpy()

        # applying argmax along the appropriate dimensions
        pred_rels_geo = rel_scores[has_rel == True, first_range].argmax(1) + 1
        pred_rels_pos = rel_scores[has_rel == True, second_range].argmax(1) + num_rel_geometric + 1
        pred_rels_sem = rel_scores[has_rel == True, third_range].argmax(1) + num_rel_geometric + num_rel_possessive + 1
        pred_rels = np.concatenate((rel_scores[has_rel == False].argmax(1) + 1, np.concatenate((pred_rels_geo, pred_rels_pos, pred_rels_sem), axis=0)), axis=0)

        # same for predicate_scores
        predicate_scores_geo = rel_scores[has_rel == True, first_range].max(1) * rel_scores_prior[has_rel == True, 0]
        predicate_scores_pos = rel_scores[has_rel == True, second_range].max(1) * rel_scores_prior[has_rel == True, 1]
        predicate_scores_sem = rel_scores[has_rel == True, third_range].max(1) * rel_scores_prior[has_rel == True, 2]
        predicate_scores = np.concatenate((predicate_scores_geo[:, np.newaxis], predicate_scores_pos[:, np.newaxis], predicate_scores_sem[:, np.newaxis]), axis=1)
        predicate_scores = torch.from_numpy(predicate_scores).softmax(-1).numpy()
        predicate_scores = np.transpose(predicate_scores).flatten()
        predicate_scores = np.concatenate((rel_scores[has_rel == False].max(1), predicate_scores), axis=0)

        sub_boxes = np.concatenate((sub_boxes[has_rel == False], np.tile(sub_boxes[has_rel == True], (num_rel_prior-1, 1))), axis=0)
        obj_boxes = np.concatenate((obj_boxes[has_rel == False], np.tile(obj_boxes[has_rel == True], (num_rel_prior-1, 1))), axis=0)
        sub_score = np.concatenate((sub_score[has_rel == False], np.tile(sub_score[has_rel == True], num_rel_prior-1)), axis=0)
        obj_score = np.concatenate((obj_score[has_rel == False], np.tile(obj_score[has_rel == True], num_rel_prior-1)), axis=0)
        sub_class = np.concatenate((sub_class[has_rel == False], np.tile(sub_class[has_rel == True], num_rel_prior-1)), axis=0)
        obj_class = np.concatenate((obj_class[has_rel == False], np.tile(obj_class[has_rel == True], num_rel_prior-1)), axis=0)


        # # apply softmax
        # rel_scores[has_rel == True, first_range] = rel_scores[has_rel == True, first_range].softmax(-1)
        # rel_scores[has_rel == True, second_range] = rel_scores[has_rel == True, second_range].softmax(-1)
        # rel_scores[has_rel == True, third_range] = rel_scores[has_rel == True, third_range].softmax(-1)
        # rel_scores = rel_scores.numpy()
        #
        # # applying argmax along the appropriate dimensions
        # pred_rels_geo = rel_scores[has_rel == True, first_range].argmax(1) + 1
        # pred_rels_pos = rel_scores[has_rel == True, second_range].argmax(1) + num_rel_geometric + 1
        # pred_rels_sem = rel_scores[has_rel == True, third_range].argmax(1) + num_rel_geometric + num_rel_possessive + 1
        #
        # # concatenating the predictions along the batch dimension
        # pred_rels = np.concatenate((pred_rels_geo, pred_rels_pos, pred_rels_sem), axis=0)
        #
        # # same for predicate_scores
        # predicate_scores_geo = rel_scores[has_rel == True, first_range].max(1) * rel_scores_prior[has_rel == True, 0]
        # predicate_scores_pos = rel_scores[has_rel == True, second_range].max(1) * rel_scores_prior[has_rel == True, 1]
        # predicate_scores_sem = rel_scores[has_rel == True, third_range].max(1) * rel_scores_prior[has_rel == True, 2]
        # predicate_scores = np.concatenate((predicate_scores_geo, predicate_scores_pos, predicate_scores_sem), axis=0)
        #
        # sub_boxes = np.tile(sub_boxes[has_rel == True], (num_rel_prior-1, 1))
        # obj_boxes = np.tile(obj_boxes[has_rel == True], (num_rel_prior-1, 1))
        # sub_score = np.tile(sub_score[has_rel == True], num_rel_prior-1)
        # obj_score = np.tile(obj_score[has_rel == True], num_rel_prior-1)
        # sub_class = np.tile(sub_class[has_rel == True], num_rel_prior-1)
        # obj_class = np.tile(obj_class[has_rel == True], num_rel_prior-1)


        # if has_rel_only:
        #     # has_rel = np.zeros(rel_scores.shape[0], dtype=np.uint8)
        #     has_rel = np.logical_and(rel_scores_prior.argmax(1) != 3, rel_scores_prior.argmax(1) != 4)
        # else:
        #     has_rel = np.ones(rel_scores_prior.shape[0], dtype=np.uint8)
        # rel_scores_prior_has_rel = rel_scores_prior[:, :3].softmax(-1).numpy()
        # rel_scores_prior = rel_scores_prior.softmax(-1).numpy()
        # rel_scores = rel_scores[:, 1:-1]
        #
        # # splitting the tensor into three parts based on the specified ranges
        # first_range = slice(0, num_rel_geometric)  # 0:15
        # second_range = slice(num_rel_geometric, num_rel_geometric + num_rel_possessive)  # 15:26
        # third_range = slice(num_rel_geometric + num_rel_possessive, rel_scores.shape[1])  # 26:50
        #
        # # apply softmax
        # rel_scores[has_rel==True, first_range] = rel_scores[has_rel==True, first_range].softmax(-1)
        # rel_scores[has_rel==True, second_range] = rel_scores[has_rel==True, second_range].softmax(-1)
        # rel_scores[has_rel==True, third_range] = rel_scores[has_rel==True, third_range].softmax(-1)
        # rel_scores[has_rel==False, :] = rel_scores[has_rel==False, :].softmax(-1)
        # rel_scores = rel_scores.numpy()
        #
        # # applying max along the appropriate dimensions
        # predicate_scores_geo = rel_scores[has_rel==True, first_range].max(1) * rel_scores_prior_has_rel[has_rel==True, 0]
        # predicate_scores_pos = rel_scores[has_rel==True, second_range].max(1) * rel_scores_prior_has_rel[has_rel==True, 1]
        # predicate_scores_sem = rel_scores[has_rel==True, third_range].max(1) * rel_scores_prior_has_rel[has_rel==True, 2]
        # # predicate_scores = np.concatenate((predicate_scores[no_rel == True], np.concatenate((predicate_scores_geo, predicate_scores_pos, predicate_scores_sem), axis=0)), axis=0)
        #
        # # find the two highest scores
        # predicate_scores_concat = np.concatenate([predicate_scores_geo[:, np.newaxis],
        #                                           predicate_scores_pos[:, np.newaxis],
        #                                           predicate_scores_sem[:, np.newaxis]], axis=1)
        # # print('predicate_scores_concat', predicate_scores_concat[:2, :])
        # # print('predicate_scores_concat', predicate_scores_concat.shape)
        # topk_indices = np.argsort(predicate_scores_concat, axis=1)[:, -2:]
        # topk_values = np.take_along_axis(predicate_scores_concat, topk_indices, axis=1)
        # # print('topk_values', topk_values.shape, 'topk_indices', topk_indices.shape)
        # predicate_scores = topk_values.flatten()
        # predicate_scores = np.concatenate((rel_scores[has_rel==False, :].max(1) * rel_scores_prior_has_rel[has_rel==False, -1], predicate_scores), axis=0)
        # # print('predicate_scores', predicate_scores[:4])
        # # print('predicate_scores', predicate_scores.shape, predicate_scores[:10])
        #
        # # applying argmax along the appropriate dimensions
        # pred_rels_geo = rel_scores[has_rel==True, first_range].argmax(1) + 1
        # pred_rels_pos = rel_scores[has_rel==True, second_range].argmax(1) + num_rel_geometric + 1
        # pred_rels_sem = rel_scores[has_rel==True, third_range].argmax(1) + num_rel_geometric + num_rel_possessive + 1
        #
        # pred_rels_concat = np.concatenate([pred_rels_geo[:, np.newaxis],
        #                                    pred_rels_pos[:, np.newaxis],
        #                                    pred_rels_sem[:, np.newaxis]], axis=1)
        # # print('pred_rels_concat', pred_rels_concat[:2, :])
        # # print('pred_rels_concat', pred_rels_concat.shape)
        # pred_rels = np.take_along_axis(pred_rels_concat, topk_indices, axis=1)
        # # print('pred_rels', pred_rels.shape)
        # pred_rels = pred_rels.flatten()
        # pred_rels = np.concatenate((rel_scores[has_rel==False].argmax(1)+1, pred_rels), axis=0)
        # # print('pred_rels', pred_rels[:4])
        # # print('pred_rels', pred_rels.shape, pred_rels[:10])
        #
        # # concatenating the predictions along the batch dimension
        # # pred_rels = np.concatenate((pred_rels[no_rel == True], np.concatenate((pred_rels_geo, pred_rels_pos, pred_rels_sem), axis=0)), axis=0)
        #
        # # print('sub_boxes', sub_boxes.shape, 'sub_score', sub_score.shape, 'sub_class', sub_class.shape)
        # sub_boxes = np.concatenate((sub_boxes[has_rel==False], np.repeat(sub_boxes[has_rel==True], 2, axis=0)), axis=0)
        # # print('sub_boxes', sub_boxes[:4])
        # obj_boxes = np.concatenate((obj_boxes[has_rel==False], np.repeat(obj_boxes[has_rel==True], 2, axis=0)), axis=0)
        # sub_score = np.concatenate((sub_score[has_rel==False], np.repeat(sub_score[has_rel==True], 2)), axis=0)
        # # print('sub_score', sub_score[:4], '\n')
        # obj_score = np.concatenate((obj_score[has_rel==False], np.repeat(obj_score[has_rel==True], 2)), axis=0)
        # sub_class = np.concatenate((sub_class[has_rel==False], np.repeat(sub_class[has_rel==True], 2)), axis=0)
        # obj_class = np.concatenate((obj_class[has_rel==False], np.repeat(obj_class[has_rel==True], 2)), axis=0)
        # # print('sub_boxes', sub_boxes.shape, 'sub_score', sub_score.shape, 'sub_class', sub_class.shape)
    ############################################################

    pred_to_gt, _, rel_scores = evaluate_recall(
                gt_rels, gt_boxes, gt_classes,
                pred_rels, sub_boxes, obj_boxes, sub_score, obj_score, predicate_scores, sub_class, obj_class, phrdet= mode=='phrdet',
                **kwargs)

    for k in result_dict[mode + '_recall']:

        match = reduce(np.union1d, pred_to_gt[:k])

        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)
    return pred_to_gt, _, rel_scores


def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, sub_boxes, obj_boxes, sub_score, obj_score, predicate_scores, sub_class, obj_class,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)


    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])

    pred_triplets = np.column_stack((sub_class, pred_rels, obj_class))
    pred_triplet_boxes = np.column_stack((sub_boxes, obj_boxes))
    relation_scores = np.column_stack((sub_score, obj_score, predicate_scores))  #TODO!!!! do not * 0.1 finally


    sorted_scores = relation_scores.prod(1)
    pred_triplets = pred_triplets[sorted_scores.argsort()[::-1],:]
    pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1],:]
    relation_scores = relation_scores[sorted_scores.argsort()[::-1],:]
    scores_overall = relation_scores.prod(1)


    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    return pred_to_gt, None, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: 
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt



def calculate_mR_from_evaluator_list(evaluator_list, mode, multiple_preds=False, save_file=None):
    all_rel_results = {}
    for (pred_id, pred_name, evaluator_rel) in evaluator_list:
        print('\n')
        print('relationship: ', pred_name)
        rel_results = evaluator_rel[mode].print_stats()
        all_rel_results[pred_name] = rel_results

    mean_recall = {}
    mR20 = 0.0
    mR50 = 0.0
    mR100 = 0.0
    for key, value in all_rel_results.items():
        if math.isnan(value['R@100']):
            continue
        mR20 += value['R@20']
        mR50 += value['R@50']
        mR100 += value['R@100']
    rel_num = len(evaluator_list)
    mR20 /= rel_num
    mR50 /= rel_num
    mR100 /= rel_num
    mean_recall['R@20'] = mR20
    mean_recall['R@50'] = mR50
    mean_recall['R@100'] = mR100
    all_rel_results['mean_recall'] = mean_recall

    if multiple_preds:
        recall_mode = 'mean recall without constraint'
    else:
        recall_mode = 'mean recall with constraint'
    print('\n')
    print('======================' + mode + '  ' + recall_mode + '============================')
    print('mR@20: ', mR20)
    print('mR@50: ', mR50)
    print('mR@100: ', mR100)

    return mean_recall
