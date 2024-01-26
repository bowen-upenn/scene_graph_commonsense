# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from .utils_relation import obj_prediction_nms

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        attribute_on,
        use_gt_box=False,
        later_nms_pred_thres=0.3,
        rel_inference="SOFTMAX"
    ):
        """
        Arguments:

        """
        super(PostProcessor, self).__init__()
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres
        self.rel_inference = rel_inference

    def forward(self, x, rel_pair_idxs, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, refine_logits = x
        
        if self.attribute_on:
            if isinstance(refine_logits[0], (list, tuple)):
                finetune_obj_logits, finetune_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attribute_on = False
                finetune_obj_logits = refine_logits
        else:
            finetune_obj_logits = refine_logits

        results = []
        for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(zip(
            relation_logits, finetune_obj_logits, rel_pair_idxs, boxes
        )):
            if self.attribute_on:
                att_logit = finetune_att_logits[i]
                att_prob = torch.sigmoid(att_logit)
            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, 0] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                obj_pred = obj_pred + 1
            else:
                # NOTE: by kaihua, apply late nms for object prediction
                if box.get_field('boxes_per_cls').size(1) != obj_logit.size(1):
                    obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                    obj_pred = obj_pred + 1
                else:
                    obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit, self.later_nms_pred_thres)
                    obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                    obj_scores = obj_class_prob.view(-1)[obj_score_ind]

            
            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            if self.use_gt_box:
                boxlist = box
            else:
                if box.get_field('boxes_per_cls').size(1) != obj_logit.size(1):
                    boxlist = box
                else:
                    # mode==sgdet
                    # apply regression based on finetuned object class
                    device = obj_class.device
                    batch_size = obj_class.shape[0]
                    regressed_box_idxs = obj_class
                    boxlist = BoxList(box.get_field('boxes_per_cls')[torch.arange(batch_size, device=device), regressed_box_idxs], box.size, 'xyxy')
            boxlist.add_field('pred_labels', obj_class) # (#obj, )
            boxlist.add_field('pred_scores', obj_scores) # (#obj, )

            if self.attribute_on:
                boxlist.add_field('pred_attributes', att_prob)
            
            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            if self.rel_inference == "SOFTMAX":
                rel_class_prob = F.softmax(rel_logit, -1)
            else:
                rel_class_prob = rel_logit
            rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1
            # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
            triple_scores = rel_scores * obj_scores0 * obj_scores1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            boxlist.add_field('rel_pair_idxs', rel_pair_idx) # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob) # (#rel, #rel_class)
            boxlist.add_field('pred_rel_labels', rel_labels) # (#rel, )
            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            # Note
            # TODO Kaihua: add a new type of element, which can have different length with boxlist (similar to field, except that once 
            # the boxlist has such an element, the slicing operation should be forbidden.)
            # it is not safe to add fields about relation into boxlist!
            results.append(boxlist)
        return results


class HierarchPostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
            self,
            attribute_on,
            use_gt_box=False,
            later_nms_pred_thres=0.3,
            rel_inference="SOFTMAX"
    ):
        """
        Arguments:

        """
        super(HierarchPostProcessor, self).__init__()
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres
        self.rel_inference = rel_inference
        self.geo_label = ['1', '2', '3', '4', '5', '6', '8', '10', '22', '23',
                          '29', '31', '32', '33', '43']
        self.pos_label = ['9', '16', '17', '20', '27', '30', '36', '42', '48',
                          '49', '50']
        self.sem_label = ['7', '11', '12', '13', '14', '15', '18', '19', '21',
                          '24', '25', '26', '28', '34', '35', '37',
                          '38', '39', '40', '41', '44', '45', '46', '47']
        self.geo_label_tensor = torch.tensor([int(x) for x in self.geo_label])
        self.pos_label_tensor = torch.tensor([int(x) for x in self.pos_label])
        self.sem_label_tensor = torch.tensor([int(x) for x in self.sem_label])

        self.common_sense_filter = False

        self.label_semantic = {"1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", "11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", "21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", "31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", "41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}
        self.label_semantic = {int(k): v for k, v in self.label_semantic.items()}
        self.obj_sementic = {"1": "airplane", "2": "animal", "3": "arm", "4": "bag", "5": "banana", "6": "basket", "7": "beach", "8": "bear", "9": "bed", "10": "bench", "11": "bike", "12": "bird", "13": "board", "14": "boat", "15": "book", "16": "boot", "17": "bottle", "18": "bowl", "19": "box", "20": "boy", "21": "branch", "22": "building", "23": "bus", "24": "cabinet", "25": "cap", "26": "car", "27": "cat", "28": "chair", "29": "child", "30": "clock", "31": "coat", "32": "counter", "33": "cow", "34": "cup", "35": "curtain", "36": "desk", "37": "dog", "38": "door", "39": "drawer", "40": "ear", "41": "elephant", "42": "engine", "43": "eye", "44": "face", "45": "fence", "46": "finger", "47": "flag", "48": "flower", "49": "food", "50": "fork", "51": "fruit", "52": "giraffe", "53": "girl", "54": "glass", "55": "glove", "56": "guy", "57": "hair", "58": "hand", "59": "handle", "60": "hat", "61": "head", "62": "helmet", "63": "hill", "64": "horse", "65": "house", "66": "jacket", "67": "jean", "68": "kid", "69": "kite", "70": "lady", "71": "lamp", "72": "laptop", "73": "leaf", "74": "leg", "75": "letter", "76": "light", "77": "logo", "78": "man", "79": "men", "80": "motorcycle", "81": "mountain", "82": "mouth", "83": "neck", "84": "nose", "85": "number", "86": "orange", "87": "pant", "88": "paper", "89": "paw", "90": "people", "91": "person", "92": "phone", "93": "pillow", "94": "pizza", "95": "plane", "96": "plant", "97": "plate", "98": "player", "99": "pole", "100": "post", "101": "pot", "102": "racket", "103": "railing", "104": "rock", "105": "roof", "106": "room", "107": "screen", "108": "seat", "109": "sheep", "110": "shelf", "111": "shirt", "112": "shoe", "113": "short", "114": "sidewalk", "115": "sign", "116": "sink", "117": "skateboard", "118": "ski", "119": "skier", "120": "sneaker", "121": "snow", "122": "sock", "123": "stand", "124": "street", "125": "surfboard", "126": "table", "127": "tail", "128": "tie", "129": "tile", "130": "tire", "131": "toilet", "132": "towel", "133": "tower", "134": "track", "135": "train", "136": "tree", "137": "truck", "138": "trunk", "139": "umbrella", "140": "vase", "141": "vegetable", "142": "vehicle", "143": "wave", "144": "wheel", "145": "window", "146": "windshield", "147": "wing", "148": "wire", "149": "woman", "150": "zebra"}
        self.obj_sementic = {int(k): v for k, v in self.obj_sementic.items()}

    def forward(self, x, rel_pair_idxs, boxes):
        """
        Arguments:
            x: rel1_prob, rel2_prob, rel3_prob, super_rel_prob, refine_logits
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        rel1_probs, rel2_probs, rel3_probs, super_rel_probs, refine_logits = x
        # Assume no attr
        finetune_obj_logits = refine_logits

        results = []
        for i, (rel1_prob, rel2_prob, rel3_prob, super_rel_prob, obj_logit,
                rel_pair_idx, box) in enumerate(zip(
                rel1_probs, rel2_probs, rel3_probs, super_rel_probs,
                finetune_obj_logits, rel_pair_idxs, boxes
        )):
            # i: index of image
            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, 0] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                obj_pred = obj_pred + 1
            else:
                # NOTE: by kaihua, apply late nms for object prediction
                obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'),
                                              obj_logit,
                                              self.later_nms_pred_thres)
                obj_score_ind = torch.arange(num_obj_bbox,
                                             device=obj_logit.device) * num_obj_class + obj_pred
                obj_scores = obj_class_prob.view(-1)[obj_score_ind]

            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            if self.use_gt_box:
                boxlist = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                device = obj_class.device
                batch_size = obj_class.shape[0]
                regressed_box_idxs = obj_class
                boxlist = BoxList(
                    box.get_field('boxes_per_cls')[torch.arange(batch_size,
                                                                device=device), regressed_box_idxs],
                    box.size, 'xyxy')
            boxlist.add_field('pred_labels', obj_class)  # (#obj, )
            boxlist.add_field('pred_scores', obj_scores)  # (#obj, )

            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]

            self.geo_label_tensor.to(rel1_prob.device)
            self.pos_label_tensor.to(rel1_prob.device)
            self.sem_label_tensor.to(rel1_prob.device)
            rel1_prob = torch.exp(rel1_prob)
            rel2_prob = torch.exp(rel2_prob)
            rel3_prob = torch.exp(rel3_prob)

            # For Bayesian classification head, we predict three edges for one pair(each edge for one super category), then gather all the predictions for ranking.
            rel1_scores, rel1_class = rel1_prob.max(dim=1)
            rel1_class = self.geo_label_tensor[rel1_class]
            rel2_scores, rel2_class = rel2_prob.max(dim=1)
            rel2_class = self.pos_label_tensor[rel2_class]
            rel3_scores, rel3_class = rel3_prob.max(dim=1)
            rel3_class = self.sem_label_tensor[rel3_class]

            cat_class_prob = torch.cat((rel1_prob, rel2_prob, rel3_prob),
                                       dim=1)
            cat_class_prob = torch.cat(
                (cat_class_prob, cat_class_prob, cat_class_prob), dim=0)
            cat_rel_pair_idx = torch.cat(
                (rel_pair_idx, rel_pair_idx, rel_pair_idx), dim=0)
            cat_obj_score0 = torch.cat((obj_scores0, obj_scores0, obj_scores0),
                                       dim=0)
            cat_obj_score1 = torch.cat((obj_scores1, obj_scores1, obj_scores1),
                                       dim=0)
            cat_labels = torch.cat((rel1_class, rel2_class, rel3_class), dim=0)
            cat_scores = torch.cat((rel1_scores, rel2_scores, rel3_scores),
                                   dim=0)

            triple_scores = cat_scores * cat_obj_score0 * cat_obj_score1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0,
                                        descending=True)
            rel_pair_idx = cat_rel_pair_idx[sorting_idx]
            rel_class_prob = cat_class_prob[sorting_idx]
            rel_labels = cat_labels[sorting_idx]

            boxlist.add_field('rel_pair_idxs', rel_pair_idx)  # (#rel, 2)
            boxlist.add_field('pred_rel_scores',
                              rel_class_prob)  # (#rel, #rel_class)
            boxlist.add_field('pred_rel_labels', rel_labels)  # (#rel, )

            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            results.append(boxlist)
        return results



def make_roi_relation_post_processor(cfg):
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    use_gt_box = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
    later_nms_pred_thres = cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
    inference = cfg.TEST.INFERENCE

    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "MotifHierarchicalPredictor" or \
            cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "TransformerHierPredictor" or \
            cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "VCTreeHierPredictor":
        postprocessor = HierarchPostProcessor(
            attribute_on,
            use_gt_box,
            later_nms_pred_thres,
            infernece
        )
    else:
        postprocessor = PostProcessor(
            attribute_on,
            use_gt_box,
            later_nms_pred_thres,
            inference
        )
    return postprocessor
