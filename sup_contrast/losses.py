from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLossGraph(nn.Module):
    def __init__(self, clip_model, tokenizer, all_labels_geometric, all_labels_possessive, all_labels_semantic, rank,
                 num_geom=15, num_poss=11, num_sem=24, base_temperature=0.07):
        super(SupConLossGraph, self).__init__()

        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.base_temperature = base_temperature

        self.num_geom = num_geom
        self.num_poss = num_poss
        self.num_sem = num_sem

        # Initialize all possible embeddings under each supercategory
        self.all_embeddings_geometric, self.all_embeddings_possessive, self.all_embeddings_semantic = self._initialize_embeddings(
            all_labels_geometric, all_labels_possessive, all_labels_semantic, rank)

    def _initialize_embeddings(self, all_labels_geometric, all_labels_possessive, all_labels_semantic, rank):
        # Geometric embeddings
        queries_geom = [f"{label}" for label in all_labels_geometric]
        inputs_geom = self.tokenizer(queries_geom, padding=True, return_tensors="pt").to(rank)
        with torch.no_grad():
            embeddings_geometric = self.clip_model.module.get_text_features(**inputs_geom)
        embeddings_geometric = F.normalize(embeddings_geometric, p=2, dim=1)

        # Possessive embeddings
        queries_poss = [f"{label}" for label in all_labels_possessive]
        inputs_poss = self.tokenizer(queries_poss, padding=True, return_tensors="pt").to(rank)
        with torch.no_grad():
            embeddings_possessive = self.clip_model.module.get_text_features(**inputs_poss)
        embeddings_possessive = F.normalize(embeddings_possessive, p=2, dim=1)

        # Semantic embeddings
        queries_sem = [f"{label}" for label in all_labels_semantic]
        inputs_sem = self.tokenizer(queries_sem, padding=True, return_tensors="pt").to(rank)
        with torch.no_grad():
            embeddings_semantic = self.clip_model.module.get_text_features(**inputs_sem)
        embeddings_semantic = F.normalize(embeddings_semantic, p=2, dim=1)

        return embeddings_geometric, embeddings_possessive, embeddings_semantic

    def forward(self, predicted_txt_embeds, curr_relation_ids, rank, temperature=0.07):
        batch_size = len(predicted_txt_embeds)
        mean_log_contrasts = 0.0

        for bid in range(batch_size):
            curr_features = predicted_txt_embeds[bid].to(rank)
            curr_features = F.normalize(curr_features, p=2, dim=0)  # without normalization, the exponential will be numerically too large
            relation_id = curr_relation_ids[bid]

            # Retrieve the initialized embeddings based on the relation_id
            if relation_id < self.num_geom:
                positive_anchors = self.all_embeddings_geometric[relation_id].to(rank)
                negative_anchors = self.all_embeddings_geometric[~torch.isin(torch.arange(self.num_geom), relation_id)].to(rank)
            elif self.num_geom <= relation_id < self.num_geom + self.num_poss:
                positive_anchors = self.all_embeddings_possessive[relation_id - self.num_geom].to(rank)
                negative_anchors = self.all_embeddings_possessive[~torch.isin(torch.arange(self.num_poss), relation_id - self.num_geom)].to(rank)
            else:
                positive_anchors = self.all_embeddings_semantic[relation_id - self.num_geom - self.num_poss].to(rank)
                negative_anchors = self.all_embeddings_semantic[~torch.isin(torch.arange(self.num_sem), relation_id - self.num_geom - self.num_poss)].to(rank)

            num_negative = negative_anchors.shape[0]

            contrast_numerator = torch.sum(curr_features * positive_anchors)
            contrast_numerator = torch.exp(contrast_numerator / temperature)

            contrast_denominator = curr_features @ negative_anchors.T
            contrast_denominator = torch.sum(torch.exp(contrast_denominator / temperature), dim=0)

            log_contrasts = torch.sum(torch.log(contrast_numerator + 1e-7) - torch.log(contrast_denominator + 1e-7))
            log_contrasts /= -1 * num_negative
            mean_log_contrasts += log_contrasts

        loss = (temperature / self.base_temperature) * mean_log_contrasts
        return loss


class SupConLossHierar(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossHierar, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def get_parent_label(self, labels):
        parent_labels = torch.clone(labels)
        parent_labels[labels < 15] = 0
        parent_labels[(labels >= 15) & (labels < 26)] = 1
        parent_labels[labels >= 26] = 2
        return parent_labels

    def forward(self, rank, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(rank)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            # compute parent labels
            parent_labels = self.get_parent_label(labels)
            mask_same_parent = torch.eq(parent_labels, parent_labels.T).float().to(rank)  # 1 if same parent label

            mask = torch.eq(labels, labels.T).float().to(rank)  # 1 if same label, 0 otherwise
        else:
            mask = mask.float().to(rank)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        mask_same_parent = mask_same_parent.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),  # create a tensor of all ones with the same shape of mask
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(rank),   # fill the diagonal with zeros
            0
        )
        mask = mask * logits_mask   # no need to contrast with itself (diff must be 0)

        # compute log_prob, where only different labels under the same parent class appear in the denominator
        logits_mask = logits_mask * mask_same_parent
        exp_logits = torch.exp(logits) * logits_mask    # for each sample in the row, logits_mask selects all its negative samples
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)  # mask selects all positive (augmented) samples

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, rank, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(rank)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(rank)
        else:
            mask = mask.float().to(rank)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print('test1', torch.mean(logits))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(rank),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print('test1', torch.mean(exp_logits))
        # print('test2', torch.mean(torch.log(exp_logits.sum(1, keepdim=True))) + 1e-7)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)
        # print('test3', torch.mean(log_prob))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)
        # print('test4', torch.mean(mean_log_prob_pos))

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print('test5', torch.mean(loss))
        loss = loss.view(anchor_count, batch_size).mean()
        # print('test6', loss)

        return loss
