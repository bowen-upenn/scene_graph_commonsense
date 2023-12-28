import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import process_super_class


class BayesianHead(nn.Module):
    """
    The prediction head with a hierarchical classification when the optional transformer encoder is used.
    """
    def __init__(self, input_dim=512, num_geometric=15, num_possessive=11, num_semantic=24, T1=1, T2=1, T3=1):
        super(BayesianHead, self).__init__()
        self.fc3_1 = nn.Linear(input_dim, num_geometric)
        self.fc3_2 = nn.Linear(input_dim, num_possessive)
        self.fc3_3 = nn.Linear(input_dim, num_semantic)
        self.fc5 = nn.Linear(input_dim, 3)
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

    def forward(self, h):
        super_relation = F.log_softmax(self.fc5(h), dim=1)

        # By Bayes rule, log p(relation_n, super_n) = log p(relation_1 | super_1) + log p(super_1)
        relation_1 = self.fc3_1(h)           # geometric
        relation_1 = F.log_softmax(relation_1 / self.T1, dim=1) + super_relation[:, 0].view(-1, 1)
        relation_2 = self.fc3_2(h)           # possessive
        relation_2 = F.log_softmax(relation_2 / self.T2, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_3 = self.fc3_3(h)           # semantic
        relation_3 = F.log_softmax(relation_3 / self.T3, dim=1) + super_relation[:, 2].view(-1, 1)
        return relation_1, relation_2, relation_3, super_relation


class FlatRelationClassifier(nn.Module):
    """
    The local prediction module with a flat classification.
    """
    def __init__(self, args, input_dim=128, output_dim=50, feature_size=32, num_classes=150, num_super_classes=17):
        super(FlatRelationClassifier, self).__init__()
        self.num_classes = num_classes
        self.num_super_classes = num_super_classes
        self.conv1_1 = nn.Conv2d(2 * input_dim + 1, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(2 * input_dim + 1, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(2 * input_dim, 4 * input_dim, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(4 * input_dim, 8 * input_dim, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(8 * input_dim * (feature_size // 4) ** 2, 4096)
        if args['dataset']['dataset'] == 'vg':
            self.fc2 = nn.Linear(4096 + 2 * (num_classes+num_super_classes), 512)
        else:
            self.fc2 = nn.Linear(4096 + 2 * num_classes, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.fc4 = nn.Linear(512, 1)

    def conv_layers(self, h_sub, h_obj):
        h_sub = torch.tanh(self.conv1_1(h_sub))
        h_obj = torch.tanh(self.conv1_2(h_obj))
        h = torch.cat((h_sub, h_obj), dim=1)  # (batch_size, 256, 32, 32)

        h = F.relu(self.conv2_1(h))           # (batch_size, 512, 32, 32)
        h = self.maxpool(h)                   # (batch_size, 512, 16, 16)
        h = F.relu(self.conv3_1(h))           # (batch_size, 1024,16, 16)
        h = self.maxpool(h)                   # (batch_size, 1024, 8,  8)

        h = torch.reshape(h, (h.shape[0], -1))
        h = self.dropout1(F.relu(self.fc1(h)))
        return h

    def concat_labels(self, h, c1, c2, s1, s2, rank, h_aug=None):
        c1 = F.one_hot(c1, num_classes=self.num_classes)
        c2 = F.one_hot(c2, num_classes=self.num_classes)
        if s1 is not None:  # concatenate super-class labels as well
            s1, s2 = process_super_class(s1, s2, self.num_super_classes, rank)
            hc = torch.cat((h, c1, c2, s1, s2), dim=1)

            if h_aug is not None:
                h_aug = torch.cat((h_aug, c1, c2, s1, s2), dim=1)
                h_aug = self.dropout2(F.relu(self.fc2(h_aug)))
        else:
            hc = torch.cat((h, c1, c2), dim=1)

            if h_aug is not None:
                h_aug = torch.cat((h_aug, c1, c2), dim=1)
                h_aug = self.dropout2(F.relu(self.fc2(h_aug)))
        return hc, h_aug

    def forward(self, h_sub, h_obj, c1, c2, s1, s2, rank, h_sub_aug=None, h_obj_aug=None, one_hot=True):
        h = self.conv_layers(h_sub, h_obj)
        h_aug = self.conv_layers(h_sub_aug, h_obj_aug) if h_sub_aug is not None else None   # need data augmentation in contrastive learning
        hc, pred_aug = self.concat_labels(h, c1, c2, s1, s2, rank, h_aug)

        pred = self.dropout2(F.relu(self.fc2(hc)))
        relation = self.fc3(pred)       # (batch_size, 50)
        connectivity = self.fc4(pred)   # (batch_size, 1)
        return relation, connectivity, pred, pred_aug


class BayesianRelationClassifier(nn.Module):
    """
    The local prediction module with a hierarchical classification.
    """
    def __init__(self, args, input_dim=128, feature_size=32, num_classes=150, num_super_classes=17, num_geometric=15,
                 num_possessive=11, num_semantic=24, T1=1, T2=1, T3=1):
        super(BayesianRelationClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_super_classes = num_super_classes
        self.conv1_1 = nn.Conv2d(2 * input_dim + 1, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(2 * input_dim + 1, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(2 * input_dim, 4 * input_dim, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(4 * input_dim, 8 * input_dim, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(8 * input_dim * (feature_size // 4) ** 2, 4096)
        if args['dataset']['dataset'] == 'vg':
            self.fc2 = nn.Linear(4096 + 2 * (num_classes+num_super_classes), 512)
        else:
            self.fc2 = nn.Linear(4096 + 2 * num_classes, 512)
        self.fc3_1 = nn.Linear(512, num_geometric)
        self.fc3_2 = nn.Linear(512, num_possessive)
        self.fc3_3 = nn.Linear(512, num_semantic)
        self.fc4 = nn.Linear(512, 1)
        self.fc5 = nn.Linear(512, 3)
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        
    def conv_layers(self, h_sub, h_obj):
        h_sub = torch.tanh(self.conv1_1(h_sub))
        h_obj = torch.tanh(self.conv1_2(h_obj))
        h = torch.cat((h_sub, h_obj), dim=1)   # (batch_size, 256, 32, 32)

        h = F.relu(self.conv2_1(h))            # (batch_size, 512, 32, 32)
        h = self.maxpool(h)                    # (batch_size, 512, 16, 16)
        h = F.relu(self.conv3_1(h))            # (batch_size, 1024,16, 16)
        h = self.maxpool(h)                    # (batch_size, 1024, 8,  8)

        h = torch.reshape(h, (h.shape[0], -1))
        h = self.dropout1(F.relu(self.fc1(h)))
        return h

    def concat_labels(self, h, c1, c2, s1, s2, rank, h_aug=None):
        c1 = F.one_hot(c1, num_classes=self.num_classes)
        c2 = F.one_hot(c2, num_classes=self.num_classes)
        if s1 is not None:  # concatenate super-class labels as well
            s1, s2 = process_super_class(s1, s2, self.num_super_classes, rank)
            hc = torch.cat((h, c1, c2, s1, s2), dim=1)

            if h_aug is not None:
                h_aug = torch.cat((h_aug, c1, c2, s1, s2), dim=1)
                h_aug = self.dropout2(F.relu(self.fc2(h_aug)))
        else:
            hc = torch.cat((h, c1, c2), dim=1)

            if h_aug is not None:
                h_aug = torch.cat((h_aug, c1, c2), dim=1)
                h_aug = self.dropout2(F.relu(self.fc2(h_aug)))
        return hc, h_aug

    def forward(self, h_sub, h_obj, c1, c2, s1, s2, rank, h_sub_aug=None, h_obj_aug=None):
        h = self.conv_layers(h_sub, h_obj)
        h_aug = self.conv_layers(h_sub_aug, h_obj_aug) if h_sub_aug is not None else None   # need data augmentation in contrastive learning
        hc, pred_aug = self.concat_labels(h, c1, c2, s1, s2, rank, h_aug)

        pred = self.dropout2(F.relu(self.fc2(hc)))
        connectivity = self.fc4(pred)   # (batch_size, 1)
        super_relation = F.log_softmax(self.fc5(pred), dim=1)

        relation_1 = self.fc3_1(pred)   # geometric
        relation_1 = F.log_softmax(relation_1 / self.T1, dim=1) + super_relation[:, 0].view(-1, 1)
        relation_2 = self.fc3_2(pred)   # possessive
        relation_2 = F.log_softmax(relation_2 / self.T2, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_3 = self.fc3_3(pred)   # semantic
        relation_3 = F.log_softmax(relation_3 / self.T3, dim=1) + super_relation[:, 2].view(-1, 1)

        return relation_1, relation_2, relation_3, super_relation, connectivity, pred, pred_aug
