import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotifHead(nn.Module):
    """
    The prediction head with a flat classification when the optional transformer encoder is used.
    """
    def __init__(self, input_dim=256, output_dim=50):
        super(MotifHead, self).__init__()
        self.fc3 = nn.Linear(2 * input_dim, output_dim)
        self.fc4 = nn.Linear(2 * input_dim, 1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, h):
        relation = self.dropout2(F.relu(self.fc3(h)))
        connectivity = self.dropout2(F.relu(self.fc4(h)))
        return relation, connectivity


class MotifHeadHier(nn.Module):
    """
    The prediction head with a hierarchical classification when the optional transformer encoder is used.
    """
    def __init__(self, input_dim=256, T1=1, T2=1, T3=1):
        super(MotifHeadHier, self).__init__()
        self.fc3_1 = nn.Linear(2 * input_dim, 15)
        self.fc3_2 = nn.Linear(2 * input_dim, 11)
        self.fc3_3 = nn.Linear(2 * input_dim, 24)
        self.fc4 = nn.Linear(2 * input_dim, 1)
        self.fc5 = nn.Linear(2 * input_dim, 3)
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

    def forward(self, h):
        connectivity = self.fc4(h)
        super_relation = F.log_softmax(self.fc5(h), dim=1)

        # By Bayes rule, log p(relation_n, super_n) = log p(relation_1 | super_1) + log p(super_1)
        relation_1 = self.fc3_1(h)           # geometric
        relation_1 = F.log_softmax(relation_1 / self.T1, dim=1) + super_relation[:, 0].view(-1, 1)
        relation_2 = self.fc3_2(h)           # possessive
        relation_2 = F.log_softmax(relation_2 / self.T2, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_3 = self.fc3_3(h)           # semantic
        relation_3 = F.log_softmax(relation_3 / self.T3, dim=1) + super_relation[:, 2].view(-1, 1)
        return relation_1, relation_2, relation_3, super_relation, connectivity


class MotifEmbed(nn.Module):
    """
    The local prediction module with a flat classification when the optional transformer encoder is used.
    Its model parameter is always pretrained and frozen.
    It provides hidden states to the transformer encoder.
    """
    def __init__(self, input_dim=128, output_dim=50, feature_size=32, num_classes=150, num_super_classes=17):
        super(MotifEmbed, self).__init__()
        self.num_classes = num_classes
        self.num_super_classes = num_super_classes
        self.conv1_1 = nn.Conv2d(2 * input_dim + 1, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(2 * input_dim + 1, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(2 * input_dim, 4 * input_dim, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(4 * input_dim, 8 * input_dim, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(8 * input_dim * (feature_size // 4) ** 2, 4096)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc2 = nn.Linear(4096 + 334, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.fc4 = nn.Linear(512, 1)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, h_graph, h_edge, c1, c2, s1, s2, rank, one_hot=True):
        h_graph = torch.tanh(self.conv1_1(h_graph))
        h_edge = torch.tanh(self.conv1_2(h_edge))
        h = torch.cat((h_graph, h_edge), dim=1)   # (batch_size, 512,  16, 16)
        h = F.relu(self.conv2_1(h))               # (batch_size, 1024, 16, 16)
        h = self.maxpool(h)                       # (batch_size, 1024,  8,  8)
        h = F.relu(self.conv3_1(h))               # (batch_size, 2048,  8,  8)
        h = self.maxpool(h)                       # (batch_size, 2048,  4,  4)

        h = h.view(h.shape[0], -1)
        h = self.dropout1(F.relu(self.fc1(h)))     # (batch_size, 4096)

        if one_hot:
            c1 = F.one_hot(c1, num_classes=self.num_classes)
            c2 = F.one_hot(c2, num_classes=self.num_classes)
            sc1 = F.one_hot(torch.tensor([s[0] for s in s1]), num_classes=self.num_super_classes)
            for i in range(1, 4):  # at most 4 diff super class for each sub class instance
                idx = torch.nonzero(torch.tensor([len(s) == i + 1 for s in s1])).view(-1)
                if len(idx) > 0:
                    sc1[idx] += F.one_hot(torch.tensor([s[i] for s in [s1[j] for j in idx]]), num_classes=self.num_super_classes)
            sc2 = F.one_hot(torch.tensor([s[0] for s in s2]), num_classes=self.num_super_classes)
            for i in range(1, 4):
                idx = torch.nonzero(torch.tensor([len(s) == i + 1 for s in s2])).view(-1)
                if len(idx) > 0:
                    sc2[idx] += F.one_hot(torch.tensor([s[i] for s in [s2[j] for j in idx]]), num_classes=self.num_super_classes)
            hc = torch.cat((h, c1, c2, sc1.to(rank), sc2.to(rank)), dim=1)   # (batch_size, 4096+334)
        else:
            hc = torch.cat((h, c1, c2, s1, s2), dim=1)

        pred = self.dropout2(F.relu(self.fc2(hc)))  # (batch_size, 512)
        relation = self.fc3(pred)
        connectivity = self.fc4(pred)
        hcr = torch.cat((pred, relation, connectivity), dim=1)   # (batch_size, 512+51)
        return hcr


class MotifEmbedHier(nn.Module):
    """
    The local prediction module with a hierarchical classification when the optional transformer encoder is used.
    Its model parameter is always pretrained and frozen.
    It provides hidden states to the transformer encoder.
    """
    def __init__(self, input_dim=128, feature_size=32, num_classes=150, num_super_classes=17, T1=1, T2=1, T3=1):
        super(MotifEmbedHier, self).__init__()
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
        self.fc2 = nn.Linear(4096 + 2 * num_classes + 2 * num_super_classes, 512)
        self.fc3_1 = nn.Linear(512, 15)
        self.fc3_2 = nn.Linear(512, 11)
        self.fc3_3 = nn.Linear(512, 24)
        self.fc4 = nn.Linear(512, 1)
        self.fc5 = nn.Linear(512, 3)
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

    def forward(self, h_graph, h_edge, c1, c2, s1, s2, rank, one_hot=True):
        h_graph = torch.tanh(self.conv1_1(h_graph))
        h_edge = torch.tanh(self.conv1_2(h_edge))
        h = torch.cat((h_graph, h_edge), dim=1)   # (batch_size, 512,  16, 16)
        h = F.relu(self.conv2_1(h))               # (batch_size, 1024, 16, 16)
        h = self.maxpool(h)                       # (batch_size, 1024,  8,  8)
        h = F.relu(self.conv3_1(h))               # (batch_size, 2048,  8,  8)
        h = self.maxpool(h)                       # (batch_size, 2048,  4,  4)

        h = h.view(h.shape[0], -1)
        h = self.dropout1(F.relu(self.fc1(h)))     # (batch_size, 4096)

        if one_hot:
            c1 = F.one_hot(c1, num_classes=self.num_classes)
            c2 = F.one_hot(c2, num_classes=self.num_classes)
            sc1 = F.one_hot(torch.tensor([s[0] for s in s1]), num_classes=self.num_super_classes)
            for i in range(1, 4):  # at most 4 diff super class for each sub class instance
                idx = torch.nonzero(torch.tensor([len(s) == i + 1 for s in s1])).view(-1)
                if len(idx) > 0:
                    sc1[idx] += F.one_hot(torch.tensor([s[i] for s in [s1[j] for j in idx]]), num_classes=self.num_super_classes)
            sc2 = F.one_hot(torch.tensor([s[0] for s in s2]), num_classes=self.num_super_classes)
            for i in range(1, 4):
                idx = torch.nonzero(torch.tensor([len(s) == i + 1 for s in s2])).view(-1)
                if len(idx) > 0:
                    sc2[idx] += F.one_hot(torch.tensor([s[i] for s in [s2[j] for j in idx]]), num_classes=self.num_super_classes)
            hc = torch.cat((h, c1, c2, sc1.to(rank), sc2.to(rank)), dim=1)   # (batch_size, 4096+334)
        else:
            hc = torch.cat((h, c1, c2, s1, s2), dim=1)

        pred = self.dropout2(F.relu(self.fc2(hc)))
        connectivity = self.fc4(pred)  # (batch_size, 1)
        super_relation = F.log_softmax(self.fc5(pred), dim=1)

        relation_1 = self.fc3_1(pred)  # geometric
        relation_1 = F.log_softmax(relation_1 / self.T1, dim=1) + super_relation[:, 0].view(-1, 1)
        relation_2 = self.fc3_2(pred)  # possessive
        relation_2 = F.log_softmax(relation_2 / self.T2, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_3 = self.fc3_3(pred)  # semantic
        relation_3 = F.log_softmax(relation_3 / self.T3, dim=1) + super_relation[:, 2].view(-1, 1)
        hcr = torch.cat((pred, relation_1, relation_2, relation_3, super_relation, connectivity), dim=1)  # (batch_size, 512+54)
        return hcr


class EdgeHead(nn.Module):
    """
    The local prediction module with a flat classification.
    """
    def __init__(self, input_dim=128, output_dim=50, feature_size=32, num_classes=150, num_super_classes=17):
        super(EdgeHead, self).__init__()
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
        self.fc2 = nn.Linear(4096 + 2 * num_classes + 2 * num_super_classes, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, h_graph, h_edge, c1, c2, s1, s2, rank, one_hot=True):
        h_graph = torch.tanh(self.conv1_1(h_graph))
        h_edge = torch.tanh(self.conv1_2(h_edge))
        h = torch.cat((h_graph, h_edge), dim=1)   # (batch_size, 256, 32, 32)
        h = F.relu(self.conv2_1(h))         # (batch_size, 512, 32, 32)
        h = self.maxpool(h)                 # (batch_size, 512, 16, 16)
        h = F.relu(self.conv3_1(h))         # (batch_size, 1024,16, 16)
        h = self.maxpool(h)                 # (batch_size, 1024, 8,  8)

        h = h.view(h.shape[0], -1)
        h = self.dropout1(F.relu(self.fc1(h)))

        if one_hot:
            c1 = F.one_hot(c1, num_classes=self.num_classes)
            c2 = F.one_hot(c2, num_classes=self.num_classes)
            sc1 = F.one_hot(torch.tensor([s[0] for s in s1]), num_classes=self.num_super_classes)
            for i in range(1, 4):  # at most 4 diff super class for each sub class instance
                idx = torch.nonzero(torch.tensor([len(s) == i + 1 for s in s1])).view(-1)
                if len(idx) > 0:
                    sc1[idx] += F.one_hot(torch.tensor([s[i] for s in [s1[j] for j in idx]]), num_classes=self.num_super_classes)
            sc2 = F.one_hot(torch.tensor([s[0] for s in s2]), num_classes=self.num_super_classes)
            for i in range(1, 4):
                idx = torch.nonzero(torch.tensor([len(s) == i + 1 for s in s2])).view(-1)
                if len(idx) > 0:
                    sc2[idx] += F.one_hot(torch.tensor([s[i] for s in [s2[j] for j in idx]]), num_classes=self.num_super_classes)
            hc = torch.cat((h, c1, c2, sc1.to(rank), sc2.to(rank)), dim=1)
        else:
            hc = torch.cat((h, c1, c2, s1, s2), dim=1)

        pred = self.dropout2(F.relu(self.fc2(hc)))
        relation = self.fc3(pred)       # (batch_size, 50)
        connectivity = self.fc4(pred)   # (batch_size, 1)
        return relation, connectivity


class EdgeHeadHier(nn.Module):
    """
    The local prediction module with a hierarchical classification.
    """
    def __init__(self, input_dim=128, feature_size=32, num_classes=150, num_super_classes=17, T1=1, T2=1, T3=1):
        super(EdgeHeadHier, self).__init__()
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
        self.fc2 = nn.Linear(4096 + 2 * num_classes + 2 * num_super_classes, 512)
        self.fc3_1 = nn.Linear(512, 15)
        self.fc3_2 = nn.Linear(512, 11)
        self.fc3_3 = nn.Linear(512, 24)
        self.fc4 = nn.Linear(512, 1)
        self.fc5 = nn.Linear(512, 3)
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

    def forward(self, h_graph, h_edge, c1, c2, s1, s2, rank, one_hot=True):
        h_graph = torch.tanh(self.conv1_1(h_graph))
        h_edge = torch.tanh(self.conv1_2(h_edge))
        h = torch.cat((h_graph, h_edge), dim=1)   # (batch_size, 256, 32, 32)
        h = F.relu(self.conv2_1(h))         # (batch_size, 512, 32, 32)
        # h = F.relu(self.conv2_2(h))       # (batch_size, 512, 32, 32)
        h = self.maxpool(h)                 # (batch_size, 512, 16, 16)
        h = F.relu(self.conv3_1(h))         # (batch_size, 1024,16, 16)
        # h = F.relu(self.conv3_2(h))       # (batch_size, 1024,16, 16)
        h = self.maxpool(h)                 # (batch_size, 1024, 8,  8)

        h = h.view(h.shape[0], -1)
        h = self.dropout1(F.relu(self.fc1(h)))

        if one_hot:
            c1 = F.one_hot(c1, num_classes=self.num_classes)
            c2 = F.one_hot(c2, num_classes=self.num_classes)
            sc1 = F.one_hot(torch.tensor([s[0] for s in s1]), num_classes=self.num_super_classes)
            for i in range(1, 4):  # at most 4 diff super class for each sub class instance
                idx = torch.nonzero(torch.tensor([len(s) == i + 1 for s in s1])).view(-1)
                if len(idx) > 0:
                    sc1[idx] += F.one_hot(torch.tensor([s[i] for s in [s1[j] for j in idx]]), num_classes=self.num_super_classes)
            sc2 = F.one_hot(torch.tensor([s[0] for s in s2]), num_classes=self.num_super_classes)
            for i in range(1, 4):
                idx = torch.nonzero(torch.tensor([len(s) == i + 1 for s in s2])).view(-1)
                if len(idx) > 0:
                    sc2[idx] += F.one_hot(torch.tensor([s[i] for s in [s2[j] for j in idx]]), num_classes=self.num_super_classes)
            hc = torch.cat((h, c1, c2, sc1.to(rank), sc2.to(rank)), dim=1)
        else:
            hc = torch.cat((h, c1, c2, s1, s2), dim=1)

        pred = self.dropout2(F.relu(self.fc2(hc)))
        connectivity = self.fc4(pred)   # (batch_size, 1)
        super_relation = F.log_softmax(self.fc5(pred), dim=1)

        relation_1 = self.fc3_1(pred)   # geometric
        relation_1 = F.log_softmax(relation_1 / self.T1, dim=1) + super_relation[:, 0].view(-1, 1)
        relation_2 = self.fc3_2(pred)   # possessive
        relation_2 = F.log_softmax(relation_2 / self.T2, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_3 = self.fc3_3(pred)   # semantic
        relation_3 = F.log_softmax(relation_3 / self.T3, dim=1) + super_relation[:, 2].view(-1, 1)
        return relation_1, relation_2, relation_3, super_relation, connectivity
