import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesHead(nn.Module):
    """
    The prediction head with a hierarchical classification when the optional transformer encoder is used.
    """
    def __init__(self, input_dim=512, num_geometric=15, num_possessive=11, num_semantic=24, T1=1, T2=1, T3=1):
        super(BayesHead, self).__init__()
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


class FlatMotif(nn.Module):
    """
    The local prediction module with a flat classification.
    """
    def __init__(self, args, input_dim=128, output_dim=50, feature_size=32, num_classes=150, num_super_classes=17):
        super(FlatMotif, self).__init__()
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

    def forward(self, h_graph, h_edge, c1, c2, s1, s2, rank, h_graph_aug=None, h_edge_aug=None, one_hot=True):
        h_graph = torch.tanh(self.conv1_1(h_graph))
        h_edge = torch.tanh(self.conv1_2(h_edge))
        h = torch.cat((h_graph, h_edge), dim=1)   # (batch_size, 256, 32, 32)
        h = F.relu(self.conv2_1(h))         # (batch_size, 512, 32, 32)
        h = self.maxpool(h)                 # (batch_size, 512, 16, 16)
        h = F.relu(self.conv3_1(h))         # (batch_size, 1024,16, 16)
        h = self.maxpool(h)                 # (batch_size, 1024, 8,  8)

        h = torch.reshape(h, (h.shape[0], -1))
        h = self.dropout1(F.relu(self.fc1(h)))

        if one_hot:
            c1 = F.one_hot(c1, num_classes=self.num_classes)
            c2 = F.one_hot(c2, num_classes=self.num_classes)
            if s1 is not None:
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
                hc = torch.cat((h, c1, c2), dim=1)
        else:
            if s1 is not None:
                hc = torch.cat((h, c1, c2, s1, s2), dim=1)
            else:
                hc = torch.cat((h, c1, c2), dim=1)

        if h_graph_aug is not None:
            h_graph_aug = torch.tanh(self.conv1_1(h_graph_aug))
            h_edge_aug = torch.tanh(self.conv1_2(h_edge_aug))

            h_aug = torch.cat((h_graph_aug, h_edge_aug), dim=1)  # (batch_size, 256, 32, 32)
            h_aug = F.relu(self.conv2_1(h_aug))  # (batch_size, 512, 32, 32)
            h_aug = self.maxpool(h_aug)  # (batch_size, 512, 16, 16)
            h_aug = F.relu(self.conv3_1(h_aug))  # (batch_size, 1024,16, 16)
            h_aug = self.maxpool(h_aug)  # (batch_size, 1024, 8,  8)

            h_aug = torch.reshape(h_aug, (h_aug.shape[0], -1))
            h_aug = self.dropout1(F.relu(self.fc1(h_aug)))

            if s1 is not None:
                hc_aug = torch.cat((h_aug, c1, c2, sc1.to(rank), sc2.to(rank)), dim=1)
            else:
                hc_aug = torch.cat((h_aug, c1, c2), dim=1)
            pred_aug = self.dropout2(F.relu(self.fc2(hc_aug)))
        else:
            pred_aug = None

        pred = self.dropout2(F.relu(self.fc2(hc)))
        relation = self.fc3(pred)       # (batch_size, 50)
        connectivity = self.fc4(pred)   # (batch_size, 1)
        return relation, connectivity, pred, pred_aug


class HierMotif(nn.Module):
    """
    The local prediction module with a hierarchical classification.
    """
    def __init__(self, args, input_dim=128, feature_size=32, num_classes=150, num_super_classes=17, num_geometric=15,
                 num_possessive=11, num_semantic=24, T1=1, T2=1, T3=1):
        super(HierMotif, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_super_classes = num_super_classes
        self.conv1_1 = nn.Conv2d(2 * input_dim + 1, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(2 * input_dim + 1, input_dim, kernel_size=1, stride=1, padding=0)
        # self.conv1_3 = nn.Conv2d(args['models']['faster_rcnn_hidden_dim'] + 1, input_dim, kernel_size=1, stride=1, padding=0)
        # self.conv1_4 = nn.Conv2d(args['models']['faster_rcnn_hidden_dim'] + 1, input_dim, kernel_size=1, stride=1, padding=0)
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

    def forward(self, h_graph, h_edge, c1, c2, s1, s2, rank, h_graph_aug=None, h_edge_aug=None, one_hot=True):
        if h_graph.shape[1] == 2 * self.input_dim + 1:
            h_graph = torch.tanh(self.conv1_1(h_graph))
            h_edge = torch.tanh(self.conv1_2(h_edge))
        else:   # faster rcnn image feature has 2048 channels, DETR has 256 instead
            h_graph = torch.tanh(self.conv1_3(h_graph))
            h_edge = torch.tanh(self.conv1_4(h_edge))
        h = torch.cat((h_graph, h_edge), dim=1)   # (batch_size, 256, 32, 32)
        h = F.relu(self.conv2_1(h))         # (batch_size, 512, 32, 32)
        h = self.maxpool(h)                 # (batch_size, 512, 16, 16)
        h = F.relu(self.conv3_1(h))         # (batch_size, 1024,16, 16)
        h = self.maxpool(h)                 # (batch_size, 1024, 8,  8)

        h = torch.reshape(h, (h.shape[0], -1))
        h = self.dropout1(F.relu(self.fc1(h)))

        if one_hot:
            c1 = F.one_hot(c1, num_classes=self.num_classes)
            c2 = F.one_hot(c2, num_classes=self.num_classes)
            if s1 is not None:
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
                hc = torch.cat((h, c1, c2), dim=1)
        else:
            if s1 is not None:
                hc = torch.cat((h, c1, c2, s1, s2), dim=1)
            else:
                hc = torch.cat((h, c1, c2), dim=1)

        if h_graph_aug is not None:
            if h_graph_aug.shape[1] == 2 * self.input_dim + 1:
                h_graph_aug = torch.tanh(self.conv1_1(h_graph_aug))
                h_edge_aug = torch.tanh(self.conv1_2(h_edge_aug))
            else:   # faster rcnn image feature has 2048 channels, DETR has 256 instead
                h_graph_aug = torch.tanh(self.conv1_3(h_graph_aug))
                h_edge_aug = torch.tanh(self.conv1_4(h_edge_aug))

            h_aug = torch.cat((h_graph_aug, h_edge_aug), dim=1)   # (batch_size, 256, 32, 32)
            h_aug = F.relu(self.conv2_1(h_aug))         # (batch_size, 512, 32, 32)
            h_aug = self.maxpool(h_aug)                 # (batch_size, 512, 16, 16)
            h_aug = F.relu(self.conv3_1(h_aug))         # (batch_size, 1024,16, 16)
            h_aug = self.maxpool(h_aug)                 # (batch_size, 1024, 8,  8)

            h_aug = torch.reshape(h_aug, (h_aug.shape[0], -1))
            h_aug = self.dropout1(F.relu(self.fc1(h_aug)))

            if s1 is not None:
                hc_aug = torch.cat((h_aug, c1, c2, sc1.to(rank), sc2.to(rank)), dim=1)
            else:
                hc_aug = torch.cat((h_aug, c1, c2), dim=1)
            pred_aug = self.dropout2(F.relu(self.fc2(hc_aug)))
        else:
            pred_aug = None

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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # if error "some elements of the input tensor and the written-to tensor refer to a single memory location", add requires_grad=False
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.3,
                 num_geometric=15, num_possessive=11, num_semantic=24, output_dim=50, hierar=True):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.hierar = hierar
        if self.hierar:
            self.bayes_head = BayesHead(d_model, num_geometric, num_possessive, num_semantic)
        else:
            self.flat_head = nn.Linear(d_model, output_dim)

    def forward(self, src, src_key_padding_mask):
        src = self.positional_encoding(src)
        hidden = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        hidden = torch.permute(hidden, (1, 0, 2))
        hidden = hidden[~src_key_padding_mask]

        if self.hierar:
            relation_1, relation_2, relation_3, super_relation = self.bayes_head(hidden)
            return relation_1, relation_2, relation_3, super_relation, hidden
        else:
            relation = self.flat_head(hidden)
            return relation, hidden


class SimpleSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(SimpleSelfAttention, self).__init__()
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, x):
        # x shape: (seq_len, batch_size, hidden_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output


class RelationshipRefiner(nn.Module):
    def __init__(self, hidden_dim):
        super(RelationshipRefiner, self).__init__()

        # additional layers to predict relationship
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.3)

        self.rel_tokens = nn.Parameter(torch.rand(1, hidden_dim), requires_grad=True)

    def forward(self, img_embed, neighbor_txt_embed, query_embed):
        hidden = self.rel_tokens + query_embed
        hidden = torch.cat((hidden, img_embed, neighbor_txt_embed), dim=-1)
        hidden = F.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc2(hidden))
        return hidden


