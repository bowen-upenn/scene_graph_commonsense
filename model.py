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
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, x, key_padding_mask):
        # x shape: (seq_len, batch_size, hidden_dim)
        x = self.linear(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, key_padding_mask=key_padding_mask)
        return attn_output


class MultimodalTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_layers=3, dropout=0.1):
        super(MultimodalTransformerEncoder, self).__init__()

        self.positional_encoding = PositionalEncoding(hidden_dim)

        # # Initialize learned modality encodings for image, query, and text
        # self.image_modality_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        # self.query_modality_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        # self.text_modality_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, key_padding_mask):
        # Apply positional encoding
        # init_pred = x[2].unsqueeze(0)
        # x = self.positional_encoding(x)

        # # Add learned modality encodings to image, query, and text features
        # image_feat = x[0] + self.image_modality_encoding
        # query_feat = x[1] + self.query_modality_encoding
        # text_feat = x[2:] + self.text_modality_encoding
        #
        # # Concatenate back into the original sequence
        # x = torch.cat([image_feat, query_feat, text_feat], dim=0)

        # Perform self-attention with transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        # output += init_pred  # skip connection

        return output


class GATLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=1, alpha=0.2, concat=False):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_dim, hidden_dim)) for _ in range(num_heads)])
        self.a = nn.ParameterList([nn.Parameter(torch.Tensor(1, 2 * hidden_dim)) for _ in range(num_heads)])
        self.leakyrelu = nn.LeakyReLU(alpha)

        for i in range(num_heads):
            nn.init.xavier_uniform_(self.W[i].data)
            nn.init.xavier_uniform_(self.a[i].data)

    def forward(self, x, key_padding_mask=None):
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()
        # if key_padding_mask is not None:
        #     # print('key_padding_mask', key_padding_mask.shape)
        #     key_padding_mask = key_padding_mask.transpose(0, 1)  # swap seq_len and batch_size dimensions
        #     key_padding_mask = key_padding_mask.unsqueeze(-1).repeat(1, 1, seq_len).view(batch_size, -1)

        outputs = []
        for i in range(self.num_heads):
            h = torch.matmul(x, self.W[i].t())  # (batch_size, seq_len, hidden_dim)
            # print('h', h.shape, 'x', x.shape, 'W', self.W[i].t().shape)
            a_input = torch.cat([h.repeat(1, 1, seq_len).view(batch_size, seq_len * seq_len, -1),
                                 h.repeat(1, seq_len, 1)], dim=2)  # (batch_size, seq_len * seq_len, 2 * hidden_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a[i].t()).squeeze(2))  # (batch_size, seq_len * seq_len)
            # print('e', e.shape, 'key_padding_mask', key_padding_mask.shape)

            # if key_padding_mask is not None:
            #     e = e.masked_fill(key_padding_mask, float('-inf'))  # mask out the padded elements
            #     # print('e', e.shape, 'key_padding_mask', key_padding_mask.shape)

            attention = F.softmax(e.view(-1, seq_len), dim=1).view(batch_size, seq_len, seq_len)
            # print('attention', attention.shape, 'h', h.shape)
            output = torch.matmul(attention, h)  # (batch_size, seq_len, hidden_dim)
            # print('output', output.shape)
            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)  # (batch_size, seq_len, num_heads * hidden_dim)
        else:
            return sum(outputs) / self.num_heads  # (batch_size, seq_len, hidden_dim)


class RelationshipRefiner(nn.Module):
    def __init__(self, hidden_dim):
        super(RelationshipRefiner, self).__init__()

        # additional layers to predict relationship
        # self.fc_img = nn.Linear(hidden_dim * 3, hidden_dim)
        # self.fc_txt = nn.Linear(hidden_dim * 3, hidden_dim)
        # self.fc_out = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.dropout = nn.Dropout(p=0.3)
        # self.fc_glo_img = nn.Linear(hidden_dim, hidden_dim)
        # self.fc_sub_img = nn.Linear(hidden_dim, hidden_dim)
        # self.fc_obj_img = nn.Linear(hidden_dim, hidden_dim)
        # self.fc_img = nn.Linear(hidden_dim, hidden_dim)
        #
        # self.fc_glo_txt = nn.Linear(hidden_dim, hidden_dim)
        # self.fc_sub_txt = nn.Linear(hidden_dim, hidden_dim)
        # self.fc_obj_txt = nn.Linear(hidden_dim, hidden_dim)
        # self.fc_txt = nn.Linear(hidden_dim, hidden_dim)

        self.fc_sub = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_obj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_fuse = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.3)

        # learnable parameters to balance contributions
        # self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # self.gamma = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, glob_imge_embed, sub_img_embed, obj_img_embed, current_txt_embed, sub_txt_embed, obj_txt_embed, neighbor_txt_embed):
        hidden_sub = torch.cat((sub_img_embed, sub_txt_embed), dim=-1)
        hidden_sub = F.relu(self.fc_sub(hidden_sub))
        hidden_sub = self.dropout(hidden_sub)

        hidden_obj = torch.cat((obj_img_embed, obj_txt_embed), dim=-1)
        hidden_obj = F.relu(self.fc_obj(hidden_obj))
        hidden_obj = self.dropout(hidden_obj)

        # hidden_img = self.fc_img(self.fc_glo_img(glob_imge_embed) - self.fc_sub_img(sub_img_embed) - self.fc_obj_img(obj_img_embed))
        # hidden_img = self.dropout(F.relu(hidden_img))
        #
        # hidden_txt = self.fc_txt(self.fc_glo_txt(neighbor_txt_embed) - self.fc_sub_txt(sub_txt_embed) - self.fc_obj_txt(obj_txt_embed))
        # hidden_txt = self.dropout(F.relu(hidden_txt))

        # hidden_img = torch.cat((glob_imge_embed, sub_img_embed, obj_img_embed), dim=-1)
        # hidden_img = F.relu(self.fc_img(hidden_img))
        # hidden_img = self.dropout(hidden_img)
        #
        # hidden_txt = torch.cat((sub_txt_embed, obj_txt_embed, neighbor_txt_embed), dim=-1)
        # hidden_txt = F.relu(self.fc_txt(hidden_txt))
        # hidden_txt = self.dropout(hidden_txt) #+ current_txt_embed  # skip connection

        # Balance contributions with learnable parameters
        # hidden = self.alpha * hidden_img + self.beta * hidden_txt + self.gamma * current_txt_embed
        # hidden = hidden_img + hidden_txt + current_txt_embed

        # hidden = hidden_img + hidden_txt + self.dropout(current_txt_embed)
        hidden = torch.cat((hidden_sub, hidden_obj, neighbor_txt_embed), dim=-1)
        hidden = F.relu(self.fc_fuse(hidden))
        hidden = self.dropout(hidden)
        hidden = self.fc_out(hidden)

        hidden += F.normalize(current_txt_embed, dim=1, p=2)  # skip connection

        return hidden


class EdgeAttentionModel(nn.Module):
    def __init__(self, d_model, nhead=8):
        super(EdgeAttentionModel, self).__init__()
        self.d_model = d_model

        # Transformer components
        self.multihead_attn_self = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1)
        self.multihead_attn_cross = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1)

        # Feedforward layers for transformation
        self.in_proj_query = nn.Linear(d_model * 3, d_model)
        self.in_proj_key = nn.Linear(d_model * 3, d_model)

        self.feed_forward_self = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model)
        )
        self.feed_forward_cross = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model)
        )
        self.tanh = nn.Tanh()

    def forward(self, queries, keys, values, init_pred, key_padding_mask=None):
        """
        queries: (1, batch_size, d_model * 3) for the current edge embeddings
        keys: (max_neighbors, batch_size, d_model * 3) for the neighbor edge embeddings
        values: (max_neighbors, batch_size, d_model) for the neighbor relation embeddings
        init_pred: (1, batch_size, d_model) for the current relation embeddings from initial predictions
        """
        queries = self.in_proj_query(queries)
        keys = self.in_proj_key(keys)

        memory, _ = self.multihead_attn_self(query=keys, key=keys, value=values, key_padding_mask=key_padding_mask)
        memory = self.feed_forward_self(memory.squeeze(dim=0))

        attn_output, _ = self.multihead_attn_cross(query=queries, key=memory, value=memory, key_padding_mask=key_padding_mask)
        output = self.feed_forward_cross(attn_output.squeeze(dim=0))

        output = self.tanh(output) + init_pred.squeeze(dim=0)  # skip connection
        return output

# class EdgeAttentionModel(nn.Module):
#     def __init__(self, d_model, nhead=8, num_decoder_layers=3):
#         super(EdgeAttentionModel, self).__init__()
#         self.d_model = d_model
#
#         # Transformer components
#         decoder_layer = nn.TransformerDecoderLayer(d_model * 3, nhead)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
#         self.output = nn.Linear(d_model * 3, d_model)
#
#     def forward(self, tgt, memory, init_pred, memory_key_padding_mask=None):
#         # tgt: (1, batch_size, d_model) for the query embedding
#         # memory: (max_neighbors, batch_size, d_model) for the neighbor embeddings
#         output = self.transformer_decoder(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
#         output = self.output(output) + init_pred  # skip connection
#         return output
