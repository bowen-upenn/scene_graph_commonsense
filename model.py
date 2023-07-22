import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from reformer_pytorch import Reformer, Autopadder


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, embedding_dim, seq_len]``
        """
        x = x + self.pe[:x.size(2)].permute(2, 1, 0)
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # create constant 'pe' buffer with sinusoid values
        position_i = torch.arange(max_len).unsqueeze(1).repeat(1, max_len)
        position_j = torch.arange(max_len).unsqueeze(0).repeat(max_len, 1)
        position = position_i + position_j
        position = position.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)[:, :, :int(pe.shape[-1]/2)]  # odd number

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, embedding_dim, width, height]``
        """
        x = x + self.pe[:x.size(2), :x.size(3)].permute(2, 1, 0)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args
        self.num_classes = self.args['models']['num_classes']
        self.num_super_classes = self.args['models']['num_super_classes']
        self.d_model = self.args['transformer']['d_model']
        self.input_dim = self.args['models']['num_img_feature']
        self.hidden_dim = self.args['models']['hidden_dim']

        self.pos_encoder = PositionalEncoding2D(self.d_model-1)
        self.transformer_encoder = Reformer(dim=self.d_model, depth=self.args['transformer']['num_layers'], bucket_size=16,
                                            heads=self.args['transformer']['nhead'], lsh_dropout=self.args['transformer']['dropout'], causal=False)
        self.transformer_encoder = Autopadder(self.transformer_encoder)
        # encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.args['transformer']['nhead'],
        #                 dim_feedforward=self.args['transformer']['dim_feedforward'], dropout=self.args['transformer']['dropout'])
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.args['transformer']['num_layers'])

        # (1) crop out a minimum union bounding box
        # (2) two parallel conv layers to reduce dim
        #     conv only, no linear layers to keep 2d spatial dim, mega-pix tokens, and 2d positional embeds
        # (3) optional global features, one more parallel conv but larger pooling stride
        # (4) no dynamic batch sizing necessary

        # self.conv1 = nn.Conv2d(self.input_dim+1, self.input_dim+1, kernel_size=3, stride=2, padding=0)
        # self.conv2 = nn.Conv2d(self.input_dim+1, self.input_dim+1, kernel_size=3, stride=2, padding=0)
        # self.dropout1 = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.subject_proj = nn.Linear(self.args['models']['num_img_feature'])

        if args['dataset']['dataset'] == 'vg':
            self.fc1 = nn.Linear(self.d_model + 2 * (self.num_classes+self.num_super_classes), self.d_model)
        else:
            self.fc1 = nn.Linear(self.d_model + 2 * self.num_classes, self.d_model)
        if args['models']['hierarchical_pred']:
            self.fc2_1 = nn.Linear(self.d_model, self.args['models']['num_geometric'])
            self.fc2_2 = nn.Linear(self.d_model, self.args['models']['num_possessive'])
            self.fc2_3 = nn.Linear(self.d_model, self.args['models']['num_semantic'])
            self.fc3 = nn.Linear(self.d_model, 3)
        else:
            self.fc_flat = nn.Linear(self.d_model, self.args['models']['num_relations'])
        self.fc4 = nn.Linear(self.d_model, 1)
        self.dropout2 = nn.Dropout(p=0.5)

    def bayesian_head(self, pred):
        pred = self.dropout2(F.relu(self.fc1(pred)))
        connectivity = self.fc4(pred)  # (batch_size, 1)

        super_relation = F.log_softmax(self.fc3(pred), dim=1)
        relation_1 = self.fc2_1(pred)  # geometric
        relation_1 = F.log_softmax(relation_1, dim=1) + super_relation[:, 0].view(-1, 1)
        relation_2 = self.fc2_2(pred)  # possessive
        relation_2 = F.log_softmax(relation_2, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_3 = self.fc2_3(pred)  # semantic
        relation_3 = F.log_softmax(relation_3, dim=1) + super_relation[:, 2].view(-1, 1)
        return relation_1, relation_2, relation_3, super_relation, connectivity

    def flat_head(self, pred):
        pred = self.dropout(F.relu(self.fc1(pred)))
        connectivity = self.fc4(pred)  # (batch_size, 1)
        relation = self.fc_flat(pred)
        return relation, connectivity

    def concat_class_labels(self, h, c1, c2, s1, s2, rank, one_hot=True):
        if one_hot:
            c1 = F.one_hot(c1, num_classes=self.num_classes)
            c2 = F.one_hot(c2, num_classes=self.num_classes)
            if s1 is not None:
                sc1 = F.one_hot(torch.tensor([s[0] for s in s1]), num_classes=self.num_super_classes)
                for i in range(1, 4):  # at most 4 diff super class for each subclass instance
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
        return hc

    def organize_src(self, src_sub, src_obj, rank):
        # assuming src is of shape (batch_size, channels, height, width)
        src_sub = self.maxpool(src_sub)
        src_obj = self.maxpool(src_obj)

        has_value = torch.cat((torch.sum(src_sub.flatten(start_dim=2), dim=1) != 0,
                               torch.sum(src_obj.flatten(start_dim=2), dim=1) != 0), dim=1)  # size [bs, 2048]
        has_value = has_value.permute(1, 0)
        seq_lens = torch.sum(has_value, dim=0)
        max_len = torch.max(seq_lens)

        # apply conv layer to reduce spatial dimensions
        # print('src_sub 1', src_sub.shape, 'src_obj', src_obj.shape)
        # src_sub = self.maxpool(F.relu(self.conv1(src_sub)))
        # src_obj = self.maxpool(F.relu(self.conv2(src_obj)))
        # print('src_sub 2', src_sub.shape, 'src_obj', src_obj.shape)

        # linearize subject and object separately with their own positional encoding
        src_sub = self.pos_encoder(src_sub)
        src_sub = src_sub.flatten(start_dim=2)  # flattening height and width into one dimension
        src_sub = torch.cat((src_sub, torch.ones(src_sub.shape[0], 1, src_sub.shape[2]).to(rank)), dim=1)  # add subject or object marker
        src_sub = src_sub.permute(0, 2, 1)  # transformer expects batch, seq_len, embedding # (2, 0, 1) transformer expects seq_len, batch, embedding

        src_obj = self.pos_encoder(src_obj)
        src_obj = src_obj.flatten(start_dim=2)  # flattening height and width into one dimension
        src_obj = torch.cat((src_obj, torch.ones(src_obj.shape[0], 1, src_obj.shape[2]).to(rank)), dim=1)  # add subject or object marker
        src_obj = src_obj.permute(0, 2, 1)  # transformer expects batch, seq_len, embedding

        src = torch.cat((src_sub, src_obj), dim=1)  # size [2048, bs, 258]
        # print('src 1', src.shape)

        # filter out zeros the transformer for better efficiency but keep the original positional encoding
        src_filt_pad = torch.zeros(src.shape[0], max_len, src.shape[2]).to(rank)  # size [batch, seq_len, embedding]
        src_key_padding_mask = torch.ones((src.shape[0], max_len), dtype=torch.bool).to(rank)  # size [batch, seq_len]
        for bid in range(src.shape[0]):  # src.shape[1] is batch size
            src_filt_pad[bid, :seq_lens[bid], :] = src[bid, has_value[:, bid], :]
            src_key_padding_mask[bid, seq_lens[bid]:] = False  # 0 represents padding
        src = src_filt_pad
        # print('src 2', src.shape)
        # src_filt_pad = torch.zeros(max_len, src.shape[1], src.shape[2]).to(rank)  # size [seq_len, batch, embedding]
        # src_key_padding_mask = torch.zeros((src.shape[1], max_len), dtype=torch.bool).to(rank)  # size [batch, seq_len]
        # for bid in range(src.shape[1]):  # src.shape[1] is batch size
        #     src_filt_pad[:seq_lens[bid], bid, :] = src[has_value[:, bid], bid, :]
        #     src_key_padding_mask[bid, seq_lens[bid]:] = 1  # 1 represents padding
        # src = src_filt_pad

        return src, src_key_padding_mask

    def forward(self, src_sub, src_obj, cat_sub, cat_obj, scat_sub, scat_obj, rank):
        src, input_mask = self.organize_src(src_sub, src_obj, rank)
        del src_sub, src_obj

        # hidden = self.transformer_encoder(src=src, mask=None, src_key_padding_mask=src_key_padding_mask)  # keys, values, and queries are all the same
        hidden = self.transformer_encoder(src, input_mask=input_mask)  # keys, values, and queries are all the same
        hidden = hidden[:, -1, :]   # single output
        # print('hidden 1', hidden.shape)

        hidden = self.concat_class_labels(hidden, cat_sub, cat_obj, scat_sub, scat_obj, rank)
        # print('hidden 2', hidden.shape)

        if self.args['models']['hierarchical_pred']:
            # add the bayesian head for hierarchical predictions
            relation_1, relation_2, relation_3, super_relation, connectivity = self.bayesian_head(hidden)
            return relation_1, relation_2, relation_3, super_relation, connectivity
        else:
            relation, connectivity = self.flat_head(hidden)
            return relation, connectivity

    # def format_src(self, src, bbox_sub, bbox_obj, rank):
    #     """ src expects [batch_size, channels, height, width], bbox expects [batch_size, [xmin, xmax, ymin, ymax]]
    #     """
    #     # add 2d positional encoding
    #     src = self.pos_encoder(src)
    #
    #     # add marker encoding to distinguish subject [1], object [2], and union regions [3] in src
    #     src = torch.cat((src, torch.zeros(src.shape[0], 1, src.shape[2], src.shape[3]).to(rank)), dim=1)
    #     for bid in range(src.shape[0]):  # src.shape[0] is batch_size
    #         src[bid, -1, bbox_sub[bid, 2]:bbox_sub[bid, 3], bbox_sub[bid, 0]:bbox_sub[bid, 1]] += 1
    #         src[bid, -1, bbox_obj[bid, 2]:bbox_obj[bid, 3], bbox_obj[bid, 0]:bbox_obj[bid, 1]] += 2
    #
    #     # find zero paddings outside subject and object, remove them to reduce transformer input sequence length
    #     src = src.flatten(start_dim=2)  # size [batch_size, channels, seq_len]
    #     has_value = src[:, -1] != 0  # size [batch_size, seq_len]
    #     seq_lens = torch.sum(has_value, dim=1)
    #     max_len = torch.max(seq_lens)
    #
    #     # transformer expects [seq_len, batch_size, embedding]
    #     src = src.permute(2, 0, 1)
    #
    #     # filter out zeros the transformer for better efficiency but keep the original positional encoding
    #     src_filt_pad = torch.zeros(max_len, src.shape[1], src.shape[2]).to(rank)  # size [seq_len, batch_size, channels]
    #     src_key_padding_mask = torch.zeros((src.shape[1], max_len), dtype=torch.bool).to(rank)  # size [batch_size, seq_len]
    #     for bid in range(src.shape[1]):  # src.shape[1] is now batch_size
    #         src_filt_pad[:seq_lens[bid], bid, :] = src[has_value[bid], bid, :]
    #         src_key_padding_mask[bid, seq_lens[bid]:] = 1  # 1 represents padding
    #     src = src_filt_pad
    #
    #     return src, src_key_padding_mask
    #
    # def forward(self, image_feature, bbox_sub, bbox_obj, cat_sub, cat_obj, scat_sub, scat_obj, rank):
    #     src, src_key_padding_mask = self.format_src(image_feature, bbox_sub, bbox_obj, rank)
    #     del image_feature
    #
    #     hidden = self.transformer_encoder(src=src, mask=None, src_key_padding_mask=src_key_padding_mask)  # keys, values, and queries are all the same
    #     hidden = hidden[-1, :, :]   # single output
    #
    #     hidden = self.concat_class_labels(hidden, cat_sub, cat_obj, scat_sub, scat_obj, rank)
    #
    #     if self.args['models']['hierarchical_pred']:
    #         # add the bayesian head for hierarchical predictions
    #         relation_1, relation_2, relation_3, super_relation, connectivity = self.bayesian_head(hidden)
    #         return relation_1, relation_2, relation_3, super_relation, connectivity
    #     else:
    #         relation, connectivity = self.flat_head(hidden)
    #         return relation, connectivity


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
    def __init__(self, input_dim=256, num_geometric=15, num_possessive=11, num_semantic=24, T1=1, T2=1, T3=1):
        super(MotifHeadHier, self).__init__()
        self.fc3_1 = nn.Linear(2 * input_dim, num_geometric)
        self.fc3_2 = nn.Linear(2 * input_dim, num_possessive)
        self.fc3_3 = nn.Linear(2 * input_dim, num_semantic)
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

        self.fc2 = nn.Linear(4096 + 2 * (num_classes+num_super_classes), 512)
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
    def __init__(self, input_dim=128, feature_size=32, num_classes=150, num_super_classes=17, num_geometric=15, num_possessive=11, num_semantic=24, T1=1, T2=1, T3=1):
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
        self.fc2 = nn.Linear(4096 + 2 * (num_classes+num_super_classes), 512)
        self.fc3_1 = nn.Linear(512, num_geometric)
        self.fc3_2 = nn.Linear(512, num_possessive)
        self.fc3_3 = nn.Linear(512, num_semantic)
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
    def __init__(self, args, input_dim=128, output_dim=50, feature_size=32, num_classes=150, num_super_classes=17):
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
        if args['dataset']['dataset'] == 'vg':
            self.fc2 = nn.Linear(4096 + 2 * (num_classes+num_super_classes), 512)
        else:
            self.fc2 = nn.Linear(4096 + 2 * num_classes, 512)
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

        pred = self.dropout2(F.relu(self.fc2(hc)))
        relation = self.fc3(pred)       # (batch_size, 50)
        connectivity = self.fc4(pred)   # (batch_size, 1)
        return relation, connectivity


class EdgeHeadHier(nn.Module):
    """
    The local prediction module with a hierarchical classification.
    """
    def __init__(self, args, input_dim=128, feature_size=32, num_classes=150, num_super_classes=17, num_geometric=15, num_possessive=11, num_semantic=24, T1=1, T2=1, T3=1):
        super(EdgeHeadHier, self).__init__()
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

        pred = self.dropout2(F.relu(self.fc2(hc)))
        connectivity = self.fc4(pred)   # (batch_size, 1)
        super_relation = F.log_softmax(self.fc5(pred), dim=1)

        relation_1 = self.fc3_1(pred)   # geometric
        relation_1 = F.log_softmax(relation_1 / self.T1, dim=1) + super_relation[:, 0].view(-1, 1)
        relation_2 = self.fc3_2(pred)   # possessive
        relation_2 = F.log_softmax(relation_2 / self.T2, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_3 = self.fc3_3(pred)   # semantic
        relation_3 = F.log_softmax(relation_3 / self.T3, dim=1) + super_relation[:, 2].view(-1, 1)

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

        return relation_1, relation_2, relation_3, super_relation, connectivity, pred, pred_aug
