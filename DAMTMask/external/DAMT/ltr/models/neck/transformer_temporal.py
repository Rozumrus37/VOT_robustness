# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT TransformerTemporal class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor
import torch
import math

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

class TransformerTemporal(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_selfattention_layers=1,num_temporalattention_layers = 1,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()


        selfattention_layer = SelfAttentionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(selfattention_layer, num_selfattention_layers)

        temporalattention_layer = TemporalAttentionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        TemporalAttention_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(temporalattention_layer, TemporalAttention_norm, num_temporalattention_layers)

        fusion_layer = FusionLayer(d_model, dim_feedforward)
        self.fusion = Fusion(fusion_layer,d_model)

        self.response = MLP(d_model, d_model//2, 1, 3)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src_temp, mask_temp, pos_temp, response_temp, src_search, mask_search,  pos_search):
        # src_temp [b,num_image,c,2*h,2*w]
        # mask_temp [b,num_image,h,w]
        # pos_temp [b,num_image,c,h,w]
        # response_temp [b,num_image,h,w]

        # src_search [b,c,2*h,2*w]
        # mask_search [b,h,w]
        # pos_search [b,c,h,w]


        src_temp = src_temp.flatten(3).permute(1, 3, 0, 2).flatten(0,1)
        pos_temp = pos_temp.flatten(3).permute(1, 3, 0, 2).flatten(0,1) #模板的位置信息给的可能不好，分不清每个模板，可以给绝对准确的模板加个标志
        mask_temp = mask_temp.flatten(1)
        response_temp = response_temp.flatten(2).permute(1, 2, 0).flatten(0,1).unsqueeze(-1)

        search_shape = src_search.shape #torch.Size([16, 256, 32, 32])
        src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_search = mask_search.flatten(1)


        src_temp = self.encoder(src=src_temp,padding_mask=None,pos = pos_temp)
        src_search = self.encoder(src=src_search,padding_mask=None,pos = pos_search)

        src_temp = self.fusion(src_temp, src_temp+src_temp*response_temp, response_temp,
                                pos_query = pos_temp,
                                pos_key = pos_temp)


        src_search = self.decoder(query = src_search,
                          key = src_temp, value = src_temp,
                          query_padding_mask = None,
                          key_padding_mask = None,
                          pos_query = pos_search,
                          pos_key = pos_temp)

        responses_out = self.response(src_search.transpose(0, 1))
        responses_out = responses_out.permute(0, 2, 1).reshape((search_shape[0],-1,search_shape[2],search_shape[3]))

        return responses_out


class Decoder(nn.Module):

    def __init__(self, temporalattention_layer, norm=None,num_temporalattention_layers =1):
        super().__init__()
        self.layers = _get_clones(temporalattention_layer, num_temporalattention_layers)
        self.norm = norm

    def forward(self, query, key, value,
                     query_padding_mask: Optional[Tensor] = None,
                     key_padding_mask: Optional[Tensor] = None,
                     pos_query: Optional[Tensor] = None,
                     pos_key: Optional[Tensor] = None):
        output = query

        for layer in self.layers:
            output = layer(output, key, value,
                           query_padding_mask=query_padding_mask,
                           key_padding_mask=key_padding_mask,
                           pos_query=pos_query, pos_key=pos_key)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Encoder(nn.Module):

    def __init__(self, selfattention_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(selfattention_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src,
                padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output,padding_mask=padding_mask,pos=pos)
        return output

class FusionLayer(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64):
        super().__init__()
        self.temp = 30
        self.WK = nn.Linear(feature_dim, key_feature_dim)  # bias=False
        self.WQ = nn.Linear(feature_dim, key_feature_dim)

        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        # Init weights
        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, value,
                     pos_query: Optional[Tensor] = None,
                     pos_key: Optional[Tensor] = None):

        query = self.with_pos_embed(query, pos_query)
        key = self.with_pos_embed(key, pos_key)

        w_k = self.WK(key)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1,2,0) # Batch, Dim, Len_1

        w_q = self.WQ(query)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1,0,2) # Batch, Len_2, Dim

        dot_prod = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1

        # dot_prod = dot_prod * self.temp
        # dot_prod_max = torch.max(dot_prod,-1)[0].unsqueeze(-1)
        # dot_prod = dot_prod - dot_prod_max

        affinity = F.softmax(dot_prod * self.temp, dim=-1)

        w_v = value.permute(1,0,2) # Batch, Len_1, Dim
        # print(affinity.shape,w_v.shape)
        output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
        output = output.permute(1,0,2)

        return output




class Fusion(nn.Module):
    def __init__(self, fusion_layer,d_model):
        super().__init__()
        self.cross_attn = fusion_layer

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.linear = nn.Linear(d_model, d_model)


    def forward(self, query, key, value,
                     pos_query: Optional[Tensor] = None,
                     pos_key: Optional[Tensor] = None):

        output = self.layer(query, key, value,
                           pos_query=pos_query, pos_key=pos_key)

        return output

    def forward(self, query, key, value,
                pos_query: Optional[Tensor] = None,
                pos_key: Optional[Tensor] = None):

        mask_response = self.cross_attn(query=query, key=key, value=value)
        feature_1 = query * mask_response
        feature_1 = self.norm1(feature_1)

        mask_feature = self.cross_attn(query=query, key=key, value=key*value)
        feature_2 = query + mask_feature
        feature_2 = self.norm2(feature_2)

        feature = self.linear(feature_1 + feature_2)
        feature = self.norm3(feature)
        return feature


class TemporalAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, value,
                     query_padding_mask: Optional[Tensor] = None,
                     key_padding_mask: Optional[Tensor] = None,
                     pos_query: Optional[Tensor] = None,
                     pos_key: Optional[Tensor] = None,
                     ):
        query_v = self.multihead_attn(query=self.with_pos_embed(query, pos_query),
                                   key=self.with_pos_embed(key, pos_key),
                                   value=value,
                                   key_padding_mask=key_padding_mask)[0]
        query = query + self.dropout1(query_v)
        query = self.norm1(query)
        query_v = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(query_v)
        query = self.norm2(query)
        return query

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                     padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        query = key = value = src

        src2 = self.multihead_attn(query=self.with_pos_embed(query, pos),
                                   key=self.with_pos_embed(key, pos),
                                   value=value,
                                   key_padding_mask=padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_featurefusion_network(settings):
    return TransformerTemporal(
        d_model=settings.hidden_dim,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedforward=settings.dim_feedforward,
        num_selfattention_layers=settings.selfattention_layers
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
