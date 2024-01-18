import torch.nn as nn
import numpy as np

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2, interpolate,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network

from ltr.models.neck.transformer_temporal import TransformerTemporal

class TransTCVTiouhsegmResponse(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, transt, freeze_transt=False,settings = None):
        super().__init__()
        self.featurefusion_network = transt.featurefusion_network
        self.class_embed = transt.class_embed
        self.bbox_embed = transt.bbox_embed
        self.input_proj = transt.input_proj
        self.backbone = transt.backbone
        self.iou_embed = transt.iou_embed

        self.bbox_attention_self = transt.bbox_attention_self
        self.bbox_attention_cross = transt.bbox_attention_cross
        self.mask_head = transt.mask_head

        if freeze_transt:
            for p in self.parameters():
                p.requires_grad_(False)

        self.transformer_temporal = TransformerTemporal(d_model=settings.hidden_dim, nhead=settings.nheads,dim_feedforward=512, num_selfattention_layers=1,num_temporalattention_layers = 2)

    def forward(self, searches, templates, response_anno, abspositions, paddingmasks):

        src_templates_list = []
        mask_templates_list = []
        pos_templates_list = []

        src_search_list = []
        mask_search_list = []
        pos_search_list = []
        src_search_list_levels = []
        bs, n_t, c, h, w = templates.shape
        for i in range(searches.shape[1]):
            templates_tmp = templates.reshape(bs * n_t, c, h, w).clone()
            if not isinstance(templates_tmp, NestedTensor):
                templates_tmp = nested_tensor_from_tensor(templates_tmp)

            search_tmp = searches[:,i,...]
            if not isinstance(search_tmp, NestedTensor):
                search_tmp = nested_tensor_from_tensor(search_tmp)

            outs, pos = self.backbone(templates_tmp, search_tmp)

            feature_templates, feature_d_templates, feature_search = outs
            pos_templates, pos_d_templates, pos_search  = pos

            #==============template
            src_templates = torch.cat([feature_templates[-1].tensors.unsqueeze(1), feature_d_templates[-1].tensors.unsqueeze(1)], axis=1)
            mask_templates = torch.cat([feature_templates[-1].mask.unsqueeze(1), feature_d_templates[-1].mask.unsqueeze(1)], axis=1)
            src_templates = src_templates.view([-1] + list(src_templates.shape[-3:]))
            mask_templates = mask_templates.view([-1] + list(mask_templates.shape[-2:]))
            pos_templates = torch.cat([pos_templates[-1].unsqueeze(1), pos_d_templates[-1].unsqueeze(1)], axis=1)

            src_templates = self.input_proj(src_templates)
            _, c_src, h_src, w_src = src_templates.shape
            pos_templates = pos_templates.reshape(bs, n_t, c_src, h_src, w_src)
            src_templates = src_templates.reshape(bs, n_t, c_src, h_src, w_src)
            mask_templates = mask_templates.reshape(bs, n_t, h_src, w_src)

            src_templates_list.append(src_templates)
            mask_templates_list.append(mask_templates)
            pos_templates_list.append(pos_templates)
            #==============template

            #==============search
            src_search, mask_search= feature_search[-1].decompose()
            src_search = self.input_proj(src_search)
            pos_search = pos_search[-1]

            src_search_list.append(src_search)
            mask_search_list.append(mask_search)
            pos_search_list.append(pos_search)
            src_search_list_levels.append(feature_search)
            #==============search

        src_templates = src_templates_list[-1]
        mask_templates = mask_templates_list[-1]
        pos_templates = pos_templates_list[-1]

        src_search = src_search_list[-1]
        mask_search = mask_search_list[-1]
        pos_search = pos_search_list[-1]
        feature_search = src_search_list_levels[-1]

        temporal_searches = torch.stack(src_search_list[:-1],1)

        hs_search, hs_temp, memory = self.featurefusion_network(src_templates, mask_templates, pos_templates,
                                              src_search, mask_search, pos_search)

        outputs_class = self.class_embed(hs_search)
        outputs_coord = self.bbox_embed(hs_search).sigmoid()
        outputs_iouh = self.iou_embed(torch.cat((hs_search, outputs_coord), 3)).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_iouh': outputs_iouh[-1]}



        #====================mask
        score = self._convert_score(outputs_class[-1]) # (16,1024)
        index_1 = score.argmax(axis=1)
        index_0 = np.arange(0, bs, 1)
        hs_search_choosen = hs_search[-1][index_0, index_1, :].unsqueeze(1)
        hs_temp_choosen = hs_temp[-1][:,119:120,:]

        #这里如果把hs[-1]共1024个向量全输入进取占显存太多，暂时取hs_temp中心的一个，并不是非常合理，后边修改
        bbox_mask_self = self.bbox_attention_self(hs_search_choosen, memory, mask=mask_search) #torch.Size([bs, n_q, nhead, h, w]) attention score 8-head 100-query
        bbox_mask_cross = self.bbox_attention_cross(hs_temp_choosen, memory, mask=mask_search) #torch.Size([bs, n_q, nhead, h, w]) attention score 8-head 100-query
        bbox_mask = torch.cat([bbox_mask_self, bbox_mask_cross], dim=2)

        #fpn的选择不一定合理
        seg_masks = self.mask_head(memory, bbox_mask, [feature_search[2].tensors, feature_search[1].tensors, feature_search[0].tensors])
        num_queries_seg = 1
        outputs_seg_masks = seg_masks.view(bs, num_queries_seg, seg_masks.shape[-2], seg_masks.shape[-1])
        out["pred_masks"] = outputs_seg_masks
        #====================mask



        # #================temporal
        # temporal_search_memorys = []
        # for i in range(len(src_search_list)):
        #     pos_temporal_search = pos_search_list[i]
        #     src_temporal_search = src_search_list[i]
        #     mask_temporal_search = mask_search_list[i]
        #     _, _, temporal_search_memory = self.featurefusion_network(src_templates, mask_templates, pos_templates,
        #                                       src_temporal_search, mask_temporal_search, pos_temporal_search)
        #     temporal_search_memorys.append(temporal_search_memory)

        # # abspositions = self.absposition_map(abscoords)
        # src_key = torch.stack(temporal_search_memorys,dim = 1)
        # mask_key = paddingmasks[:,:-1,...]
        # pos_key = abspositions[:,:-1,...]
        # response_key = response_anno[:,:-1,...]

        # src_query = memory
        # mask_query = paddingmasks[:,-1,...]
        # pos_query = abspositions[:,-1,...]
        # response_query = response_anno[:,-1,...]

        # response = self.transformer_temporal(src_key, mask_key, pos_key, response_key, src_query, mask_query,  pos_query)
        # out["pred_responses"] = response
        # #================temporal


        #================temporal
        src_key = temporal_searches
        mask_key = paddingmasks[:,:-1,...]
        pos_key = abspositions[:,:-1,...]
        response_key = response_anno[:,:-1,...]

        src_query = src_search
        mask_query = paddingmasks[:,-1,...]
        pos_query = abspositions[:,-1,...]
        response_query = response_anno[:,-1,...]

        response = self.transformer_temporal(src_key, mask_key, pos_key, response_key, src_query, mask_query,  pos_query)
        out["pred_responses"] = response
        #================temporal



        return out


        # out["pred_masks"].shape
        # torch.Size([16, 1, 64, 64])
        # out["pred_boxes"].shape
        # torch.Size([16, 1024, 4])
        # out["pred_logits"].shape
        # torch.Size([16, 1024, 2])

    def track(self, search, templates: list):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        feature_search, pos_search = self.backbone(search)
        src_search, mask_search= feature_search[-1].decompose()
        src_search = self.input_proj(src_search)

        for i in range(len(templates)):
            if i == 0:
                src_templates = templates[i]['src']
                mask_templates = templates[i]['mask']
                pos_templates = templates[i]['pos']
            else:
                src_templates = torch.cat((src_templates, templates[i]['src']), 1)
                mask_templates = torch.cat((mask_templates, templates[i]['mask']), 1)
                pos_templates = torch.cat((pos_templates, templates[i]['pos']), 1)

        hs, _, _ = self.featurefusion_network(src_templates, mask_templates, pos_templates,
                                              src_search, mask_search, pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_iouh = self.iou_embed(torch.cat((hs, outputs_coord), 3)).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_iouh': outputs_iouh[-1]}
        return out

    def track_seg(self, search, templates: list, mask=False):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        feature_search, pos_search = self.backbone(search)
        bs = feature_search[-1].tensors.shape[0]
        src_search, mask_search= feature_search[-1].decompose()
        src_search = self.input_proj(src_search)

        for i in range(len(templates)):
            if i == 0:
                src_templates = templates[i]['src']
                mask_templates = templates[i]['mask']
                pos_templates = templates[i]['pos']
            else:
                src_templates = torch.cat((src_templates, templates[i]['src']), 1)
                mask_templates = torch.cat((mask_templates, templates[i]['mask']), 1)
                pos_templates = torch.cat((pos_templates, templates[i]['pos']), 1)

        hs_search, hs_temp, memory = self.featurefusion_network(src_templates, mask_templates, pos_templates,
                                              src_search, mask_search, pos_search[-1])

        outputs_class = self.class_embed(hs_search)
        outputs_coord = self.bbox_embed(hs_search).sigmoid()
        outputs_iouh = self.iou_embed(torch.cat((hs_search, outputs_coord), 3)).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_iouh': outputs_iouh[-1]}

        if mask==True:
            score = self._convert_score(outputs_class[-1]) # (16,1024)
            index_1 = score.argmax(axis=1)
            index_0 = np.arange(0, bs, 1)
            hs_search_choosen = hs_search[-1][index_0, index_1, :].unsqueeze(1)
            hs_temp_choosen = hs_temp[-1][:,119:120,:]

            #这里如果把hs[-1]共1024个向量全输入进取占显存太多，暂时取hs_temp中心的一个，并不是非常合理，后边修改
            bbox_mask_self = self.bbox_attention_self(hs_search_choosen, memory, mask=mask_search) #torch.Size([bs, n_q, nhead, h, w]) attention score 8-head 100-query
            bbox_mask_cross = self.bbox_attention_cross(hs_temp_choosen, memory, mask=mask_search) #torch.Size([bs, n_q, nhead, h, w]) attention score 8-head 100-query
            bbox_mask = torch.cat([bbox_mask_self, bbox_mask_cross], dim=2)

            #fpn的选择不一定合理
            seg_masks = self.mask_head(memory, bbox_mask, [feature_search[3].tensors, feature_search[2].tensors, feature_search[1].tensors, feature_search[0].tensors])
            num_queries_seg = 1
            outputs_seg_masks = seg_masks.view(bs, num_queries_seg, seg_masks.shape[-2], seg_masks.shape[-1])
            #
            outputs_seg_masks = interpolate(outputs_seg_masks, size=search.tensors.shape[-2:],
                                    mode="bilinear", align_corners=False)
            out["pred_masks"] = outputs_seg_masks.sigmoid()

        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        feature_template, pos_template = self.backbone(z)
        src_template, mask_template = feature_template[-1].decompose()
        template_out = {
            'pos': pos_template[-1].unsqueeze(1),
            'src': self.input_proj(src_template).unsqueeze(1),
            'mask': mask_template.unsqueeze(1)
        }
        return template_out

    def _convert_score(self, score):
        score = F.softmax(score, dim=2).data[:, :, 0].cpu().numpy()
        return score
        #score.shape (16, 1024)
class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        num_positive = (prediction > self.threshold).float().sum()

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)
        if num_positive > 0:
            loss = loss.sum()/num_positive
        else:
            loss = loss.sum()
        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss
