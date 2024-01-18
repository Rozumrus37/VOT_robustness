import torch.nn as nn
import numpy as np

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2, interpolate,
                       accuracy)

class TransTFusionSegm(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, transt, freeze_transt=False):
        super().__init__()
        self.backbone = transt.backbone
        if freeze_transt:
            for p in self.parameters():
                p.requires_grad_(False)

        self.featurefusion_network = transt.featurefusion_network
        self.input_proj = transt.input_proj

        hidden_dim, nheads = transt.featurefusion_network.d_model, transt.featurefusion_network.nhead
        self.mask_head = MaskHeadSmallConv(hidden_dim, [1024, 512, 256, 64], hidden_dim)


    def forward(self, search, template, template_seg):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        feature_search, pos_search = self.backbone(search)
        src_search, mask_search= feature_search[-1].decompose()
        src_search = self.input_proj(src_search)

        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feature_template, pos_template = self.backbone(template)
        src_template, mask_template= feature_template[-1].decompose()
        src_template = self.input_proj(src_template)

        hs_search, hs_temp, memory = self.featurefusion_network(
                                    src_template, mask_template, pos_template[-1],template_seg,
                                    src_search, mask_search, pos_search[-1])

        seg_masks = self.mask_head(memory, [feature_search[3].tensors, feature_search[2].tensors, feature_search[1].tensors, feature_search[0].tensors])
        num_queries_seg = 1
        outputs_seg_masks = seg_masks.view(-1, num_queries_seg, seg_masks.shape[-2], seg_masks.shape[-1])
        #
        out = {"pred_masks":outputs_seg_masks}
        return out

    def get_template(self,template, template_seg):

        if len(template.shape) == 5:
            template = template[:,0,...]
            template_seg = template_seg[:,0,...]

        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feature_template, pos_template = self.backbone(template)
        src_template, mask_template= feature_template[-1].decompose()
        src_template = self.input_proj(src_template)

        out = {
            'src_temp':src_template,
            'paddingmask_temp':mask_template,
            'pos_temp':pos_template[-1],
            'mask_init':template_seg
        }
        return out
    def get_mask(self, search, template_hub):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        feature_search, pos_search = self.backbone(search)
        src_search, mask_search= feature_search[-1].decompose()
        src_search = self.input_proj(src_search)

        hs_search, hs_temp, memory = self.featurefusion_network(
                                    template_hub['src_temp'], template_hub['paddingmask_temp'],
                                    template_hub['pos_temp'],template_hub['mask_init'],
                                    src_search, mask_search, pos_search[-1])

        seg_masks = self.mask_head(memory, [feature_search[3].tensors, feature_search[2].tensors, feature_search[1].tensors, feature_search[0].tensors])
        num_queries_seg = 1
        outputs_seg_masks = seg_masks.view(-1, num_queries_seg, seg_masks.shape[-2], seg_masks.shape[-1]).sigmoid()
        #
        out = {"pred_masks":outputs_seg_masks}
        return out

class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 32] # [264, 128, 64, 32, 16, 8]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1) # 264->264
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)  # 264->128
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)   # 128->64
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)   # 64->32
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)   # 32->16
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.lay6 = torch.nn.Conv2d(inter_dims[4], inter_dims[5], 3, padding=1)   # 16->8
        self.gn6 = torch.nn.GroupNorm(8, inter_dims[5])
        self.out_lay = torch.nn.Conv2d(inter_dims[5], 1, 3, padding=1)   # 8->1

        self.dim = dim #264

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)   # 1024->128
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)   # 512->64
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)   # 256->32
        self.adapter4 = torch.nn.Conv2d(fpn_dims[3], inter_dims[4], 1)   # 64->16

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, fpns):
        # x torch.Size([16, 256, 32, 32])
        # bbox_mask torch.Size([16, 1024, 8, 32, 32])
        # fpns[0] torch.Size([16, 1024, 32, 32])
        # fpns[1] torch.Size([16, 512, 32, 32])
        # fpns[2] torch.Size([16, 256, 64, 64])

        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

        # x = torch.cat([expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        # 把attention score和主干网络的特征cat到一起
        # expand(x, bbox_mask.shape[1]).shape
        # torch.Size([bs*1024, 256, 32, 232])
        # bbox_mask.flatten(0, 1).shape
        # torch.Size([bs*1024, 8, 32, 32])
        # x.shape
        # torch.Size([bs*1024, 264, 32, 32])

        x = self.lay1(x) #torch.Size([bs*1024, 264, 32, 32])
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x) #torch.Size([bs*1024, 128, 32, 32])
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0]) #torch.Size([bs, 1024, 32, 32])-> torch.Size([bs, 128, 32, 32])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 128, 32, 32])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest") #与浅层的特征相加
        x = self.lay3(x)   # 128->64
        x = self.gn3(x)
        x = F.relu(x) #torch.Size([bs*1024, 64, 32, 32])

        cur_fpn = self.adapter2(fpns[1]) #torch.Size([bs, 512, 32, 32]) -> torch.Size([bs, 64, 32, 32])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 64, 32, 32])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)   # 64->32
        x = self.gn4(x)
        x = F.relu(x) #torch.Size([bs*1024, 32, 32, 32])

        cur_fpn = self.adapter3(fpns[2]) #torch.Size([16, 256, 64, 64]) -> torch.Size([16, 32, 64, 64])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 32, 64, 64])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest") #torch.Size([bs*1024, 32, 64, 64])
        x = self.lay5(x)   # 32->16
        x = self.gn5(x)
        x = F.relu(x)

        cur_fpn = self.adapter4(fpns[3]) #torch.Size([16, 64, 128, 128]) -> torch.Size([16, 16, 128, 128])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 16, 128, 128])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest") #torch.Size([bs*1024, 16, 128, 128])
        x = self.lay6(x)   # 16->8
        x = self.gn6(x)
        x = F.relu(x)

        x = self.out_lay(x)   # 8->1 #torch.Size([bs*1024, 1, 128, 128])
        return x



