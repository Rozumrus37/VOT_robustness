# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
import ltr.models.backbone as backbones

from util.misc import NestedTensor

from ltr.models.neck.position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias



class BackboneBaseSegm(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor, target_mask=None):
        if target_mask is None:
            xs = self.body(tensor_list.tensors)
        else:
            xs = self.body(tensor_list.tensors, target_mask)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class SegmBackbone(BackboneBaseSegm):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 output_layers,
                 pretrained,
                 frozen_layers,
                 branch,
                 use_dilated = False):
        if branch == "template":
            backbone = backbones.EncoderM(depth=50, pretrained=pretrained, output_layers=output_layers,use_dilated = use_dilated)
        elif branch == "search":
            backbone = backbones.EncoderQ(depth=50, pretrained=pretrained, output_layers=output_layers,use_dilated = use_dilated)
        else:
            raise ValueError("Unsupported branch value: {}".format(branch))
        num_channels = 1024
        super().__init__(backbone, num_channels)


class JoinerSegm(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, target_mask=None):
        xs = self[0](tensor_list, target_mask)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(settings, backbone_pretrained=True, frozen_backbone_layers=(),use_dilated = False):
    position_embedding = build_position_encoding(settings)

    backbone_template = SegmBackbone(output_layers=['conv1','layer1','layer2','layer3'], pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers,branch = 'template',use_dilated = use_dilated)
    model_template = JoinerSegm(backbone_template, position_embedding)
    model_template.num_channels = backbone_template.num_channels

    backbone_search = SegmBackbone(output_layers=['conv1','layer1','layer2','layer3'], pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers,branch = 'search',use_dilated = use_dilated)
    model_search = JoinerSegm(backbone_search, position_embedding)
    model_search.num_channels = backbone_search.num_channels
    return model_template, model_search
