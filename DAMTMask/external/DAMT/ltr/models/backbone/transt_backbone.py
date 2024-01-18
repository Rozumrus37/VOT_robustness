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


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 output_layers,
                 pretrained,
                 frozen_layers):
        backbone = backbones.resnet50(output_layers=output_layers, pretrained=pretrained,
                                      frozen_layers=frozen_layers)
        num_channels = 1024
        super().__init__(backbone, num_channels)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class JoinerCVT(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, template_tensor_list: NestedTensor, search_tensor_list: NestedTensor):
        outputs = list()
        zs, dzs, xs = self[0](template_tensor_list, search_tensor_list)

        outs = list()
        poses = list()

        out: List[NestedTensor] = []
        pos = []
        # template
        for name, x in zs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        outs.append(out)
        poses.append(pos)

        out: List[NestedTensor] = []
        pos = []
        # dynamic template
        for name, x in dzs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        outs.append(out)
        poses.append(pos)

        out: List[NestedTensor] = []
        pos = []
        # search
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        outs.append(out)
        poses.append(pos)

        return outs, poses

class JoinerCVTSegm(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, template_tensor_list: NestedTensor, search_tensor_list: NestedTensor, templates_seg):
        outputs = list()
        zs, dzs, xs = self[0](template_tensor_list, search_tensor_list, templates_seg)

        outs = list()
        poses = list()

        out: List[NestedTensor] = []
        pos = []
        # template
        for name, x in zs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        outs.append(out)
        poses.append(pos)

        out: List[NestedTensor] = []
        pos = []
        # dynamic template
        for name, x in dzs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        outs.append(out)
        poses.append(pos)

        out: List[NestedTensor] = []
        pos = []
        # search
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        outs.append(out)
        poses.append(pos)

        return outs, poses

class CVTSegmBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 pretrained=None,
                 frozen_layers=None,
                 use_large_backbone = False):
        super().__init__()

        from .mixformer_seg import get_mixformer_model
        # TODO: pretrained
        backbone = get_mixformer_model(use_large_backbone = use_large_backbone)
        self.body = backbone
        self.num_channels = 384

    def forward(
        self,
        template_tensor_list: NestedTensor,
        search_tensor_list: NestedTensor,
        template_seg_list
    ):

        Nt, C, H, W = template_tensor_list.tensors.shape
        Ns, _, _, _ = search_tensor_list.tensors.shape
        # single template
        if Ns == Nt:
            template = template_tensor_list.tensors
            dynamic_template = template_tensor_list.tensors
            template_mask = template_tensor_list.mask
            dynamic_template_mask = template_tensor_list.mask

            template_seg = template_seg_list if len(template_seg_list.shape) == 4 else template_seg_list.unsqueeze(1)
            dynamic_template_seg = template_seg_list if len(template_seg_list.shape) == 4 else template_seg_list.unsqueeze(1)

        # multiple template
        else:
            template_tensor_list.tensors = template_tensor_list.tensors.view(Ns, -1, C, H, W)
            template_tensor_list.mask = template_tensor_list.mask.view(Ns, -1, H, W)
            template = template_tensor_list.tensors[:, 0, :, :, :]
            dynamic_template = template_tensor_list.tensors[:, 1, :, :, :]
            template_mask = template_tensor_list.mask[:, 0, :, :]
            dynamic_template_mask = template_tensor_list.mask[:, 1, :, :]

            template_seg = template_seg_list[:,0,:,:,:]
            dynamic_template_seg = template_seg_list[:,1,:,:,:]

        zs, dzs, xs = self.body(template, dynamic_template,
                       search_tensor_list.tensors,
                       template_seg, dynamic_template_seg)
        # TODO: fit multi-resolution out in CVT
        if not isinstance(zs, dict):
            zs = {'0': zs}
        if not isinstance(dzs, dict):
            dzs = {'0': dzs}
        if not isinstance(xs, dict):
            xs = {'0': xs}

        template_out: Dict[str, NestedTensor] = {}
        dynamic_template_out: Dict[str, NestedTensor] = {}
        search_out: Dict[str, NestedTensor] = {}
        for name, z in zs.items():
            m = template_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=z.shape[-2:]).to(torch.bool)[0]
            template_out[name] = NestedTensor(z, mask)

        for name, z in dzs.items():
            # TODO: make sure
            m = dynamic_template_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=z.shape[-2:]).to(torch.bool)[0]
            dynamic_template_out[name] = NestedTensor(z, mask)

        for name, x in xs.items():
            m = search_tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            search_out[name] = NestedTensor(x, mask)
        return template_out, dynamic_template_out, search_out



class CVTBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 pretrained=None,
                 frozen_layers=None):
        super().__init__()

        from .mixformer import get_mixformer_model
        # TODO: pretrained
        backbone = get_mixformer_model()
        self.body = backbone
        self.num_channels = 384

    def forward(
        self,
        template_tensor_list: NestedTensor,
        search_tensor_list: NestedTensor
    ):

        Nt, C, H, W = template_tensor_list.tensors.shape
        Ns, _, _, _ = search_tensor_list.tensors.shape
        # single template
        if Ns == Nt:
            template = template_tensor_list.tensors
            dynamic_template = template_tensor_list.tensors
            template_mask = template_tensor_list.mask
            dynamic_template_mask = template_tensor_list.mask
        # multiple template
        else:
            template_tensor_list.tensors = template_tensor_list.tensors.view(Ns, -1, C, H, W)
            template_tensor_list.mask = template_tensor_list.mask.view(Ns, -1, H, W)
            template = template_tensor_list.tensors[:, 0, :, :, :]
            dynamic_template = template_tensor_list.tensors[:, 1, :, :, :]
            template_mask = template_tensor_list.mask[:, 0, :, :]
            dynamic_template_mask = template_tensor_list.mask[:, 1, :, :]

        zs, dzs, xs = self.body(template, dynamic_template,
                       search_tensor_list.tensors)
        # TODO: fit multi-resolution out in CVT
        if not isinstance(zs, dict):
            zs = {'0': zs}
        if not isinstance(dzs, dict):
            dzs = {'0': dzs}
        if not isinstance(xs, dict):
            xs = {'0': xs}

        template_out: Dict[str, NestedTensor] = {}
        dynamic_template_out: Dict[str, NestedTensor] = {}
        search_out: Dict[str, NestedTensor] = {}
        for name, z in zs.items():
            m = template_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=z.shape[-2:]).to(torch.bool)[0]
            template_out[name] = NestedTensor(z, mask)

        for name, z in dzs.items():
            # TODO: make sure
            m = dynamic_template_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=z.shape[-2:]).to(torch.bool)[0]
            dynamic_template_out[name] = NestedTensor(z, mask)

        for name, x in xs.items():
            m = search_tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            search_out[name] = NestedTensor(x, mask)
        return template_out, dynamic_template_out, search_out





def build_backbone(settings, backbone_pretrained=True, frozen_backbone_layers=(), use_cvt=False, use_cvt_seg = False, use_large_backbone = False):
    position_embedding = build_position_encoding(settings)

    if use_cvt_seg:
        backbone = CVTSegmBackbone(use_large_backbone = use_large_backbone)
        model = JoinerCVTSegm(backbone, position_embedding)
    elif use_cvt:
        backbone = CVTBackbone()
        # TODO
        model = JoinerCVT(backbone, position_embedding)
    else:
        backbone = Backbone(output_layers=['conv1','layer1','layer2','layer3'], pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
        model = Joiner(backbone, position_embedding)

    model.num_channels = backbone.num_channels
    return model
