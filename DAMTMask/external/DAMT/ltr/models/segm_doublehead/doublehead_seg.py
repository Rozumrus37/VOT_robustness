import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2, interpolate,
                       accuracy)



class ResSegm(nn.Module):
    def __init__(self,transt,freeze_transt=False, mdim = 256,use_dilated = False):
        super(ResSegm, self).__init__()
        self.backbone_template = transt.backbone_template
        self.backbone_search = transt.backbone_search
        self.featurefusion_network_cnn = transt.featurefusion_network_cnn

        if freeze_transt:
            for p in self.parameters():
                p.requires_grad_(False)

        self.cnn_mask = ResMask(mdim = mdim, use_dilated = use_dilated)
        self._init_weights()

    def _init_weights(self):
        conv_weight_std = 0.01
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=conv_weight_std)  # conv_weight_std=0.01

    def forward(self, search, template, template_seg):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        feature_search, pos_search = self.backbone_search(search)
        src_search, mask_search= feature_search[-1].decompose()


        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feature_template, pos_template = self.backbone_template(template,template_seg)
        src_template, mask_template= feature_template[-1].decompose()


        mem_out = self.featurefusion_network_cnn(src_template,src_search)

        out = self.cnn_mask(mem_out,[feature_search[2].tensors,feature_search[1].tensors,feature_search[0].tensors])
        # out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        # out = torch.sigmoid(out)
        out = {"pred_masks":out}
        return out


class ResMask(nn.Module):
    def __init__(self,mdim,use_dilated = False):
        super(ResMask, self).__init__()
        self.aspp = ASPP([2, 4, 8], [2, 4, 8], mdim)
        # self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)

        self.rf_16_8 = Refine(in_planes=512, hid_planes=mdim, out_planes=mdim, use_dilated = use_dilated)  # 1/16 -> 1/8
        self.rf_8_4 = Refine(in_planes=256, hid_planes=mdim, out_planes=64)  # 1/8 -> 1/4
        self.rf_4_2 = Refine(in_planes=64, hid_planes=64, out_planes=32)  # 1/4 -> 1/2

        self.pred2 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
    def forward(self,mem_out,fpns):
        mem_out = self.ResMM(self.aspp(mem_out))
        mo_f2 = self.rf_16_8(fpns[0], mem_out)  # out: size=(H/8, W/8), channel=mdim
        mo_f1 = self.rf_8_4(fpns[1], mo_f2)  # out: size=(H/4, W/4), channel=mdim
        mo_f0 = self.rf_4_2(fpns[2], mo_f1)  # out: size=(H/2, W/2), channel=32
        out = self.pred2(F.relu(mo_f0))
        return out


class ASPP(nn.Module):
    def __init__(self, dilation_series, padding_series, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(1024, depth, 1, 1)
        self.bn_x = nn.BatchNorm2d(depth)
        self.conv2d_0 = nn.Conv2d(1024, depth, kernel_size=1, stride=1)
        self.bn_0 = nn.BatchNorm2d(depth)
        self.conv2d_1 = nn.Conv2d(1024, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                  dilation=dilation_series[0])
        self.bn_1 = nn.BatchNorm2d(depth)
        self.conv2d_2 = nn.Conv2d(1024, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                  dilation=dilation_series[1])
        self.bn_2 = nn.BatchNorm2d(depth)
        self.conv2d_3 = nn.Conv2d(1024, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                  dilation=dilation_series[2])
        self.bn_3 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d(depth * 5, depth, kernel_size=3, padding=1)  # 512 1x1Conv
        # self.bn = nn.BatchNorm2d(depth)
        # self.prelu = nn.PReLU()
        # for m in self.conv2d_list:
        #    m.weight.data.normal_(0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_stage_(self, dilation1, padding1):
        Conv = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=padding1, dilation=dilation1, bias=True)  # classes
        Bn = nn.BatchNorm2d(256)
        Relu = nn.ReLU(inplace=True)
        return nn.Sequential(Conv, Bn, Relu)

    def forward(self, x):
        # out = self.conv2d_list[0](x)
        # mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)
        out_0 = self.conv2d_0(x)
        out_0 = self.bn_0(out_0)
        out_0 = self.relu(out_0)
        out_1 = self.conv2d_1(x)
        out_1 = self.bn_1(out_1)
        out_1 = self.relu(out_1)
        out_2 = self.conv2d_2(x)
        out_2 = self.bn_2(out_2)
        out_2 = self.relu(out_2)
        out_3 = self.conv2d_3(x)
        out_3 = self.bn_3(out_3)
        out_3 = self.relu(out_3)
        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1)
        out = self.bottleneck(out)
        # out = self.bn(out)
        # out = self.prelu(out)
        # for i in range(len(self.conv2d_list) - 1):
        #    out += self.conv2d_list[i + 1](x)

        return out

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Refine(nn.Module):
    def __init__(self, in_planes, hid_planes, out_planes, scale_factor=2, use_dilated = False):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(in_planes, hid_planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(hid_planes, hid_planes)
        self.ResMM = ResBlock(hid_planes, out_planes)
        self.scale_factor = scale_factor
        self.use_dilated = use_dilated

    def forward(self, feat_from_backbone, feat_from_deeper):
        s = self.ResFS(self.convFS(feat_from_backbone))
        if self.use_dilated:
            m = s + feat_from_deeper
        else:
            m = s + F.interpolate(feat_from_deeper, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


