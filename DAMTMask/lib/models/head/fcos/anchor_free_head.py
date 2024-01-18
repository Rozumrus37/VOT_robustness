import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_head.taskhead_base import TRACK_HEADS


class ConvModule(nn.Module):
    def __init__(self, in_channels, mdim):
        super(ConvModule, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, mdim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mdim, mdim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mdim, mdim, kernel_size=3, padding=1),
        )

    def forward(self, m):
        m = self.conv_layers(m)
        return m


def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = np.linspace(0., fm_height - 1.,
                         fm_height).reshape(1, fm_height, 1, 1)
    y_list = y_list.repeat(fm_width, axis=2)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, fm_width, 1)
    x_list = x_list.repeat(fm_height, axis=1)
    xy_list = score_offset + np.concatenate((x_list, y_list), 3) * total_stride
    xy_ctr = np.repeat(xy_list, batch, axis=0).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    # TODO: consider use float32 type from the beginning of this function
    xy_ctr = torch.from_numpy(xy_ctr.astype(np.float32))
    return xy_ctr


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred


@TRACK_HEADS.register
class AnchorFreeHead(ModuleBase):
    default_hyper_params = dict(
        total_stride=8,
        score_size=0,
        x_size=0,
        input_size_adapt=False,
        in_channels=512,
    )

    def __init__(self, ):
        super(AnchorFreeHead, self).__init__()
        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))

    def solve(self, y):
        cc = self.cls_ctr(y)
        rr = self.reg(y)
        classification_score = self.cls_score(cc)
        centerness = self.ctr_score(cc)
        regression = self.reg_offsets(rr)

        return classification_score, centerness, regression

    def forward(self, dec_output, x_size=0):
        cls_score, ctr_score, offsets = self.solve(dec_output)

        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)

        ctr_score = ctr_score.permute(0, 2, 3, 1)
        ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)

        offsets = torch.exp(self.si * offsets + self.bi) * self.total_stride

        # bbox decoding
        if self._hyper_params["input_size_adapt"] and x_size > 0:
            score_offset = (x_size - 1 - (offsets.size(-1) - 1) * self.total_stride) // 2
            fm_ctr = get_xy_ctr_np(offsets.size(-1), score_offset, self.total_stride)
            fm_ctr = fm_ctr.to(offsets.device)
        else:
            fm_ctr = self.fm_ctr.to(offsets.device)
        bbox = get_box(fm_ctr, offsets)

        return [cls_score, ctr_score, bbox]

    def update_params(self):
        super().update_params()

        x_size = self._hyper_params["x_size"]
        self.score_size = self._hyper_params["score_size"]
        self.total_stride = self._hyper_params["total_stride"]
        score_offset = (x_size - 1 - (self.score_size - 1) * self.total_stride) // 2
        self._hyper_params["score_offset"] = score_offset
        self.score_offset = self._hyper_params["score_offset"]

        ctr = get_xy_ctr_np(self.score_size, self.score_offset, self.total_stride)
        self.fm_ctr = ctr
        self.fm_ctr.require_grad = False

        self._make_net()
        self._initialize_conv()

    def _make_net(self):
        self.in_channels = self._hyper_params["in_channels"]
        self.cls_ctr = ConvModule(self.in_channels * 2, self.in_channels)
        self.reg = ConvModule(self.in_channels * 2, self.in_channels)

        # has bn, no relu
        self.cls_score = conv_bn_relu(self.in_channels, 1, stride=1, kszie=1, pad=0, has_relu=False)
        self.ctr_score = conv_bn_relu(self.in_channels, 1, stride=1, kszie=1, pad=0, has_relu=False)
        self.reg_offsets = conv_bn_relu(self.in_channels, 4, stride=1, kszie=1, pad=0, has_relu=False)

    def _initialize_conv(self, ):
        # initialze head
        conv_list = []
        for m in self.cls_ctr.modules():
            if isinstance(m, nn.Conv2d):
                conv_list.append(m)
        for m in self.reg.modules():
            if isinstance(m, nn.Conv2d):
                conv_list.append(m)
        conv_list.append(self.cls_score.conv)
        conv_list.append(self.ctr_score.conv)
        conv_list.append(self.reg_offsets.conv)
        conv_classifier = [self.cls_score.conv]
        assert all(elem in conv_list for elem in conv_classifier)

        conv_weight_std = 0.0001
        pi = 0.01
        bv = -np.log((1 - pi) / pi)
        for ith in range(len(conv_list)):
            # fetch conv from list
            conv = conv_list[ith]
            # torch.nn.init.normal_(conv.weight, std=0.01) # from megdl impl.
            torch.nn.init.normal_(conv.weight, std=conv_weight_std)  # conv_weight_std = 0.0001
            # nn.init.kaiming_uniform_(conv.weight, a=np.sqrt(5))  # from PyTorch default implementation
            # nn.init.kaiming_uniform_(conv.weight, a=0)  # from PyTorch default implementation
            if conv in conv_classifier:
                torch.nn.init.constant_(conv.bias, torch.tensor(bv))
            else:
                # torch.nn.init.constant_(conv.bias, 0)  # from PyTorch default implementation
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(conv.bias, -bound, bound)
