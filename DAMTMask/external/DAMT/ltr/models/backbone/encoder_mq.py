import torch
import torch.nn as nn
from collections import OrderedDict
from .resnext import resnext50_32x4d, resnext101_32x8d


class EncoderM(nn.Module):
    def __init__(self, depth=50, pretrained=True,output_layers = ['conv1','layer1','layer2','layer3'],use_dilated = False):
        super(EncoderM, self).__init__()
        self.output_layers = output_layers
        if depth == 50:
            self.resnet = resnext50_32x4d(pretrained=pretrained, use_dilated = use_dilated)
        elif depth == 101:
            self.resnet = resnext101_32x8d(pretrained=pretrained)
        else:
            raise ValueError("Undefined depth: {}".format(depth))

        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool

        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, image, mask, output_layers=None):
        r"""
        Args:
            image: already converted channel order, scaled and normalized.
                (B, C, H, W)
            mask: has the same shape with image.
                (B, 1, H, W)
        Returns:

        """
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(image) + self.conv1_m(mask)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


class EncoderQ(nn.Module):
    def __init__(self, depth=50, pretrained=True,output_layers = ['conv1','layer1','layer2','layer3'],use_dilated = False):
        super(EncoderQ, self).__init__()
        self.output_layers = output_layers
        if depth == 50:
            self.resnet = resnext50_32x4d(pretrained=pretrained, use_dilated = use_dilated)
        elif depth == 101:
            self.resnet = resnext101_32x8d(pretrained=pretrained)
        else:
            raise ValueError("Undefined depth: {}".format(depth))

        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool

        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, image, output_layers=None):
        r"""
        Args:
            image: already converted channel order, scaled and normalized.
                (B, C, H, W)
            mask: has the same shape with image.
                (B, 1, H, W)
        Returns:

        """
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')
