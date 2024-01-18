import torch
import torch.nn as nn
from collections import OrderedDict
from .resnext import resnext50_32x4d, resnext101_32x8d


class Focus(nn.Module):
    """Focus width and height information into channel space.
    """

    def __init__(self,
                 out_channels,
                 downrate = 2,
                 kernel_size=1,
                 padding = 0):
        super().__init__()
        self.count_flod = 0
        while (downrate % 2 == 0):
            self.count_flod += 1
            downrate = downrate // 2

        self.conv = nn.Conv2d(
                    4**self.count_flod,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=1,
                )
    def forward(self, x):
        for _ in range(self.count_flod):
            # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
            patch_top_left = x[..., ::2, ::2]
            patch_top_right = x[..., ::2, 1::2]
            patch_bot_left = x[..., 1::2, ::2]
            patch_bot_right = x[..., 1::2, 1::2]
            x = torch.cat(
                (
                    patch_top_left,
                    patch_bot_left,
                    patch_top_right,
                    patch_bot_right,
                ),
                dim=1,
            )
        return self.conv(x)




class EncoderTemplate(nn.Module):
    def __init__(self, depth=50, pretrained=True, output_layers=['conv1','layer1','layer2','layer3']):
        super(EncoderTemplate, self).__init__()
        if depth == 50:
            self.resnet = resnext50_32x4d(pretrained=pretrained)
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


        self.stem0 = Focus(out_channels = 64,downrate = 2)
        self.stem1 = Focus(out_channels = 256,downrate = 4)
        self.stem2 = Focus(out_channels = 512,downrate = 8)
        self.stem3 = Focus(out_channels = 1024,downrate = 16)

        self.output_layers = output_layers

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, image, mask):
        r"""
        Args:
            image: already converted channel order, scaled and normalized.
                (B, C, H, W)
            mask: has the same shape with image.
                (B, 1, H, W)
        Returns:

        """
        outputs = OrderedDict()

        if self.output_layers is None:
            self.output_layers = self.resnet.output_layers

        x = self.conv1(image) + self.stem0(mask)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, self.output_layers):
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x) + self.stem1(mask)

        if self._add_output_and_check('layer1', x, outputs, self.output_layers):
            return outputs

        x = self.layer2(x) + self.stem2(mask)

        if self._add_output_and_check('layer2', x, outputs, self.output_layers):
            return outputs

        x = self.layer3(x) + self.stem3(mask)

        if self._add_output_and_check('layer3', x, outputs, self.output_layers):
            return outputs

        if len(self.output_layers) == 1 and self.output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


class EncoderSearch(nn.Module):
    def __init__(self, depth=50, pretrained=True, output_layers=['conv1','layer1','layer2','layer3']):
        super(EncoderSearch, self).__init__()
        if depth == 50:
            self.resnet = resnext50_32x4d(pretrained=pretrained)
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

        self.output_layers = output_layers

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, image):
        r"""
        Args:
            image: already converted channel order, scaled and normalized.
                (B, C, H, W)
            mask: has the same shape with image.
                (B, 1, H, W)
        Returns:

        """
        outputs = OrderedDict()

        if self.output_layers is None:
            self.output_layers = self.resnet.output_layers

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, self.output_layers):
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, self.output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, self.output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, self.output_layers):
            return outputs

        if len(self.output_layers) == 1 and self.output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')
