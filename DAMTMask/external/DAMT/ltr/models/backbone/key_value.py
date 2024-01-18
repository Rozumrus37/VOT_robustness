import torch
import torch.nn as nn


class KeyValue(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels):
        super(KeyValue, self).__init__()
        self.key = nn.Conv2d(in_channels, key_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.value = nn.Conv2d(in_channels, value_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self._init_weights()

    def forward(self, x):
        return self.key(x), self.value(x)

    def _init_weights(self):
        conv_weight_std = 0.01
        for m in [self.key, self.value]:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=conv_weight_std)  # conv_weight_std=0.01
