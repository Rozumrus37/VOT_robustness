import torch
from torch import nn
import torch.nn.functional as F
class Merge(nn.Module):
    def __init__(self,inchannels,outchannel,midchannel):
        super(Merge, self).__init__()
        self.inchannels = inchannels
        if len(inchannels) > 1:
            # Top layer
            self.toplayer = nn.Conv2d(inchannels[-1], midchannel, kernel_size=1, stride=1, padding=0)  # Reduce channels
            # Lateral layers
            self.latlayer = nn.Conv2d(inchannels[-2], midchannel, kernel_size=1, stride=1, padding=0)
            # Smooth layers
            self.smooth = nn.Conv2d(midchannel, outchannel, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        assert isinstance(x,list)
        if len(self.inchannels) > 1:
            # Top-down
            merge = self._upsample_add(self.toplayer(x[-1]), self.latlayer(x[-2]))
            # Smooth
            merge = self.smooth(merge)
        else:
            merge = x[-1]

        return merge
