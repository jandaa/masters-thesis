from typing import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from torchvision.models.resnet import conv3x3
from torchvision.models.feature_extraction import create_feature_extractor


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()

        self.conv = conv3x3(
            in_planes,
            out_planes,
        )
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


class FeatureDecoder(nn.Module):
    def __init__(self):
        super(FeatureDecoder, self).__init__()

        self.encoder = create_feature_extractor(
            resnet34(pretrained=True),
            return_nodes={
                "relu": "relu",
                "layer1": "layer1",
                "layer2": "layer2",
                "layer3": "layer3",
                "layer4": "layer4",
            },
        )

        self.planes = np.array([512, 256, 128, 64, 32, 32])
        self.skip_planes = np.array([0, 256, 128, 64, 0, 0])
        self.conv = OrderedDict()
        self.encoder_order = ["layer4", "layer3", "layer2", "layer1"]

        # Build decoder
        for i in range(self.planes.size - 1):
            in_planes = self.planes[i]
            out_planes = self.planes[i + 1]
            self.conv[(i, 0)] = ConvBlock(in_planes, out_planes)

            in_planes = out_planes + self.skip_planes[i + 1]
            self.conv[(i, 1)] = ConvBlock(in_planes, out_planes)

        self.decoder = nn.ModuleList(list(self.conv.values()))

    def forward(self, x):
        encoder_features = self.encoder(x)

        out = encoder_features[self.encoder_order[0]]
        for i in range(self.planes.size - 1):
            out = self.conv[(i, 0)](out)
            out = [upsample(out)]
            if self.skip_planes[i + 1]:
                out += [encoder_features[self.encoder_order[i + 1]]]
            out = torch.cat(out, 1)
            out = self.conv[(i, 1)](out)

        return out
