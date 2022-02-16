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


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True
    )


class MLP2d(nn.Module):
    """Taken from pixpro"""

    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d, self).__init__()

        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = conv1x1(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x


class FeatureDecoder(nn.Module):
    def __init__(self, freeze_encoder=True):
        super(FeatureDecoder, self).__init__()

        self.freeze_encoder = freeze_encoder
        self.encoder = create_feature_extractor(
            resnet34(pretrained=True),
            return_nodes={
                "relu": "relu",
                "layer1": "layer1",
                "layer2": "layer2",
                "layer3": "layer3",
                "layer4": "layer4",
            },
        ).eval()

        # Freeze encoder if desired
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

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

        # Different ways to ensure encoder is frozen
        # TODO: Need to check
        # if self.freeze_encoder:
        #     with torch.no_grad:
        #         encoder_features = self.encoder(x)
        # else:
        encoder_features = self.encoder(x)

        if self.freeze_encoder:
            for feature in encoder_features.values():
                feature.requires_grad_ = False

        out = encoder_features[self.encoder_order[0]]
        for i in range(self.planes.size - 1):
            out = self.conv[(i, 0)](out)
            out = [upsample(out)]
            if self.skip_planes[i + 1]:
                out += [encoder_features[self.encoder_order[i + 1]]]
            out = torch.cat(out, 1)
            out = self.conv[(i, 1)](out)

        return out
