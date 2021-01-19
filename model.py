"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_backbone(self.architecture)
        self.fcs = self._create_head(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(x)

    def _create_backbone(self, architecture):
        ''' Exercise 1: complete the backbone construction'''
        layers = []

        return nn.Sequential(*layers)

    def _create_head(self, split_size, num_boxes, num_classes):
        ''' Exercise 1: complete the head construction'''
        S, B, C = split_size, num_boxes, num_classes


        return nn.Sequential()
