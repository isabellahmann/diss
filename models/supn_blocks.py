# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Blocks of networks
import torch.nn as nn
from enum import Enum


class ResNetType(Enum):
    UPSAMPLE = 0
    DOWNSAMPLE = 1
    SAME = 2


class ResnetBlock(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, block_type, use_bnorm=True, use_bias=True, layers_per_group=1, use_3D=False):
        super().__init__()
        assert isinstance(block_type, ResNetType)
        self.type = block_type

        self._use_bnorm = use_bnorm
        
        if use_3D:
            self._conv1 = nn.Conv3d(int(in_channels), int(out_channels), 3, bias=use_bias, padding=1)
            self._conv2 = nn.Conv3d(int(out_channels), int(out_channels), 3, bias=use_bias, padding=1)
        else:
            self._conv1 = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=use_bias, padding=1)
            self._conv2 = nn.Conv2d(int(out_channels), int(out_channels), 3, bias=use_bias, padding=1)

        if self._use_bnorm:
            self._bn1 = nn.GroupNorm(in_channels//layers_per_group, in_channels)#nn.BatchNorm2d(in_channels)
            self._bn2 = nn.GroupNorm(out_channels//layers_per_group, out_channels)#nn.BatchNorm2d(out_channels)

        self._relu1 = nn.LeakyReLU(0.2)
        self._relu2 = nn.LeakyReLU(0.2)
        self._resample = None

        if block_type == ResNetType.UPSAMPLE:
            if use_3D:
                self._resample = nn.Upsample(scale_factor=2, mode='trilinear')
            else:
                self._resample = nn.UpsamplingBilinear2d(scale_factor=2)

        elif block_type == ResNetType.DOWNSAMPLE:
            if use_3D:
                self._resample = nn.AvgPool3d(2)
                self._resample2 = nn.AvgPool3d(2)
            else:
                self._resample = nn.AvgPool2d(2)
                self._resample2 = nn.AvgPool2d(2)

        if use_3D:
            self._resid_conv = nn.Conv3d(int(in_channels), int(out_channels), 1, bias=use_bias)
        else:
            self._resid_conv = nn.Conv2d(int(in_channels), int(out_channels), 1, bias=use_bias)

    def forward(self, input):
        resid_connection = None

        if self.type == ResNetType.UPSAMPLE or self.type == ResNetType.SAME:
            if self._resample:
                input = self._resample(input)
            resid_connection = self._resid_conv(input)

        net = input
        if self._use_bnorm:
            net = self._bn1(net)
        net = self._relu1(net)
        net = self._conv1(net)

        if self._use_bnorm:
            net = self._bn2(net)
        net = self._relu2(net)
        net = self._conv2(net)

        if self.type == ResNetType.DOWNSAMPLE:
            net = self._resample(net)
            resid_connection = self._resid_conv(input)
            resid_connection = self._resample2(resid_connection)

        net = net + resid_connection
        return net