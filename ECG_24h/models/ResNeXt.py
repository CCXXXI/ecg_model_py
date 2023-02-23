# -*- coding: utf-8 -*-
from __future__ import division

""" 
Creates a ResNeXt Model as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor, kernal_size=3):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv1d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm1d(D)
        # self.conv_conv = nn.Conv1d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.conv_conv = nn.Conv1d(D, D, kernel_size=kernal_size, stride=stride, padding=kernal_size//2,
                                   groups=cardinality, bias=False)
        self.bn = nn.BatchNorm1d(D)
        self.conv_expand = nn.Conv1d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm1d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class ECGResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ECGResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        leads_num = 8

        self.conv_1_3x3 = nn.Conv1d(leads_num, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm1d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.

        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.

        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = self.avgpool(x)

        x = x.view(-1, self.stages[3])
        return self.classifier(x)


class ECGResNeXt32(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, nlabels, base_width, widen_factor, kernal_size):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ECGResNeXt32, self).__init__()

        self.kernal_size = kernal_size

        self.cardinality = cardinality
        self.depth = 32
        # self.block_depth = (self.depth - 2) // 9
        self.block_depth = 2
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        # self.output_size = 64
        self.stages_num = (self.depth - 2) // 6
        # self.stages = [64 * (1 if i == 0 else self.widen_factor) * (2 ** i) for i in range(self.stages_num+1)]
        # print(self.stages)
        self.stages = [64 * (self.widen_factor ** i) for i in range(self.stages_num+1)]
        # print(self.stages)
        # self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        leads_num = 8

        self.conv_1_k = nn.Conv1d(leads_num, 64, kernal_size, 1, kernal_size//2, bias=False)
        self.bn_1 = nn.BatchNorm1d(64)

        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.stage_4 = self.block('stage_4', self.stages[3], self.stages[4], 2)
        self.stage_5 = self.block('stage_5', self.stages[4], self.stages[5], 2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(self.stages[5], nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.

        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.

        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor, kernal_size=self.kernal_size))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor, kernal_size=self.kernal_size))
        return block

    def forward(self, x):
        x = self.conv_1_k.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = self.stage_4.forward(x)
        x = self.stage_5.forward(x)
        print(x.shape)
        x = self.avgpool(x)

        x = x.view(-1, self.stages[5])
        return self.classifier(x)


def ResNeXt29_conv3_cd8(num_classes: int) -> nn.Module:
    model = ECGResNeXt(cardinality=8, depth=29, nlabels=num_classes, base_width=64, widen_factor=4)
    return model


def ResNeXt32_conv7_cd8(num_classes: int) -> nn.Module:
    model = ECGResNeXt32(cardinality=8, nlabels=num_classes, base_width=64, widen_factor=2, kernal_size=7)
    return model


if __name__ == '__main__':
    # model = ECGResNeXt(cardinality=8, depth=29, nlabels=26, base_width=64, widen_factor=4)
    # model = ECGResNeXt(cardinality=8*2, depth=29+9, nlabels=26, base_width=64*2, widen_factor=4*2)
    # model = ResNeXt29_conv3_cd8(26)
    model = ResNeXt32_conv7_cd8(26)
    import torch
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    print(model)



    x = torch.randn(1, 8, 2048)
    x = model(x)
    print(x.shape)
    print(x)
