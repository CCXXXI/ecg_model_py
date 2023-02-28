# -*- coding: utf-8 -*-
"""
@time: 2019/9/8 20:14
直接修改torch的resnet
@ author: javis
"""

import torch.nn as nn
from models.cbam import CBAM
from torch.nn import init

__all__ = [
    "BasicBlock",
    "ResNet",
    "resnet34_cbam_ch1",
]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        use_cbam=False,
        dropout_rate=0.2,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        # self.dropout = nn.Dropout(.2)
        self.dropout = nn.Dropout(dropout_rate)

        if use_cbam:
            self.cbam = CBAM(planes, 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if not self.cbam is None:
            out = self.cbam(out)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=55,
        att_type=None,
        dropout_rate=0.2,
        initial_channel=8,
    ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(
            initial_channel, 64, kernel_size=15, stride=2, padding=7, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer1 = self._make_layer(
            block, 64, layers[0], att_type=att_type, dropout_rate=dropout_rate
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            att_type=att_type,
            dropout_rate=dropout_rate,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            att_type=att_type,
            dropout_rate=dropout_rate,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            att_type=att_type,
            dropout_rate=dropout_rate,
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode="fan_out")
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0

    def _make_layer(
        self, block, planes, blocks, stride=1, att_type=None, dropout_rate=0.2
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                use_cbam=att_type == "CBAM",
                dropout_rate=dropout_rate,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    use_cbam=att_type == "CBAM",
                    dropout_rate=dropout_rate,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet34_cbam_ch1(pretrained=False, **kwargs):
    """Constructs a ResNet-34-CBAM model with 1 channels.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(
        BasicBlock, [3, 4, 6, 3], att_type="CBAM", initial_channel=1, **kwargs
    )
    return model
