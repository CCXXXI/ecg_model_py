import torch.nn as nn
import torch
import numpy as np
import os
from models.resnet_cbam import resnet34_cbam
import torch.nn.functional as F

__all__ = ["Supcon_resnet34_cbam"]


class EncoderNet_resnet34_cbam(nn.Module):
    def __init__(self, encoder_feature_dim=512):
        super(EncoderNet_resnet34_cbam, self).__init__()
        encoder_model = resnet34_cbam()
        self.encoder = nn.Sequential(
            encoder_model.conv1,
            encoder_model.bn1,
            encoder_model.relu,
            encoder_model.maxpool,
            encoder_model.layer1,
            encoder_model.layer2,
            encoder_model.layer3,
            encoder_model.layer4,
            encoder_model.avgpool,
        )
        self.encoder_feature_dim = encoder_feature_dim

    def forward(self, x):
        """
        @param x:
        @return: shape(batchsize, 512)
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x


class Supcon(nn.Module):
    def __init__(self, encoder_model, head="mlp", feature_dim=128):
        super(Supcon, self).__init__()
        self.encoder = encoder_model
        self.feature_dim = feature_dim
        if head == "linear":
            self.head = nn.Linear(self.encoder.encoder_feature_dim, self.feature_dim)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(
                    self.encoder.encoder_feature_dim, self.encoder.encoder_feature_dim
                ),
                nn.ReLU(inplace=True),
                nn.Linear(self.encoder.encoder_feature_dim, self.feature_dim),
            )

    def forward(self, x, pretrain=False):
        out = self.encoder(x)
        out = self.head(out)
        if pretrain:
            out = F.normalize(out, p=2, dim=1)
        return out


def Supcon_resnet34_cbam(**kwargs):
    rs34_cbam_encoder = EncoderNet_resnet34_cbam()
    model = Supcon(rs34_cbam_encoder, head="mlp", feature_dim=128)
    return model
