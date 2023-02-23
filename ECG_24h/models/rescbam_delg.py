import torch.nn as nn
import torch
import numpy as np
import os
from models.resnet_cbam import ResNet, BasicBlock
from models.vit import ViT
import torch.nn.functional as F
from torch.nn import init

__all__ = [
    "resnet34_cbam_global",
    "resnet34_cbam_global_nw",
    "resnet34_cbam_local",
    "resnet34_cbam_local_2",
    "resnet34_cbam_local_3",
    "resnet34_cbam_local_4",
    "resnet34_cbam_local_5",
    "resnet34_cbam_local_6",
    "resnet34_cbam_local_7",
    "resnet34_cbam_delg",
    "resnet34_cbam_delg_2",
    "resnet34_cbam_delg_3",
    "resnet34_cbam_delg_4",
    "resnet34_cbam_delg_6",
]


class Resnet34CBAMBackbone(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(Resnet34CBAMBackbone, self).__init__()
        resnet34_cbam = ResNet(
            BasicBlock, [3, 4, 6, 3], att_type="CBAM", dropout_rate=dropout_rate
        )

        self.conv1 = resnet34_cbam.conv1
        self.bn1 = resnet34_cbam.bn1
        self.relu = resnet34_cbam.relu
        self.maxpool = resnet34_cbam.maxpool
        self.layer1 = resnet34_cbam.layer1
        self.layer2 = resnet34_cbam.layer2
        self.layer3 = resnet34_cbam.layer3
        self.layer4 = resnet34_cbam.layer4
        self.layer3_output_channel_size = 256
        self.layer3_output_featuremap_size = 128
        self.output_channel_size = 512

    def forward(self, x):
        x = self.conv1(x)  # shape = [bsz, 64, 1024]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # shape = [bsz, 64, 512]

        x1 = self.layer1(x)  # shape = [bsz, 64, 512]
        x2 = self.layer2(x1)  # shape = [bsz, 128, 256]
        x3 = self.layer3(x2)  # shape = [bsz, 256, 128]
        x4 = self.layer4(x3)  # shape = [bsz, 512, 64]
        # print(x3.shape)
        # print(x4.shape)
        return x4, x3  # 分别返回深层特征 和 浅层特征


class GeneralizedMeanPooling(nn.Module):
    r"""注意下面是个二维版本的解释Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.adaptive_avg_pool1d(x, self.output_size).pow(1.0 / self.p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.p)
            + ", "
            + "output_size="
            + str(self.output_size)
            + ")"
        )


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """Same, but norm is trainable"""

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class GlobalHead(nn.Module):
    def __init__(self, in_channel_size=512, global_dim=256, use_whiten=True):
        super(GlobalHead, self).__init__()
        self.pool = GeneralizedMeanPoolingP()
        self.use_whiten = use_whiten
        if use_whiten:
            self.fc = nn.Linear(in_channel_size, global_dim, bias=True)
            init.kaiming_normal(self.fc.weight)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.use_whiten:
            x = self.fc(x)
        return x


class Resnet34CBAMGlobalModel(nn.Module):
    def __init__(
        self, num_classes=55, global_dim=256, use_whiten=True, dropout_rate=0.2
    ):
        super(Resnet34CBAMGlobalModel, self).__init__()
        self.backbone = Resnet34CBAMBackbone(dropout_rate=dropout_rate)
        if not use_whiten:
            global_dim = self.backbone.output_channel_size
        self.globalhead = GlobalHead(
            in_channel_size=self.backbone.output_channel_size,
            global_dim=global_dim,
            use_whiten=use_whiten,
        )
        self.fc = nn.Linear(global_dim, num_classes)

        init.kaiming_normal(self.fc.weight)

    def forward(self, x):
        deeper_feature, shallower_feature = self.backbone(
            x
        )  # [bsz, 512, 64], [bsz, 256, 128]
        global_feature = self.globalhead(deeper_feature)
        ret = self.fc(global_feature)
        return ret


class Resnet34CBAMLocalModel(nn.Module):
    def __init__(
        self,
        num_classes=55,
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=4,
        multi_heads=12,
        transfomer_mlp_dim=512,
    ):
        super(Resnet34CBAMLocalModel, self).__init__()
        self.backbone = Resnet34CBAMBackbone(dropout_rate=dropout_rate)
        self.backbone.layer4 = nn.Identity()  # 弃用掉Resnet的layer4
        self.transformer_model = ViT(
            num_patches=self.backbone.layer3_output_featuremap_size,
            dim=self.backbone.layer3_output_channel_size,
            depth=transformer_depth,
            heads=multi_heads,
            mlp_dim=transfomer_mlp_dim,
            dropout=tranformer_dropout_rate,
        )
        self.fc = nn.Linear(self.backbone.layer3_output_channel_size, num_classes)

        init.kaiming_normal(self.fc.weight)

    def forward(self, x):
        _, shallower_feature = self.backbone(x)  # [bsz, 512, 64], [bsz, 256, 128]
        output = self.transformer_model(shallower_feature)
        output = self.fc(output)
        return output


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x):
        out = F.normalize(x, p=2, dim=1)
        return out


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=-1)


class GlobalReduce(nn.Module):
    def __init__(self, op):
        super(GlobalReduce, self).__init__()
        self.op = op

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.op(x)
        x = torch.squeeze(x, dim=1)
        return x


class Resnet34CBAMDELGModel(nn.Module):
    def __init__(
        self,
        num_classes=55,
        global_pool="gemp",  # 'gemp', 'gem', 'avg'
        global_reduction="None",  # 'None', 'maxpool', 'avgpool', 'conv', 'fc'
        norm_mode="layer",  # 'None', 'layer', 'batch', 'l2'
        fusion_mode="con",  # 'con'
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=4,
        multi_heads=12,
        transfomer_mlp_dim=512,
    ):
        super(Resnet34CBAMDELGModel, self).__init__()
        self.backbone = Resnet34CBAMBackbone(dropout_rate=dropout_rate)

        self.transformer_model = ViT(
            num_patches=self.backbone.layer3_output_featuremap_size,
            dim=self.backbone.layer3_output_channel_size,
            depth=transformer_depth,
            heads=multi_heads,
            mlp_dim=transfomer_mlp_dim,
            dropout=tranformer_dropout_rate,
        )

        if global_pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif global_pool == "gem":
            self.global_pool = GeneralizedMeanPooling(norm=3)
        elif global_pool == "gemp":
            self.global_pool = GeneralizedMeanPoolingP()

        self.global_dim = self.backbone.output_channel_size
        self.global_reduce = nn.Identity()
        if global_reduction == "maxpool":
            self.global_reduce = GlobalReduce(
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )
            self.global_dim = self.global_dim // 2
        elif global_reduction == "avgpool":
            self.global_reduce = GlobalReduce(
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
            )
            self.global_dim = self.global_dim // 2
        elif global_reduction == "conv":
            self.global_reduce = GlobalReduce(
                nn.Conv1d(
                    in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1
                )
            )
            self.global_dim = self.global_dim // 2
        elif global_reduction == "fc":
            self.global_reduce = nn.Linear(self.global_dim, self.global_dim // 2)
            self.global_dim = self.global_dim // 2

        self.global_norm = nn.Identity()
        self.local_norm = nn.Identity()
        if norm_mode == "layer":
            self.global_norm = nn.LayerNorm(self.global_dim)
            self.local_norm = nn.LayerNorm(self.backbone.layer3_output_channel_size)
        elif norm_mode == "batch":
            self.global_norm = nn.BatchNorm1d(self.global_dim)
            self.local_norm = nn.BatchNorm1d(self.backbone.layer3_output_channel_size)
        elif norm_mode == "l2":
            self.global_norm = L2Norm()
            self.local_norm = L2Norm()

        self.fusion_layer = Concatenate()

        self.fc = nn.Linear(
            self.global_dim + self.backbone.layer3_output_channel_size, num_classes
        )

        init.kaiming_normal(self.fc.weight)

    def forward(self, x):
        deeper_feature, shallower_feature = self.backbone(
            x
        )  # [bsz, 512, 64], [bsz, 256, 128]
        global_feature = self.global_pool(deeper_feature)
        global_feature = global_feature.view(global_feature.size(0), -1)
        global_feature = self.global_reduce(global_feature)
        global_feature = self.global_norm(global_feature)

        local_feature = self.transformer_model(shallower_feature)
        local_feature = self.local_norm(local_feature)

        fusion_feature = self.fusion_layer(global_feature, local_feature)
        output = self.fc(fusion_feature)
        return output, global_feature, local_feature


def resnet34_cbam_global(**kwargs):
    """Constructs a ResNet-34_CBAM_Global model.
    dropout_rate = 0.2
    global_dim = 256

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet34CBAMGlobalModel(global_dim=256, dropout_rate=0.2, **kwargs)
    return model


def resnet34_cbam_global_nw(**kwargs):
    """Constructs a ResNet-34_CBAM_Global model without whiten layer (FC)
    dropout_rate = 0.2
    global_dim = 512

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet34CBAMGlobalModel(use_whiten=False, dropout_rate=0.2, **kwargs)
    return model


def resnet34_cbam_local(**kwargs):
    """Constructs a ResNet-34_CBAM with ViT local model"""
    model = Resnet34CBAMLocalModel(
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=4,
        multi_heads=12,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_local_2(**kwargs):
    """Constructs a ResNet-34_CBAM with ViT local model"""
    model = Resnet34CBAMLocalModel(
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=4,
        multi_heads=8,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_local_3(**kwargs):
    """Constructs a ResNet-34_CBAM with ViT local model"""
    model = Resnet34CBAMLocalModel(
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=3,
        multi_heads=12,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_local_4(**kwargs):
    """Constructs a ResNet-34_CBAM with ViT local model"""
    model = Resnet34CBAMLocalModel(
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=4,
        multi_heads=4,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_local_5(**kwargs):
    """Constructs a ResNet-34_CBAM with ViT local model"""
    model = Resnet34CBAMLocalModel(
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=6,
        multi_heads=4,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_local_6(**kwargs):
    """Constructs a ResNet-34_CBAM with ViT local model"""
    model = Resnet34CBAMLocalModel(
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=6,
        multi_heads=8,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_local_7(**kwargs):
    """Constructs a ResNet-34_CBAM with ViT local model"""
    model = Resnet34CBAMLocalModel(
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=8,
        multi_heads=4,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_delg(**kwargs):
    model = Resnet34CBAMDELGModel(
        global_pool="gemp",  # 'gemp', 'gem', 'avg'
        global_reduction="None",  # 'None', 'maxpool', 'avgpool', 'conv', 'fc'
        norm_mode="layer",  # 'None', 'layer', 'batch', 'l2'
        fusion_mode="con",  # 'con'
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=3,
        multi_heads=8,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_delg_2(**kwargs):
    model = Resnet34CBAMDELGModel(
        global_pool="gemp",  # 'gemp', 'gem', 'avg'
        global_reduction="None",  # 'None', 'maxpool', 'avgpool', 'conv', 'fc'
        norm_mode="batch",  # 'None', 'layer', 'batch', 'l2'
        fusion_mode="con",  # 'con'
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=3,
        multi_heads=8,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_delg_3(**kwargs):
    # 对齐resnet34_cbam_local_7的Transformer部分的参数
    model = Resnet34CBAMDELGModel(
        global_pool="gemp",  # 'gemp', 'gem', 'avg'
        global_reduction="None",  # 'None', 'maxpool', 'avgpool', 'conv', 'fc'
        norm_mode="layer",  # 'None', 'layer', 'batch', 'l2'
        fusion_mode="con",  # 'con'
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=8,
        multi_heads=4,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_delg_4(**kwargs):
    # resnet34_cbam_delg_3上换标准化方式为L2标准化
    model = Resnet34CBAMDELGModel(
        global_pool="gemp",  # 'gemp', 'gem', 'avg'
        global_reduction="None",  # 'None', 'maxpool', 'avgpool', 'conv', 'fc'
        norm_mode="l2",  # 'None', 'layer', 'batch', 'l2'
        fusion_mode="con",  # 'con'
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=8,
        multi_heads=4,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_delg_5(**kwargs):
    # resnet34_cbam_delg_4上加avgpool降维  # 实验做错，没加降维，实际和delg_4一样
    model = Resnet34CBAMDELGModel(
        global_pool="gemp",  # 'gemp', 'gem', 'avg'
        global_reduction="None",  # 'None', 'maxpool', 'avgpool', 'conv', 'fc'
        norm_mode="l2",  # 'None', 'layer', 'batch', 'l2'
        fusion_mode="con",  # 'con'
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=8,
        multi_heads=4,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


def resnet34_cbam_delg_6(**kwargs):
    # resnet34_cbam_delg_4上加avgpool降维
    model = Resnet34CBAMDELGModel(
        global_pool="gemp",  # 'gemp', 'gem', 'avg'
        global_reduction="avgpool",  # 'None', 'maxpool', 'avgpool', 'conv', 'fc'
        norm_mode="l2",  # 'None', 'layer', 'batch', 'l2'
        fusion_mode="con",  # 'con'
        dropout_rate=0.2,
        tranformer_dropout_rate=0.1,
        transformer_depth=8,
        multi_heads=4,
        transfomer_mlp_dim=512,
        **kwargs
    )
    return model


if __name__ == "__main__":
    import torch

    x = torch.randn(2, 8, 2048)
    # m = resnet34_cbam_global()
    # m = resnet34_cbam_delg()
    m = resnet34_cbam_local()
    print(m)
    y = m(x)
    print(y.shape)
    print("----------------")
    # print(m.transformer_model.transformer.layers[-1][0].norm)
    print(m.backbone.layer3[-1])
