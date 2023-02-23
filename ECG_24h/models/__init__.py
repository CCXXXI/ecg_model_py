# -*- coding: utf-8 -*-
"""
@time: 2019/9/8 20:13

@ author: javis
"""

from .ECGNet import ECGNet
from .MLGCN import MLGCN_resnet34_cbam
from .ResNeXt import ResNeXt29_conv3_cd8, ResNeXt32_conv7_cd8
from .SE_ECGNet import SE_ECGNet
from .SE_ECGNet import SE_ECGNet
from .Supcon import Supcon_resnet34_cbam
from .cbam import CBAM
from .rescbam_delg import (
    resnet34_cbam_global,
    resnet34_cbam_global_nw,
    resnet34_cbam_local,
    resnet34_cbam_local_2,
    resnet34_cbam_local_3,
    resnet34_cbam_local_4,
    resnet34_cbam_local_5,
    resnet34_cbam_local_6,
    resnet34_cbam_local_7,
    resnet34_cbam_delg,
    resnet34_cbam_delg_2,
    resnet34_cbam_delg_3,
    resnet34_cbam_delg_4,
    resnet34_cbam_delg_5,
    resnet34_cbam_delg_6,
)
from .resnet import resnet18, resnet34, resnet50, resnet101
from .resnet_cbam import (
    resnet18_cbam,
    resnet34_cbam,
    resnet50_cbam,
    resnet34_cbam_drp3,
    resnet34_cbam_ch9,
    resnet34_cbam_ch1,
)
from .resnet_cbam_sma import resnet34_cbam_sma5
from .resnet_rv1 import resnet34_rv1
