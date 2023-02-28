import torch
from scipy import signal

import config


def resample(sig, target_point_num=None):
    """
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    """
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def transform(sig, train=False):
    # 前置不可或缺的步骤
    sig = resample(sig, config.target_point_num)

    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig
