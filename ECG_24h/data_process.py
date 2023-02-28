import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

# 保证每次划分数据一致
np.random.seed(41)


def BSW(data, band_hz=0.5, fs=240):
    wn1 = 2 * band_hz / fs  # 只截取5hz以上的数据
    b, a = signal.butter(1, wn1, btype="high")
    filteddata = signal.filtfilt(b, a, data)
    return filteddata


def resample(sig, target_point_num=None):
    """
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    """
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def output_sliding_voting_v2(ori_output, window=5, type_num=4):
    output = np.array(ori_output)
    n = len(output)
    half_window = int(window / 2)
    cnt = np.zeros((type_num,), dtype=np.int32)
    l_index = 0
    r_index = -1
    for i in range(n):
        if r_index - l_index + 1 == window and half_window < i < n - half_window:
            cnt[ori_output[l_index]] -= 1
            l_index += 1
        while r_index - l_index + 1 < window and r_index + 1 < n:
            r_index += 1
            cnt[ori_output[r_index]] += 1
        output[i] = np.argmax(cnt)
    return output


def U_net_peak(
    data,
    input_fs,
    band_Hz=0.5,
    del_drift=True,
    target_fs=240,
    model=None,
    gpu=False,
    device="cpu",
):
    # 提取U-net波群信息

    x = data.copy()
    if not input_fs == 240:
        if input_fs < target_fs:
            print("！ERROR：目标采样率大于原始采样率，无法重采样")
            return
        x = resample(x, len(x) * target_fs // input_fs)
    lenx = len(x)
    if del_drift:
        wn1 = 2 * band_Hz / target_fs
        b, a = signal.butter(1, wn1, btype="high")
        x = signal.filtfilt(b, a, x)
    # 标准化
    x = (x - np.mean(x)) / np.std(x)
    x = torch.tensor(x)
    x = torch.unsqueeze(x, 0)
    x = torch.unsqueeze(x, 0)
    x = x.to(device)

    pred = model(x)
    out_pred = F.softmax(pred, 1).detach().cpu().numpy().argmax(axis=1)
    out_pred = np.reshape(out_pred, lenx)
    output = output_sliding_voting_v2(out_pred, 9)

    p = output == 0  # P波
    N = output == 1  # QRS
    t = output == 2  # t波
    r = output == 3  # 其他

    return p, N, t, r


def R_Detection_U_net(data, N):
    # 获取R波波峰
    x = data.copy()
    lenx = len(x)
    N_ = np.array(N)
    N_ = np.insert(N_, lenx, False)
    N_ = np.insert(N_, 0, False)
    R_start = []
    R_end = []
    R = []
    for i in range(lenx):
        idx_ = i + 1
        if N_[idx_] == 1 and (N_[idx_ - 1] == 1 or N_[idx_ + 1] == 1):
            if N_[idx_ - 1] == 0:
                R_start.append(i)
            elif N_[idx_ + 1] == 0:
                R_end.append(i)
    if not len(R_start) == len(R_end):
        print("error，R波起点和终点数目不同")
        return

    for i in range(lenx):
        x[i] = (
            x[max(i - 2, 0)]
            + x[max(i - 1, 0)]
            + x[i]
            + x[min(i + 1, lenx - 1)]
            + x[min(i + 2, lenx - 1)]
        ) / 5
    for i in range(len(R_start)):
        R_candidate = []
        peak_candate = []

        for idx in range(R_start[i], R_end[i]):
            if idx <= 0 or idx >= lenx - 1:
                continue

            if x[idx] >= x[idx - 1] and x[idx] >= x[idx + 1]:
                R_candidate.append(idx)
                peak_candate.append(x[idx])
        if len(R_candidate) == 0:
            R.append(R_start[i] + np.argmax(x[R_start[i] : R_end[i]]))
        else:
            R.append(R_candidate[np.argmax(peak_candate)])
    return R


def U_net_RPEAK(x):
    # 获取心拍

    lenx = len(x)
    x_ = np.array(x)
    x_ = np.insert(x_, lenx, False)
    x_ = np.insert(x_, 0, False)

    y = np.zeros_like(x)
    flag = 0
    for i in range(lenx):
        idx_ = i + 1
        if x_[idx_] == 1 and (x_[idx_ - 1] == 1 or x_[idx_ + 1] == 1):
            if x_[idx_ - 1] == 0 or x_[idx_ + 1] == 0:
                y[i] = 1
            else:
                y[i] = 0

    start = 0
    end = 0
    flag = 0
    r_list = []
    for i in range(lenx):
        if y[i] == 1 and flag == 0:
            flag = 1
            start = i
        elif y[i] == 1 and flag == 1:
            flag = 0
            end = i

            r_list.append(start + math.floor((end - start) / 2))
    return r_list


def name2index(path):
    """
    把类别名称转换为index索引
    :param path: 文件路径
    :return: 字典
    """
    list_name = []
    for line in open(path, encoding="utf-8"):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx
