import logging
import math
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import signal
from torch import Tensor
from torch.nn.functional import softmax

import models
from config import device, fs


def _u_net_peak(
    data: NDArray[float],
) -> tuple[NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool]]:
    # 提取U-net波群信息
    x: NDArray[float] = data.copy()
    wn1: float = 1 / fs
    b: NDArray[float]
    a: NDArray[float]
    # noinspection PyTupleAssignmentBalance
    b, a = signal.butter(1, wn1, btype="high")
    x: NDArray[float] = signal.filtfilt(b, a, x)
    # 标准化
    x: NDArray[float] = (x - np.mean(x)) / np.std(x)
    x_tensor: Tensor = torch.tensor(x)
    x_tensor: Tensor = torch.unsqueeze(x_tensor, 0)
    x_tensor: Tensor = torch.unsqueeze(x_tensor, 0)
    x_tensor: Tensor = x_tensor.to(device)

    pred: Tensor = models.u_net(x_tensor)
    out_pred: NDArray[int] = softmax(pred, 1).detach().cpu().numpy().argmax(axis=1)
    out_pred: NDArray[int] = np.reshape(out_pred, len(x))
    output: NDArray[int] = _output_sliding_voting_v2(out_pred)

    p: NDArray[bool] = output == 0  # P波
    n: NDArray[bool] = output == 1  # QRS
    t: NDArray[bool] = output == 2  # t波
    r: NDArray[bool] = output == 3  # 其他

    return p, n, t, r


def _u_net_r_peak(x: NDArray[bool]) -> list[int]:
    """获取心拍"""

    x_: NDArray[bool] = np.array(x)
    x_: NDArray[bool] = np.insert(x_, len(x), False)
    x_: NDArray[bool] = np.insert(x_, 0, False)

    y: NDArray[bool] = np.zeros_like(x)
    for i in range(len(x)):
        idx_: int = i + 1
        if x_[idx_] == 1 and (x_[idx_ - 1] == 1 or x_[idx_ + 1] == 1):
            if x_[idx_ - 1] == 0 or x_[idx_ + 1] == 0:
                y[i] = 1
            else:
                y[i] = 0

    start: int = 0
    flag: int = 0
    r_list: list[int] = []
    for i in range(len(x)):
        if y[i] == 1 and flag == 0:
            flag = 1
            start = i
        elif y[i] == 1 and flag == 1:
            flag = 0
            end: int = i

            r_list.append(start + math.floor((end - start) / 2))
    return r_list


def _r_detection_u_net(data: NDArray[float], n: NDArray[bool]) -> list[int]:
    # 获取R波波峰
    x: NDArray[float] = data.copy()
    n_: NDArray[bool] = np.array(n)
    n_: NDArray[bool] = np.insert(n_, len(x), False)
    n_: NDArray[bool] = np.insert(n_, 0, False)
    r_start: list[int] = []
    r_end: list[int] = []
    r: list[int] = []
    for i in range(len(x)):
        idx_: int = i + 1
        if n_[idx_] == 1 and (n_[idx_ - 1] == 1 or n_[idx_ + 1] == 1):
            if n_[idx_ - 1] == 0:
                r_start.append(i)
            elif n_[idx_ + 1] == 0:
                r_end.append(i)

    assert len(r_start) == len(r_end), "R 波起点和终点数目不同"

    for i in range(len(x)):
        x[i] = (
            x[max(i - 2, 0)]
            + x[max(i - 1, 0)]
            + x[i]
            + x[min(i + 1, len(x) - 1)]
            + x[min(i + 2, len(x) - 1)]
        ) / 5
    for i in range(len(r_start)):
        r_candidate: list[int] = []
        peak_candidate: list[float] = []

        for idx in range(r_start[i], r_end[i]):
            if idx <= 0 or idx >= len(x) - 1:
                continue

            if x[idx] >= x[idx - 1] and x[idx] >= x[idx + 1]:
                r_candidate.append(idx)
                peak_candidate.append(x[idx])
        if len(r_candidate) == 0:
            r.append(r_start[i] + np.argmax(x[r_start[i] : r_end[i]]))
        else:
            r.append(r_candidate[np.argmax(peak_candidate)])
    return r


def _output_sliding_voting_v2(
    ori_output: NDArray[int],
) -> NDArray[int]:
    window: Final[int] = 9

    output: NDArray[int] = np.array(ori_output)
    n: int = len(output)
    half_window: int = int(window / 2)
    cnt: NDArray[int] = np.zeros((4,), dtype=int)
    l_index: int = 0
    r_index: int = -1
    for i in range(n):
        if r_index - l_index + 1 == window and half_window < i < n - half_window:
            cnt[ori_output[l_index]] -= 1
            l_index += 1
        while r_index - l_index + 1 < window and r_index + 1 < n:
            r_index += 1
            cnt[ori_output[r_index]] += 1
        output[i] = np.argmax(cnt)
    return output


def get_r_peaks(data: NDArray[float], ori_fs: int) -> tuple[list[int], list[int]]:
    """提取R波切分心拍"""
    logging.info("重采样原始信号")
    data: NDArray[float] = signal.resample(data, len(data) * fs // ori_fs)
    len_u_net: int = 10 * 60 * fs
    logging.info(f"重采样成功，采样后数据长度：{data.shape[0]}")

    logging.info("提取波群信息")
    len_data: int = data.shape[0]
    beats: list[int] = []
    r_peaks: list[int] = []
    cur_s: int = 0
    while cur_s < len_data:
        if cur_s + len_u_net <= len_data:
            now_s: int = cur_s + len_u_net
        else:
            break
        p: NDArray[bool]
        n: NDArray[bool]
        t: NDArray[bool]
        r: NDArray[bool]
        p, n, t, r = _u_net_peak(data[cur_s:now_s])

        beat_list: list[int] = _u_net_r_peak(n)
        r_list: list[int] = _r_detection_u_net(data[cur_s:now_s], n)
        # 记录QRS波中点，以该点标识心拍     之后两边扩展
        beat_list: NDArray[int] = np.array(beat_list)
        r_list: NDArray[int] = np.array(r_list)

        append_start: int = int(0.5 * 60 * fs)
        append_end: int = int(9.5 * 60 * fs)
        if cur_s == 0:
            append_start = 0

        for beat in beat_list:
            if append_start < beat <= append_end:
                beats.append(beat + cur_s)
        for r in r_list:
            if append_start < r <= append_end:
                r_peaks.append(r + cur_s)

        cur_s += 9 * 60 * fs
    logging.info(f"提取成功，提取出{len(beats)}个心拍")
    return beats, r_peaks
