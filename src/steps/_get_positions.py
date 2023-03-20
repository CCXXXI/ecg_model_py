import math
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import signal
from torch import Tensor
from torch.nn.functional import softmax

from utils import fs, load_model


def _u_net_peak(data: NDArray[float]) -> NDArray[bool]:
    """QRS 提取"""
    # 提取U-net波群信息
    x: NDArray[float] = data.copy()
    wn1 = 1 / fs
    b: NDArray[float]
    a: NDArray[float]
    # noinspection PyTupleAssignmentBalance
    b, a = signal.butter(1, wn1, btype="high")
    x = signal.filtfilt(b, a, x)
    # 标准化
    x = (x - np.mean(x)) / np.std(x)
    x_tensor: Tensor = torch.tensor(x)
    x_tensor = torch.unsqueeze(x_tensor, 0)
    x_tensor = torch.unsqueeze(x_tensor, 0)

    model = load_model("u_net.pt")
    pred: Tensor = model(x_tensor)
    out_pred: NDArray[int] = softmax(pred, 1).detach().cpu().numpy().argmax(axis=1)
    out_pred = np.reshape(out_pred, len(x))
    output = _output_sliding_voting_v2(out_pred)

    qrs: NDArray[bool] = output == 1

    return qrs


def _u_net_r_peak(qrs: NDArray[bool]) -> list[int]:
    """获取心拍"""

    qrs: NDArray[bool] = np.array(qrs)
    qrs = np.insert(qrs, len(qrs), False)
    qrs = np.insert(qrs, 0, False)

    y: NDArray[bool] = np.zeros_like(qrs)
    for i in range(len(qrs)):
        idx_ = i + 1
        if qrs[idx_] == 1 and (qrs[idx_ - 1] == 1 or qrs[idx_ + 1] == 1):
            if qrs[idx_ - 1] == 0 or qrs[idx_ + 1] == 0:
                y[i] = 1
            else:
                y[i] = 0

    start = 0
    flag = False
    r_list: list[int] = []
    for i in range(len(qrs)):
        if y[i] == 1 and not flag:
            flag = True
            start = i
        elif y[i] == 1 and flag:
            flag = False
            end = i

            r_list.append(start + math.floor((end - start) / 2))
    return r_list


def _output_sliding_voting_v2(
    ori_output: NDArray[int],
) -> NDArray[int]:
    window: Final[int] = 9

    output: NDArray[int] = np.array(ori_output)
    n = len(output)
    half_window = int(window / 2)
    cnt: NDArray[int] = np.zeros((4,), dtype=int)
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


def get_positions(data: NDArray[float], ori_fs: int) -> list[int]:
    """切分心拍"""
    data: NDArray[float] = signal.resample(data, len(data) * fs // ori_fs)
    len_u_net = 10 * 60 * fs

    len_data: int = data.shape[0]
    beats: list[int] = []
    cur_s = 0
    while cur_s < len_data:
        if cur_s + len_u_net <= len_data:
            now_s: int = cur_s + len_u_net
        else:
            break
        qrs: NDArray[bool]
        qrs = _u_net_peak(data[cur_s:now_s])

        r_list: list[int] = _u_net_r_peak(qrs)
        # 记录QRS波中点，以该点标识心拍     之后两边扩展
        r_list: NDArray[int] = np.array(r_list)

        append_start = int(0.5 * 60 * fs)
        append_end = int(9.5 * 60 * fs)
        if cur_s == 0:
            append_start = 0

        for beat in r_list:
            if append_start < beat <= append_end:
                beats.append(beat + cur_s)

        cur_s += 9 * 60 * fs

    return beats
