import logging
import math
from dataclasses import dataclass
from pprint import pp
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import integrate, signal
from scipy.interpolate import interp1d
from torch import Tensor
from torch.nn.functional import softmax

import models
from models.CMI_ECG_segmentation_CNV2 import CBR_1D, Unet_1D
from models.resnet_cbam import ResNet

# This must be imported, otherwise the model cannot be loaded.
# noinspection PyStatementEffect
CBR_1D

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fs: Final[int] = 240


@dataclass
class Beat:
    position: int
    r_peak: int
    is_new: bool
    label: str = ""


def transform(sig: NDArray[float]) -> Tensor:
    # 前置不可或缺的步骤
    sig: NDArray[float] = signal.resample(sig, 360)

    # 后置不可或缺的步骤
    sig: NDArray[float] = sig.transpose()
    return torch.tensor(sig.copy(), dtype=torch.float)


def bsw(data: NDArray[float], band_hz: float) -> NDArray[float]:
    wn1: float = 2 * band_hz / fs  # 只截取5hz以上的数据
    b: NDArray[float]
    a: NDArray[float]
    # noinspection PyTupleAssignmentBalance
    b, a = signal.butter(1, wn1, btype="high")
    return signal.filtfilt(b, a, data)


def output_sliding_voting_v2(
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


def u_net_peak(
    data: NDArray[float],
    model: Unet_1D,
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

    pred: Tensor = model(x_tensor)
    out_pred: NDArray[int] = softmax(pred, 1).detach().cpu().numpy().argmax(axis=1)
    out_pred: NDArray[int] = np.reshape(out_pred, len(x))
    output: NDArray[int] = output_sliding_voting_v2(out_pred)

    p: NDArray[bool] = output == 0  # P波
    n: NDArray[bool] = output == 1  # QRS
    t: NDArray[bool] = output == 2  # t波
    r: NDArray[bool] = output == 3  # 其他

    return p, n, t, r


def r_detection_u_net(data: NDArray[float], n: NDArray[bool]) -> list[int]:
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


def u_net_r_peak(x: NDArray[bool]) -> list[int]:
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


def get_24h_beats(
    data: NDArray[float], u_net: Unet_1D, ori_fs: int
) -> tuple[list[int], list[int]]:
    """提取R波和心拍"""
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
        p, n, t, r = u_net_peak(data[cur_s:now_s], model=u_net)

        beat_list: list[int] = u_net_r_peak(n)
        r_list: list[int] = r_detection_u_net(data[cur_s:now_s], n)
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


def check_beats(beats: NDArray[int], r_peaks: NDArray[int]) -> tuple[int, list[Beat]]:
    beats: NDArray[int] = np.array(beats, dtype=int)
    r_peaks: NDArray[int] = np.array(r_peaks, dtype=int)
    checked_beats: list[Beat] = [
        Beat(position=beats[0], r_peak=r_peaks[0], is_new=False)
    ]
    limit: float = 2 * 1.5 * fs
    beats_diff: NDArray[int] = np.diff(beats)
    add_num: int = 0
    for index, diff in enumerate(beats_diff):
        if diff >= limit:
            start = beats[index]
            cur = start
            end = beats[index + 1]
            while (end - cur) >= limit:
                new_beat = cur + int(limit / 2)
                checked_beats.append(Beat(position=new_beat, r_peak=-1, is_new=True))
                cur = new_beat
                add_num += 1
            checked_beats.append(
                Beat(position=beats[index + 1], r_peak=r_peaks[index + 1], is_new=False)
            )
        else:
            checked_beats.append(
                Beat(position=beats[index + 1], r_peak=r_peaks[index + 1], is_new=False)
            )
    return add_num, checked_beats


def classification_beats(
    data: NDArray[float],
    beats: list[Beat],
    resnet: ResNet,
    ori_fs: int,
) -> tuple[list[Beat], dict[str, int]]:
    half_len: int = int(0.75 * fs)

    logging.info("重采样原始信号")

    data: NDArray[float] = signal.resample(data, len(data) * fs // ori_fs)
    data: NDArray[float] = bsw(data, band_hz=0.5)

    logging.info(f"重采样成功，采样后数据长度：{data.shape[0]}")

    logging.info("分类心拍")

    labels: list[str] = [
        "窦性心律",
        "房性早搏",
        "心房扑动",
        "心房颤动",
        "室性早搏",
        "阵发性室上性心动过速",
        "心室预激",
        "室扑室颤",
        "房室传导阻滞",
        "噪声",
    ]
    label_cnt: dict[str, int] = {label: 0 for label in labels}

    batch_size: int = 64
    input_tensor: list[Tensor] = []
    input_beats: list[Beat] = []

    beat: Beat
    for idx, beat in enumerate(beats):
        if beat.position < half_len or beat.position >= data.shape[0] - half_len:
            beat.label = ""
            continue

        x: NDArray[float] = data[beat.position - half_len : beat.position + half_len]
        x: NDArray[float] = np.reshape(x, (1, half_len * 2))
        x: NDArray[float] = (x - np.mean(x)) / np.std(x)
        x: NDArray[float] = x.T
        x_tensor: Tensor = transform(x).unsqueeze(0).to(device)
        input_tensor.append(x_tensor)
        input_beats.append(beat)

        if len(input_tensor) % batch_size == 0 or idx == len(beats) - 1:
            x_tensor = torch.vstack(input_tensor)
            output: Tensor = torch.softmax(resnet(x_tensor), dim=1).squeeze()

            # 修改维度
            y_pred: Tensor = torch.argmax(output, dim=1, keepdim=False)
            for i, pred in enumerate(y_pred):
                pred: Tensor
                pred: int = pred.item()
                beat = input_beats[i]
                label_cnt[labels[pred]] += 1
                beat.label = labels[pred]
            input_tensor = []
            input_beats = []

    logging.info("分类结束")

    return beats, label_cnt


def get_lf_hf(
    rr_intervals: NDArray[float], rr_interval_times: NDArray[float]
) -> tuple[int, float]:
    resampling_period: float = 0.5
    interpolated_rr_intervals = interp1d(rr_interval_times, rr_intervals, kind="cubic")
    # fft conversion
    start_time: float = interpolated_rr_intervals.x[0]
    end_time: float = interpolated_rr_intervals.x[-1]
    fixed_times: NDArray[float] = np.arange(start_time, end_time, resampling_period)
    num_samples: int = fixed_times.shape[0]
    resampled_rr_intervals = interpolated_rr_intervals(fixed_times)
    frequencies: NDArray[float] = np.fft.fftfreq(num_samples, d=resampling_period)
    non_negative_frequency_index: NDArray[bool] = frequencies >= 0

    frequencies = frequencies[non_negative_frequency_index]
    fft_converted: NDArray[np.complex128] = np.fft.fft(resampled_rr_intervals)[
        non_negative_frequency_index
    ]
    amplitudes: NDArray[float] = np.abs(fft_converted)
    powers: NDArray[float] = amplitudes**2

    minimum_frequency = 0.05
    boundary_frequency = 0.15
    maximum_frequency = 0.4

    try:
        start_index = np.where(frequencies >= minimum_frequency)[0][0]
        boundary_index = np.where(frequencies >= boundary_frequency)[0][0]
        end_index = np.where(frequencies <= maximum_frequency)[0][-1]
    except IndexError:
        return -1, -1

    # 利用积分来代替个数
    if not start_index >= boundary_index:
        lf_integrated = integrate.simps(
            powers[start_index:boundary_index], frequencies[start_index:boundary_index]
        )
    else:
        lf_integrated = -1
    if not end_index <= boundary_index:
        hf_integrated = integrate.simps(
            powers[boundary_index : end_index + 1],
            frequencies[boundary_index : end_index + 1],
        )
    else:
        hf_integrated = -1
    return lf_integrated, hf_integrated


def sample_to_time(position: int) -> tuple[int, int, int]:
    total_seconds: float = position / fs
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int((total_seconds % 3600 % 60))
    return h, m, s


def analyze_beats(my_beats, output_path):
    """统计带有标签的 beats"""
    n_diff: list[NDArray[int]] = []
    n_time: list[NDArray[float]] = []
    n_diff_flatten_with_r_peak: list[list[int]] = []
    n_flag: bool = False
    n_continuous_beats: list[int] = []
    n_stop_beats: list[list[int]] = []
    n_num: int = 0
    rr: list[int] = []
    af_diff: list[NDArray[int]] = []  # 房扑房颤
    af_flag: bool = False
    af_continuous_beats: list[int] = []
    vf: list[list[int]] = []

    qrs_num: int = 0
    apb: list[int] = []
    apb_single: list[int] = []  # 单发房早-次数
    apb_double: list[int] = []  # 成对房早-次数
    apb_double_rhythm: list[int] = []  # 房早二联律-次数、持续时间
    apb_triple_rhythm: list[int] = []  # 房早三联律-次数、持续时间
    apb_short_array: list[list[int]] = []  # 短阵房早-次数、持续时间
    vpb: list[int] = []
    vpb_single: list[int] = []  # 单发室早
    vpb_double: list[int] = []  # 成对室早
    vpb_double_rhythm: list[int] = []  # 室早二联律
    vpb_triple_rhythm: list[int] = []  # 室早三联律
    vpb_short_array: list[list[int]] = []  # 短阵室早
    iteration_num: int = 0
    len_my_beats: int = len(my_beats)

    for index, my_beat in enumerate(my_beats):
        if my_beat.label == "":
            continue
        rr.append(my_beat.r_peak if not my_beat.r_peak == -1 else my_beat.position)
        if not my_beat.is_new:
            qrs_num += 1
        if my_beat.label == "房性早搏":  # 单发、成对、二联律、三联律、短阵
            apb.append(my_beat.position)
            if iteration_num > 0:
                iteration_num -= 1
                continue
            if index + 1 < len_my_beats and my_beats[index + 1].label == "房性早搏":
                apb_probe = 2
                apb_short_array_list = [my_beat.position, my_beats[index + 1].position]
                while True:
                    if (
                        index + apb_probe < len_my_beats
                        and my_beats[index + apb_probe].label == "房性早搏"
                    ):
                        apb_short_array_list.append(
                            my_beats[index + apb_probe].position
                        )  # 出现短阵房早
                        apb_probe += 1
                    else:
                        break
                if apb_probe == 2:
                    apb_double.append(my_beat.position)
                    iteration_num = 1
                else:
                    apb_short_array.append(apb_short_array_list)
                    iteration_num = apb_probe - 1
            else:
                if index + 2 < len_my_beats and my_beats[index + 2].label == "房性早搏":
                    apb_double_rhythm.append(my_beat.position)  # 出现房早二联律
                    iteration_num = 2
                else:
                    if index + 3 < len_my_beats and my_beats[index + 3].label == "房性早搏":
                        if (
                            index + 6 < len_my_beats
                            and my_beats[index + 6].label == "房性早搏"
                        ):
                            apb_triple_rhythm.append(my_beat.position)  # 出现房早三联律
                            iteration_num = 6
                    else:
                        apb_single.append(my_beat.position)  # 单发房早
        if my_beat.label == "室性早搏":  # 单发、成对、二联律、三联律、短阵
            vpb.append(my_beat.position)
            if iteration_num > 0:
                iteration_num -= 1
                continue
            if index + 1 < len_my_beats and my_beats[index + 1].label == "室性早搏":
                vpb_probe = 2
                vpb_short_array_list = [my_beat.position, my_beats[index + 1].position]
                while True:
                    if (
                        index + vpb_probe < len_my_beats
                        and my_beats[index + vpb_probe].label == "室性早搏"
                    ):
                        vpb_short_array_list.append(
                            my_beats[index + vpb_probe].position
                        )  # 出现短阵室早
                        vpb_probe += 1
                    else:
                        break
                if vpb_probe == 2:
                    vpb_double.append(my_beat.position)  # 出现双发室早
                    iteration_num = 1
                else:
                    vpb_short_array.append(vpb_short_array_list)
                    iteration_num = vpb_probe - 1
            else:
                if index + 2 < len_my_beats and my_beats[index + 2].label == "室性早搏":
                    vpb_double_rhythm.append(my_beat.position)  # 出现室早二联律
                    iteration_num = 2
                else:
                    if index + 3 < len_my_beats and my_beats[index + 3].label == "室性早搏":
                        if (
                            index + 6 < len_my_beats
                            and my_beats[index + 6].label == "室性早搏"
                        ):
                            vpb_triple_rhythm.append(my_beat.position)  # 出现室早三联律
                            iteration_num = 6
                    else:
                        vpb_single.append(my_beat.position)  # 单发室早
        if my_beat.label == "窦性心律" and not my_beat.is_new and index < len_my_beats - 1:
            n_num += 1
            if not n_flag:
                if index + 1 < len_my_beats and my_beats[index + 1].label == "窦性心律":
                    n_continuous_beats.append(my_beat.r_peak)
                    n_flag = True
            else:
                n_continuous_beats.append(my_beat.r_peak)
        else:
            if index == len_my_beats - 1:
                if my_beat.label == "窦性心律" and not my_beat.is_new:
                    n_num += 1
                    n_continuous_beats.append(my_beat.r_peak)
            if n_flag:
                n_time.append(np.array(n_continuous_beats) / fs)
                n_continuous_diff = np.diff(np.array(n_continuous_beats))
                for i, diff in enumerate(n_continuous_diff):
                    n_diff_flatten_with_r_peak.append([n_continuous_beats[i + 1], diff])
                    if diff > 2 * fs:
                        n_stop_beats.append([n_continuous_beats[i + 1], diff])
                n_diff.append(n_continuous_diff)
                n_continuous_beats.clear()
                n_flag = False

        if my_beat.label == "心房扑动" or my_beat.label == "心房颤动":
            if not af_flag:
                if (
                    index + 1 < len_my_beats
                    and my_beats[index + 1].label == "心房扑动"
                    or "心房颤动"
                    and index < len_my_beats - 1
                ):
                    af_continuous_beats.append(
                        my_beat.r_peak if not my_beat.r_peak == -1 else my_beat.position
                    )
                    af_flag = True
            else:
                af_continuous_beats.append(
                    my_beat.r_peak if not my_beat.r_peak == -1 else my_beat.position
                )
        else:
            if index == len_my_beats - 1:
                if my_beat.label == "心房扑动" or my_beat.label == "心房颤动":
                    af_continuous_beats.append(
                        my_beats.r_peak
                        if not my_beat.r_peak == -1
                        else my_beat.position
                    )
            if af_flag:
                af_continuous_diff = np.diff(np.array(af_continuous_beats))
                af_diff.append(af_continuous_diff)
                af_continuous_beats.clear()
                af_flag = False
        # 室扑室颤噪声检查
        if my_beat.label == "室扑室颤":
            if iteration_num > 0:
                iteration_num -= 1
                continue
            vf_probe = index
            while (
                vf_probe + 1 < len_my_beats and my_beats[vf_probe + 1].label == "室扑室颤"
            ):
                vf_probe += 1
            vf_time = int((my_beats[vf_probe].position - my_beat.position) / fs)
            if vf_time <= 2:
                continue
            vf.append([my_beat.position, vf_time])
            iteration_num = vf_probe

    len_h = ((my_beats[-1].position / fs) + 0.75) / (60 * 60)

    with open(output_path, "w", encoding="utf-8") as f_out:
        # 计算窦性心室率
        n_max_diff = 0
        n_min_diff = 10000
        n_diff_sum = 0
        n_diff_num = 0
        for diffs in n_diff:
            for diff in diffs:
                if n_max_diff < diff < 2 * fs:
                    n_max_diff = diff
                if 0.3 * fs < diff < n_min_diff:
                    n_min_diff = diff
                n_diff_sum += diff
                n_diff_num += 1
        n_ventricular_mean_rate = int(60 / (n_diff_sum / n_diff_num / fs))
        n_ventricular_max_rate = int(60 / (n_min_diff / fs))
        n_ventricular_min_rate = int(60 / (n_max_diff / fs))
        f_out.write("------数据------\n")
        display_number = 1
        if 100 >= n_ventricular_mean_rate >= 60:
            n_state = "窦性心律"
        elif n_ventricular_mean_rate < 60:
            n_state = "窦性心动过缓"
        elif n_ventricular_mean_rate > 100:
            n_state = "窦性心动过速"
        f_out.write(
            "{}、{}:平均心室率：{}，最快心室率：{}，最慢心室率：{}\n".format(
                display_number,
                n_state,
                n_ventricular_mean_rate,
                n_ventricular_max_rate,
                n_ventricular_min_rate,
            )
        )

        if not len(n_stop_beats) == 0:
            n_stop_max = 0
            n_stop_index = 0
            for index, beats in enumerate(n_stop_beats):
                if beats[1] > n_stop_max:
                    n_stop_max = beats[1]
                    n_stop_index = beats[0]
            n_stop_max_seconds = n_stop_max / fs
            n_stop_max_h, n_stop_max_m, n_stop_max_s = sample_to_time(n_stop_index)
            f_out.write(
                "    发生了{}次窦性停搏,最长的一次为：{:.1f}s，发生于:{}h-{}m-{}s\n".format(
                    len(n_stop_beats),
                    n_stop_max_seconds,
                    n_stop_max_h,
                    n_stop_max_m,
                    n_stop_max_s,
                )
            )
        # 计算房扑房颤心室率
        if not len(af_diff) == 0:
            display_number += 1
            af_max_diff = 0
            af_min_diff = 10000
            af_diff_sum = 0
            af_diff_num = 0
            for diffs in af_diff:
                for diff in diffs:
                    if af_max_diff < diff < 1.5 * fs:  # 最慢不能低于100心率
                        af_max_diff = diff
                    if 0.2 * fs < diff < af_min_diff:  # 最快不能高于300心率
                        af_min_diff = diff
                    af_diff_sum += diff
                    af_diff_num += 1
            af_ventricular_mean_rate = int(60 / (af_diff_sum / af_diff_num / fs))
            af_ventricular_max_rate = int(60 / (af_min_diff / fs))
            af_ventricular_min_rate = int(60 / (af_max_diff / fs))
            f_out.write(
                "{}、房扑房颤的平均心室率：{}次/分，最慢心室率：{}次/分，最快心室率：{}次/分\n".format(
                    display_number,
                    af_ventricular_mean_rate,
                    af_ventricular_min_rate,
                    af_ventricular_max_rate,
                )
            )
        if not len(apb) == 0:
            display_number += 1
            f_out.write(
                "{}、房性早搏{}次/24h,成对房早{}次/24h,房早二联律{}次/24h,房早三联律{}次/24h,短阵房速{}阵/24h\n".format(
                    display_number,
                    int(len(apb) * 24 / len_h),
                    int(len(apb_double) * 24 / len_h),
                    int(len(apb_double_rhythm) * 24 / len_h),
                    int(len(apb_triple_rhythm) * 24 / len_h),
                    math.ceil((len(apb_short_array) * 24 / len_h)),
                )
            )
            # 短阵房早
            if not len(apb_short_array) == 0:
                apb_short_array_diff = []
                apb_short_array_min_diff = 10000
                apb_short_array_min_index = 0
                for apb_s_a in apb_short_array:
                    apb_short_array_diff.append(np.diff(apb_s_a))
                for index, diffs in enumerate(apb_short_array_diff):
                    for diff in diffs:
                        if 0.2 * fs < diff < apb_short_array_min_diff:
                            apb_short_array_min_index = index
                            apb_short_array_min_diff = diff
                apb_short_array_ventricular_max_rate = 60 / (
                    apb_short_array_min_diff / fs
                )
                apb_shor_array_h, apb_shor_array_m, apb_shor_array_s = sample_to_time(
                    apb_short_array[apb_short_array_min_index][0]
                )
                f_out.write(
                    "其中，短阵房速最快心室率为：{},由{}个QRS波组成,发生于：{}(采样点)/{}h-{}m-{}s(时间)\n".format(
                        int(apb_short_array_ventricular_max_rate),
                        len(apb_short_array[apb_short_array_min_index]),
                        int(apb_short_array[apb_short_array_min_index][0] * 250 / fs),
                        apb_shor_array_h,
                        apb_shor_array_m,
                        apb_shor_array_s,
                    )
                )
        if not len(vpb) == 0:
            display_number += 1
            f_out.write(
                "{}、室性早搏{}次/24h,成对室早{}次/24h,室早二联律{}次/24h,室早三联律{}次/24h,短阵室速{}阵/24h\n".format(
                    display_number,
                    int(len(vpb) * 24 / len_h),
                    int(len(vpb_double) * 24 / len_h),
                    int(len(vpb_double_rhythm) * 24 / len_h),
                    int(len(vpb_triple_rhythm) * 24 / len_h),
                    math.ceil((len(vpb_short_array) * 24 / len_h)),
                )
            )
            # 短阵室早
            if not len(vpb_short_array) == 0:
                vpb_short_array_diff = []
                vpb_short_array_min_diff = 10000
                vpb_short_array_min_index = 0
                for VPB_s_a in vpb_short_array:
                    vpb_short_array_diff.append(np.diff(VPB_s_a))
                for index, diffs in enumerate(vpb_short_array_diff):
                    for diff in diffs:
                        if diff < vpb_short_array_min_diff:
                            vpb_short_array_min_index = index
                            vpb_short_array_min_diff = diff
                vpb_short_array_ventricular_max_rate = 60 / (
                    vpb_short_array_min_diff / fs
                )
                vpb_shor_array_h, vpb_shor_array_m, vpb_shor_array_s = sample_to_time(
                    vpb_short_array[vpb_short_array_min_index][0]
                )
                f_out.write(
                    "其中，短阵室速最快心室率为：{},由{}个QRS波组成,发生于：{}(采样点)/{}h-{}m-{}s(时间)\n".format(
                        int(vpb_short_array_ventricular_max_rate),
                        len(vpb_short_array[vpb_short_array_min_index]),
                        int(vpb_short_array[vpb_short_array_min_index][0] * 250 / fs),
                        vpb_shor_array_h,
                        vpb_shor_array_m,
                        vpb_shor_array_s,
                    )
                )
        # 室扑室颤
        if not len(vf) == 0:
            display_number += 1
            f_out.write("{}、出现室扑室颤，如下：\n".format(display_number))
            for v in vf:
                vf_h, vf_m, vf_s = sample_to_time(v[0])
                f_out.write(
                    "在{}h-{}m-{}s发生室扑室颤，持续时长{}s\n".format(vf_h, vf_m, vf_s, vf[1])
                )
        # 计算HRV

        # lf-hf
        lf = 0
        hf = 0
        for index in range(len(n_diff)):
            rr_interval = np.array(n_diff[index])
            rr_interval = rr_interval / fs
            rr_interval_times = np.array(n_time[index])[:-1]
            if len(rr_interval) < 4:
                continue
            lf_, hf_ = get_lf_hf(rr_interval, rr_interval_times)
            if lf_ == -1 or hf_ == -1:
                continue
            else:
                lf += lf_
                hf += hf_

        n_diff_r_peak = [i[0] for i in n_diff_flatten_with_r_peak]
        n_diff_flatten = [i[1] for i in n_diff_flatten_with_r_peak]

        # RMSSD
        n_diff_diff = []
        for diffs in n_diff:
            diff_diffs = np.diff(diffs)
            for diff_diff in diff_diffs:
                n_diff_diff.append(diff_diff)

        # SDANN
        start_r_peak = n_diff_r_peak[0]
        end_r_peak = n_diff_r_peak[-1]
        nn_5minute_mean = []
        nn_5minute_sum = 0
        nn_5minute_num = 0

        for index, r_peak in enumerate(n_diff_r_peak):
            if (r_peak - start_r_peak) <= 300 * fs and not r_peak == end_r_peak:
                nn_5minute_sum += n_diff_flatten[index]
                nn_5minute_num += 1
            else:
                if r_peak == end_r_peak:
                    if (r_peak - start_r_peak) <= 300 * fs:
                        nn_5minute_num += 1
                        nn_5minute_sum += n_diff_flatten[index]
                if not nn_5minute_num == 0:
                    nn_5minute_mean.append(nn_5minute_sum / nn_5minute_num)
                nn_5minute_sum = n_diff_flatten[index]
                nn_5minute_num = 1
                start_r_peak = r_peak
        # PNN50
        n_pnn50_num = 0
        for diff_diff in n_diff_diff:
            if diff_diff >= 0.05 * fs:
                n_pnn50_num += 1

        sdnn = np.std(n_diff_flatten)
        sdann = np.std(nn_5minute_mean)
        rmssd = math.sqrt(sum([x**2 for x in n_diff_diff]) / len(n_diff_diff))
        pnn50 = n_pnn50_num / n_num
        display_number += 1
        f_out.write("{}、心率变异性指标：\n".format(display_number))
        f_out.write("    SDNN:{:.2f}ms\n".format(sdnn))
        f_out.write("    SDANN:{:.2f}ms\n".format(sdann))
        f_out.write("    RMSSD:{:.2f}ms\n".format(rmssd))
        f_out.write("    PNN50:{:.2f}%\n".format(pnn50 * 100))
        f_out.write("    lf:{}".format(int(lf)))
        f_out.write("    lf/hf:{}".format(lf / hf))


def get_r_peaks(data, ori_fs) -> tuple[list[int], list[int]]:
    """提取R波切分心拍"""
    u_net = torch.load("../assets/240HZ_t+c_v2_best.pt", map_location=device)
    u_net.eval()

    return get_24h_beats(
        data,
        u_net=u_net,
        ori_fs=ori_fs,
    )


def get_checked_beats(beats, r_peaks):
    """补充心拍"""
    assert len(beats) == len(r_peaks), "提取出的心拍数量与 R 波数量不同"

    add_num, checked_beats = check_beats(beats, r_peaks)
    logging.info(f"补充了{add_num}个心拍")
    return checked_beats


def get_labels(data, checked_beats, ori_fs):
    """进行预测，获取标签"""
    resnet = getattr(models, "resnet34_cbam_ch1")(num_classes=10)
    resnet.load_state_dict(
        torch.load(
            "../assets/best_w.pth",
            map_location="cpu",
        )["state_dict"]
    )
    resnet = resnet.to(device)
    resnet.eval()
    return classification_beats(
        data,
        checked_beats,
        resnet=resnet,
        ori_fs=ori_fs,
    )


def infer(data, ori_fs):
    beats: list[int]
    r_peaks: list[int]
    beats, r_peaks = get_r_peaks(data, ori_fs)

    checked_beats: list[Beat] = get_checked_beats(beats, r_peaks)

    labelled_beats: list[Beat]
    label_cnt: dict[str, int]
    labelled_beats, label_cnt = get_labels(data, checked_beats, ori_fs)

    return label_cnt, labelled_beats


def main():
    with torch.no_grad():
        label_cnt, labelled_beats = infer(np.loadtxt("../assets/input.txt"), 250)

    analyze_beats(labelled_beats, output_path="../assets/output/report.txt")

    with open("../assets/output/labelled_beats.txt", "w", encoding="utf-8") as f:
        print(*labelled_beats, sep="\n", file=f)
    with open("../assets/output/label_cnt.txt", "w", encoding="utf-8") as f:
        pp(label_cnt, stream=f)


if __name__ == "__main__":
    main()
