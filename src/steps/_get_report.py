import math

import numpy as np
from numpy.typing import NDArray
from scipy import integrate
from scipy.interpolate import interp1d
from utils import Beat
from utils import fs
from utils import Label


def _sample_to_time(position: int) -> tuple[int, int, int]:
    total_seconds = position // fs
    h = total_seconds // 3600
    m = total_seconds % 3600 // 60
    s = total_seconds % 60
    return h, m, s


def _get_lf_hf(rr_intervals: NDArray[float],
               rr_interval_times: NDArray[float]) -> tuple[int, float]:
    resampling_period = 0.5
    interpolated_rr_intervals = interp1d(rr_interval_times,
                                         rr_intervals,
                                         kind="cubic")
    # fft conversion
    start_time: float = interpolated_rr_intervals.x[0]
    end_time: float = interpolated_rr_intervals.x[-1]
    fixed_times: NDArray[float] = np.arange(start_time, end_time,
                                            resampling_period)
    num_samples: int = fixed_times.shape[0]
    resampled_rr_intervals = interpolated_rr_intervals(fixed_times)
    frequencies: NDArray[float] = np.fft.fftfreq(num_samples,
                                                 d=resampling_period)
    non_negative_frequency_index: NDArray[bool] = frequencies >= 0

    frequencies = frequencies[non_negative_frequency_index]
    fft_converted: NDArray[np.complex128] = np.fft.fft(
        resampled_rr_intervals)[non_negative_frequency_index]
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
            powers[start_index:boundary_index],
            frequencies[start_index:boundary_index])
    else:
        lf_integrated = -1
    if not end_index <= boundary_index:
        hf_integrated = integrate.simps(
            powers[boundary_index:end_index + 1],
            frequencies[boundary_index:end_index + 1],
        )
    else:
        hf_integrated = -1
    return lf_integrated, hf_integrated


def get_report(beats: list[Beat]) -> str:
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
    len_my_beats: int = len(beats)

    for index, beat in enumerate(beats):
        if beat.label is Label.未知:
            continue
        rr.append(beat.r_peak if beat.r_peak != -1 else beat.position)
        if not beat.is_new:
            qrs_num += 1
        if beat.label is Label.房性早搏:  # 单发、成对、二联律、三联律、短阵
            apb.append(beat.position)
            if iteration_num > 0:
                iteration_num -= 1
                continue
            if index + 1 < len_my_beats and beats[index +
                                                  1].label is Label.房性早搏:
                apb_probe = 2
                apb_short_array_list = [
                    beat.position, beats[index + 1].position
                ]
                while True:
                    if (index + apb_probe < len_my_beats
                            and beats[index + apb_probe].label is Label.房性早搏):
                        apb_short_array_list.append(
                            beats[index + apb_probe].position)  # 出现短阵房早
                        apb_probe += 1
                    else:
                        break
                if apb_probe == 2:
                    apb_double.append(beat.position)
                    iteration_num = 1
                else:
                    apb_short_array.append(apb_short_array_list)
                    iteration_num = apb_probe - 1
            else:
                if index + 2 < len_my_beats and beats[index +
                                                      2].label is Label.房性早搏:
                    apb_double_rhythm.append(beat.position)  # 出现房早二联律
                    iteration_num = 2
                else:
                    if (index + 3 < len_my_beats
                            and beats[index + 3].label is Label.房性早搏):
                        if (index + 6 < len_my_beats
                                and beats[index + 6].label is Label.房性早搏):
                            apb_triple_rhythm.append(beat.position)  # 出现房早三联律
                            iteration_num = 6
                    else:
                        apb_single.append(beat.position)  # 单发房早
        if beat.label is Label.室性早搏:  # 单发、成对、二联律、三联律、短阵
            vpb.append(beat.position)
            if iteration_num > 0:
                iteration_num -= 1
                continue
            if index + 1 < len_my_beats and beats[index +
                                                  1].label is Label.室性早搏:
                vpb_probe = 2
                vpb_short_array_list = [
                    beat.position, beats[index + 1].position
                ]
                while True:
                    if (index + vpb_probe < len_my_beats
                            and beats[index + vpb_probe].label is Label.室性早搏):
                        vpb_short_array_list.append(
                            beats[index + vpb_probe].position)  # 出现短阵室早
                        vpb_probe += 1
                    else:
                        break
                if vpb_probe == 2:
                    vpb_double.append(beat.position)  # 出现双发室早
                    iteration_num = 1
                else:
                    vpb_short_array.append(vpb_short_array_list)
                    iteration_num = vpb_probe - 1
            else:
                if index + 2 < len_my_beats and beats[index +
                                                      2].label is Label.室性早搏:
                    vpb_double_rhythm.append(beat.position)  # 出现室早二联律
                    iteration_num = 2
                else:
                    if (index + 3 < len_my_beats
                            and beats[index + 3].label is Label.室性早搏):
                        if (index + 6 < len_my_beats
                                and beats[index + 6].label is Label.室性早搏):
                            vpb_triple_rhythm.append(beat.position)  # 出现室早三联律
                            iteration_num = 6
                    else:
                        vpb_single.append(beat.position)  # 单发室早
        if beat.label is Label.窦性心律 and not beat.is_new and index < len_my_beats - 1:
            n_num += 1
            if not n_flag:
                if index + 1 < len_my_beats and beats[index +
                                                      1].label is Label.窦性心律:
                    n_continuous_beats.append(beat.r_peak)
                    n_flag = True
            else:
                n_continuous_beats.append(beat.r_peak)
        else:
            if index == len_my_beats - 1:
                if beat.label is Label.窦性心律 and not beat.is_new:
                    n_num += 1
                    n_continuous_beats.append(beat.r_peak)
            if n_flag:
                n_time.append(np.array(n_continuous_beats) / fs)
                n_continuous_diff = np.diff(np.array(n_continuous_beats))
                for i, diff in enumerate(n_continuous_diff):
                    n_diff_flatten_with_r_peak.append(
                        [n_continuous_beats[i + 1], diff])
                    if diff > 2 * fs:
                        n_stop_beats.append([n_continuous_beats[i + 1], diff])
                n_diff.append(n_continuous_diff)
                n_continuous_beats.clear()
                n_flag = False

        if beat.label in (Label.心房扑动, Label.心房颤动):
            if not af_flag:
                if (next_ := index + 1) < len_my_beats and (beats[next_].label
                                                            in (Label.心房扑动,
                                                                Label.心房颤动)):
                    af_continuous_beats.append(
                        beat.r_peak if beat.r_peak != -1 else beat.position)
                    af_flag = True
            else:
                af_continuous_beats.append(
                    beat.r_peak if beat.r_peak != -1 else beat.position)
        else:
            if index == len_my_beats - 1:
                if beat.label in (Label.心房扑动, Label.心房颤动):
                    af_continuous_beats.append(
                        beat.r_peak if beat.r_peak != -1 else beat.position)
            if af_flag:
                af_continuous_diff = np.diff(np.array(af_continuous_beats))
                af_diff.append(af_continuous_diff)
                af_continuous_beats.clear()
                af_flag = False
        # 室扑室颤噪声检查
        if beat.label is Label.室扑室颤:
            if iteration_num > 0:
                iteration_num -= 1
                continue
            vf_probe = index
            while (vf_probe + 1 < len_my_beats
                   and beats[vf_probe + 1].label is Label.室扑室颤):
                vf_probe += 1
            vf_time = int((beats[vf_probe].position - beat.position) / fs)
            if vf_time <= 2:
                continue
            vf.append([beat.position, vf_time])
            iteration_num = vf_probe

    len_h = ((beats[-1].position / fs) + 0.75) / (60 * 60)

    buffer: list[str] = []

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
    display_number = 1
    if 60 <= n_ventricular_mean_rate <= 100:
        n_state = "窦性心律"
    elif n_ventricular_mean_rate < 60:
        n_state = "窦性心动过缓"
    elif n_ventricular_mean_rate > 100:
        n_state = "窦性心动过速"
    else:
        assert False
    buffer.append("{}、{}:平均心室率：{}，最快心室率：{}，最慢心室率：{}\n".format(
        display_number,
        n_state,
        n_ventricular_mean_rate,
        n_ventricular_max_rate,
        n_ventricular_min_rate,
    ))

    if not len(n_stop_beats) == 0:
        n_stop_max = 0
        n_stop_index = 0
        for index, beats in enumerate(n_stop_beats):
            if beats[1] > n_stop_max:
                n_stop_max = beats[1]
                n_stop_index = beats[0]
        n_stop_max_seconds = n_stop_max / fs
        n_stop_max_h, n_stop_max_m, n_stop_max_s = _sample_to_time(
            n_stop_index)
        buffer.append("    发生了{}次窦性停搏,最长的一次为：{:.1f}s，发生于:{}h-{}m-{}s\n".format(
            len(n_stop_beats),
            n_stop_max_seconds,
            n_stop_max_h,
            n_stop_max_m,
            n_stop_max_s,
        ))
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
        buffer.append("{}、房扑房颤的平均心室率：{}次/分，最慢心室率：{}次/分，最快心室率：{}次/分\n".format(
            display_number,
            af_ventricular_mean_rate,
            af_ventricular_min_rate,
            af_ventricular_max_rate,
        ))
    if not len(apb) == 0:
        display_number += 1
        buffer.append(
            "{}、房性早搏{}次/24h,成对房早{}次/24h,房早二联律{}次/24h,房早三联律{}次/24h,短阵房速{}阵/24h\n"
            .format(
                display_number,
                int(len(apb) * 24 / len_h),
                int(len(apb_double) * 24 / len_h),
                int(len(apb_double_rhythm) * 24 / len_h),
                int(len(apb_triple_rhythm) * 24 / len_h),
                math.ceil((len(apb_short_array) * 24 / len_h)),
            ))
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
            apb_short_array_ventricular_max_rate = 60 / \
                (apb_short_array_min_diff / fs)
            apb_shor_array_h, apb_shor_array_m, apb_shor_array_s = _sample_to_time(
                apb_short_array[apb_short_array_min_index][0])
            buffer.append(
                "其中，短阵房速最快心室率为：{},由{}个QRS波组成,发生于：{}(采样点)/{}h-{}m-{}s(时间)\n".
                format(
                    int(apb_short_array_ventricular_max_rate),
                    len(apb_short_array[apb_short_array_min_index]),
                    int(apb_short_array[apb_short_array_min_index][0] * 250 /
                        fs),
                    apb_shor_array_h,
                    apb_shor_array_m,
                    apb_shor_array_s,
                ))
    if not len(vpb) == 0:
        display_number += 1
        buffer.append(
            "{}、室性早搏{}次/24h,成对室早{}次/24h,室早二联律{}次/24h,室早三联律{}次/24h,短阵室速{}阵/24h\n"
            .format(
                display_number,
                int(len(vpb) * 24 / len_h),
                int(len(vpb_double) * 24 / len_h),
                int(len(vpb_double_rhythm) * 24 / len_h),
                int(len(vpb_triple_rhythm) * 24 / len_h),
                math.ceil((len(vpb_short_array) * 24 / len_h)),
            ))
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
            vpb_short_array_ventricular_max_rate = 60 / \
                (vpb_short_array_min_diff / fs)
            vpb_shor_array_h, vpb_shor_array_m, vpb_shor_array_s = _sample_to_time(
                vpb_short_array[vpb_short_array_min_index][0])
            buffer.append(
                "其中，短阵室速最快心室率为：{},由{}个QRS波组成,发生于：{}(采样点)/{}h-{}m-{}s(时间)\n".
                format(
                    int(vpb_short_array_ventricular_max_rate),
                    len(vpb_short_array[vpb_short_array_min_index]),
                    int(vpb_short_array[vpb_short_array_min_index][0] * 250 /
                        fs),
                    vpb_shor_array_h,
                    vpb_shor_array_m,
                    vpb_shor_array_s,
                ))
    # 室扑室颤
    if not len(vf) == 0:
        display_number += 1
        buffer.append("{}、出现室扑室颤，如下：\n".format(display_number))
        for v in vf:
            vf_h, vf_m, vf_s = _sample_to_time(v[0])
            buffer.append("在{}h-{}m-{}s发生室扑室颤，持续时长{}s\n".format(
                vf_h, vf_m, vf_s, vf[1]))
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
        lf_, hf_ = _get_lf_hf(rr_interval, rr_interval_times)
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
    buffer.append("{}、心率变异性指标：\n".format(display_number))
    buffer.append("    SDNN:{:.2f}ms\n".format(sdnn))
    buffer.append("    SDANN:{:.2f}ms\n".format(sdann))
    buffer.append("    RMSSD:{:.2f}ms\n".format(rmssd))
    buffer.append("    PNN50:{:.2f}%\n".format(pnn50 * 100))
    buffer.append("    lf:{}".format(int(lf)))
    buffer.append("    hf:{}".format(hf))

    return "".join(buffer)
