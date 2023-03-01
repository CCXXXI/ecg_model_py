import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from scipy import integrate
from scipy import signal
from scipy.interpolate import interp1d

import config
import models
from U_net.CMI_ECG_segmentation_CNV2 import CBR_1D, Unet_1D
from dataset import transform

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


# The two classes must be imported, otherwise the model cannot be loaded.
# noinspection PyStatementEffect
CBR_1D, Unet_1D


class Mybeat:
    position = 0
    rpeak = -1
    new = False
    label = ""

    def __init__(self, position=0, rpeak=-1, label="", new=False):
        self.position = position
        # 定义补充心拍  处理噪声和室扑，室颤
        self.new = new
        self.rpeak = rpeak
        self.label = label


def get_24h_Beats(data_name, data_dir_path, Unet=None, device=None, fs=240, ori_fs=250):
    # 提取R波和心拍

    data = load_input(data_dir_path, data_name)

    print("###正在重采样原始信号###")
    start = time.time()
    data = resample(data, len(data) * fs // ori_fs)
    lenunet = 10 * 60 * fs
    end = time.time()
    print("###重采样成功，采样后数据长度：{}###，耗时：{}s".format(data.shape[0], end - start))

    print("###正在提取波群信息###")
    lendata = data.shape[0]
    start = time.time()
    beats = []
    R_peaks = []
    cur_s = 0
    pbar = tqdm.tqdm(total=lendata)
    while cur_s < lendata:
        if cur_s + lenunet <= lendata:
            pbar.update(lenunet)
            now_s = cur_s + lenunet
        else:
            now_s = lendata
            break
        p, N, t, r = U_net_peak(
            data[cur_s:now_s], input_fs=fs, del_drift=True, model=Unet, device=device
        )

        beat_list = U_net_RPEAK(N)
        r_list = R_Detection_U_net(data[cur_s:now_s], N)
        # 记录QRS波中点，以该点标识心拍     之后两边扩展
        beat_list = np.array(beat_list)
        r_list = np.array(r_list)

        append_start = int(0.5 * 60 * fs)
        append_end = int(9.5 * 60 * fs)
        if cur_s == 0:
            append_start = 0

        for beat in beat_list:
            if append_start < beat <= append_end:
                beats.append(beat + cur_s)
        for r in r_list:
            if append_start < r <= append_end:
                R_peaks.append(r + cur_s)

        cur_s += 9 * 60 * fs
    end = time.time()
    pbar.close()
    print("###提取成功，提取出{}个心拍，耗时：{}s###".format(len(beats), end - start))
    return beats, R_peaks


def check_beats(beats, rpeaks, fs=240):
    beats = np.array(beats, dtype=int)
    rpeaks = np.array(rpeaks, dtype=int)
    checked_beats = [Mybeat(position=beats[0], rpeak=rpeaks[0], new=False)]
    limit = 2 * 1.5 * fs
    beats_diff = np.diff(beats)
    add_num = 0
    for index, diff in enumerate(beats_diff):
        if diff >= limit:
            start = beats[index]
            cur = start
            end = beats[index + 1]
            while (end - cur) >= limit:
                new_beat = cur + int(limit / 2)
                checked_beats.append(Mybeat(position=new_beat, new=True))
                cur = new_beat
                add_num += 1
            checked_beats.append(
                Mybeat(position=beats[index + 1], rpeak=rpeaks[index + 1], new=False)
            )
        else:
            checked_beats.append(
                Mybeat(position=beats[index + 1], rpeak=rpeaks[index + 1], new=False)
            )
    return add_num, checked_beats


def classification_beats(
    data_name,
    data_dir_path,
    save_dir,
    beats,
    Resnet=None,
    device=None,
    fs=240,
    ori_fs=250,
):
    half_len = int(0.75 * fs)
    data = load_input(data_dir_path, data_name)

    print("###正在重采样原始信号###")
    start = time.time()
    data = resample(data, len(data) * fs // ori_fs)
    data = BSW(data, band_hz=0.5, fs=fs)
    end = time.time()
    print("###重采样成功，采样后数据长度：{}###，耗时：{}s".format(data.shape[0], end - start))

    print("###正在分类心拍###")
    start = time.time()

    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    name2cnt = {name: 0 for name in name2idx.keys()}
    pbar = tqdm.tqdm(total=len(beats))
    pbar_num = 0

    batch_size = 64
    input_tensor = []
    input_beats = []
    with torch.no_grad():
        for idx, beat in enumerate(beats):
            pbar_num += 1
            if pbar_num >= 100:
                pbar_num = 0
                pbar.update(100)

            if beat.position < half_len or beat.position >= data.shape[0] - half_len:
                beat.label = ""
                continue

            x = data[beat.position - half_len : beat.position + half_len]
            x.astype(np.float32)
            x = np.reshape(x, (1, half_len * 2))
            if config.data_standardization:
                x = (x - np.mean(x)) / np.std(x)
            x = x.T
            x = transform(x).unsqueeze(0).to(device)
            input_tensor.append(x)
            input_beats.append(beat)

            if len(input_tensor) % batch_size == 0 or idx == len(beats) - 1:
                x = torch.vstack(input_tensor)
                output = torch.softmax(Resnet(x), dim=1).squeeze()

                # 修改维度
                y_pred = torch.argmax(output, dim=1, keepdim=False)
                for i, pred in enumerate(y_pred):
                    pred = pred.item()
                    beat = input_beats[i]
                    name2cnt[idx2name[pred]] += 1
                    beat.label = idx2name[pred]
                input_tensor = []
                input_beats = []

    save_mybeats(
        mybeats=beats,
        data_name=data_name.split(".")[0] + "_mybeats_withlabel_v1.3.txt",
        save_dir=save_dir,
    )
    end = time.time()
    print("###{} 分类结束，耗时：{}###".format(data_name, end - start))
    with open(
        os.path.join(
            save_dir, data_name.split(".")[0] + "_classification_cnt_v1.3.txt"
        ),
        "w",
        encoding="utf-8",
    ) as fout:
        for name, cnt in name2cnt.items():
            fout.write(name)
            fout.write(":")
            fout.write(str(cnt))
            fout.write("\n")


def save_mybeats(mybeats, data_name, save_dir):
    with open(os.path.join(save_dir, data_name), "w", encoding="utf-8") as fout:
        for mybeat in mybeats:
            fout.write(str(mybeat.position))
            fout.write(",")
            fout.write(str(mybeat.rpeak))
            fout.write(",")
            fout.write(mybeat.label)
            fout.write(",")
            fout.write(str(mybeat.new))
            fout.write("\n")


def load_mybeats(data_name, load_dir):
    fin = open(os.path.join(load_dir, data_name), encoding="utf-8")
    my_beats = []
    for line in fin:
        line = line.strip().split(",")

        my_beat = Mybeat(
            position=int(line[0]),
            rpeak=int(line[1]),
            label=line[2],
            new=True if line[3] == "True" else False,
        )
        my_beats.append(my_beat)

    return my_beats


def get_lfhf(rr_intervals, rr_interval_times):
    resampling_period = 0.5
    interpolation_method = "spline"
    if interpolation_method == "spline":
        interpolated_rr_intervals = interp1d(
            rr_interval_times, rr_intervals, kind="cubic"
        )
    elif interpolation_method == "linear":
        interpolated_rr_intervals = interp1d(rr_interval_times, rr_intervals)
    # fft conversion
    start_time = interpolated_rr_intervals.x[0]
    end_time = interpolated_rr_intervals.x[-1]
    fixed_times = np.arange(start_time, end_time, resampling_period)
    num_samples = fixed_times.shape[0]
    resampled_rr_intervals = interpolated_rr_intervals(fixed_times)
    frequencies = np.fft.fftfreq(num_samples, d=resampling_period)
    non_negative_frequency_index = frequencies >= 0

    frequencies = frequencies[non_negative_frequency_index]
    fft_converted = np.fft.fft(resampled_rr_intervals)[non_negative_frequency_index]
    amplitudes = np.abs(fft_converted)
    powers = amplitudes**2
    if frequencies.shape[0] >= 2 and powers.shape[0] >= 2:
        interpolated_powers = interp1d(frequencies, powers)

    lfhf_configuration = {
        "minimum_frequency": 0.05,
        "boundary_frequency": 0.15,
        "maximum_frequency": 0.4,
    }
    minimum_frequency = lfhf_configuration["minimum_frequency"]
    boundary_frequency = lfhf_configuration["boundary_frequency"]
    maximum_frequency = lfhf_configuration["maximum_frequency"]

    try:
        start_index = np.where(frequencies >= minimum_frequency)[0][0]
        boundary_index = np.where(frequencies >= boundary_frequency)[0][0]
        end_index = np.where(frequencies <= maximum_frequency)[0][-1]
    except:
        return -1, -1

    low_frequency_component_powers = powers[start_index:boundary_index]
    high_frequency_component_powers = powers[boundary_index : end_index + 1]
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


def sampletotime(position, fs):
    total_seconds = position / fs
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int((total_seconds % 3600 % 60))
    return h, m, s


def analyze_mybeats(mybeats, data_name, save_dir, fs=240):
    N_diff = []
    N_time = []
    N_diff_flatten_with_rpeak = []
    N_flag = False
    N_continuous_beats = []
    N_continuous_time = []
    N_stop_beats = []
    N_num = 0
    RR = []
    AF_diff = []  # 房扑房颤
    AF_flag = False
    AF_continuous_beats = []
    VF = []

    QRS_num = 0
    APB = []
    APB_single = []  # 单发房早-次数
    APB_double = []  # 成对房早-次数
    APB_double_rhythm = []  # 房早二联律-次数、持续时间
    APB_trible_rhythm = []  # 房早三联律-次数、持续时间
    APB_short_array = []  # 短阵房早-次数、持续时间
    VPB = []
    VPB_single = []  # 单发室早
    VPB_double = []  # 成对室早
    VPB_double_rhythm = []  # 室早二联律
    VPB_trible_rhythm = []  # 室早三联律
    VPB_short_array = []  # 短阵室早
    iteration_num = 0
    lenmybeats = len(mybeats)

    pbar = tqdm.tqdm(total=len(mybeats))
    pbar_num = 0
    for index, mybeat in enumerate(mybeats):
        pbar_num += 1
        if pbar_num >= 100:
            pbar_num = 0
            pbar.update(100)
        if mybeat.label == "":
            continue
        RR.append(mybeat.rpeak if not mybeat.rpeak == -1 else mybeat.position)
        if mybeat.new == False:
            QRS_num += 1
        if mybeat.label == "房性早搏":  # 单发、成对、二联律、三联律、短阵
            APB.append(mybeat.position)
            if iteration_num > 0:
                iteration_num -= 1
                continue
            if index + 1 < lenmybeats and mybeats[index + 1].label == "房性早搏":
                APB_probe = 2
                APB_short_array_list = [mybeat.position, mybeats[index + 1].position]
                while True:
                    if (
                        index + APB_probe < lenmybeats
                        and mybeats[index + APB_probe].label == "房性早搏"
                    ):
                        APB_short_array_list.append(
                            mybeats[index + APB_probe].position
                        )  # 出现短阵房早
                        APB_probe += 1
                    else:
                        break
                if APB_probe == 2:
                    APB_double.append(mybeat.position)
                    iteration_num = 1
                else:
                    APB_short_array.append(APB_short_array_list)
                    iteration_num = APB_probe - 1
            else:
                if index + 2 < lenmybeats and mybeats[index + 2].label == "房性早搏":
                    APB_double_rhythm.append(mybeat.position)  # 出现房早二联律
                    iteration_num = 2
                else:
                    if index + 3 < lenmybeats and mybeats[index + 3].label == "房性早搏":
                        if (
                            index + 6 < lenmybeats
                            and mybeats[index + 6].label == "房性早搏"
                        ):
                            APB_trible_rhythm.append(mybeat.position)  # 出现房早三联律
                            iteration_num = 6
                    else:
                        APB_single.append(mybeat.position)  # 单发房早
        if mybeat.label == "室性早搏":  # 单发、成对、二联律、三联律、短阵
            VPB.append(mybeat.position)
            if iteration_num > 0:
                iteration_num -= 1
                continue
            if index + 1 < lenmybeats and mybeats[index + 1].label == "室性早搏":
                VPB_probe = 2
                VPB_short_array_list = [mybeat.position, mybeats[index + 1].position]
                while True:
                    if (
                        index + VPB_probe < lenmybeats
                        and mybeats[index + VPB_probe].label == "室性早搏"
                    ):
                        VPB_short_array_list.append(
                            mybeats[index + VPB_probe].position
                        )  # 出现短阵室早
                        VPB_probe += 1
                    else:
                        break
                if VPB_probe == 2:
                    VPB_double.append(mybeat.position)  # 出现双发室早
                    iteration_num = 1
                else:
                    VPB_short_array.append(VPB_short_array_list)
                    iteration_num = VPB_probe - 1
            else:
                if index + 2 < lenmybeats and mybeats[index + 2].label == "室性早搏":
                    VPB_double_rhythm.append(mybeat.position)  # 出现室早二联律
                    iteration_num = 2
                else:
                    if index + 3 < lenmybeats and mybeats[index + 3].label == "室性早搏":
                        if (
                            index + 6 < lenmybeats
                            and mybeats[index + 6].label == "室性早搏"
                        ):
                            VPB_trible_rhythm.append(mybeat.position)  # 出现室早三联律
                            iteration_num = 6
                    else:
                        VPB_single.append(mybeat.position)  # 单发室早
        if mybeat.label == "窦性心律" and mybeat.new == False and index < lenmybeats - 1:
            N_num += 1
            if N_flag == False:
                if index + 1 < lenmybeats and mybeats[index + 1].label == "窦性心律":
                    N_continuous_beats.append(mybeat.rpeak)
                    N_flag = True
            else:
                N_continuous_beats.append(mybeat.rpeak)
        else:
            if index == lenmybeats - 1:
                if mybeat.label == "窦性心律" and mybeat.new == False:
                    N_num += 1
                    N_continuous_beats.append(mybeat.rpeak)
            if N_flag == True:
                N_time.append(np.array(N_continuous_beats) / fs)
                N_continuous_diff = np.diff(np.array(N_continuous_beats))
                for index, diff in enumerate(N_continuous_diff):
                    N_diff_flatten_with_rpeak.append(
                        [N_continuous_beats[index + 1], diff]
                    )
                    if diff > 2 * fs:
                        N_stop_beats.append([N_continuous_beats[index + 1], diff])
                # if len(N_continuous_diff)>1:
                N_diff.append(N_continuous_diff)
                N_continuous_beats.clear()
                N_flag = False

        if mybeat.label == "心房扑动" or mybeat.label == "心房颤动":
            if AF_flag == False:
                if (
                    index + 1 < lenmybeats
                    and mybeats[index + 1].label == "心房扑动"
                    or "心房颤动"
                    and index < lenmybeats - 1
                ):
                    AF_continuous_beats.append(
                        mybeat.rpeak if not mybeat.rpeak == -1 else mybeat.position
                    )
                    AF_flag = True
            else:
                AF_continuous_beats.append(
                    mybeat.rpeak if not mybeat.rpeak == -1 else mybeat.position
                )
        else:
            if index == lenmybeats - 1:
                if mybeat.label == "心房扑动" or mybeat.label == "心房颤动":
                    AF_continuous_beats.append(
                        mybeats.rpeak if not mybeat.rpeak == -1 else mybeat.position
                    )
            if AF_flag == True:
                AF_continuous_diff = np.diff(np.array(AF_continuous_beats))
                AF_diff.append(AF_continuous_diff)
                AF_continuous_beats.clear()
                AF_flag = False
        # 室扑室颤噪声检查
        if mybeat.label == "室扑室颤":
            if iteration_num > 0:
                iteration_num -= 1
                continue
            VF_probe = index
            while VF_probe + 1 < lenmybeats and mybeats[VF_probe + 1].label == "室扑室颤":
                VF_probe += 1
            VF_time = int((mybeats[VF_probe].position - mybeat.position) / fs)
            if VF_time <= 2:
                continue
            VF.append([mybeat.position, VF_time])
            iteration_num = VF_probe

    lenh = ((mybeats[-1].position / fs) + 0.75) / (60 * 60)

    with open(
        os.path.join(save_dir, data_name + "_report_lfhf.txt"), "w", encoding="utf-8"
    ) as fout:
        # 计算窦性心室率
        N_max_diff = 0
        N_min_diff = 10000
        N_diff_sum = 0
        N_diff_num = 0
        for diffs in N_diff:
            for diff in diffs:
                if diff > N_max_diff and diff < 2 * fs:
                    N_max_diff = diff
                if diff < N_min_diff and diff > 0.3 * fs:
                    N_min_diff = diff
                N_diff_sum += diff
                N_diff_num += 1
        N_ventricular_mean_rate = int(60 / (N_diff_sum / N_diff_num / fs))
        N_ventricular_max_rate = int(60 / (N_min_diff / fs))
        N_ventricular_min_rate = int(60 / (N_max_diff / fs))
        fout.write("------数据{}------\n".format(data_name))
        display_number = 1
        if 100 >= N_ventricular_mean_rate >= 60:
            N_state = "窦性心律"
        elif N_ventricular_mean_rate < 60:
            N_state = "窦性心动过缓"
        elif N_ventricular_mean_rate > 100:
            N_state = "窦性心动过速"
        fout.write(
            "{}、{}:平均心室率：{}，最快心室率：{}，最慢心室率：{}\n".format(
                display_number,
                N_state,
                N_ventricular_mean_rate,
                N_ventricular_max_rate,
                N_ventricular_min_rate,
            )
        )

        if not len(N_stop_beats) == 0:
            N_stop_max = 0
            N_stop_index = 0
            for index, beats in enumerate(N_stop_beats):
                if beats[1] > N_stop_max:
                    N_stop_max = beats[1]
                    N_stop_index = beats[0]
            N_stop_max_seconds = N_stop_max / fs
            N_stop_max_h, N_stop_max_m, N_stop_max_s = sampletotime(N_stop_index, fs)
            fout.write(
                "    发生了{}次窦性停搏,最长的一次为：{:.1f}s，发生于:{}h-{}m-{}s\n".format(
                    len(N_stop_beats),
                    N_stop_max_seconds,
                    N_stop_max_h,
                    N_stop_max_m,
                    N_stop_max_s,
                )
            )
        # 计算房扑房颤心室率
        if not len(AF_diff) == 0:
            display_number += 1
            AF_max_diff = 0
            AF_min_diff = 10000
            AF_diff_sum = 0
            AF_diff_num = 0
            for diffs in AF_diff:
                for diff in diffs:
                    if diff > AF_max_diff and diff < 1.5 * fs:  # 最慢不能低于100心率
                        AF_max_diff = diff
                    if diff < AF_min_diff and diff > 0.2 * fs:  # 最快不能高于300心率
                        AF_min_diff = diff
                    AF_diff_sum += diff
                    AF_diff_num += 1
            AF_ventricular_mean_rate = int(60 / (AF_diff_sum / AF_diff_num / fs))
            AF_ventricular_max_rate = int(60 / (AF_min_diff / fs))
            AF_ventricular_min_rate = int(60 / (AF_max_diff / fs))
            fout.write(
                "{}、房扑房颤的平均心室率：{}次/分，最慢心室率：{}次/分，最快心室率：{}次/分\n".format(
                    display_number,
                    AF_ventricular_mean_rate,
                    AF_ventricular_min_rate,
                    AF_ventricular_max_rate,
                )
            )
        if not len(APB) == 0:
            display_number += 1
            fout.write(
                "{}、房性早搏{}次/24h,成对房早{}次/24h,房早二联律{}次/24h,房早三联律{}次/24h,短阵房速{}阵/24h\n".format(
                    display_number,
                    int(len(APB) * 24 / lenh),
                    int(len(APB_double) * 24 / lenh),
                    int(len(APB_double_rhythm) * 24 / lenh),
                    int(len(APB_trible_rhythm) * 24 / lenh),
                    math.ceil((len(APB_short_array) * 24 / lenh)),
                )
            )
            # 短阵房早
            if not len(APB_short_array) == 0:
                APB_short_array_diff = []
                APB_short_array_min_diff = 10000
                APB_short_array_min_index = 0
                for APB_s_a in APB_short_array:
                    APB_short_array_diff.append(np.diff(APB_s_a))
                for index, diffs in enumerate(APB_short_array_diff):
                    for diff in diffs:
                        if diff < APB_short_array_min_diff and diff > 0.2 * fs:
                            APB_short_array_min_index = index
                            APB_short_array_min_diff = diff
                APB_short_array_ventricular_max_rate = 60 / (
                    APB_short_array_min_diff / fs
                )
                APB_shor_array_h, APB_shor_array_m, APB_shor_array_s = sampletotime(
                    APB_short_array[APB_short_array_min_index][0], fs
                )
                fout.write(
                    "其中，短阵房速最快心室率为：{},由{}个QRS波组成,发生于：{}(采样点)/{}h-{}m-{}s(时间)\n".format(
                        int(APB_short_array_ventricular_max_rate),
                        len(APB_short_array[APB_short_array_min_index]),
                        int(APB_short_array[APB_short_array_min_index][0] * 250 / fs),
                        APB_shor_array_h,
                        APB_shor_array_m,
                        APB_shor_array_s,
                    )
                )
        if not len(VPB) == 0:
            display_number += 1
            fout.write(
                "{}、室性早搏{}次/24h,成对室早{}次/24h,室早二联律{}次/24h,室早三联律{}次/24h,短阵室速{}阵/24h\n".format(
                    display_number,
                    int(len(VPB) * 24 / lenh),
                    int(len(VPB_double) * 24 / lenh),
                    int(len(VPB_double_rhythm) * 24 / lenh),
                    int(len(VPB_trible_rhythm) * 24 / lenh),
                    math.ceil((len(VPB_short_array) * 24 / lenh)),
                )
            )
            # 短阵室早
            if not len(VPB_short_array) == 0:
                VPB_short_array_diff = []
                VPB_short_array_min_diff = 10000
                VPB_short_array_min_index = 0
                for VPB_s_a in VPB_short_array:
                    VPB_short_array_diff.append(np.diff(VPB_s_a))
                for index, diffs in enumerate(VPB_short_array_diff):
                    for diff in diffs:
                        if diff < VPB_short_array_min_diff:
                            VPB_short_array_min_index = index
                            VPB_short_array_min_diff = diff
                VPB_short_array_ventricular_max_rate = 60 / (
                    VPB_short_array_min_diff / fs
                )
                VPB_shor_array_h, VPB_shor_array_m, VPB_shor_array_s = sampletotime(
                    VPB_short_array[VPB_short_array_min_index][0], fs
                )
                fout.write(
                    "其中，短阵室速最快心室率为：{},由{}个QRS波组成,发生于：{}(采样点)/{}h-{}m-{}s(时间)\n".format(
                        int(VPB_short_array_ventricular_max_rate),
                        len(VPB_short_array[VPB_short_array_min_index]),
                        int(VPB_short_array[VPB_short_array_min_index][0] * 250 / fs),
                        VPB_shor_array_h,
                        VPB_shor_array_m,
                        VPB_shor_array_s,
                    )
                )
        # 室扑室颤
        if not len(VF) == 0:
            display_number += 1
            fout.write("{}、出现室扑室颤，如下：\n".format(display_number))
            for vf in VF:
                VF_h, VF_m, VF_S = sampletotime(vf[0], fs=240)
                fout.write(
                    "在{}h-{}m-{}s发生室扑室颤，持续时长{}s\n".format(VF_h, VF_m, VF_S, vf[1])
                )
        # 计算HRV

        # lf-hf
        lf = 0
        hf = 0
        for index in range(len(N_diff)):
            rr_interval = np.array(N_diff[index])
            rr_interval = rr_interval / fs
            rr_interval_times = np.array(N_time[index])[:-1]
            if len(rr_interval) < 4:
                continue
            lf_, hf_ = get_lfhf(rr_interval, rr_interval_times)
            if lf_ == -1 or hf_ == -1:
                continue
            else:
                lf += lf_
                hf += hf_

        N_diff_rpeak = [i[0] for i in N_diff_flatten_with_rpeak]
        N_diff_flatten = [i[1] for i in N_diff_flatten_with_rpeak]

        # RMSSD
        N_diff_diff = []
        for diffs in N_diff:
            diff_diffs = np.diff(diffs)
            for diff_diff in diff_diffs:
                N_diff_diff.append(diff_diff)

        # SDANN
        start_rpeak = N_diff_rpeak[0]
        end_rpeak = N_diff_rpeak[-1]
        NN_5minute_mean = []
        NN_5minute_sum = 0
        NN_5minute_num = 0

        for index, rpeak in enumerate(N_diff_rpeak):
            if (rpeak - start_rpeak) <= 300 * fs and not rpeak == end_rpeak:
                NN_5minute_sum += N_diff_flatten[index]
                NN_5minute_num += 1
            else:
                if rpeak == end_rpeak:
                    if (rpeak - start_rpeak) <= 300 * fs:
                        NN_5minute_num += 1
                        NN_5minute_sum += N_diff_flatten[index]
                if not NN_5minute_num == 0:
                    NN_5minute_mean.append(NN_5minute_sum / NN_5minute_num)
                NN_5minute_sum = N_diff_flatten[index]
                NN_5minute_num = 1
                start_rpeak = rpeak
        # PNN50
        N_PNN50_num = 0
        for diff_diff in N_diff_diff:
            if diff_diff >= 0.05 * fs:
                N_PNN50_num += 1

        SDNN = np.std(N_diff_flatten)
        SDANN = np.std(NN_5minute_mean)
        RMSSD = math.sqrt(sum([x**2 for x in N_diff_diff]) / len(N_diff_diff))
        PNN50 = N_PNN50_num / N_num
        display_number += 1
        fout.write("{}、心率变异性指标：\n".format(display_number))
        fout.write("    SDNN:{:.2f}ms\n".format(SDNN))
        fout.write("    SDANN:{:.2f}ms\n".format(SDANN))
        fout.write("    RMSSD:{:.2f}ms\n".format(RMSSD))
        fout.write("    PNN50:{:.2f}%\n".format(PNN50 * 100))
        fout.write("    lf:{}".format(int(lf)))
        fout.write("    lf/hf:{}".format(lf / hf))


def load_input(data_dir_path, data_name):
    return np.loadtxt(os.path.join(data_dir_path, data_name))


def main():
    os.makedirs(config.beats_24h, exist_ok=True)
    os.makedirs(config.R_24h, exist_ok=True)
    os.makedirs(config.mybeats_24h, exist_ok=True)
    os.makedirs(config.report_24h, exist_ok=True)
    fs = 240
    ori_fs = 250
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_24h_file_dir = config.dir_24h
    # ①提取R波切分心拍
    with torch.no_grad():
        Unet = torch.load("U_net/240HZ_t+c_v2_best.pt", map_location=device)
        Unet.eval()
        for i in os.listdir(data_24h_file_dir):
            with open(
                os.path.join(config.beats_24h, "{}-beats.txt".format(i.split(".")[0])),
                "w",
                encoding="utf-8",
            ) as output1:
                with open(
                    os.path.join(config.R_24h, "{}-Rpeaks.txt".format(i.split(".")[0])),
                    "w",
                    encoding="utf-8",
                ) as output2:
                    beats, R_peaks = get_24h_Beats(
                        i,
                        data_24h_file_dir,
                        Unet=Unet,
                        device=device,
                        fs=fs,
                        ori_fs=ori_fs,
                    )
                    print("{}-{}".format(i, len(beats)))
                    output1.write(i)
                    for beat in beats:
                        output1.write(",")
                        output1.write(str(beat))
                    output1.write("\n")
                    output2.write(i)
                    for R_peak in R_peaks:
                        output2.write(",")
                        output2.write(str(R_peak))
                    output2.write("\n")
                output2.close()
            output1.close()
    # ②补充心拍
    for beat_file in os.listdir(config.beats_24h):
        data_name = beat_file.split("-")[0]
        beats_in = open(os.path.join(config.beats_24h, beat_file))
        rpeak_in = open(os.path.join(config.R_24h, data_name + "-Rpeaks.txt"))
        beats = beats_in.readline().split(",")[1:]
        rpeaks = rpeak_in.readline().split(",")[1:]
        if not len(beats) == len(rpeaks):
            print("error:提取出的心拍数量{}与R波数量{}不同".format(len(beats), len(rpeaks)))
            continue
        add_num, checked_mybeats = check_beats(beats, rpeaks, fs=240)
        print("补充了{}个心拍".format(add_num))
        save_mybeats(checked_mybeats, data_name + "-mybeats.txt", config.mybeats_24h)
    # ③读取mybeats并进行预测，存储标签
    Resnet = getattr(models, config.model_name)(num_classes=config.num_classes)
    Resnet.load_state_dict(
        torch.load(
            "assets/best_w.pth",
            map_location="cpu",
        )["state_dict"]
    )
    Resnet = Resnet.to(device)
    Resnet.eval()
    with torch.no_grad():
        for data_name in os.listdir(data_24h_file_dir):
            name = data_name.split(".")[0] + "-mybeats.txt"
            print(name)
            my_beats = load_mybeats(name, config.mybeats_24h)
            classification_beats(
                data_name,
                data_24h_file_dir,
                config.mybeats_24h,
                my_beats,
                Resnet=Resnet,
                device=device,
                fs=fs,
                ori_fs=ori_fs,
            )
    # ④读取带有标签的mybeats，并进行统计
    load_dir = config.mybeats_24h
    for data_name in os.listdir(data_24h_file_dir):
        name = data_name.split(".")[0] + "_mybeats_withlabel_v1.3.txt"
        my_beats = load_mybeats(name, load_dir)
        analyze_mybeats(
            my_beats, data_name.split(".")[0], save_dir=config.report_24h, fs=fs
        )


if __name__ == "__main__":
    main()
