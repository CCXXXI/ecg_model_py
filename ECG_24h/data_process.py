# -*- coding: utf-8 -*-
"""
@time: 2019/9/8 18:44
数据预处理：
    1.构建label2index和index2label
    2.划分数据集
@ author: javis
"""
import os, torch
import numpy as np
from config import config
import scipy.signal as signal
import torch.nn.functional as F

# 保证每次划分数据一致
np.random.seed(41)


def BSW(data, band_hz=0.5, fs=240):
    from scipy import signal

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


def output_sliding_voting(output, window=5):
    # window size must be odd number 奇数
    import pandas as pd
    from scipy.stats import mode

    output = (
        pd.Series(output)
        .rolling(window)
        .apply(lambda x: mode(x)[0][0])
        .fillna(method="bfill")
    )
    return output.values


def my_output_sliding_voting(ori_output, window=5):
    from scipy.stats import mode

    output = np.array(ori_output)
    leno = len(output)
    half_window = int(window / 2)
    for index, value in enumerate(output):
        if index < half_window:
            value = mode(output[index : index + window])[0][0]
        elif index >= leno - half_window:
            value = mode(output[index - window : index])[0][0]
        else:
            value = mode(output[index - half_window : index + half_window])[0][0]
        output[index] = value  # 漏了一句
    return output


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
    from dataset import transform

    x = data.copy()
    if not input_fs == 240:
        if input_fs < target_fs:
            print("！ERROR：目标采样率大于原始采样率，无法重采样")
            return
        x = resample(x, len(x) * target_fs // input_fs)
    lenx = len(x)
    # import time  #
    if del_drift:
        # st_time = time.time()  #
        wn1 = 2 * band_Hz / target_fs
        b, a = signal.butter(1, wn1, btype="high")
        x = signal.filtfilt(b, a, x)
        # ed_time = time.time()  #
        # delta_del_drift = ed_time - st_time
    # 标准化
    x = (x - np.mean(x)) / np.std(x)
    x = torch.tensor(x)
    x = torch.unsqueeze(x, 0)
    x = torch.unsqueeze(x, 0)
    x = x.to(device)

    # st_time = time.time() #
    pred = model(x)
    # ed_time = time.time() #
    # delta_predict = ed_time - st_time #
    out_pred = F.softmax(pred, 1).detach().cpu().numpy().argmax(axis=1)
    out_pred = np.reshape(out_pred, lenx)
    # output = output_sliding_voting(out_pred,15)
    # st_time = time.time() #
    # output = my_output_sliding_voting(out_pred,9)
    output = output_sliding_voting_v2(out_pred, 9)
    # ed_time = time.time() #
    # delta_vote = ed_time - st_time  #

    # print('高通滤波时间：{}s，推理时间：{}s，投票时间：{}s'.format(delta_del_drift, delta_predict, delta_vote)) #
    # output = out_pred
    p = output == 0  # P波
    N = output == 1  # QRS
    t = output == 2  # t波
    r = output == 3  # 其他

    # return p,N,t,r, out_pred
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
    import math

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
            # print(end-start)

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


def split_data_spv1(file2idx, val_ratio):
    from skmultilearn.model_selection import iterative_train_test_split

    data = list(file2idx.keys())

    X = np.array([[i] for i in range(len(data))], dtype=np.int32)
    Y = []
    for filename in data:
        y = np.zeros(shape=(config.num_classes,), dtype=np.int32)
        y[file2idx[filename]] = 1
        y = y.tolist()
        Y.append(y)
    Y = np.array(Y, dtype=np.int32)
    X_train, y_train, X_val, y_test = iterative_train_test_split(X, Y, val_ratio)
    train = [data[tid[0]] for tid in X_train]
    val = [data[tid[0]] for tid in X_val]
    return train, val


# 在spv1的基础上，将空标签数据（如果有）按比例划分至两数据集内
def split_data_spv2(file2idx, val_ratio):
    from skmultilearn.model_selection import iterative_train_test_split

    nolabeldata = []
    data = []

    for rid, labels in file2idx.items():
        if len(labels) == 0:
            nolabeldata.append(rid)
        else:
            data.append(rid)

    X = np.array([[i] for i in range(len(data))], dtype=np.int32)
    Y = []
    for filename in data:
        y = np.zeros(shape=(config.num_classes,), dtype=np.int32)
        y[file2idx[filename]] = 1
        y = y.tolist()
        Y.append(y)
    Y = np.array(Y, dtype=np.int32)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X, Y, val_ratio)
    train = [data[tid[0]] for tid in X_train]
    val = [data[tid[0]] for tid in X_val]

    if len(nolabeldata) > 0:
        import random

        random.shuffle(nolabeldata)
        val_extend_len = int(len(nolabeldata) * val_ratio)
        val.extend(nolabeldata[:val_extend_len])
        train.extend(nolabeldata[val_extend_len:])

    return train, val


def file2index(path, name2idx):
    """
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label列表的字段
    """

    file2index = dict()
    for line in open(path, encoding="utf-8"):
        if len(line.strip()) == 0:
            continue
        arr = line.strip().split(",")
        file = arr[0]
        labels = []
        if len(arr) > 1 and arr[1] != "":
            labels = [name2idx[name] for name in arr[1:]]
        # print(id, labels)
        file2index[file] = labels
    return file2index


def file2classname(path, name2idx):
    """
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label名称的列表
    """

    file2cname = dict()
    for line in open(path, encoding="utf-8"):
        if len(line.strip()) == 0:
            continue
        arr = line.strip().split(",")
        file = arr[0]
        labels = []
        if len(arr) > 1 and arr[1] != "":
            labels = arr[1:]
        # print(id, labels)
        file2cname[file] = labels
    return file2cname


def read_file2value(path: str):
    value_dict = {}
    for line in open(path, "r", encoding="UTF-8"):
        line = line.strip()
        if line:
            xml, val = line.split(",")
            value_dict[xml] = float(val)
    return value_dict


def count_labels(data, file2idx, num_classes=config.num_classes):
    """
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    """
    cc = [0] * num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)


def count_others(data: list, file2idx: dict) -> int:
    cnt = 0
    for rid in data:
        if len(file2idx[rid]) == 0:
            cnt += 1
    return cnt


def train(name2idx, idx2name):
    file2idx = file2index(config.train_label, name2idx)
    train, val = split_data_spv2(file2idx, config.val_ratio)
    wc = count_labels(train, file2idx)
    print(wc)
    dd = {
        "train": train,
        "val": val,
        "idx2name": idx2name,
        "file2idx": file2idx,
        "wc": wc,
    }
    # print(dd)
    torch.save(dd, config.train_data)

    val_cnt = count_labels(val, file2idx)

    with open(config.train_data.split(".pth")[0] + "_cnt.csv", "w") as output:
        output.write("idx,arraythmia,quantity,train,val\n")
        for i in range(wc.shape[0]):
            output.write(
                "{},{},{},{},{}\n".format(
                    i, idx2name[i], wc[i] + val_cnt[i], wc[i], val_cnt[i]
                )
            )
        train_others_cnt = count_others(train, file2idx)
        val_others_cnt = count_others(val, file2idx)
        if train_others_cnt > 0 or val_others_cnt > 0:
            output.write(
                "{},{},{},{},{}\n".format(
                    "-",
                    "其它",
                    train_others_cnt + val_others_cnt,
                    train_others_cnt,
                    val_others_cnt,
                )
            )
        output.write(
            "全部记录,{},{},{},{}".format(
                wc.shape[0], len(train) + len(val), len(train), len(val)
            )
        )


def save_label(savepath, data, file2idx, idx2name):
    with open(savepath, "w", encoding="utf-8") as output:
        for file in data:
            idxs = file2idx[file]
            output.write(file)
            for idx in idxs:
                output.write(",{}".format(idx2name[idx]))
            output.write("\n")


def trainval_test(idx2name):
    """
    将整个数据集划分为“训练集验证集”和“测试集两部分”，test_ratio是测试集的比例
    """
    file2idx = file2index(config.alldata_label, name2idx)
    trainval, test = split_data_spv2(file2idx, config.test_ratio)

    test_cnt = count_labels(test, file2idx)
    test_others_cnt = count_others(test, file2idx)

    with open(config.test_label.split(".txt")[0] + "_cnt.csv", "w") as output:
        output.write("idx,arraythmia,quantity\n")
        for i in range(test_cnt.shape[0]):
            output.write("{},{},{}\n".format(i, idx2name[i], test_cnt[i]))
        if test_others_cnt > 0:
            output.write("{},{},{}\n".format("-", "其它", test_others_cnt))
        output.write("总计,{},{}".format(config.num_classes, len(test)))

    save_label(config.test_label, test, file2idx, idx2name)
    save_label(config.train_label, trainval, file2idx, idx2name)


if __name__ == "__main__":
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    trainval_test(idx2name)
    train(name2idx, idx2name)
