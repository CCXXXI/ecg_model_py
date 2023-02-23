# -*- coding: utf-8 -*-
"""
@time: 2019/9/8 19:47

@ author: javis
"""
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy import signal
from torch.utils.data import Dataset

from config import config


def load_wordembedding(word_embedding_path, idx2name) -> np.ndarray:
    n, dim, wordlist, word2embedding = None, None, None, None
    with open(word_embedding_path, "r", encoding="utf-8") as cin:
        lines = cin.readlines()
        n, dim = list(map(int, lines[0].strip().split(",")))
        wordlist = []
        word2embedding = {}
        for line in lines[1:]:
            line = line.strip()
            arr = line.split(",")
            word = arr[0]
            embedding = np.array(list(map(float, arr[1:])), dtype=np.float32)
            wordlist.append(word)
            word2embedding[word] = embedding
    embeddings = []
    for i in range(config.num_classes):
        name = idx2name[i]
        embedding = word2embedding[name]
        embedding = embedding[None, :]
        embeddings.append(embedding)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def load_Ridx_dict(Ridx_path: str) -> {str: list}:
    Ridx_dict = {}
    with open(Ridx_path, "r") as cin:
        for line in cin.readlines():
            line = line.strip()
            if line:
                rid, Ridxs = line.split(",")
                Ridxs = list(map(int, Ridxs.split(" "))) if len(Ridxs) else []
                Ridx_dict[rid] = Ridxs
    return Ridx_dict


def resample(sig, target_point_num=None):
    """
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    """
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def verflip(sig):
    """
    信号竖直翻转  x 信号水平翻转 √
    :param sig:
    :return:
    """
    return sig[::-1, :]


def shift(sig, interval=20):
    """
    上下平移
    :param sig:
    :return:
    """
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):
    # 前置不可或缺的步骤
    sig = resample(sig, config.target_point_num)

    # 数据增强
    if "data_augmentation" not in dir(config) or config.data_augmentation:
        if train:
            if np.random.randn() > 0.5:
                sig = scaling(sig)
            if np.random.randn() > 0.5:
                sig = verflip(sig)
            if np.random.randn() > 0.5:
                sig = shift(sig)

    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class Transformation:
    def __init__(self, *args, **kwargs):
        self.params = kwargs

    def get_params(self):
        return self.params


class TGaussianNoise(Transformation):
    """Add gaussian noise to sample."""

    def __init__(self, scale=0.01):
        super(TGaussianNoise, self).__init__(scale=scale)
        self.scale = scale

    def __call__(self, sample):
        if self.scale == 0:
            return sample
        else:
            data = sample
            data = data + self.scale * torch.randn(data.shape)
            return data

    def __str__(self):
        return "GaussianNoise_scale{}".format(self.scale)


class TChannelResize(Transformation):
    """Scale amplitude of sample (per channel) by random factor in given magnitude range"""

    def __init__(self, magnitude_range=(0.33, 3)):
        super(TChannelResize, self).__init__(magnitude_range=magnitude_range)
        self.log_magnitude_range = torch.log(torch.tensor(magnitude_range))

    def __call__(self, sample):
        data = sample
        timesteps, channels = data.shape
        resize_factors = torch.exp(
            torch.empty(channels).uniform_(*self.log_magnitude_range)
        )
        resize_factors_same_shape = resize_factors.repeat(timesteps).reshape(data.shape)
        data = resize_factors_same_shape * data
        return data

    def __str__(self):
        return "ChannelResize"


class Shift(Transformation):
    """
    上下平移
    """

    def __init__(self, interval=20):
        super(Shift, self).__init__(interval=interval)
        self.interval = interval

    def __call__(self, sample):
        data = sample
        for col in range(data.shape[1]):
            offset = np.random.choice(range(-self.interval, self.interval))
            data[:, col] += offset
        return data

    def __str__(self):
        return "Shift_in{}".format(self.interval)


class TRandomCrop(object):
    """Crop randomly the image in a sample."""

    def __init__(self, output_size, annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data = sample

        timesteps, _ = data.shape
        assert timesteps >= self.output_size
        if timesteps == self.output_size:
            start = 0
        else:
            start = random.randint(
                0, timesteps - self.output_size - 1
            )  # np.random.randint(0, timesteps - self.output_size)

        data = data[start : start + self.output_size, :]

        return data

    def __str__(self):
        return "RandomCrop"


def Tinterpolate(data, marker):
    timesteps, channels = data.shape
    data = data.transpose(0, 1).flatten()
    ndata = data.numpy()
    interpolation = torch.from_numpy(
        np.interp(
            np.where(ndata == marker)[0],
            np.where(ndata != marker)[0],
            ndata[ndata != marker],
        )
    )
    data[data == marker] = interpolation.type(data.type())
    data = data.reshape(channels, timesteps).T
    return data


class TRandomResizedCrop(Transformation):
    """Extract crop at random position and resize it to full size"""

    def __init__(self, crop_ratio_range=[0.5, 1.0]):
        super(TRandomResizedCrop, self).__init__(crop_ratio_range=crop_ratio_range)
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, sample):
        output = torch.full(sample.shape, float("inf")).type(sample[0].type())
        timesteps, channels = output.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        data = TRandomCrop(int(crop_ratio * timesteps))(sample)  # apply random crop
        cropped_timesteps = data.shape[0]
        indices = torch.sort(
            (torch.randperm(timesteps - 2) + 1)[: cropped_timesteps - 2]
        )[0]
        indices = torch.cat([torch.tensor([0]), indices, torch.tensor([timesteps - 1])])
        output[
            indices, :
        ] = data  # fill output array randomly (but in right order) with values from random crop

        # use interpolation to resize random crop
        output = Tinterpolate(output, float("inf"))
        return output

    def __str__(self):
        return "RandomResizedCrop"


class Verflip(Transformation):
    """
    信号竖直翻转  x 信号水平翻转 √
    """

    def __init__(self):
        super(Verflip, self).__init__()

    def __call__(self, sample):
        data = sample
        return torch.flip(data, dims=[0])

    def __str__(self):
        return "Verflip"


class Scaling(Transformation):
    """
    信号竖直翻转  x 信号水平翻转 √
    """

    def __init__(self, std=0.1):
        super(Scaling, self).__init__(std=std)
        self.std = std

    def __call__(self, sample):
        data = sample
        scalingFactor = torch.normal(1.0, self.std, (1, data.shape[1]))
        myNoise = torch.mm(torch.ones([data.shape[0], 1]), scalingFactor)
        data = data.mul(myNoise)
        return data

    def __str__(self):
        return "Scaling_std{}".format(self.std)


def get_transfroms_from_string_list(strlist):
    transfromlist = []
    for name in strlist:
        if name == "Shift":
            transfromlist.append(
                Shift(
                    interval=config.Shift_interval
                    if "Shift_interval" in dir(config)
                    else 20
                )
            )
        elif name == "GaussianNoise":
            transfromlist.append(
                TGaussianNoise(
                    scale=config.GaussianNoise_scale
                    if "GaussianNoise_scale" in dir(config)
                    else 0.005
                )
            )
        elif name == "ChannelResize":
            transfromlist.append(
                TChannelResize(
                    magnitude_range=config.ChannelResize_magnitude_range
                    if "ChannelResize_magnitude_range" in dir(config)
                    else [0.5, 2]
                )
            )
        elif name == "RandomResizedCrop":
            transfromlist.append(
                TRandomResizedCrop(
                    crop_ratio_range=config.RandomResizedCrop_crop_ratio_range
                    if "RandomResizedCrop_crop_ratio_range" in dir(config)
                    else [0.5, 1.0]
                )
            )
        elif name == "Verflip":
            transfromlist.append(Verflip())
        elif name == "Scaling":
            transfromlist.append(
                Scaling(std=config.Scaling_std if "Scaling_std" in dir(config) else 0.1)
            )
    return transfromlist


def apply_transformation(
    sig, transformlist, two_crop_transform, half_chance_data_augmentation=False
):
    def tsf(sig, transformlist, half_chance_data_augmentation):
        ans = sig.clone()
        for tran in transformlist:
            if (not half_chance_data_augmentation) or np.random.rand() > 0.5:
                ans = tran(ans)
        return ans

    sig = torch.tensor(sig.copy(), dtype=torch.float)
    x1 = tsf(sig, transformlist, half_chance_data_augmentation)
    x1 = x1.transpose(0, 1)
    if two_crop_transform:
        x2 = tsf(sig, transformlist, half_chance_data_augmentation)
        x2 = x2.transpose(0, 1)
        return [x1, x2]
    return x1


def add_channel9(x: np.ndarray, Ridxs: list) -> np.ndarray:
    if config.ch9_type == "Rdecay":
        if config.ch9_decay_type == "linear":
            # 仅限十秒的数据可用
            channel9 = np.zeros_like(x[0], dtype=np.float32)

            for Ridx in Ridxs:
                half_interval = int(
                    math.ceil(config.ch9_decay_Rtime_ms / (10000 / x.shape[1]) / 2)
                )
                for i in range(half_interval):
                    channel9[min(Ridx + i, x.shape[1] - 1)] = (
                        config.ch9_decay_init
                        - config.ch9_decay_init / half_interval * i
                    )
                    channel9[max(Ridx - i, 0)] = (
                        config.ch9_decay_init
                        - config.ch9_decay_init / half_interval * i
                    )
            x = np.concatenate([x, channel9[None, :]], axis=0)
        elif config.ch9_decay_type == "const":
            # 仅限十秒的数据可用
            channel9 = np.zeros_like(x[0], dtype=np.float32)

            for Ridx in Ridxs:
                half_interval = int(
                    math.ceil(config.ch9_decay_Rtime_ms / (10000 / x.shape[1]) / 2)
                )
                for i in range(half_interval):
                    channel9[min(Ridx + i, x.shape[1] - 1)] = config.ch9_decay_init
                    channel9[max(Ridx - i, 0)] = config.ch9_decay_init
            x = np.concatenate([x, channel9[None, :]], axis=0)
    if config.ch9_type == "STdecay":
        if config.ch9_decay_type == "const":
            channel9 = np.zeros_like(x[0], dtype=np.float32)
            # 仅限十秒的数据可用
            for Ridx in Ridxs:
                ST_lenth = int(
                    math.ceil(config.ch9_decay_STtime_ms / (10000 / x.shape[1]))
                )
                for i in range(ST_lenth):
                    channel9[min(Ridx + i, x.shape[1] - 1)] = config.ch9_decay_init
            x = np.concatenate([x, channel9[None, :]], axis=0)
    return x


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, data_path, train=True, two_crop_transform=False):
        super(ECGDataset, self).__init__()
        dd = torch.load(data_path)
        self.train = train
        self.data = dd["train"] if train else dd["val"]
        self.idx2name = dd["idx2name"]
        self.file2idx = dd["file2idx"]
        self.wc = 1.0 / np.log(dd["wc"])
        if "transforms" in dir(config):
            self.transforms = get_transfroms_from_string_list(config.transforms)
            self.half_chance_data_augmentation = (
                config.half_chance_data_augmentation
                if "half_chance_data_augmentation" in dir(config)
                else False
            )
        self.two_crop_transform = two_crop_transform

        if (
            "loss_function" in dir(config)
            and config.loss_function == "WeightSinglelabel"
        ):
            from data_process import file2index, name2index

            train_label_w = config.alldata_full_label  # 获得训练数据的多标签
            name2idx_w = name2index(config.all_arrythmia)  # 得到异常索引字典
            file2idx_w = file2index(train_label_w, name2idx_w)  # 得到数据异常索引字典
            self.file2idx_w = file2idx_w
            self.class_weight_dict = dict(
                pd.read_csv(
                    config.class_weight_path, encoding="utf-8", header=None
                ).values
            )

        if "loss_function" in dir(config) and config.loss_function == "MLB_SupConLoss":
            from data_process import file2index, name2index

            self.name2idx_full = name2index(config.all_arrythmia)
            self.idx2name_full = {idx: name for name, idx in self.name2idx_full.items()}
            self.file2idx_full = file2index(
                config.alldata_full_label, self.name2idx_full
            )

            self.error_rate_list = []
            for line in open(config.MLB_error_rate_path, "r", encoding="UTF-8"):
                line = line.strip()
                if line:
                    arr = line.split(",")
                    self.error_rate_list.append(float(arr[-1]))

        if config.model_name == "resnet34_cbam_ch9":
            self.Ridx_dict = load_Ridx_dict(config.Ridx_path)

    def __getitem__(self, index):
        fid = self.data[index % len(self.data)]
        file_path = os.path.join(config.train_dir, fid + ".npy")
        x = np.load(file_path)

        # 处理基线漂移
        if "del_drift" in dir(config) and config.del_drift:
            wn1 = 2 * config.del_drift_band_Hz / config.sampling_rate  # 只截取band_Hz以上的数据
            b, a = signal.butter(1, wn1, btype="high")
            x = signal.filtfilt(b, a, x)

        x = x.astype(np.float32)
        # x = x * config.inputUnit_uv / config.targetUnit_uv  # 符值转换
        if config.data_standardization:
            x = (x - np.mean(x)) / x.std()
        if config.model_name == "resnet34_cbam_ch9":
            Ridxs = self.Ridx_dict[fid]
            x = add_channel9(x, Ridxs)
        x = x.T  # 注意这次转置

        if "transforms" in dir(self):
            x = resample(x, config.target_point_num)
            if self.train:
                if (
                    "twice_data_augmentation" in dir(config)
                    and config.twice_data_augmentation
                ):
                    curtranforms = []
                    if index >= len(self.data):
                        curtranforms = self.transforms
                    x = apply_transformation(
                        x,
                        curtranforms,
                        self.two_crop_transform,
                        self.half_chance_data_augmentation,
                    )
                else:
                    x = apply_transformation(
                        x,
                        self.transforms,
                        self.two_crop_transform,
                        self.half_chance_data_augmentation,
                    )
            else:
                x = x.T
                x = torch.tensor(x, dtype=torch.float32)
                if self.two_crop_transform:
                    x = [x.clone(), x.clone()]
        else:
            x = transform(x, self.train)

        target = None
        # 为单标签多分类提供标量的loss，0 <= target < num_classes
        if "loss_function" in dir(config) and (
            config.loss_function == "WeightedCrossEntropyLoss"
            or config.loss_function == "MultiClassFocalLoss"
        ):
            target = self.file2idx[fid][0]
            target = torch.tensor(target, dtype=torch.long)
        else:
            target = np.zeros(config.num_classes)
            target[self.file2idx[fid]] = 1
            target = torch.tensor(target, dtype=torch.float32)

        if (
            "loss_function" in dir(config)
            and config.loss_function == "WeightSinglelabel"
        ):
            weight = -1
            for i in self.file2idx_w[fid]:
                if weight < self.class_weight_dict[i]:
                    weight = self.class_weight_dict[i]  # 取多标签中最大权重为当前权重
            weight = weight + 1
            target = np.append(target, weight)
            targetwithw = torch.tensor(target, dtype=torch.float32)
            return x, targetwithw
        elif (
            self.train
            and "loss_function" in dir(config)
            and config.loss_function == "MLB_SupConLoss"
        ):
            cur_error_rate = np.ones(1, dtype=np.float32)
            # 处理标签为空即'其他'的情况
            if len(self.file2idx_full[fid]) == 0:
                cur_error_rate[0] = self.error_rate_list[-1]
            else:
                cur_error_rate_list = []
                for idx in self.file2idx_full[fid]:
                    cur_error_rate_list.append(self.error_rate_list[idx])

                if config.MLB_multi_label_error_rate_mode == "aver":
                    cur_error_rate[0] = sum(cur_error_rate_list) / len(
                        self.file2idx_full[fid]
                    )
                elif config.MLB_multi_label_error_rate_mode == "max":
                    cur_error_rate[0] = max(cur_error_rate_list)

            target = torch.tensor(target, dtype=torch.float32)
            cur_error_rate = torch.tensor(cur_error_rate, dtype=torch.float32)
            return x, [target, cur_error_rate]

        return x, target

    def __len__(self):
        if "twice_data_augmentation" in dir(config) and config.twice_data_augmentation:
            return len(self.data) * 2
        return len(self.data)


if __name__ == "__main__":
    d = ECGDataset(config.train_data, two_crop_transform=True)
    print(d[0])
