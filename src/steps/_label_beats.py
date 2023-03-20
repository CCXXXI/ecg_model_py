import numpy as np
import torch
from numpy.typing import NDArray
from scipy import signal
from torch import Tensor
from utils import Beat
from utils import fs
from utils import Label
from utils import load_model


def _bsw(data: NDArray[float], band_hz: float) -> NDArray[float]:
    wn1 = 2 * band_hz / fs  # 只截取5hz以上的数据
    b: NDArray[float]
    a: NDArray[float]
    # noinspection PyTupleAssignmentBalance
    b, a = signal.butter(1, wn1, btype="high")
    return signal.filtfilt(b, a, data)


def _transform(sig: NDArray[float]) -> Tensor:
    sig = signal.resample(sig, 360).T
    return torch.tensor(sig.copy(), dtype=torch.float)


def label_beats(data: NDArray[float], beats: list[Beat], ori_fs: int) -> list[Beat]:
    """进行预测，获取标签"""
    half_len = int(0.75 * fs)

    data = signal.resample(data, len(data) * fs // ori_fs)
    data = _bsw(data, band_hz=0.5)

    batch_size = 64
    input_tensor: list[Tensor] = []
    input_beats: list[Beat] = []

    for idx, beat in enumerate(beats):
        if beat.position < half_len or beat.position >= data.shape[0] - half_len:
            beat.label = Label.未知
            continue

        x: NDArray[float] = data[beat.position - half_len : beat.position + half_len]
        x = np.reshape(x, (1, half_len * 2))
        x = (x - np.mean(x)) / np.std(x)
        x = x.T
        x_tensor: Tensor = _transform(x).unsqueeze(0)
        input_tensor.append(x_tensor)
        input_beats.append(beat)

        if len(input_tensor) % batch_size == 0 or idx == len(beats) - 1:
            x_tensor = torch.vstack(input_tensor)
            model = load_model("res_net.pt")
            output: Tensor = torch.softmax(model(x_tensor), dim=1).squeeze()

            # 修改维度
            y_pred: Tensor = torch.argmax(output, dim=1, keepdim=False)
            for i, pred in enumerate(y_pred):
                pred: Tensor
                pred_i: int = pred.item()
                beat = input_beats[i]
                beat.label = Label(pred_i)
            input_tensor = []
            input_beats = []

    return beats
