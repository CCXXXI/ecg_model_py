from dataclasses import dataclass
from typing import Final

import torch
from enum import Enum


# noinspection NonAsciiCharacters
class Label(Enum):
    未知 = -1
    窦性心律 = 0
    房性早搏 = 1
    心房扑动 = 2
    心房颤动 = 3
    室性早搏 = 4
    阵发性室上性心动过速 = 5
    心室预激 = 6
    室扑室颤 = 7
    房室传导阻滞 = 8
    噪声 = 9


@dataclass
class Beat:
    position: int
    r_peak: int
    is_new: bool
    label: Label = Label.未知


fs: Final[int] = 240


_models_path: str


def set_models_path(path: str) -> None:
    global _models_path
    _models_path = path


def load_model(filename: str) -> torch.nn.Module:
    model = torch.jit.load(_models_path + filename)
    model.eval()
    return model
