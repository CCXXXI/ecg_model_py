from dataclasses import dataclass
from typing import Final

import torch


@dataclass
class Beat:
    position: int
    r_peak: int
    is_new: bool
    label: str = ""


fs: Final[int] = 240


_models_path: str


def set_models_path(path: str) -> None:
    global _models_path
    _models_path = path


def load_model(filename: str) -> torch.nn.Module:
    model = torch.jit.load(_models_path + filename)
    model.eval()
    return model
