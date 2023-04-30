from dataclasses import dataclass
from enum import Enum
from typing import Final

import torch


class Label(Enum):
    sinus_rhythm = 0
    """窦性心律 Sinus rhythm"""

    atrial_premature_beat = 1
    """房性早搏 Atrial premature beat"""

    atrial_flutter = 2
    """心房扑动 Atrial flutter"""

    atrial_fibrillation = 3
    """心房颤动 Atrial fibrillation"""

    ventricular_premature_beat = 4
    """室性早搏 Ventricular premature beat"""

    paroxysmal_supra_ventricular_tachycardia = 5
    """阵发性室上性心动过速 Paroxysmal supra-ventricular tachycardia"""

    ventricular_pre_excitation = 6
    """心室预激 Ventricular pre-excitation"""

    ventricular_flutter_and_fibrillation = 7
    """室扑室颤 Ventricular flutter and fibrillation"""

    atrioventricular_block = 8
    """房室传导阻滞 Atrioventricular block"""

    noise = 9
    """噪声 Noise"""

    unknown = 10
    """未知 Unknown"""


@dataclass
class Beat:
    position: int
    label: Label

    def to_dict(self):
        return {
            "millisecondsSinceStart": self.position * 1000 // fs,
            "label": self.label.value,
        }


fs: Final[int] = 240

_models_path: str


def set_models_path(path: str) -> None:
    global _models_path
    _models_path = path


def load_model(filename: str) -> torch.nn.Module:
    model = torch.jit.load(_models_path + filename)
    model.eval()
    return model
