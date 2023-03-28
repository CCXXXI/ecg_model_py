from json import dump
from json import load

import numpy as np
import torch
from numpy.typing import NDArray

from steps import get_beats
from steps import label_beats
from utils import Beat
from utils import set_models_path


def infer(data: NDArray[float], ori_fs: int) -> list[Beat]:
    beats = get_beats(data, ori_fs)
    labelled_beats = label_beats(data, beats, ori_fs)
    return labelled_beats


def get_input() -> NDArray[float]:
    with open("../assets/ecg_data/assets/data.json") as f:
        points = load(f)
    # The model output of lead-I seems incorrect.
    # See https://github.com/CCXXXI/ecg_models/commit/eb5904809e087ac575d41d9f5fe9d8f8ee044aa9.
    return np.array([p["leadII"] for p in points])


def main() -> None:
    set_models_path("../assets/ecg_models/models/")

    with torch.no_grad():
        beats = infer(get_input(), 125)

    # human-readable output
    with open("../assets/ecg_models/output/beats.txt", "w", encoding="utf-8") as f:
        print(*beats, sep="\n", file=f)

    # machine-readable output
    with open("../assets/ecg_models/output/beats.json", "w", encoding="utf-8") as f:
        dump([b.to_dict() for b in beats], f, indent=2)


if __name__ == "__main__":
    main()
