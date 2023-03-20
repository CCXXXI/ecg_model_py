import numpy as np
import torch
from numpy.typing import NDArray

from steps import get_labelled_beats
from steps import get_r_peaks
from utils import Beat
from utils import set_models_path


def infer(data: NDArray[float], ori_fs: int) -> list[Beat]:
    positions, r_peaks = get_r_peaks(data, ori_fs)

    beats = [
        Beat(position=position, r_peak=r_peak)
        for position, r_peak in zip(positions, r_peaks)
    ]

    labelled_beats = get_labelled_beats(data, beats, ori_fs)

    return labelled_beats


def main() -> None:
    # The model output of "lead I.txt" seems incorrect.
    # See https://github.com/CCXXXI/ecg_models/commit/eb5904809e087ac575d41d9f5fe9d8f8ee044aa9.
    input_path = "../assets/ecg_data/assets/lead II.txt"

    set_models_path("../assets/ecg_models/models/")

    with torch.no_grad():
        labelled_beats = infer(np.loadtxt(input_path), 125)

    with open(
        "../assets/ecg_models/output/labelled_beats.txt", "w", encoding="utf-8"
    ) as f:
        print(*labelled_beats, sep="\n", file=f)


if __name__ == "__main__":
    main()
