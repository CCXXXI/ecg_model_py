from pprint import pp

import numpy as np
import torch
from numpy.typing import NDArray

from models import CBR_1D, Unet_1D
from steps import get_report, get_checked_beats, get_labelled_beats, get_r_peaks
from utils.data_types import Beat
from utils.models import load_models

# These classes must be imported, otherwise the model cannot be loaded.
# noinspection PyStatementEffect
CBR_1D, Unet_1D


def infer(data: NDArray[float], ori_fs: int) -> tuple[dict[str, int], list[Beat], str]:
    beats: list[int]
    r_peaks: list[int]
    beats, r_peaks = get_r_peaks(data, ori_fs)

    checked_beats: list[Beat] = get_checked_beats(beats, r_peaks)

    labelled_beats: list[Beat]
    label_cnt: dict[str, int]
    labelled_beats, label_cnt = get_labelled_beats(data, checked_beats, ori_fs)

    report: str = get_report(labelled_beats)

    return label_cnt, labelled_beats, report


def main():
    input_path = "../assets/input/107_leadII_10min.txt"

    load_models("../assets/")

    with torch.no_grad():
        label_cnt, labelled_beats, report = infer(np.loadtxt(input_path), 250)

    with open("../assets/output/labelled_beats.txt", "w", encoding="utf-8") as f:
        print(*labelled_beats, sep="\n", file=f)
    with open("../assets/output/label_cnt.txt", "w", encoding="utf-8") as f:
        pp(label_cnt, stream=f)
    with open("../assets/output/report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
