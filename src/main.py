import numpy as np
import torch
from numpy.typing import NDArray

from steps import get_report, get_checked_beats, get_labelled_beats, get_r_peaks
from utils import Beat, set_models_path


def infer(data: NDArray[float], ori_fs: int) -> tuple[list[Beat], str]:
    beats, r_peaks = get_r_peaks(data, ori_fs)
    checked_beats = get_checked_beats(beats, r_peaks)
    labelled_beats = get_labelled_beats(data, checked_beats, ori_fs)
    report = get_report(labelled_beats)

    return labelled_beats, report


def main() -> None:
    input_path = "../assets/input/107_leadII_10min.txt"

    set_models_path("../assets/models/")

    with torch.no_grad():
        labelled_beats, report = infer(np.loadtxt(input_path), 250)

    with open("../assets/output/labelled_beats.txt", "w", encoding="utf-8") as f:
        print(*labelled_beats, sep="\n", file=f)
    with open("../assets/output/report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
