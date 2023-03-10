import numpy as np
import torch
from numpy.typing import NDArray
from steps import get_checked_beats
from steps import get_labelled_beats
from steps import get_r_peaks
from steps import get_report
from utils import Beat
from utils import set_models_path


def infer(data: NDArray[float], ori_fs: int) -> tuple[list[Beat], str]:
    beats, r_peaks = get_r_peaks(data, ori_fs)
    checked_beats = get_checked_beats(beats, r_peaks)
    labelled_beats = get_labelled_beats(data, checked_beats, ori_fs)
    report = get_report(labelled_beats)

    return labelled_beats, report


def main() -> None:
    # The model output of "lead I.txt" seems incorrect.
    # See https://github.com/CCXXXI/ecg_models/commit/eb5904809e087ac575d41d9f5fe9d8f8ee044aa9.
    input_path = "../assets/ecg_data/assets/lead II.txt"

    set_models_path("../assets/ecg_models/models/")

    with torch.no_grad():
        labelled_beats, report = infer(np.loadtxt(input_path), 125)

    with open(
        "../assets/ecg_models/output/labelled_beats.txt", "w", encoding="utf-8"
    ) as f:
        print(*labelled_beats, sep="\n", file=f)
    with open("../assets/ecg_models/output/report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
