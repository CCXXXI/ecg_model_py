import numpy as np
from numpy.typing import NDArray

from utils import Beat, fs


def get_checked_beats(beats: list[int], r_peaks: list[int]) -> list[Beat]:
    """补充心拍"""
    assert len(beats) == len(r_peaks), "提取出的心拍数量与 R 波数量不同"

    checked_beats: list[Beat] = [
        Beat(position=beats[0], r_peak=r_peaks[0], is_new=False)
    ]
    limit = 2 * 1.5 * fs
    beats_diff: NDArray[int] = np.diff(beats)
    num = 0
    for index, diff in enumerate(beats_diff):
        if diff >= limit:
            start = beats[index]
            cur = start
            end = beats[index + 1]
            while (end - cur) >= limit:
                new_beat = cur + int(limit / 2)
                checked_beats.append(Beat(position=new_beat, r_peak=-1, is_new=True))
                cur = new_beat
                num += 1
            checked_beats.append(
                Beat(position=beats[index + 1], r_peak=r_peaks[index + 1], is_new=False)
            )
        else:
            checked_beats.append(
                Beat(position=beats[index + 1], r_peak=r_peaks[index + 1], is_new=False)
            )
    return checked_beats
