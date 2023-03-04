from dataclasses import dataclass


@dataclass
class Beat:
    position: int
    r_peak: int
    is_new: bool
    label: str = ""
