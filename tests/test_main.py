from dataclasses import asdict
from json import load

import numpy as np
import torch
from main import infer
from utils import set_models_path


def test_infer():
    # set up
    input_path = "../assets/ecg_data/assets/lead II.txt"
    set_models_path("../assets/ecg_models/models/")

    # get actual
    with torch.no_grad():
        beats = infer(np.loadtxt(input_path), 125)
    actual = [asdict(b) for b in beats]

    # get expected
    with open("../assets/ecg_models/output/beats.json", encoding="utf-8") as f:
        expected = load(f)

    # compare
    assert actual == expected
