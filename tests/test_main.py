from json import load

import torch
from main import get_input
from main import infer
from utils import set_models_path


def test_infer():
    # set up
    set_models_path("../assets/ecg_models/models/")

    # get actual
    with torch.no_grad():
        beats = infer(get_input(), 125)
    actual = [b.to_dict() for b in beats]

    # get expected
    with open("../assets/ecg_models/output/beats.json", encoding="utf-8") as f:
        expected = load(f)

    # compare
    assert actual == expected
