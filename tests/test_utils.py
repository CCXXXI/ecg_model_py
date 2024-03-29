from pytest import mark
from utils import load_model
from utils import set_models_path


@mark.parametrize("filename", ["u_net.pt", "res_net.pt"])
def test_load_model(filename: str):
    set_models_path("../assets/ecg_models/models/")
    model = load_model(filename)
    assert not model.training
