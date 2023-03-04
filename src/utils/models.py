import torch

from models import resnet34_cbam_ch1
from .config import device

u_net: torch.nn.Module
res_net: torch.nn.Module


def load_models(path: str) -> None:
    global u_net
    u_net = torch.jit.load(path + "u_net.pt", map_location=device)
    u_net.eval()

    global res_net
    res_net = resnet34_cbam_ch1(num_classes=10)
    res_net.load_state_dict(
        torch.load(
            path + "best_w.pth",
            map_location="cpu",
        )["state_dict"]
    )
    res_net = res_net.to(device)
    res_net.eval()
