import torch

from config import device
from models_ import resnet34_cbam_ch1

u_net: torch.nn.Module
res_net: torch.nn.Module


def load_models(path: str) -> None:
    global u_net
    u_net = torch.load(path + "240HZ_t+c_v2_best.pt", map_location=device)
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
