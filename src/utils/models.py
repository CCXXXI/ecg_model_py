import torch

from .config import device

u_net: torch.nn.Module
res_net: torch.nn.Module


def load_models(path: str) -> None:
    global u_net
    u_net = torch.jit.load(path + "u_net.pt", map_location=device)
    u_net.eval()

    global res_net
    res_net = torch.jit.load(path + "res_net.pt", map_location=device)
    res_net.eval()
