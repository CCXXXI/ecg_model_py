from typing import Final

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fs: Final[int] = 240
