from typing import Callable

import torch.nn as nn


class LambdaModule(nn.Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.f = f

    def forward(self, *args):
        return self.f(*args)
