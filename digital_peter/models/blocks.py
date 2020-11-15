from typing import Callable

import torch.nn as nn


class LambdaModule(nn.Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.f = f

    def forward(self, *args):
        return self.f(*args)


class SequentialLinear(nn.Module):
    def __init__(self, num_inputs, num_outputs, pre_activation=True):
        super().__init__()
        modules = []
        if pre_activation:
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_inputs, num_outputs))
        self.model = nn.Sequential(*modules)

    def forward(self, sequences, sequence_lengths):
        return self.model(sequences), sequence_lengths
