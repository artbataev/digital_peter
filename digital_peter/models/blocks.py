from typing import Callable, List

import torch.nn as nn


class LambdaModule(nn.Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.f = f

    def forward(self, *args):
        return self.f(*args)


class SequentialSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            inputs = module(*inputs)
        return inputs


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


class MixtureModel(nn.Module):
    def __init__(self, models: List[nn.Module], weights: List[float]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights
        assert len(models) == len(weights)

    def forward(self, images, image_lengths):
        logits_mix = None
        logits_lengths = None
        for model, weight in zip(self.models, self.weights):
            logits, logits_lengths = model(images, image_lengths)
            if logits_mix is None:
                logits_mix = logits * weight
            else:
                logits_mix += logits * weight
        return logits_mix, logits_lengths
