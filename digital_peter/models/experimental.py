import torch.nn as nn
import torch.nn.functional as F

from digital_peter.models.blocks import LambdaModule, SequentialLinear, SequentialSequential
from digital_peter.models.conv import ConvExtractor, ResnetExtractor
from digital_peter.models.rnn import RNNEncoder


# basic model modularized
def conv__gru_2x256_drop02(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ConvExtractor(),
        LambdaModule(lambda seq, seq_len: (F.relu(seq), seq_len)),
        RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=2, hidden_size=256, input_size=512),
        SequentialLinear(256 * 2, num_outputs, pre_activation=True)
    ])
    return model


# also good model
def conv__gru_2x368_drop02(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ConvExtractor(),
        LambdaModule(lambda seq, seq_len: (F.relu(seq), seq_len)),
        RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=2, hidden_size=368, input_size=512),
        SequentialLinear(368 * 2, num_outputs, pre_activation=True)
    ])
    return model


def resnet__gru_2x256_drop02(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ResnetExtractor(),
        LambdaModule(lambda seq, seq_len: (F.relu(seq), seq_len)),
        RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=2, hidden_size=256, input_size=512),
        SequentialLinear(256 * 2, num_outputs, pre_activation=True)
    ])
    return model


def resnet_pure(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ResnetExtractor(),
        SequentialLinear(512, num_outputs, pre_activation=True)
    ])
    return model

# def convhires__gru_2x256_drop02(num_outputs) -> nn.Module:
#     model = SequentialSequential(*[
#         ConvExtractor(),
#         LambdaModule(lambda seq, seq_len: (F.relu(seq), seq_len)),
#         RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=2, hidden_size=256, input_size=512),
#         SequentialLinear(256 * 2, num_outputs, pre_activation=True)
#     ])
#     return model
