import torch.nn as nn
import torch.nn.functional as F

from digital_peter.models.base import BaselineModelBnAllNoTimePad
from digital_peter.models.blocks import LambdaModule
from digital_peter.models.blocks import SequentialLinear
from digital_peter.models.conv import ConvExtractor, ResnetExtractor
from digital_peter.models.rnn import RNNEncoder


# basic model
def base__drop02__gru_2x256(num_outputs) -> nn.Module:
    model = BaselineModelBnAllNoTimePad(num_outputs=num_outputs, dropout=0.2, n_rnn=2, rnn_type="GRU", rnn_dim=256)
    return model


def base__drop02__gru_4x128(num_outputs) -> nn.Module:
    model = BaselineModelBnAllNoTimePad(num_outputs=num_outputs, dropout=0.2, n_rnn=4, rnn_type="GRU", rnn_dim=128)
    return model


def base__drop02__lstm_2x256(num_outputs) -> nn.Module:
    model = BaselineModelBnAllNoTimePad(num_outputs=num_outputs, dropout=0.2, n_rnn=2, rnn_type="LSTM", rnn_dim=256)
    return model


def base__drop02__lstm_4x128(num_outputs) -> nn.Module:
    model = BaselineModelBnAllNoTimePad(num_outputs=num_outputs, dropout=0.2, n_rnn=4, rnn_type="LSTM", rnn_dim=128)
    return model


# basic model modularized
def conv__gru_2x256_drop02(num_outputs) -> nn.Module:
    model = nn.Sequential(*[
        ConvExtractor(),
        LambdaModule(lambda seq, seq_len: (F.relu(seq), seq_len)),
        RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=2, hidden_size=256, input_size=512),
        SequentialLinear(256 * 2, num_outputs, pre_activation=True)
    ])
    return model


def resnet__gru_2x256_drop02(num_outputs) -> nn.Module:
    model = nn.Sequential(*[
        ResnetExtractor(),
        LambdaModule(lambda seq, seq_len: (F.relu(seq), seq_len)),
        RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=2, hidden_size=256, input_size=512),
        SequentialLinear(256 * 2, num_outputs, pre_activation=True)
    ])
    return model
