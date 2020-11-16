"""Not very good models, just for history"""
import torch.nn as nn
import torch.nn.functional as F

from digital_peter.models.base import BaselineModelBnAllNoTimePad
from digital_peter.models.blocks import LambdaModule, SequentialLinear, SequentialSequential
from digital_peter.models.conv import ConvExtractor, ResnetExtractor
from digital_peter.models.rnn import RNNEncoder, TransformerEncoder


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


def conv_instnorm__gru_2x256_drop02(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ConvExtractor(norm=nn.InstanceNorm2d),
        LambdaModule(lambda seq, seq_len: (F.relu(seq), seq_len)),
        RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=2, hidden_size=256, input_size=512),
        SequentialLinear(256 * 2, num_outputs, pre_activation=True)
    ])
    return model


def conv_pure(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ConvExtractor(),
        SequentialLinear(512, num_outputs, pre_activation=True)
    ])
    return model


def conv_instnorm_pure(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ConvExtractor(norm=nn.InstanceNorm2d),
        SequentialLinear(512, num_outputs, pre_activation=True)
    ])
    return model


def conv__gru_2x256_drop02__transf_2x4x128x256_drop01(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ConvExtractor(),
        LambdaModule(lambda seq, seq_len: (F.relu(seq), seq_len)),
        RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=2, hidden_size=256, input_size=512),
        SequentialLinear(256 * 2, 128, pre_activation=True),
        TransformerEncoder(dropout=0.1, num_layers=2, num_heads=4, dim_model=128, dim_feedforward=256),
        SequentialLinear(128, num_outputs, pre_activation=True)
    ])
    return model


def conv__transf_2x4x128x256_drop01__gru_2x256_drop02(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ConvExtractor(),
        SequentialLinear(512, 128, pre_activation=True),
        TransformerEncoder(dropout=0.1, num_layers=2, num_heads=4, dim_model=128, dim_feedforward=256),
        RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=2, hidden_size=256, input_size=128),
        SequentialLinear(256 * 2, num_outputs, pre_activation=True)
    ])
    return model


def conv__gru_2x480_drop02(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ConvExtractor(),
        LambdaModule(lambda seq, seq_len: (F.relu(seq), seq_len)),
        RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=2, hidden_size=480, input_size=512),
        SequentialLinear(480 * 2, num_outputs, pre_activation=True)
    ])
    return model


def conv__gru_3x368_drop02(num_outputs) -> nn.Module:
    model = SequentialSequential(*[
        ConvExtractor(),
        LambdaModule(lambda seq, seq_len: (F.relu(seq), seq_len)),
        RNNEncoder(dropout=0.2, rnn_type="GRU", num_layers=3, hidden_size=368, input_size=512),
        SequentialLinear(368 * 2, num_outputs, pre_activation=True)
    ])
    return model
