import torch.nn as nn

from digital_peter.models.base import BaselineModelBnAllNoTimePad


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
