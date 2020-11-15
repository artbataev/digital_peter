import torch
import torch.nn as nn
import torch.nn.utils.rnn as utils_rnn

from digital_peter.models.transf import PositionalEncoding


class RNNEncoder(nn.Module):
    def __init__(self, dropout=0.2, num_layers=2, rnn_type="GRU", hidden_size=256):
        super().__init__()
        self.rnn_dropout = nn.Dropout(dropout)
        rnn_type = getattr(nn, rnn_type)
        self.num_layers = num_layers
        assert num_layers >= 1
        self.rnn = rnn_type(input_size=512, hidden_size=hidden_size, bidirectional=True, dropout=dropout,
                            batch_first=False,
                            num_layers=num_layers)

    def forward(self, sequences, sequence_lengths):
        output = self.rnn_dropout(sequences)
        output = utils_rnn.pack_padded_sequence(output, sequence_lengths, batch_first=False)
        output = self.rnn(output)[0]
        output = utils_rnn.pad_packed_sequence(output)[0]
        return output, sequence_lengths  # LxBxC, B


class TransformerEncoder(nn.Module):
    def __init__(self, dropout=0.1, num_layers=2, num_heads=4, dim_model=128, dim_feedforward=256):
        super().__init__()
        self.pos_encoder = PositionalEncoding(dim_model, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(dim_model, num_heads, dim_feedforward=dim_feedforward,
                                                    dropout=dropout)
        encoder_norm = nn.LayerNorm(128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers, encoder_norm)

    def forward(self, sequences, sequence_lengths):
        batch_size = sequences.shape[1]
        device = sequences.get_device()
        padding_mask = (torch.arange(sequences.shape[0], dtype=torch.long, device=device
                                     ).unsqueeze(0).expand(batch_size, -1) >= sequence_lengths.unsqueeze(1))
        output = self.transformer_encoder(sequences, src_key_padding_mask=padding_mask)
        return output, sequence_lengths  # LxBxC, B
