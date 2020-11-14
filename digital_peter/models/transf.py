import math

import torch
import torch.nn as nn

from digital_peter.models.blocks import LambdaModule


# from https://github.com/vlomme/OCR-transformer/blob/main/ocr.py
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderBase(nn.Module):
    def __init__(self, num_outputs, dropout=0.1, n_layers=2, n_head=4, dim_feedforward=512):
        super().__init__()
        # input: Bx3x128xL
        left_context = 19
        right_context = 19 + 4
        self.encoder = nn.Sequential(*[
            nn.ReplicationPad2d([left_context, right_context, 0, 0]),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=[1, 0]),  # L - 2
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(4, 2), stride=2),  # / 2
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=[1, 0]),  # -2
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(4, 2), stride=2),  # /2
            nn.Conv2d(128, 256, (3, 3), padding=[1, 0]),  # -2
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), padding=[1, 0]),  # -2
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ZeroPad2d([0, 0, 2, 1]),  # same padding for maxpool2d
            nn.MaxPool2d(kernel_size=(4, 1), padding=0),  # pool_4
            nn.Conv2d(256, 512, (3, 3), padding=[1, 0]),  # -2
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3, 3), padding=[1, 0]),  # -2
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ZeroPad2d([0, 0, 1, 2]),  # same padding for maxpool2d
            nn.MaxPool2d(kernel_size=(4, 1), padding=0),
            nn.Conv2d(512, 512, (2, 2)),  # 512x1x255 CxHxW # -1
            nn.ReLU(),
            LambdaModule(lambda x: x.squeeze(dim=2).permute(2, 0, 1)),  # LxBxC
        ])
        self.reduce_dim = nn.Linear(512, 128)

        self.pos_encoder = PositionalEncoding(128, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(128, n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers, encoder_norm)
        self.final = nn.Linear(128, num_outputs)

    def forward(self, images, image_lengths):
        output = self.encoder(images)  # LxBxC, L//4
        output = self.reduce_dim(output)
        batch_size = images.shape[0]
        device = output.get_device()
        src_lengths = image_lengths // 4
        padding_mask = (torch.arange(output.shape[0], dtype=torch.long, device=device
                                     ).unsqueeze(0).expand(batch_size, -1) >= src_lengths.unsqueeze(1))
        # padding_mask = padding_mask.transpose(0, 1)
        output = self.transformer_encoder(output, src_key_padding_mask=padding_mask)
        logits = self.final(output)
        return logits  # LxBxC
