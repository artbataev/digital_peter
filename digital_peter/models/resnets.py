import torch.nn as nn
import torch.nn.utils.rnn as utils_rnn

from digital_peter.models.blocks import LambdaModule


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride_h=1, time_dilation=1):
        super().__init__()
        layers = []
        if stride_h != 1:
            layers.append(nn.MaxPool2d(kernel_size=(stride_h, 1)))
        layers += [
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 0), dilation=(1, time_dilation)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 0)),
        ]
        self._strip = 1 + time_dilation

        self.net = nn.Sequential(*layers)
        projection = []
        if stride_h != 1:
            projection.append(nn.MaxPool2d((stride_h, 1)))
        if in_channels != out_channels:
            projection.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.projection = nn.Sequential(*projection) if projection else nn.Identity()

    def forward(self, x):
        # x: BxCxHxW
        x_transformed = self.projection(x[..., self._strip:-self._strip])
        output = self.net(x)
        output += x_transformed
        return output


class BaselineResnetIbm1(nn.Module):
    """
    See https://arxiv.org/abs/1703.02136
    """

    def __init__(self, num_outputs, dropout=0.2, n_rnn=2, rnn_type="GRU"):
        super().__init__()
        self.num_outputs = num_outputs
        left_context = 62
        right_context = 62
        self.encoder = nn.Sequential(*[
            nn.ReplicationPad2d([left_context, right_context, 0, 0]),
            nn.Conv2d(3, 64, kernel_size=(5, 5), padding=(2, 0)),  # 128, time -4
            nn.MaxPool2d(kernel_size=(2, 2)),  # to 64, time / 2 // (24+2) * 2
            ResBlock(64, 64, stride_h=2),  # to 32, time -4 // 24+2
            nn.MaxPool2d(kernel_size=(2, 2)),  # to 16, time/2 // 12*2
            ResBlock(64, 128, stride_h=2),  # to 8, time -4
            ResBlock(128, 128),  # 8, time -4
            ResBlock(128, 256, stride_h=2),  # to 4, time -4
            ResBlock(256, 256),  # 4, time -4
            ResBlock(256, 512, stride_h=2),  # to 2, time -4
            ResBlock(512, 512, stride_h=2),  # to 1, time -4
            ResBlock(512, 512),  # 1, time -4
            LambdaModule(lambda x: x.squeeze(2)),  # BxCx1xL -> BxCxL
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            LambdaModule(lambda x: x.permute(2, 0, 1)),  # LxBxC
        ])
        self.rnn_dropout = nn.Dropout(dropout)
        rnn_type = getattr(nn, rnn_type)
        self.n_rnn = n_rnn
        if n_rnn > 0:
            self.rnn = rnn_type(input_size=512, hidden_size=256, bidirectional=True, dropout=dropout, batch_first=False,
                                num_layers=n_rnn)
        else:
            self.rnn = nn.Identity()
        self.final = nn.Linear(512, num_outputs)

    def forward(self, images, image_lengths):
        output = self.encoder(images)  # LxBxC
        output = self.rnn_dropout(output)
        if self.n_rnn:
            output = utils_rnn.pack_padded_sequence(output, image_lengths // 4, batch_first=False)
            output = self.rnn(output)[0]
            output = utils_rnn.pad_packed_sequence(output)[0]
        logits = self.final(output)
        return logits  # LxBxC
