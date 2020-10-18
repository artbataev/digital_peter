import torch.nn as nn
import torch.nn.utils.rnn as utils_rnn


class BaselineModel(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        # input: Bx3x128xL
        self.encoder = nn.Sequential(*[
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2), stride=2),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.ZeroPad2d([0, 0, 2, 1]),  # same padding for maxpool2d
            nn.MaxPool2d(kernel_size=(4, 1), padding=0),  # pool_4
            nn.Conv2d(256, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ZeroPad2d([0, 0, 1, 2]),  # same padding for maxpool2d
            nn.MaxPool2d(kernel_size=(4, 1), padding=0),
            nn.Conv2d(512, 512, (2, 2)),  # 512x1x255 CxHxW
            nn.ReLU(),
        ])

        self.rnn_dropout = nn.Dropout(0.2)
        self.rnn = nn.GRU(input_size=512, hidden_size=256, bidirectional=True, dropout=0.2, batch_first=False,
                          num_layers=2)
        self.final = nn.Linear(512, num_outputs)

    def forward(self, images, image_lengths):
        output = self.encoder(images)  # Bx512x1x(L/4-1)
        output = output.squeeze(dim=2).permute(2, 0, 1)  # LxBxC
        output = self.rnn_dropout(output)
        output = utils_rnn.pack_padded_sequence(output, image_lengths // 4 - 1, batch_first=False)
        output = self.rnn(output)[0]
        output = utils_rnn.pad_packed_sequence(output)[0]
        logits = self.final(output)
        return logits  # LxBxC
