import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        # input: Bx3x128,1024
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

        self.rnn = nn.Sequential(*[
            nn.Dropout(0.2),
            nn.GRU(input_size=512, hidden_size=256, bidirectional=True, dropout=0.2, batch_first=True, num_layers=2)
        ])
        self.final = nn.Linear(512, num_outputs)

    def forward(self, x):
        output = self.encoder(x)  # Bx512x1x255
        output = output.squeeze(dim=2).transpose(1, 2)  # BxLxC
        output = self.rnn(output)[0]
        logits = self.final(output)
        return logits
