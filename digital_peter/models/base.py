import torch.nn as nn
import torch.nn.utils.rnn as utils_rnn
import torchvision as tv

from digital_peter.models.blocks import LambdaModule


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
        return logits, image_lengths // 4 - 1  # LxBxC, B


class BaselineModelBnFirst(BaselineModel):
    def __init__(self, num_outputs):
        super().__init__(num_outputs)
        self.bn1 = nn.BatchNorm2d(3)

    def forward(self, images, image_lengths):
        images = self.bn1(images)
        return super().forward(images, image_lengths)


class BaselineModelBnAll(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        # input: Bx3x128xL
        self.encoder = nn.Sequential(*[
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(4, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(4, 2), stride=2),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
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
        return logits, image_lengths // 4 - 1  # LxBxC, B


class BaselineModelBnAllNoTimePad(nn.Module):
    def __init__(self, num_outputs, dropout=0.2, n_rnn=2, rnn_type="GRU", rnn_dim=256):
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

        self.rnn_dropout = nn.Dropout(dropout)
        rnn_type = getattr(nn, rnn_type)
        self.n_rnn = n_rnn
        if n_rnn > 0:
            self.rnn = rnn_type(input_size=512, hidden_size=rnn_dim, bidirectional=True, dropout=dropout,
                                batch_first=False,
                                num_layers=n_rnn)
        else:
            self.rnn = nn.Identity()
        self.final = nn.Linear(rnn_dim * 2, num_outputs)

    def forward(self, images, image_lengths):
        output = self.encoder(images)  # LxBxC, L//4
        output = self.rnn_dropout(output)
        if self.n_rnn:
            output = utils_rnn.pack_padded_sequence(output, image_lengths // 4, batch_first=False)
            output = self.rnn(output)[0]
            output = utils_rnn.pad_packed_sequence(output)[0]
        logits = self.final(output)
        return logits, image_lengths // 4  # LxBxC, B


class BaselineModelResNet1(nn.Module):
    def __init__(self, num_outputs, dropout=0.2, n_rnn=2, rnn_type="GRU"):
        super().__init__()
        # input: Bx3x128xL
        # left_context = 19
        # right_context = 19 + 4
        self.resnet = tv.models.resnet._resnet('resnet18', tv.models.resnet.Bottleneck, [2, 2, 2, 2], False, False,
                                               replace_stride_with_dilation=[True, True, True])
        self.resnet.fc = nn.Identity()
        self.resnet.avgpool = nn.Identity()
        self.fc = nn.Conv2d(2048, 16, 1)

        self.rnn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        rnn_type = getattr(nn, rnn_type)
        self.n_rnn = n_rnn
        if n_rnn > 0:
            self.rnn = rnn_type(input_size=512, hidden_size=256, bidirectional=True, dropout=dropout, batch_first=False,
                                num_layers=n_rnn)
        else:
            self.rnn = nn.Identity()
        self.final = nn.Linear(512, num_outputs)

    def forward(self, images, image_lengths):
        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)  # LxBx512, L//4

        output = self.rnn_dropout(x)
        if self.n_rnn:
            output = utils_rnn.pack_padded_sequence(output, image_lengths // 4, batch_first=False)
            output = self.rnn(output)[0]
            output = utils_rnn.pad_packed_sequence(output)[0]
        logits = self.final(output)
        return logits, image_lengths // 4  # LxBxC, B
