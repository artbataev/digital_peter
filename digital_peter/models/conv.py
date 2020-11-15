import torch.nn as nn

from digital_peter.models.base import LambdaModule
from digital_peter.models.resnets import ResBlock


class ConvExtractor(nn.Module):
    def __init__(self, norm=nn.BatchNorm2d):
        super().__init__()
        # input: Bx3x128xL
        left_context = 19
        right_context = 19 + 4
        self.reduction_fn = lambda x: x // 4
        self.encoder = nn.Sequential(*[
            nn.ReplicationPad2d([left_context, right_context, 0, 0]),
            norm(3),
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=[1, 0]),  # L - 2
            nn.ReLU(),
            norm(64),
            nn.MaxPool2d(kernel_size=(4, 2), stride=2),  # / 2
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=[1, 0]),  # -2
            nn.ReLU(),
            norm(128),
            nn.MaxPool2d(kernel_size=(4, 2), stride=2),  # /2
            nn.Conv2d(128, 256, (3, 3), padding=[1, 0]),  # -2
            nn.ReLU(),
            norm(256),
            nn.Conv2d(256, 256, (3, 3), padding=[1, 0]),  # -2
            nn.ReLU(),
            norm(256),
            nn.ZeroPad2d([0, 0, 2, 1]),  # same padding for maxpool2d
            nn.MaxPool2d(kernel_size=(4, 1), padding=0),  # pool_4
            nn.Conv2d(256, 512, (3, 3), padding=[1, 0]),  # -2
            nn.ReLU(),
            norm(512),
            nn.Conv2d(512, 512, (3, 3), padding=[1, 0]),  # -2
            nn.ReLU(),
            norm(512),
            nn.ZeroPad2d([0, 0, 1, 2]),  # same padding for maxpool2d
            nn.MaxPool2d(kernel_size=(4, 1), padding=0),
            nn.Conv2d(512, 512, (2, 2)),  # 512x1x255 CxHxW # -1
            LambdaModule(lambda x: x.squeeze(dim=2).permute(2, 0, 1)),  # LxBxC
        ])

    def forward(self, images, image_lengths):
        output = self.encoder(images)  # LxBxC, L//4
        return output, self.reduction_fn(image_lengths)  # LxBxC, B


class ResnetExtractor(nn.Module):
    """
    See https://arxiv.org/abs/1703.02136
    """

    def __init__(self):
        super().__init__()
        left_context = 62
        right_context = 62
        self.reduction_fn = lambda x: x // 4
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
            LambdaModule(lambda x: x.permute(2, 0, 1)),  # LxBxC
        ])

    def forward(self, images, image_lengths):
        output = self.encoder(images)  # LxBxC, L//4
        return output, self.reduction_fn(image_lengths)  # LxBxC, B
