from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GoogLeNet(nn.Module):

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        blocks = [BasicConv2d, Inception, InceptionAux]
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.conv1 = conv_block(in_channels=5, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = conv_block(in_channels=64, out_channels= 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128,
                                           ch5x5red=16, ch5x5=32, pool_proj=32)
        self.inception3b = inception_block(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192,
                                           ch5x5red=32, ch5x5=96, pool_proj=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208,
                                           ch5x5red=16, ch5x5=48, pool_proj=64)
        self.inception4b = inception_block(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224,
                                           ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4c = inception_block(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256,
                                           ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4d = inception_block(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288,
                                           ch5x5red=32, ch5x5=64, pool_proj=64)
        self.inception4e = inception_block(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320,
                                           ch5x5red=32, ch5x5=128, pool_proj=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320,
                                           ch5x5red=32, ch5x5=128, pool_proj=128)
        self.inception5b = inception_block(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384,
                                           ch5x5red=48, ch5x5=128, pool_proj=128)

        self.aux1 = inception_aux_block(in_channels=512, num_classes=num_classes)
        self.aux2 = inception_aux_block(in_channels=528, num_classes=num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        # N x 4
        return x, aux2, aux1


class Inception(nn.Module):
    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3: int,
                 ch5x5red: int, ch5x5: int, pool_proj: int) -> None:
        super().__init__()
        conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 4 (num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
