import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, filters, number_of_channels):
        super(InceptionModule, self).__init__()

        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=number_of_channels, out_channels=filters[0],
                      kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=number_of_channels, out_channels=filters[1][0],
                      kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=filters[1][0], out_channels=filters[1][1],
                      kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=number_of_channels, out_channels=filters[2][0],
                      kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=filters[2][0], out_channels=filters[2][1],
                      kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=number_of_channels, out_channels=filters[3],
                      kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        # bibliografie: https://www.kaggle.com/code/luckscylla/googlenet-implementation/script

        path1 = self.path1(x)
        path2 = self.path2(x)
        path3 = self.path3(x)
        path4 = self.path4(x)

        return torch.cat([path1, path2, path3, path4], 1)


class AuxiliaryModule(nn.Module):

    def __init__(self, number_of_channels):
        super(AuxiliaryModule, self).__init__()

        self.path = nn.Sequential(
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Conv2d(in_channels=number_of_channels, out_channels=128,
                      kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=3200, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        layer = self.path(x)

        return layer


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=22, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionModule(filters=[64, (96, 128), (16, 32), 32], number_of_channels=192),
            InceptionModule(filters=[128, (128, 192), (32, 96), 64],
                            number_of_channels=64 + 128 + 32 + 32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionModule(filters=[192, (96, 208), (16, 48), 64],
                            number_of_channels=128 + 192 + 96 + 64)
        )

        self.auxiliary1 = AuxiliaryModule(number_of_channels=192 + 208 + 48 + 64)

        self.main_after_aux1 = nn.Sequential(
            InceptionModule(filters=[160, (112, 224), (24, 64), 64],
                            number_of_channels=192 + 208 + 48 + 64),
            InceptionModule(filters=[128, (128, 256), (24, 64), 64],
                            number_of_channels=160 + 224 + 64 + 64),
            InceptionModule(filters=[112, (144, 288), (32, 64), 64],
                            number_of_channels=128 + 256 + 64 + 64)
        )

        self.auxiliary2 = AuxiliaryModule(number_of_channels=112 + 288 + 64 + 64)

        self.main_after_aux2 = nn.Sequential(
            InceptionModule(filters=[256, (160, 320), (32, 128), 128],
                            number_of_channels=112 + 288 + 64 + 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionModule(filters=[256, (160, 320), (32, 128), 128],
                            number_of_channels=256 + 320 + 128 + 128),
            InceptionModule(filters=[384, (192, 384), (48, 128), 128],
                            number_of_channels=256 + 320 + 128 + 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=256)
        )

        self.linear2 = nn.Linear(in_features=256, out_features=4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        layer = self.main(x)

        aux1 = self.auxiliary1(layer)

        layer = self.main_after_aux1(layer)

        aux2 = self.auxiliary2(layer)

        layer = self.main_after_aux2(layer)

        main = self.linear2(layer)
        main = self.softmax(main)

        return main, aux1, aux2