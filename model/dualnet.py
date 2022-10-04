import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BOXNN(nn.Module):
    def __init__(self, features=3) -> None:
        super().__init__()

        self.features = features

        alpha = 3e-2

        self.box_conv1 = nn.Conv3d(1, 64, (3, 3, 3), padding=1)
        self.box_conv2 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.box_conv3 = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.box_conv4 = nn.Conv3d(256, 256, (3, 3, 3), padding=1)
        self.box_conv5 = nn.Conv3d(256, 512, (3, 3, 3), padding=1)
        self.box_conv6 = nn.Conv3d(512, 512, (3, 3, 3), padding=1)

        self.box_bn3d64 = nn.BatchNorm3d(64)
        self.box_bn3d128 = nn.BatchNorm3d(128)
        self.box_bn3d256 = nn.BatchNorm3d(256)
        self.box_bn3d512 = nn.BatchNorm3d(512)
        self.box_bn1d2048 = nn.BatchNorm1d(2048)

        self.box_avgp1 = nn.AvgPool3d((2, 1, 1))
        self.box_maxp1 = nn.MaxPool3d((1, 2, 2))
        self.box_maxp2 = nn.MaxPool3d((2, 2, 2))

        self.box_zero_padding = nn.ZeroPad2d(1)
        self.box_dense1 = nn.Linear(8192, 2048)
        self.box_drop1 = nn.Dropout(p=0.25)
        # self.box_banl1 = nn.BatchNorm1d( 2048 )
        self.box_dense2 = nn.Linear(2048, 2048)
        self.box_drop2 = nn.Dropout(p=0.25)
        # self.box_banl2 = nn.BatchNorm1d( 2048 )
        self.box_dense3 = nn.Linear(2048, 3)

        self.all_dense = nn.Linear(3, features)

        self.relu = nn.LeakyReLU(negative_slope=alpha)

    def forward(self, x):

        out = self.box_avgp1(x)
        out = self.box_conv1(out)
        out = self.relu(out)
        out = self.box_maxp1(out)
        out = self.box_bn3d64(out)
        out = self.box_conv2(out)
        out = self.relu(out)
        out = self.box_maxp2(out)
        out = self.box_bn3d128(out)
        out = self.box_conv3(out)
        out = self.relu(out)
        out = self.box_maxp2(out)
        out = self.box_bn3d256(out)
        out = self.box_conv4(out)
        out = self.relu(out)
        out = self.box_maxp2(out)
        out = self.box_bn3d256(out)
        out = self.box_conv5(out)
        out = self.relu(out)
        out = self.box_bn3d512(out)
        out = self.box_conv6(out)
        out = self.relu(out)
        out = self.box_zero_padding(out)
        out = self.box_maxp2(out)
        out = nn.Flatten()(out)
        out = self.box_dense1(out)
        out = self.relu(out)
        out = self.box_bn1d2048(out)
        out = self.box_drop1(out)
        out = self.box_dense2(out)
        out = self.relu(out)
        out = self.box_bn1d2048(out)
        out = self.box_drop2(out)
        out = self.box_dense3(out)

        out = self.all_dense(out)

        return out


class SMLNN(nn.Module):
    def __init__(self, features=3) -> None:
        super().__init__()

        self.features = features

        alpha = 5e-3

        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv6 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)

        self.sml_conv7 = nn.Conv3d(
            512, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1
        )
        self.sml_conv8 = nn.Conv3d(64, 1, kernel_size=(3, 3, 3), padding=1)

        self.bn3d64 = nn.BatchNorm3d(64)
        self.bn3d128 = nn.BatchNorm3d(128)
        self.bn3d256 = nn.BatchNorm3d(256)
        self.bn3d512 = nn.BatchNorm3d(512)

        self.avgp1 = nn.AvgPool3d((2, 1, 1))
        self.maxp1 = nn.MaxPool3d((1, 2, 2))
        self.maxp2 = nn.MaxPool3d((2, 2, 2))
        self.drop = nn.Dropout(p=0.25)

        self.all_dense = nn.Linear(1, features)

        self.relu = nn.LeakyReLU(negative_slope=alpha)

    def forward(self, y):
        # SML
        y = self.avgp1(y)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.maxp1(y)
        y = self.bn3d64(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.maxp2(y)
        y = self.bn3d128(y)
        y = self.conv3(y)
        y = self.relu(y)
        y = self.bn3d256(y)
        y = self.conv4(y)
        y = self.relu(y)
        y = self.maxp2(y)
        y = self.bn3d256(y)
        y = self.conv5(y)
        y = self.relu(y)
        y = self.bn3d512(y)
        y = self.conv6(y)
        y = self.relu(y)
        y = self.maxp2(y)
        y = self.sml_conv7(y)
        y = self.relu(y)
        y = self.bn3d64(y)
        y = self.sml_conv8(y)
        y = self.relu(y)
        y = nn.Flatten()(y)

        out = self.all_dense(y)

        return out


class DLNN(nn.Module):
    def __init__(self, features=3) -> None:
        super().__init__()

        self.features = features

        alpha = 3e-2

        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv6 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)

        self.sml_conv7 = nn.Conv3d(
            512, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1
        )
        self.sml_conv8 = nn.Conv3d(64, 1, kernel_size=(3, 3, 3), padding=1)

        self.bn3d64 = nn.BatchNorm3d(64)
        self.bn3d128 = nn.BatchNorm3d(128)
        self.bn3d256 = nn.BatchNorm3d(256)
        self.bn3d512 = nn.BatchNorm3d(512)
        self.box_bn1d2048 = nn.BatchNorm1d(2048)

        self.avgp1 = nn.AvgPool3d((2, 1, 1))
        self.maxp1 = nn.MaxPool3d((1, 2, 2))
        self.maxp2 = nn.MaxPool3d((2, 2, 2))
        self.drop = nn.Dropout(p=0.25)

        self.box_zero_padding = nn.ZeroPad2d(1)
        self.box_dense1 = nn.Linear(8192, 2048)
        self.box_dense2 = nn.Linear(2048, 2048)
        self.box_dense3 = nn.Linear(2048, 3)

        self.all_dense1 = nn.Linear(4, 4)
        self.all_dense2 = nn.Linear(4, features)

        self.relu = nn.LeakyReLU(negative_slope=alpha)

    def forward(self, x, y):
        # BOX
        x = self.avgp1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxp1(x)
        x = self.bn3d64(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxp2(x)
        x = self.bn3d128(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxp2(x)
        x = self.bn3d256(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxp2(x)
        x = self.bn3d256(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn3d512(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.box_zero_padding(x)
        x = self.maxp2(x)
        x = nn.Flatten()(x)
        x = self.box_dense1(x)
        x = self.relu(x)
        x = self.box_bn1d2048(x)
        x = self.drop(x)
        x = self.box_dense2(x)
        x = self.relu(x)
        x = self.box_bn1d2048(x)
        x = self.drop(x)
        x = self.box_dense3(x)
        # SML
        y = self.avgp1(y)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.maxp1(y)
        y = self.bn3d64(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.maxp2(y)
        y = self.bn3d128(y)
        y = self.conv3(y)
        y = self.relu(y)
        y = self.bn3d256(y)
        y = self.conv4(y)
        y = self.relu(y)
        y = self.maxp2(y)
        y = self.bn3d256(y)
        y = self.conv5(y)
        y = self.relu(y)
        y = self.bn3d512(y)
        y = self.conv6(y)
        y = self.relu(y)
        y = self.maxp2(y)
        y = self.sml_conv7(y)
        y = self.relu(y)
        y = self.bn3d64(y)
        y = self.sml_conv8(y)
        y = self.relu(y)
        y = nn.Flatten()(y)

        # CAT
        out = torch.cat(tensors=(x, y), dim=1)
        out = self.all_dense1(out)
        out = self.all_dense2(out)

        return out
