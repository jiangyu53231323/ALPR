import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3),
                               stride=stride, padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3),
                               stride=1, padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual

        return out


class BN_Conv3_BN_ReLU_Conv3_BN_Dy(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BN_Conv3_BN_ReLU_Conv3_BN_Dy, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        return out


class Res_block2_Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block2_Conv_BN_ReLU, self).__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3),
                              stride=stride, padding=(0, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)

        return out

# 主干网络
class SCRNet(nn.Module):
    def __init__(self):
        super(SCRNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(16, 93),
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(93, 93),
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(93, 93),
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(93, 93),
        )
        self.stage2 = nn.Sequential(
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(93, 176, stride=2),
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(176, 176),
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(176, 176),
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(176, 176),
        )
        self.stage3 = nn.Sequential(
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(176, 256, stride=2),
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(256, 256),
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(256, 256),
            BN_Conv3_BN_ReLU_Conv3_BN_Dy(256, 256),
        )
        self.conv2 = nn.Sequential(
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(8, 1),
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            Res_block2_Conv_BN_ReLU(256, 256),
            Res_block2_Conv_BN_ReLU(256, 256),
        )
        self.classifier1 = nn.Sequential(
            nn.Conv2d(256, 34, kernel_size=(1, 8),
                      stride=(1, 4), padding=0, bias=False),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True),
        )
        self.classifier2 = nn.Sequential(
            nn.Conv2d(256, 25, kernel_size=(1, 8),
                      stride=(1, 4), padding=0, bias=False),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.Conv2d(256, 35, kernel_size=(1, 8),
                      stride=(1, 4), padding=0, bias=False),
            nn.BatchNorm2d(35),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.conv2(out)
        out = self.stage4(out)
        y1 = self.classifier1(out)
        y2 = self.classifier2(out)
        y3 = self.classifier3(out)

        return y1, y2, y3

