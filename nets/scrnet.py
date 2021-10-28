import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from thop import profile


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class DP_Conv(nn.Module):
    def __init__(self, kernel_size, in_size, out_size, nolinear, stride):
        super(DP_Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_size, in_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=in_size, bias=False)
        self.bn1 = nn.BatchNorm2d(in_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class SAC(nn.Module):
    def __init__(self, in_size, out_size):
        super(SAC, self).__init__()
        self.pre = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1),
        )
        self.switch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, 1, kernel_size=1, stride=1),
            hsigmoid()
        )
        self.big_field = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=2, dilation=2)
        self.small_field = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, dilation=1)
        self.post = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_size, out_size, kernel_size=1, stride=1),
        )

    def forward(self, x):
        out = self.pre(x)
        out = out + x
        switch = self.switch(out)
        big_field = self.big_field(out)
        small_field = self.small_field(out)
        out = (big_field * switch) + (small_field * (1 - switch))
        out_post = self.post(out)
        out = out_post + out
        return out


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


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        if semodule != None:
            self.se = semodule(out_size)
        else:
            self.se = None

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


# 主干网络
class SCRNet(nn.Module):
    def __init__(self):
        super(SCRNet, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=4,
        #               stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        # )
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        # self.stage1 = nn.Sequential(
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(16, 93),
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(93, 93),
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(93, 93),
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(93, 93),
        # )
        # self.stage2 = nn.Sequential(
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(93, 176, stride=2),
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(176, 176),
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(176, 176),
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(176, 176),
        # )
        # self.stage3 = nn.Sequential(
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(176, 256, stride=2),
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(256, 256),
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(256, 256),
        #     BN_Conv3_BN_ReLU_Conv3_BN_Dy(256, 256),
        # )
        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule, 2),
        )
        self.bneck2 = nn.Sequential(
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
        )
        self.bneck3 = nn.Sequential(
            Block(5, 24, 96, 40, hswish(), SeModule, 2),
            Block(5, 40, 240, 40, hswish(), SeModule, 1),
            Block(5, 40, 240, 40, hswish(), SeModule, 1),
            Block(5, 40, 120, 48, hswish(), SeModule, 1),
            Block(5, 48, 144, 48, hswish(), SeModule, 1),
        )
        self.bneck4 = nn.Sequential(
            Block(5, 48, 288, 96, hswish(), SeModule, 2),
            Block(5, 96, 576, 96, hswish(), SeModule, 1),
            Block(5, 96, 576, 96, hswish(), SeModule, 1),
        )
        # self.conv2 = nn.Sequential(
        #     conv3x3(256, 256),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=(8, 1),
        #               stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # self.stage4 = nn.Sequential(
        #     Res_block2_Conv_BN_ReLU(256, 256),
        #     Res_block2_Conv_BN_ReLU(256, 256),
        # )
        self.conv2 = DP_Conv(3, 96, 128, hswish(), 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.hs2 = hswish()
        # self.classifier1 = nn.Sequential(
        #     nn.Conv2d(128, 34, kernel_size=(1, 8),
        #               stride=(1, 4), padding=0, bias=False),
        #     nn.BatchNorm2d(34),
        #     nn.ReLU(inplace=True),
        # )
        # self.classifier2 = nn.Sequential(
        #     nn.Conv2d(128, 25, kernel_size=(1, 8),
        #               stride=(1, 4), padding=0, bias=False),
        #     nn.BatchNorm2d(25),
        #     nn.ReLU(inplace=True),
        # )
        # self.classifier3 = nn.Sequential(
        #     nn.Conv2d(128, 35, kernel_size=(1, 8),
        #               stride=(1, 4), padding=0, bias=False),
        #     nn.BatchNorm2d(35),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))  # 32,96
        out = self.bneck1(out)  # 16,48
        out = self.bneck2(out)  # 8,24
        out = self.bneck3(out)  # 4,12
        out = self.bneck4(out)
        out = self.hs2(self.bn2(self.conv2(out)))

        return out


def test():
    net = SCRNet()
    x = torch.randn(1, 3, 64, 192)
    flops, params = profile(net, inputs=(x,))
    net.eval()
    y = net(x)
    print(y.size())
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    # sac = SAC(3, 16)
    # sac.eval()
    # out = sac(x)
    # print(out.size())


if __name__ == '__main__':
    test()
