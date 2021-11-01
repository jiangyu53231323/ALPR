import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from thop import profile
from lib.DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1


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


# 填充转置卷积的weight
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)  # ceil()向上取整
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            # fabs()返回绝对值
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


# 填充回归预测的卷积 weight
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def _get_deconv_cfg(deconv_kernel):
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0
    return deconv_kernel, padding, output_padding


# 创建可形变卷积层，kernel为转置卷积的卷积核大小，dcn的卷积核为固定的3×3
def _make_deconv_layer(inplanes, filter, kernel):
    layers = []
    kernel, padding, output_padding = _get_deconv_cfg(kernel)
    planes = filter
    # self.inplanes记录了上一层网络的out通道数
    fc = DCN(inplanes,
             planes,
             kernel_size=(3, 3),
             stride=1,
             padding=1,
             dilation=1,
             deformable_groups=1)
    # fc = nn.Conv2d(self.inplanes, planes,
    #         kernel_size=3, stride=1,
    #         padding=1, dilation=1, bias=False)
    # fill_fc_weights(fc)
    # 转置卷积（逆卷积、反卷积）
    up = nn.ConvTranspose2d(in_channels=planes,
                            out_channels=planes,
                            kernel_size=kernel,
                            stride=2,
                            padding=padding,
                            output_padding=output_padding,
                            bias=False)
    fill_up_weights(up)
    layers.append(fc)
    layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
    layers.append(hswish())
    # 上采样，最终将特征图恢复到layer1层之前的大小
    layers.append(up)
    layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
    layers.append(hswish())
    return nn.Sequential(*layers)


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


class Blue_ocr(nn.Module):
    def __init__(self, in_channel):
        super(Blue_ocr, self).__init__()
        self.classifier1 = nn.Sequential(
            nn.Conv2d(in_channel, 34, kernel_size=(1, 6),
                      stride=(1, 3), padding=0, bias=False),
            # nn.ReLU(inplace=True),
        )
        self.classifier2 = nn.Sequential(
            nn.Conv2d(in_channel, 25, kernel_size=(1, 6),
                      stride=(1, 3), padding=0, bias=False),
            # nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.Conv2d(in_channel, 35, kernel_size=(1, 6),
                      stride=(1, 3), padding=0, bias=False),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.classifier1(x)
        out2 = self.classifier2(x)
        out3 = self.classifier3(x)
        # 格式化输出
        y1 = out1[:, :, :, 0].view([out1.size()[0], -1])
        y2 = out2[:, :, :, 1].view([out2.size()[0], -1])
        y3 = out3[:, :, :, 2].view([out3.size()[0], -1])
        y4 = out3[:, :, :, 3].view([out3.size()[0], -1])
        y5 = out3[:, :, :, 4].view([out3.size()[0], -1])
        y6 = out3[:, :, :, 5].view([out3.size()[0], -1])
        y7 = out3[:, :, :, 6].view([out3.size()[0], -1])

        return [y1, y2, y3, y4, y5, y6, y7]


class Green_ocr(nn.Module):
    def __init__(self, in_channel):
        super(Green_ocr, self).__init__()
        self.classifier1 = nn.Sequential(
            nn.Conv2d(in_channel, 34, kernel_size=(1, 6),
                      stride=(1, 3), padding=2, bias=False),
            # nn.ReLU(inplace=True),
        )
        self.classifier2 = nn.Sequential(
            nn.Conv2d(in_channel, 25, kernel_size=(1, 6),
                      stride=(1, 3), padding=(0, 2), bias=False),
            # nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.Conv2d(in_channel, 35, kernel_size=(1, 6),
                      stride=(1, 3), padding=2, bias=False),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.classifier1(x)
        out2 = self.classifier2(x)
        out3 = self.classifier3(x)
        # 格式化输出
        y1 = out1[:, :, :, 0].view([out1.size()[0], -1])
        y2 = out2[:, :, :, 1].view([out2.size()[0], -1])
        y3 = out3[:, :, :, 2].view([out3.size()[0], -1])
        y4 = out3[:, :, :, 3].view([out3.size()[0], -1])
        y5 = out3[:, :, :, 4].view([out3.size()[0], -1])
        y6 = out3[:, :, :, 5].view([out3.size()[0], -1])
        y7 = out3[:, :, :, 6].view([out3.size()[0], -1])
        y8 = out3[:, :, :, 7].view([out3.size()[0], -1])

        return [y1, y2, y3, y4, y5, y6, y7, y8]


# 主干网络
class SCRNet(nn.Module):
    def __init__(self):
        super(SCRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 2),
            Block(3, 16, 32, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 16, nn.ReLU(inplace=True), None, 1),
        )
        self.bneck2 = nn.Sequential(
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
        )
        self.bneck3 = nn.Sequential(
            Block(5, 24, 96, 40, hswish(), SeModule, 2),
            Block(5, 40, 240, 40, hswish(), SeModule, 1),
            Block(5, 40, 240, 40, hswish(), SeModule, 1),
            Block(5, 40, 120, 48, hswish(), SeModule, 1),
            Block(5, 48, 144, 48, hswish(), SeModule, 1),
        )
        # self.bneck4 = nn.Sequential(
        #     Block(5, 48, 288, 96, hswish(), SeModule, 2),
        #     Block(5, 96, 576, 96, hswish(), SeModule, 1),
        #     Block(5, 96, 576, 96, hswish(), SeModule, 1),
        # )

        # self.conv_fpn1 = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fpn2 = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fpn3 = nn.Conv2d(48, 64, kernel_size=1, stride=1, padding=0, bias=False)

        # self.deconv_layer3 = _make_deconv_layer(128, 64, 4)
        self.deconv_layer2 = _make_deconv_layer(64, 64, 4)
        # self.deconv_layer1 = _make_deconv_layer(64, 32, 4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(8, 1), stride=1, padding=0, groups=64, bias=False),
            nn.BatchNorm2d(64),
            hswish(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            hswish(),
        )

        self.blue_classifier = Blue_ocr(64)
        self.green_classifier = Green_ocr(64)
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            hswish(),
        )
        self.category = nn.Linear(32, 2)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))  # out:32,96
        out1 = self.bneck1(out)  # out:16,48
        out2 = self.bneck2(out1)  # out:8,24
        out3 = self.bneck3(out2)  # out:4,12

        p3 = self.conv_fpn3(out3)
        p2 = self.deconv_layer2(p3) + self.conv_fpn2(out2)
        out = self.conv2(p2)
        # out4 = self.bneck4(out3)  # out:2,6
        # out = self.hs2(self.bn2(self.conv2(out4)))  # out:1,3
        c = self.conv3(out)
        c = c.view(c.size(0), -1)
        c = self.category(c)
        b = self.blue_classifier(out)
        g = self.green_classifier(out)

        return [c, b, g]


def test():
    net = SCRNet()
    x = torch.randn(1, 3, 64, 192)
    flops, params = profile(net, inputs=(x,))
    net.eval()
    y = net(x)
    print(y[0].size())
    print(y[1][0].size())
    print(y[2][0].size())
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    # sac = SAC(3, 16)
    # sac.eval()
    # out = sac(x)
    # print(out.size())


if __name__ == '__main__':
    test()
