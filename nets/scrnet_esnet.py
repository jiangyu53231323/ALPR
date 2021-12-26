import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile
from torchstat import stat

from lib.DCNv2.dcn_v2 import DCN


BN_MOMENTUM = 0.1

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # b, c, h, w =======>  b, g, c_per, h, w
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batch_size, -1, height, width)
    return x


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
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

def upsampling(inplanes, filter, kernel):
    layers = []
    kernel, padding, output_padding = _get_deconv_cfg(kernel)
    up = nn.ConvTranspose2d(in_channels=inplanes,
                            out_channels=filter,
                            kernel_size=kernel,
                            stride=2,
                            padding=padding,
                            output_padding=output_padding,
                            bias=False)
    fill_up_weights(up)
    layers.append(up)
    layers.append(nn.BatchNorm2d(filter, momentum=BN_MOMENTUM))
    layers.append(hswish())
    return nn.Sequential(*layers)

class DP_Conv(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, nolinear, stride):
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

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, inp, oup, k=3, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        self.oup = oup
        init_channels = oup // 2  # hidden channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, 1, s, 0, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, oup, k, 1, k // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class ES_SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        out = identity * x
        return out


class ES_Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ES_Bottleneck, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        # assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            # 第一条branch分支，用于stride=2的ES_Bottleneck
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Hardswish(inplace=True),
            )

        self.branch2 = nn.Sequential(
            # 第一二条branch分支，用于stride=2的ES_Bottleneck
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            ES_SEModule(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
        )

        self.branch3 = nn.Sequential(
            # 第三条branch分支，用于stride=1的ES_Bottleneck
            GhostConv(branch_features, branch_features, 3, 1),
            ES_SEModule(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
        )

        self.branch4 = nn.Sequential(
            # 第四条branch分支，用于stride=2的ES_Bottleneck的最后一次深度可分离卷积
            self.depthwise_conv(oup, oup, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.Hardswish(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size=3, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    @staticmethod
    def conv1x1(i, o, kernel_size=1, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x3 = torch.cat((x1, self.branch3(x2)), dim=1)
            out = channel_shuffle(x3, 2)
        elif self.stride == 2:
            x1 = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            out = self.branch4(x1)

        return out


class Blue_ocr(nn.Module):
    def __init__(self, in_channel):
        super(Blue_ocr, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel, kernel_size=(1, 3), stride=1, padding=(0, 1)),
        #     nn.BatchNorm2d(in_channel),
        #     nn.ReLU(inplace=True),
        # )
        self.classifier1 = nn.Sequential(
            nn.Conv2d(in_channel, 35, kernel_size=(1, 6),
                      stride=1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.Conv2d(in_channel, 35, kernel_size=(1, 6),
                      stride=(1, 4), padding=(0, 1), bias=True),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x = self.conv(x)
        x1 = x[:, :, :, :6]
        x3 = x[:, :, :, 4:]
        out1 = self.classifier1(x1)
        # out2 = self.classifier2(x)
        out3 = self.classifier3(x3)
        out = torch.cat((out1, out3), -1).squeeze(2).permute(0, 2, 1)
        return out


class Green_ocr(nn.Module):
    def __init__(self, in_channel):
        super(Green_ocr, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel, kernel_size=(1, 3), stride=1, padding=(0, 1)),
        #     nn.BatchNorm2d(in_channel),
        #     nn.ReLU(inplace=True),
        # )
        self.classifier1 = nn.Sequential(
            nn.Conv2d(in_channel, 35, kernel_size=(1, 6),
                      stride=1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.Conv2d(in_channel, 35, kernel_size=(1, 6),
                      stride=(1, 3), padding=0, bias=True),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x = self.conv(x)
        x1 = x[:, :, :, :6]
        x3 = x[:, :, :, 4:]
        out1 = self.classifier1(x1)
        # out2 = self.classifier2(x)
        out3 = self.classifier3(x3)
        # out: [b,8,c]
        out = torch.cat((out1, out3), -1).squeeze(2).permute(0, 2, 1)
        return out


class Classifier(nn.Module):
    def __init__(self, in_channel, filter, cls):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, filter, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(filter),
            hswish(),
        )
        self.category = nn.Linear(filter, cls)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.category(out)
        return out


class SCRNet_Pico(nn.Module):
    def __init__(self):
        super(SCRNet_Pico, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            # hswish(),
            nn.ReLU(inplace=True),
        )
        self.bneck1 = nn.Sequential(
            ES_Bottleneck(16, 32, 2),
            ES_Bottleneck(32, 32, 1),
            ES_Bottleneck(32, 32, 1),
            ES_Bottleneck(32, 32, 1),
        )
        self.bneck2 = nn.Sequential(
            ES_Bottleneck(32, 64, 2),
            ES_Bottleneck(64, 64, 1),
            ES_Bottleneck(64, 64, 1),
            ES_Bottleneck(64, 64, 1),
            ES_Bottleneck(64, 64, 1),
            ES_Bottleneck(64, 64, 1),
        )
        self.bneck3 = nn.Sequential(
            ES_Bottleneck(64, 128, 2),
            ES_Bottleneck(128, 128, 1),
            ES_Bottleneck(128, 128, 1),
            ES_Bottleneck(128, 128, 1),
            ES_Bottleneck(128, 128, 1),
            ES_Bottleneck(128, 128, 1),
            ES_Bottleneck(128, 128, 1),
            ES_Bottleneck(128, 128, 1),
            ES_Bottleneck(128, 128, 1),
        )
        self.bneck4 = nn.Sequential(
            ES_Bottleneck(128, 256, 2),
            ES_Bottleneck(256, 256, 1),
            ES_Bottleneck(256, 256, 1),
            ES_Bottleneck(256, 256, 1),
            ES_Bottleneck(256, 256, 1),
            ES_Bottleneck(256, 256, 1),
            ES_Bottleneck(256, 256, 1),
            ES_Bottleneck(256, 256, 1),
            ES_Bottleneck(256, 256, 1),
        )
        self.conv_fpn2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fpn3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fpn4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_fuse = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)

        self.up4 = upsampling(256, 128, 4)
        self.up3 = upsampling(128, 64, 4)

        self.conv2 = nn.Sequential(
            DP_Conv(64, 64, 3, hswish(), stride=2),
            nn.BatchNorm2d(64),
            hswish(),
        )
        self.conv3 = nn.Sequential(
            DP_Conv(64, 64, 3, hswish(), stride=1),
            nn.BatchNorm2d(64),
            hswish(),
            nn.Conv2d(64, 64, kernel_size=(8, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            hswish(),
        )
        self.blue_classifier = Blue_ocr(64)
        self.green_classifier = Green_ocr(64)
        self.category = Classifier(64, 16, 2)

    def forward(self, x):
        out = self.conv1(x)  # out:64,192  out:64,224
        out1 = self.bneck1(out)  # out:32,96  out:32,112
        out2 = self.bneck2(out1)  # out:16,48  out:16,56
        out3 = self.bneck3(out2)  # out:8,24  out:8,28
        out4 = self.bneck4(out3)  # out:4,12  out:4,14

        p4 = self.conv_fpn4(out4)
        p3 = self.up4(p4) + self.conv_fpn3(out3)
        p2 = self.up3(p3) + self.conv_fpn2(out2)
        out5 = self.conv2(p2)
        out = out5 + self.conv_fuse(out3)
        out = self.conv3(out)

        # province = self.province(out)
        # ctc_orc = self.ctc_ocr(out)
        c = self.category(out)
        b = self.blue_classifier(out)
        g = self.green_classifier(out)

        # return [province, ctc_orc]
        return [c, b, g]


def test():
    net = SCRNet_Pico()
    x = torch.randn(1, 3, 64, 224)
    net.eval()
    y = net(x)

    # flops, params = profile(net, inputs=(x,))
    # stat(net, (3, 64, 224))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    # total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
    # print("Number of parameter: %.6f" % (total))  # 输出
    # flops = FlopCountAnalysis(net, x)
    # print('FLOPs = ' + str(flops.total() / 1000 ** 3) + 'G')
    # print(flop_count_table(flops))

    time_start = time.time()
    for i in range(400):
        x = torch.randn(1, 3, 64, 224)
        y = net(x)
    time_end = time.time()
    print("time = " + str(time_end - time_start))


if __name__ == '__main__':
    test()