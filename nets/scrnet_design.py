import math

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


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _c(v, divisor=4, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


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


def deconv(inplanes, filter):
    layers = []
    fc = DCN(inplanes,
             filter,
             kernel_size=(3, 3),
             stride=1,
             padding=1,
             dilation=1,
             deformable_groups=1)
    fill_fc_weights(fc)
    layers.append(fc)
    layers.append(nn.BatchNorm2d(filter, momentum=BN_MOMENTUM))
    layers.append(hswish())
    return nn.Sequential(*layers)


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


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class Blue_ocr(nn.Module):
    def __init__(self, in_channel):
        super(Blue_ocr, self).__init__()
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


# 主干网络
class SCRNet_des(nn.Module):
    def __init__(self):
        super(SCRNet_des, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        w = 1.3

        block = GhostBottleneck
        self.bneck1 = nn.Sequential(
            block(16, _c(16 * w), _c(16 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(16 * w), _c(48 * w), _c(24 * w), dw_kernel_size=3, stride=2, se_ratio=0),
            block(_c(24 * w), _c(72 * w), _c(24 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(24 * w), _c(72 * w), _c(40 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
            block(_c(40 * w), _c(120 * w), _c(40 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
        )
        # self.bneck2 = nn.Sequential(
        #     block(_c(24 * w), _c(72 * w), _c(40 * w), dw_kernel_size=5, stride=2, se_ratio=0.25),
        #     block(_c(40 * w), _c(120 * w), _c(40 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
        # )
        self.bneck2 = nn.Sequential(
            block(_c(40 * w), _c(240 * w), _c(80 * w), dw_kernel_size=3, stride=2, se_ratio=0),
            block(_c(80 * w), _c(200 * w), _c(80 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(80 * w), _c(184 * w), _c(80 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(80 * w), _c(184 * w), _c(80 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(80 * w), _c(480 * w), _c(112 * w), dw_kernel_size=3, stride=1, se_ratio=0.25),
            block(_c(112 * w), _c(672 * w), _c(112 * w), dw_kernel_size=3, stride=1, se_ratio=0.25),
        )
        # self.bneck4 = nn.Sequential(
        #     block(_c(112 * w), _c(672 * w), _c(160 * w), dw_kernel_size=5, stride=2, se_ratio=0.25),
        #     block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0),
        #     block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
        #     block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0),
        #     block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
        # )

        # self.conv_fpn1 = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_fpn2 = nn.Conv2d(_c(40 * w), 64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_fpn3 = nn.Conv2d(_c(112 * w), 128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_fpn4 = nn.Conv2d(_c(160 * w), 256, kernel_size=1, stride=1, padding=0, bias=False)

        # self.deconv_layer4 = _make_deconv_layer(256, 128, 4)
        # self.deconv_layer3 = _make_deconv_layer(128, 64, 4)
        # self.deconv_layer2 = _make_deconv_layer(64, 32, 4)

        # self.up4 = upsampling(256, 128, 4)
        # self.up3 = upsampling(128, 64, 4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(_c(112 * w), _c(112 * w), kernel_size=(3, 3), stride=2, padding=1, groups=_c(112 * w),
                      bias=False),
            nn.BatchNorm2d(_c(112 * w)),
            # hswish(),
            nn.ReLU(inplace=True),
            nn.Conv2d(_c(112 * w), 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            # hswish(),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1, groups=64, bias=False),
            # nn.BatchNorm2d(64),
            # hswish(),
            # nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(64),
            # hswish(),
            nn.Conv2d(64, 64, kernel_size=(8, 1), stride=1, padding=0, groups=64, bias=False),
            nn.BatchNorm2d(64),
            # hswish(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            # hswish(),
            nn.ReLU(inplace=True),
        )

        self.blue_classifier = Blue_ocr(64)
        self.green_classifier = Green_ocr(64)
        self.category = Classifier(64, 16, 2)
        # self.province = Province_ocr(64, 32)
        # self.ctc_ocr = CTC_orc(64)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))  # out:64,192  out:64,224
        out1 = self.bneck1(out)  # out:32,96  out:32,112
        out2 = self.bneck2(out1)  # out:16,48  out:16,56
        # out3 = self.bneck3(out2)  # out:8,24  out:8,28
        # out4 = self.bneck4(out3)  # out:4,12  out:4,14

        # p4 = self.conv_fpn4(out4)
        # p3 = self.up4(p4) + self.conv_fpn3(out3)
        # p2 = self.up3(p3) + self.conv_fpn2(out2)
        out = self.conv2(out2)  # out:1,24  out:1,28

        # province = self.province(out)
        # ctc_orc = self.ctc_ocr(out)
        c = self.category(out)
        b = self.blue_classifier(out)
        g = self.green_classifier(out)

        # return [province, ctc_orc]
        return [c, b, g]


def test():
    net = SCRNet_des()
    x = torch.randn(1, 3, 64, 224)
    flops, params = profile(net, inputs=(x,))
    net.eval()
    y = net(x)
    # print(y[0].size())
    # print(y[1].size())
    # print(y[2][0].size())
    stat(net, (3, 64, 224))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
    print("Number of parameter: %.6f" % (total))  # 输出

    flops = FlopCountAnalysis(net, x)
    print('FLOPs = ' + str(flops.total() / 1000 ** 3) + 'G')
    print(flop_count_table(flops))

    # sac = SAC(3, 16)
    # sac.eval()
    # out = sac(x)
    # print(out.size())


if __name__ == '__main__':
    test()
