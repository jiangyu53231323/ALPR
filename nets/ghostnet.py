# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from thop import profile
from lib.DCNv2.dcn_v2 import DCN
from torchstat import stat

# __all__ = ['ghost_net']
BN_MOMENTUM = 0.1


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
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
    # up = nn.ConvTranspose2d(in_channels=planes,
    #                         out_channels=planes,
    #                         kernel_size=kernel,
    #                         stride=2,
    #                         padding=padding,
    #                         output_padding=output_padding,
    #                         bias=False)
    up = nn.UpsamplingBilinear2d(scale_factor=2)
    # fill_up_weights(up)
    layers.append(fc)
    layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
    layers.append(hswish())
    # 上采样，最终将特征图恢复到layer1层之前的大小
    layers.append(up)
    layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
    layers.append(hswish())
    return nn.Sequential(*layers)


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


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


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
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


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


class My_GhostNet(nn.Module):
    def __init__(self, num_classes=1, w=1.0):
        super(My_GhostNet, self).__init__()

        # building first layer
        self.conv_stem = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)

        # building inverted residual blocks
        block = GhostBottleneck
        self.block1 = nn.Sequential(
            block(16, _c(16 * w), _c(16 * w), dw_kernel_size=3, stride=1, se_ratio=0),
        )
        self.block2 = nn.Sequential(
            block(_c(16 * w), _c(48 * w), _c(24 * w), dw_kernel_size=3, stride=2, se_ratio=0),
            block(_c(24 * w), _c(72 * w), _c(24 * w), dw_kernel_size=3, stride=1, se_ratio=0),
        )
        self.block3 = nn.Sequential(
            block(_c(24 * w), _c(72 * w), _c(40 * w), dw_kernel_size=5, stride=2, se_ratio=0.25),
            block(_c(40 * w), _c(120 * w), _c(40 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
        )
        self.block4 = nn.Sequential(
            block(_c(40 * w), _c(240 * w), _c(80 * w), dw_kernel_size=3, stride=2, se_ratio=0),
            block(_c(80 * w), _c(200 * w), _c(80 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(80 * w), _c(184 * w), _c(80 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(80 * w), _c(184 * w), _c(80 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(80 * w), _c(480 * w), _c(112 * w), dw_kernel_size=3, stride=1, se_ratio=0.25),
            block(_c(112 * w), _c(672 * w), _c(112 * w), dw_kernel_size=3, stride=1, se_ratio=0.25),
        )
        self.block5 = nn.Sequential(
            block(_c(112 * w), _c(672 * w), _c(160 * w), dw_kernel_size=5, stride=2, se_ratio=0.25),
            block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0),
            block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
            block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0),
            block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
        )

        # self.conv_fpn2 = nn.Conv2d(_c(40 * w), 64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_fpn3 = nn.Conv2d(_c(112 * w), 96, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_fpn4 = nn.Conv2d(_c(160 * w), 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fpn2 = nn.Sequential(
            nn.Conv2d(_c(40 * w), 64, kernel_size=1, stride=1, padding=0, bias=False),
            hswish(),
        )
        self.conv_fpn3 = nn.Sequential(
            nn.Conv2d(_c(112 * w), 96, kernel_size=1, stride=1, padding=0, bias=False),
            hswish(),
        )
        self.conv_fpn4 = nn.Sequential(
            nn.Conv2d(_c(160 * w), 128, kernel_size=1, stride=1, padding=0, bias=False),
            hswish(),
        )
        # used for deconv layers 可形变卷积
        # 将主干网最终输出channel控制在64
        # self.deconv_layer3 = _make_deconv_layer(128, 96, 4)
        # self.deconv_layer2 = _make_deconv_layer(96, 64, 4)
        # self.deconv_layer1 = _make_deconv_layer(64, 64, 4)

        self.up4 = nn.Sequential(
            # upsampling(128, 96, 4),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 96, kernel_size=1, stride=1, padding=0, bias=False),
            hswish(),
        )
        self.up3 = nn.Sequential(
            # upsampling(96, 64, 4),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0, bias=False),
            hswish(),
        )
        self.up2 = nn.Sequential(
            # upsampling(96, 64, 4),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            hswish(),
        )

        self.hmap = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                  # hswish(),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, num_classes, kernel_size=1, bias=True))
        self.hmap[-1].bias.data.fill_(-2.19)
        self.cors = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                  # hswish(),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 8, kernel_size=1, bias=True))
        self.w_h_ = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                  # hswish(),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 4, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        # x = self.blocks(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        p5 = self.conv_fpn4(x5)
        # p4 = self.deconv_layer3(p5) + self.conv_fpn3(x4)
        # p3 = self.deconv_layer2(p4) + self.conv_fpn2(x3)
        # p2 = self.deconv_layer1(p3)

        p4 = self.up4(p5) + self.conv_fpn3(x4)
        p3 = self.up3(p4) + self.conv_fpn2(x3)
        p2 = self.up2(p3)

        out = [[self.hmap(p2), self.cors(p2), self.w_h_(p2)]]
        # out = [[self.hmap(p2), self.w_h_(p2)]]

        return out


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__ == '__main__':
    input = torch.randn(1, 3, 384, 256)
    model = My_GhostNet(num_classes=1, w=1.3)
    model.eval()
    # print(model)

    y = model(input)
    # print(y[0][0].size())

    flops, params = profile(model, inputs=(input,))
    stat(model, (3, 384, 256))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    total = sum([param.nelement() for param in model.parameters()])  # 计算总参数量
    print("Number of parameter: %.6f" % (total))  # 输出

    # time_start = time.time()
    # for i in range(200):
    #     x = torch.randn(1, 3, 384, 256)
    #     y = model(x)
    # time_end = time.time()
    # print("time = " + str(time_end - time_start))
