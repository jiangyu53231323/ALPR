"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021).
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']

from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile
from torchstat import stat


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _c(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


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
    layers.append(nn.BatchNorm2d(filter, momentum=0.1))
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


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


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


class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


class SCRNet_effnetv2(nn.Module):
    def __init__(self):
        super(SCRNet_effnetv2, self).__init__()
        w = 1.
        self.conv1 = conv_3x3_bn(3, _c(24 * w), 1)
        self.bneck1 = nn.Sequential(
            MBConv(_c(24 * w), _c(24 * w), 1, 1, 0),
            MBConv(_c(24 * w), _c(24 * w), 1, 1, 0),
        )
        self.bneck2 = nn.Sequential(
            MBConv(_c(24 * w), _c(48 * w), 2, 4, 0),
            MBConv(_c(48 * w), _c(48 * w), 1, 4, 0),
            MBConv(_c(48 * w), _c(48 * w), 1, 4, 0),
            MBConv(_c(48 * w), _c(48 * w), 1, 4, 0),
        )
        self.bneck3 = nn.Sequential(
            MBConv(_c(48 * w), _c(64 * w), 2, 4, 0),
            MBConv(_c(64 * w), _c(64 * w), 1, 4, 0),
            MBConv(_c(64 * w), _c(64 * w), 1, 4, 0),
            MBConv(_c(64 * w), _c(64 * w), 1, 4, 0),
        )
        self.bneck4 = nn.Sequential(
            MBConv(_c(64 * w), _c(128 * w), 2, 4, 1),
            MBConv(_c(128 * w), _c(128 * w), 1, 4, 1),
            MBConv(_c(128 * w), _c(128 * w), 1, 4, 1),
            # MBConv(_c(128 * w), _c(128 * w), 1, 4, 1),
            # MBConv(_c(128 * w), _c(128 * w), 1, 4, 1),
            # MBConv(_c(128 * w), _c(128 * w), 1, 4, 1),
        )
        self.bneck5 = nn.Sequential(
            MBConv(_c(128 * w), _c(160 * w), 1, 6, 1),
            MBConv(_c(160 * w), _c(160 * w), 1, 6, 1),
            MBConv(_c(160 * w), _c(160 * w), 1, 6, 1),
            # MBConv(_c(160 * w), _c(160 * w), 1, 6, 1),
            # MBConv(_c(160 * w), _c(160 * w), 1, 6, 1),
            # MBConv(_c(160 * w), _c(160 * w), 1, 6, 1),
            # MBConv(_c(160 * w), _c(160 * w), 1, 6, 1),
            # MBConv(_c(160 * w), _c(160 * w), 1, 6, 1),
            # MBConv(_c(160 * w), _c(160 * w), 1, 6, 1),
        )
        self.bneck6 = nn.Sequential(
            MBConv(_c(160 * w), _c(256 * w), 2, 6, 1),
            MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
            # MBConv(_c(256 * w), _c(256 * w), 1, 6, 1),
        )

        self.conv_fpn3 = nn.Conv2d(_c(64 * w), 96, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fpn5 = nn.Conv2d(_c(160 * w), 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fpn6 = nn.Conv2d(_c(256 * w), 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.up6 = upsampling(256, 128, 4)
        self.up5 = upsampling(128, 96, 4)

        self.conv2 = nn.Sequential(
            DP_Conv(96, 96, 3, hswish(), stride=2),
            nn.BatchNorm2d(96),
            hswish(),
            DP_Conv(96, 64, 3, hswish(), stride=1),
            nn.BatchNorm2d(64),
            hswish(),

            nn.Conv2d(64, 64, kernel_size=(8, 1), stride=1, padding=0, groups=64, bias=False),
            nn.BatchNorm2d(64),
            hswish(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            hswish(),
        )

        self.blue_classifier = Blue_ocr(64)
        self.green_classifier = Green_ocr(64)
        self.category = Classifier(64, 16, 2)

    def forward(self, x):
        out = self.conv1(x)  # out:64,224
        out1 = self.bneck1(out)  # out:64,224
        out2 = self.bneck2(out1)  # out:32,112
        out3 = self.bneck3(out2)  # out:16,56
        out4 = self.bneck4(out3)  # out:8,28
        out5 = self.bneck5(out4)  # out:8,28
        out6 = self.bneck6(out5)  # out:4,14

        p6 = self.conv_fpn6(out6)
        p5 = self.up6(p6) + self.conv_fpn5(out5)
        p3 = self.up5(p5) + self.conv_fpn3(out3)
        out = self.conv2(p3)

        c = self.category(out)
        b = self.blue_classifier(out)
        g = self.green_classifier(out)

        return [c, b, g]


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def test():
    # net = effnetv2_s()
    net = SCRNet_effnetv2()
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
    for i in range(50):
        x = torch.randn(1, 3, 64, 224)
        y = net(x)
    time_end = time.time()
    print("time = " + str(time_end - time_start))


if __name__ == '__main__':
    test()
