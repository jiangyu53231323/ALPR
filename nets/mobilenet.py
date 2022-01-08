'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from thop import profile
from torchstat import stat

from lib.DCNv2.dcn_v2 import DCN

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
    # up = nn.UpsamplingBilinear2d(scale_factor=2)
    fill_up_weights(up)
    layers.append(fc)
    layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
    layers.append(hswish())
    # 上采样，最终将特征图恢复到layer1层的大小
    layers.append(up)
    layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
    layers.append(hswish())
    return nn.Sequential(*layers)


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        self.deconv_with_bias = False
        expansion = 1

        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        # self.bneck = nn.Sequential(
        #     Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
        #     Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
        #     Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
        #     Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule, 2),
        #     Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule, 1),
        #     Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule, 1),
        #     Block(3, 40, 240, 80, hswish(), None, 2),
        #     Block(3, 80, 200, 80, hswish(), None, 1),
        #     Block(3, 80, 184, 80, hswish(), None, 1),
        #     Block(3, 80, 184, 80, hswish(), None, 1),
        #     Block(3, 80, 480, 112, hswish(), SeModule, 1),
        #     Block(3, 112, 672, 112, hswish(), SeModule, 1),
        #     Block(5, 112, 672, 160, hswish(), SeModule, 1),
        #     Block(5, 160, 672, 160, hswish(), SeModule, 2),
        #     Block(5, 160, 960, 160, hswish(), SeModule, 1),
        # )

        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            # Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule, 1),
        )
        self.bneck2 = nn.Sequential(
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule, 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule, 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule, 1),
            # Block(3, 40, 200, 80, hswish(), SeModule, 1),
        )
        self.bneck3 = nn.Sequential(
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule, 1),
            Block(3, 112, 672, 112, hswish(), SeModule, 1),
            Block(5, 112, 672, 160, hswish(), SeModule, 1),
        )
        self.bneck4 = nn.Sequential(
            Block(5, 160, 672, 160, hswish(), SeModule, 2),
            Block(5, 160, 960, 160, hswish(), SeModule, 1),
            Block(5, 160, 960, 160, hswish(), SeModule, 1),
        )

        # self.conv_fpn1 = nn.Sequential(
        #     nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #     hswish(),
        # )
        self.conv_fpn2 = nn.Sequential(
            nn.Conv2d(40, 64, kernel_size=1, stride=1, padding=0, bias=False),
            hswish(),
        )
        self.conv_fpn3 = nn.Sequential(
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0, bias=False),
            hswish(),
        )
        self.conv_fpn4 = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=1, stride=1, padding=0, bias=False),
            hswish(),
        )
        # self.conv_fpn2 = nn.Conv2d(80, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_fpn3 = nn.Conv2d(160, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_fpn4 = nn.Conv2d(160, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.up4 = nn.Sequential(
        #     # upsampling(96, 64, 4),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     # nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #     # hswish(),
        # )
        # self.up3 = nn.Sequential(
        #     # upsampling(96, 64, 4),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     # nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #     # hswish(),
        # )
        # self.up2 = nn.Sequential(
        #     # upsampling(96, 64, 4),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     # nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #     # hswish(),
        # )
        # used for deconv layers 可形变卷积
        # 将主干网最终输出channel控制在64
        self.deconv_layer3 = _make_deconv_layer(128, 96, 4)
        self.deconv_layer2 = _make_deconv_layer(96, 64, 4)
        self.deconv_layer1 = _make_deconv_layer(64, 64, 4)

        # self.conv_fuse = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #     hswish(),
        # )
        # 融合特征图的逐点卷积
        # self.fuse3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.fuse2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)

        self.hmap = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                  hswish(),
                                  nn.Conv2d(64, num_classes, kernel_size=1, bias=True))
        self.hmap[-1].bias.data.fill_(-2.19)
        self.cors = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                  hswish(),
                                  nn.Conv2d(64, 8, kernel_size=1, bias=True))
        self.w_h_ = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                  hswish(),
                                  nn.Conv2d(64, 4, kernel_size=1, bias=True))
        # self.hmap = nn.Sequential(DP_Conv(3, 32, 64, hswish(), 1),
        #                           hswish(),
        #                           nn.Conv2d(64, num_classes, kernel_size=1, bias=True))
        # self.cors = nn.Sequential(DP_Conv(3, 32, 64, hswish(), 1),
        #                           hswish(),
        #                           nn.Conv2d(64, 8, kernel_size=1, bias=True))
        # self.w_h_ = nn.Sequential(DP_Conv(3, 32, 64, hswish(), 1),
        #                           hswish(),
        #                           nn.Conv2d(64, 4, kernel_size=1, bias=True))

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        # out = self.bneck(out)
        # out = self.hs2(self.bn2(self.conv2(out)))
        # out = F.avg_pool2d(out, 7)
        # out = out.view(out.size(0), -1)
        # out = self.hs3(self.bn3(self.linear3(out)))
        # out = self.linear4(out)
        c1 = self.bneck1(out)
        c2 = self.bneck2(c1)
        c3 = self.bneck3(c2)
        c4 = self.bneck4(c3)

        # p4_1 = self.conv_fpn4(c4)
        # p3_1 = self.conv_fpn3(c3) + self.up4(p4_1)
        # p2_1 = self.conv_fpn2(c2) + self.up3(p3_1)
        # p1_1 = self.conv_fpn1(c1) + self.up2(p2_1)
        # p3 = torch.cat([self.deconv_layer3(p4), self.conv_fpn3(c3)], dim=1)
        # p3 = self.fuse3(p3)
        # p2 = torch.cat([self.deconv_layer2(p3), self.conv_fpn2(c2)], dim=1)
        # p2 = self.fuse2(p2)
        # p3_2 = self.deconv_layer3(c4)
        # p2_2 = self.deconv_layer2(p3_2)
        # p1_2 = self.deconv_layer1(p2_2)

        # p1 = self.conv_fuse(torch.cat((p1_2, p1_2), 1))
        # p1 = p1_1

        p4 = self.conv_fpn4(c4)
        p3 = self.deconv_layer3(p4) + self.conv_fpn3(c3)
        p2 = self.deconv_layer2(p3) + self.conv_fpn2(c2)
        p1 = self.deconv_layer1(p2)

        out = [[self.hmap(p1), self.cors(p1), self.w_h_(p1)]]

        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        self.deconv_with_bias = False
        expansion = 1  # DCN的输出通道扩张数

        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        # self.bneck = nn.Sequential(
        #     Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule, 2),
        #     Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
        #     Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
        #     Block(5, 24, 96, 40, hswish(), SeModule, 2),
        #     Block(5, 40, 240, 40, hswish(), SeModule, 1),
        #     Block(5, 40, 240, 40, hswish(), SeModule, 1),
        #     Block(5, 40, 120, 48, hswish(), SeModule, 1),
        #     Block(5, 48, 144, 48, hswish(), SeModule, 1),
        #     Block(5, 48, 288, 96, hswish(), SeModule, 2),
        #     Block(5, 96, 576, 96, hswish(), SeModule, 1),
        #     Block(5, 96, 576, 96, hswish(), SeModule, 1),
        # )

        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule, 2),
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
        self.bneck4 = nn.Sequential(
            Block(5, 48, 288, 96, hswish(), SeModule, 2),
            Block(5, 96, 576, 96, hswish(), SeModule, 1),
            Block(5, 96, 576, 96, hswish(), SeModule, 1),
        )

        self.conv_fpn2 = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fpn3 = nn.Conv2d(48, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fpn4 = nn.Conv2d(96, 256, kernel_size=1, stride=1, padding=0, bias=False)

        # used for deconv layers 可形变卷积
        # 将主干网最终输出channel控制在64
        self.deconv_layer3 = _make_deconv_layer(256, 128, 4)
        self.deconv_layer2 = _make_deconv_layer(128, 64, 4)
        self.deconv_layer1 = _make_deconv_layer(64, 64, 4)
        # 融合特征图的逐点卷积
        # self.fuse3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.fuse2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)

        self.hmap = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                  hswish(),
                                  nn.Conv2d(64, num_classes, kernel_size=1, bias=True))
        self.hmap[-1].bias.data.fill_(-2.19)
        self.cors = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                  hswish(),
                                  nn.Conv2d(64, 8, kernel_size=1, bias=True))
        self.w_h_ = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                  hswish(),
                                  nn.Conv2d(64, 4, kernel_size=1, bias=True))

        # self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn2 = nn.BatchNorm2d(576)
        # self.hs2 = hswish()
        # self.linear3 = nn.Linear(576, 1280)
        # self.bn3 = nn.BatchNorm1d(1280)
        # self.hs3 = hswish()
        # self.linear4 = nn.Linear(1280, num_classes)
        # self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        # out = self.bneck(out)
        c1 = self.bneck1(out)
        c2 = self.bneck2(c1)
        c3 = self.bneck3(c2)
        c4 = self.bneck4(c3)

        p4 = self.conv_fpn4(c4)
        # p3 = torch.cat([self.deconv_layer3(p4), self.conv_fpn3(c3)], dim=1)
        # p3 = self.fuse3(p3)
        # p2 = torch.cat([self.deconv_layer2(p3), self.conv_fpn2(c2)], dim=1)
        # p2 = self.fuse2(p2)
        p3 = self.deconv_layer3(p4) + self.conv_fpn3(c3)
        p2 = self.deconv_layer2(p3) + self.conv_fpn2(c2)
        p1 = self.deconv_layer1(p2)

        out = [[self.hmap(p1), self.cors(p1), self.w_h_(p1)]]

        # out = self.hs2(self.bn2(self.conv2(out4)))
        # out = F.avg_pool2d(out, 7)
        # out = out.view(out.size(0), -1)
        # out = self.hs3(self.bn3(self.linear3(out)))
        # out = self.linear4(out)
        return out


def test():
    net = MobileNetV3_Large(num_classes=1)
    x = torch.randn(1, 3, 384, 256)
    net.eval()
    y = net(x)
    # print(y[0][0].size())
    # print(y.size())

    flops, params = profile(net, inputs=(x,))
    stat(net, (3, 384, 256))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
    print("Number of parameter: %.6f" % (total))  # 输出

    # time_start = time.time()
    # for i in range(200):
    #     x = torch.randn(1, 3, 384, 256)
    #     y = net(x)
    # time_end = time.time()
    # print("time = " + str(time_end - time_start))


if __name__ == '__main__':
    test()
