import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torchstat import stat
from fvcore.nn import FlopCountAnalysis, flop_count_table


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)
    return x


class BasicUnit(nn.Module):
    """Basic Unit of ShuffleNet-v2"""

    def __init__(self, in_chnls, out_chnls, is_se=False, is_residual=False, c_ratio=0.5, groups=2):
        super(BasicUnit, self).__init__()
        self.is_se, self.is_res = is_se, is_residual
        self.l_chnls = int(in_chnls * c_ratio)
        self.r_chnls = in_chnls - self.l_chnls
        self.ro_chnls = out_chnls - self.l_chnls
        self.groups = groups

        # layers
        self.conv1 = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0)
        self.dwconv2 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 3, 1, 1,  # same padding, depthwise conv
                                 groups=self.ro_chnls, activation=False)
        act = False if self.is_res else True
        self.conv3 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 1, 1, 0, activation=act)
        if self.is_se:
            self.se = SE(self.ro_chnls, 16)
        if self.is_res:
            self.shortcut = nn.Sequential()
            if self.r_chnls != self.ro_chnls:
                self.shortcut = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0, activation=False)

    def forward(self, x):
        x_l = x[:, :self.l_chnls, :, :]
        x_r = x[:, self.l_chnls:, :, :]

        # right path
        out_r = self.conv1(x_r)
        out_r = self.dwconv2(out_r)
        out_r = self.conv3(out_r)
        if self.is_se:
            coefficient = self.se(out_r)
            out_r *= coefficient
        if self.is_res:
            out_r += self.shortcut(x_r)

        # concatenate
        out = torch.cat((x_l, out_r), 1)
        return shuffle_chnls(out, self.groups)


class DSampling(nn.Module):
    """Spatial down sampling of SuffleNet-v2"""

    def __init__(self, in_chnls, groups=2):
        super(DSampling, self).__init__()
        self.groups = groups
        self.dwconv_l1 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1,  # down-sampling, depth-wise conv.
                                   groups=in_chnls, activation=False)
        self.conv_l2 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.conv_r1 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.dwconv_r2 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1, groups=in_chnls, activation=False)
        self.conv_r3 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)

    def forward(self, x):
        # left path
        out_l = self.dwconv_l1(x)
        out_l = self.conv_l2(out_l)

        # right path
        out_r = self.conv_r1(x)
        out_r = self.dwconv_r2(out_r)
        out_r = self.conv_r3(out_r)

        # concatenate
        out = torch.cat((out_l, out_r), 1)
        out = shuffle_chnls(out, self.groups)
        return shuffle_chnls(out, self.groups)


class ShuffleNet_v2(nn.Module):
    """ShuffleNet-v2"""

    _defaults = {
        "sets": {0.5, 1, 1.5, 2},
        "units": [3, 7, 3],
        "chnl_sets": {0.5: [24, 48, 96, 192, 1024],
                      1: [24, 116, 232, 464, 1024],
                      1.5: [24, 176, 352, 704, 1024],
                      2: [24, 244, 488, 976, 2048]}
    }

    def __init__(self, scale, num_cls, is_se=False, is_res=False) -> object:
        super(ShuffleNet_v2, self).__init__()
        self.__dict__.update(self._defaults)
        assert (scale in self.sets)
        self.is_se = is_se
        self.is_res = is_res
        self.chnls = self.chnl_sets[scale]

        # make layers
        self.conv1 = BN_Conv2d(3, self.chnls[0], 3, 2, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.stage2 = self.__make_stage(self.chnls[0], self.chnls[1], self.units[0])
        self.stage3 = self.__make_stage(self.chnls[1], self.chnls[2], self.units[1])
        self.stage4 = self.__make_stage(self.chnls[2], self.chnls[3], self.units[2])
        self.conv5 = BN_Conv2d(self.chnls[3], self.chnls[4], 1, 1, 0)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.body = self.__make_body()
        self.fc = nn.Linear(self.chnls[4], num_cls)

    def __make_stage(self, in_chnls, out_chnls, units):
        layers = [DSampling(in_chnls),
                  BasicUnit(2 * in_chnls, out_chnls, self.is_se, self.is_res)]
        for _ in range(units - 1):
            layers.append(BasicUnit(out_chnls, out_chnls, self.is_se, self.is_res))
        return nn.Sequential(*layers)

    def __make_body(self):
        return nn.Sequential(
            self.conv1, self.maxpool, self.stage2, self.stage3,
            self.stage4, self.conv5, self.globalpool
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


"""
API
"""


def shufflenet_0_5x(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes)


def shufflenet_0_5x_se(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes, is_se=True)


def shufflenet_0_5x_res(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes, is_res=True)


def shufflenet_0_5x_se_res(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes, is_se=True, is_res=True)


def shufflenet_1x(num_classes=1000):
    return ShuffleNet_v2(1, num_classes)


def shufflenet_1x_se(num_classes=1000):
    return ShuffleNet_v2(1, num_classes, is_se=True)


def shufflenet_1x_res(num_classes=1000):
    return ShuffleNet_v2(1, num_classes, is_res=True)


def shufflenet_1x_se_res(num_classes=1000):
    return ShuffleNet_v2(1, num_classes, is_se=True, is_res=True)


def shufflenet_1_5x(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes)


def shufflenet_1_5x_se(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes, is_se=True)


def shufflenet_1_5x_res(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes, is_res=True)


def shufflenet_1_5x_se_res(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes, is_se=True, is_res=True)


def shufflenet_2x(num_classes=1000):
    return ShuffleNet_v2(2, num_classes)


def shufflenet_2x_se(num_classes=1000):
    return ShuffleNet_v2(2, num_classes, is_se=True)


def shufflenet_2x_res(num_classes=1000):
    return ShuffleNet_v2(2, num_classes, is_res=True)


def shufflenet_2x_se_res(num_classes=1000):
    return ShuffleNet_v2(2, num_classes, is_se=True, is_res=True)


def test():
    net = shufflenet_2x()
    x = torch.randn(1, 3, 384, 256)
    # flops, params = profile(net, inputs=(x,))
    net.eval()
    y = net(x)
    # print(y[0][0].size())
    print(y.size())

    stat(net, (3, 384, 256))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')

    total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
    print("Number of parameter: %.6f" % (total))  # 输出

    flops = FlopCountAnalysis(net, x)
    print('FLOPs = ' + str(flops.total() / 1000 ** 3) + 'G')
    print(flop_count_table(flops))

if __name__ == '__main__':
    test()