import math
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo

from lib.DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1
model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)

        self.ca = ChannelAttention(out_planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * self.expansion)
        self.sa = SpatialAttention

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

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


class PoseResNet(nn.Module):
    def __init__(self, block, layers, head_conv, num_classes):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.num_classes = num_classes

        super(PoseResNet, self).__init__()
        # 缩小四倍
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers 可形变卷积
        # 将主干网最终输出channel控制在64
        self.deconv_layers = self._make_deconv_layer(3, [256, 128, 64], [4, 4, 4])
        # 进行回归预测前的卷积filter个数
        if head_conv > 0:
            # heatmap layers 中心点定位
            self.hmap = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True))
            self.hmap[-1].bias.data.fill_(-2.19)
            # regression layers  角点、长宽
            self.cors = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, 8, kernel_size=1, bias=True))
            self.w_h_ = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, 4, kernel_size=1, bias=True))
        else:
            # heatmap layers
            self.hmap = nn.Conv2d(64, num_classes, kernel_size=1, bias=True)
            # corners layers
            self.cors = nn.Conv2d(64, 8, kernel_size=1, bias=True)
            # bboxes layers
            self.w_h_ = nn.Conv2d(64, 4, kernel_size=1, bias=True)

        fill_fc_weights(self.cors)
        fill_fc_weights(self.w_h_)

    # 创建普通卷积层
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)  # 表示列表元素作为多个元素传入

    def _get_deconv_cfg(self, deconv_kernel, index):
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

    # 创建可形变卷积层
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # self.inplanes记录了上一层网络的out通道数
            fc = DCN(self.inplanes,
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
                                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            # 上采样，最终将特征图恢复到layer1层之前的大小
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        x = self.deconv_layers(out4)
        out = [[self.hmap(x), self.cors(x), self.w_h_(x)]]
        return out

    # 初始化权重
    def init_weights(self, num_layers):
        url = model_urls['resnet{}'.format(num_layers)]
        pretrained_state_dict = model_zoo.load_url(url, model_dir='./model')
        print('=> loading pretrained model {}'.format(url))
        self.load_state_dict(pretrained_state_dict, strict=False)

        print('=> init deconv weights from normal distribution')
        # 正态分布初始化 可形变卷积 中的BN层
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, head_conv=64, num_classes=1):
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers, head_conv, num_classes)
    model.init_weights(num_layers)
    return model
