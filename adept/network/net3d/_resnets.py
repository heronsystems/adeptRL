# Copyright (C) 2018 Heron Systems, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, nb_input_channel, nb_output_channel, stride=1, downsample=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(nb_input_channel, nb_output_channel, stride)
        self.bn1 = nn.BatchNorm2d(nb_output_channel)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(nb_output_channel, nb_output_channel)
        self.bn2 = nn.BatchNorm2d(nb_output_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockV2(nn.Module):
    expansion = 1

    def __init__(
        self, nb_input_channel, nb_output_channel, stride=1, downsample=None
    ):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(nb_input_channel)
        self.conv1 = conv3x3(nb_input_channel, nb_output_channel, stride)

        self.bn2 = nn.BatchNorm2d(nb_output_channel)
        self.conv2 = conv3x3(nb_output_channel, nb_output_channel)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.relu(out)

        out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, nb_in_channel, nb_out_channel, stride=1, downsample=None
    ):
        super(Bottleneck, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(
            nb_in_channel, nb_out_channel, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(nb_out_channel)
        self.conv2 = nn.Conv2d(
            nb_out_channel,
            nb_out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(nb_out_channel)
        self.conv3 = nn.Conv2d(
            nb_out_channel,
            nb_out_channel * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(nb_out_channel * self.expansion)

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckV2(nn.Module):
    expansion = 4

    def __init__(
        self, nb_in_channel, nb_out_channel, stride=1, downsample=None
    ):
        super(BottleneckV2, self).__init__()
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(nb_in_channel)
        self.conv1 = nn.Conv2d(
            nb_in_channel, nb_out_channel, kernel_size=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(nb_out_channel)
        self.conv2 = nn.Conv2d(
            nb_out_channel,
            nb_out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(nb_out_channel)
        self.conv3 = nn.Conv2d(
            nb_out_channel,
            nb_out_channel * self.expansion,
            kernel_size=1,
            bias=False,
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class ResNet(nn.Module):
    def __init__(self, block, layer_sizes):
        self.nb_input_channel = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layer_sizes[0])
        self.layer2 = self._make_layer(block, 128, layer_sizes[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_sizes[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layer_sizes[3], stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.nb_output_channel = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.nb_input_channel != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.nb_input_channel,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.nb_input_channel, planes, stride, downsample)]
        self.nb_input_channel = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.nb_input_channel, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        return x


def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


def resnet18v2():
    model = ResNet(BasicBlockV2, [2, 2, 2, 2])
    return model


def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model


def resnet34v2():
    model = ResNet(BasicBlockV2, [3, 4, 6, 3])
    return model


def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def resnet50v2():
    model = ResNet(BottleneckV2, [3, 4, 6, 3])
    return model


def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


def resnet101v2():
    model = ResNet(BottleneckV2, [3, 4, 23, 3])
    return model


def resnet152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model


def resnet152v2():
    model = ResNet(BottleneckV2, [3, 8, 36, 3])
    return model
