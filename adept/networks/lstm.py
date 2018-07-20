from __future__ import division

import torch
from torch.nn import init, Conv2d, BatchNorm2d, LSTMCell, functional as F

from adept.modules import Identity, LSTMCellLayerNorm, Residual2DPreact
from ._base import Network


class FourConvLSTM(Network):
    def __init__(self, nb_input, output_shape_dict, normalize):
        self.embedding_shape = 512
        super().__init__(self.embedding_shape, output_shape_dict)
        bias = not normalize
        self.conv1 = Conv2d(nb_input, 32, 3, stride=2, padding=1, bias=bias)
        self.conv2 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)
        self.conv3 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)
        self.conv4 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(32)
            self.bn4 = BatchNorm2d(32)
            self.lstm = LSTMCellLayerNorm(800, self.embedding_shape)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.lstm = LSTMCell(800, self.embedding_shape)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

    def forward(self, xs, internals):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = F.relu(self.bn4(self.conv4(xs)))

        xs = xs.view(xs.size(0), -1)

        hxs = torch.stack(internals['hx'])
        cxs = torch.stack(internals['cx'])
        hxs, cxs = self.lstm(xs, (hxs, cxs))

        return self.network_head(hxs, {
            'hx': list(torch.unbind(hxs, dim=0)),
            'cx': list(torch.unbind(cxs, dim=0))
        })

    def new_internals(self, device):
        return {
            'hx': torch.zeros(self.embedding_shape).to(device),
            'cx': torch.zeros(self.embedding_shape).to(device)
        }


class ResNetLSTM(Network):
    def __init__(self, nb_input_chan, output_shape_dict, normalize):
        self.embedding_shape = 512
        super(ResNetLSTM, self).__init__(self.embedding_shape, output_shape_dict)
        bias = not normalize
        self.conv1 = Conv2d(nb_input_chan, 32, 3, stride=2, padding=1, bias=bias)  # 40x40
        self.res1 = Residual2DPreact(32, 32, stride=1)
        self.res2 = Residual2DPreact(32, 32, stride=2)  # 20x20
        self.res3 = Residual2DPreact(32, 32, stride=1)
        self.res4 = Residual2DPreact(32, 32, stride=2)  # 10x10
        self.res5 = Residual2DPreact(32, 32, stride=1)
        self.res6 = Residual2DPreact(32, 32, stride=2)  # 5x5

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.lstm = LSTMCellLayerNorm(800, 512)
        else:
            self.bn1 = Identity()
            self.lstm = LSTMCell(800, 512)

        if not normalize:
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    def forward(self, xs, internals):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.res1(xs))
        xs = F.relu(self.res2(xs))
        xs = F.relu(self.res3(xs))
        xs = F.relu(self.res4(xs))
        xs = F.relu(self.res5(xs))
        xs = F.relu(self.res6(xs))

        xs = xs.view(xs.size(0), -1)

        hxs = torch.stack(internals['hx'])
        cxs = torch.stack(internals['cx'])
        hxs, cxs = self.lstm(xs, (hxs, cxs))

        return self.network_head(hxs, {
            'hx': list(torch.unbind(hxs, dim=0)),
            'cx': list(torch.unbind(cxs, dim=0))
        })

    def new_internals(self, device):
        return {
            'hx': torch.zeros(self.embedding_shape).to(device),
            'cx': torch.zeros(self.embedding_shape).to(device)
        }
