import torch
from torch.nn import (
    init, Conv2d, Linear, BatchNorm2d, BatchNorm1d, LSTMCell, functional as F
)

from adept.modules import Identity, LSTMCellLayerNorm, MultiHeadSelfAttention
from ._base import Network


class AttentionCNN(Network):
    def __init__(self, nb_in_chan, output_shape_dict, nb_head, normalize):
        self.embedding_shape = 512
        super(AttentionCNN, self).__init__(self.embedding_shape, output_shape_dict)
        self.normalize = normalize
        bias = not normalize
        self.conv1 = Conv2d(nb_in_chan, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.conv2 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=bias)

        self.attention = MultiHeadSelfAttention(20 * 20, 34, 34, nb_head)
        self.mlp = Linear(34, 34)

        self.conv3 = Conv2d(34, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.conv4 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.linear = Linear(800, self.embedding_shape, bias=bias)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(32)
            self.bn4 = BatchNorm2d(32)
            self.bn_linear = BatchNorm1d(512)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn_linear = Identity()

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

    def forward(self, input, internals):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.ln(x)

        xs_chan = torch.linspace(-1, 1, 20).view(1, 1, 1, 20).expand(input.size(0), 1, 20, 20).to(input.device)
        ys_chan = torch.linspace(-1, 1, 20).view(1, 1, 20, 1).expand(input.size(0), 1, 20, 20).to(input.device)
        x = torch.cat([x, xs_chan, ys_chan], dim=1)
        h = x.size(2)
        w = x.size(3)
        # need to transpose because attention expects attention dim before channel dim
        x = x.view(x.size(0), x.size(1), h * w).transpose(1, 2)

        x = self.attention(x.contiguous())
        x = F.relu(self.mlp(x))
        # need to undo the transpose before output
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), h, w)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_linear(self.linear(x)))
        return self.network_head(x, {})

    def new_internals(self, device):
        return {}


class AttentionLSTM(Network):
    def __init__(self, nb_in_chan, output_shape_dict, nb_head, normalize):
        self.embedding_shape = 512
        super(AttentionLSTM, self).__init__(self.embedding_shape, output_shape_dict)
        self.normalize = normalize
        bias = not normalize
        self.conv1 = Conv2d(nb_in_chan, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.conv2 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=bias)

        self.attention = MultiHeadSelfAttention(20 * 20, 34, 34, nb_head)
        self.mlp = Linear(34, 34)

        self.conv3 = Conv2d(34, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.conv4 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=bias)

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

    def forward(self, input, internals):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.ln(x)

        xs_chan = torch.linspace(-1, 1, 20).view(1, 1, 1, 20).expand(input.size(0), 1, 20, 20).to(input.device)
        ys_chan = torch.linspace(-1, 1, 20).view(1, 1, 20, 1).expand(input.size(0), 1, 20, 20).to(input.device)
        x = torch.cat([x, xs_chan, ys_chan], dim=1)
        h = x.size(2)
        w = x.size(3)
        # need to transpose because attention expects attention dim before channel dim
        x = x.view(x.size(0), x.size(1), h * w).transpose(1, 2)

        x = self.attention(x.contiguous())
        x = F.relu(self.mlp(x))
        # need to undo the transpose before output
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), h, w)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)

        hxs = torch.stack(internals['hx'])
        cxs = torch.stack(internals['cx'])
        hxs, cxs = self.lstm(x, (hxs, cxs))

        return self.network_head(hxs, {
            'hx': list(torch.unbind(hxs, dim=0)),
            'cx': list(torch.unbind(cxs, dim=0))
        })

    def new_internals(self, device):
        return {
            'hx': torch.zeros(self.embedding_shape).to(device),
            'cx': torch.zeros(self.embedding_shape).to(device)
        }
