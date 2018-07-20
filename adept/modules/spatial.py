from torch import nn as nn
from torch.nn import functional as F


class Residual2DPreact(nn.Module):
    def __init__(self, nb_in_chan, nb_out_chan, stride=1):
        super(Residual2DPreact, self).__init__()

        self.nb_in_chan = nb_in_chan
        self.nb_out_chan = nb_out_chan
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(nb_in_chan)
        self.conv1 = nn.Conv2d(nb_in_chan, nb_out_chan, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nb_out_chan)
        self.conv2 = nn.Conv2d(nb_out_chan, nb_out_chan, 3, stride=1, padding=1, bias=False)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)

        self.do_projection = self.nb_in_chan != self.nb_out_chan or self.stride > 1
        if self.do_projection:
            self.projection = nn.Conv2d(nb_in_chan, nb_out_chan, 3, stride=stride, padding=1)
            self.projection.weight.data.mul_(relu_gain)

    def forward(self, x):
        first = F.relu(self.bn1(x))
        if self.do_projection:
            projection = self.projection(first)
        else:
            projection = x
        x = self.conv1(first)
        x = self.conv2(F.relu(self.bn2(x)))
        return x + projection