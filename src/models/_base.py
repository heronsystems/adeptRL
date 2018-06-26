import torch
from src.utils import weights_init, norm_col_init
from torch import nn as nn
from torch.nn import functional as F


class StateEmbedder(torch.nn.Module):
    def __init__(self, num_inputs):
        super(StateEmbedder, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)  # 40x40
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 20x20
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 10x10
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 5x5
        self.layer_norm = nn.LayerNorm([800])
        self.init()

    def init(self):
        self.apply(weights_init)
        # gain = nn.init.calculate_gain('leaky_relu')
        # self.conv1.weight.data.mul_(gain)
        # self.conv2.weight.data.mul_(gain)
        # self.conv3.weight.data.mul_(gain)
        # self.conv4.weight.data.mul_(gain)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # 1024
        x = self.layer_norm(x)
        # x = x / torch.norm(x, 2).detach()
        return x


class ForwardModel(torch.nn.Module):
    def __init__(self, nb_action):
        super(ForwardModel, self).__init__()
        self.nb_action = nb_action
        self.hidden = nn.Linear(800 + nb_action, 512)
        self.out_linear = nn.Linear(512, 800)
        self.init()

    def init(self):
        self.hidden.weight.data = norm_col_init(self.hidden.weight.data, 0.01)
        self.hidden.bias.data.fill_(0)
        self.out_linear.weight.data = norm_col_init(self.out_linear.weight.data, 0.01)
        self.out_linear.bias.data.fill_(0)

    def forward(self, state_emb, action):
        x = torch.cat([state_emb, action], dim=1)
        x = F.leaky_relu(self.hidden(x))
        x = self.out_linear(x)
        # x = x / torch.norm(x, 2).detach()
        return x


class InverseModel(torch.nn.Module):
    def __init__(self, nb_action):
        super(InverseModel, self).__init__()
        self.hidden = nn.Linear(1600, 512)
        self.out_linear = nn.Linear(512, nb_action)
        self.init()

    def init(self):
        self.hidden.weight.data = norm_col_init(self.hidden.weight.data, 0.01)
        self.hidden.bias.data.fill_(0)
        self.out_linear.weight.data = norm_col_init(self.out_linear.weight.data, 0.01)
        self.out_linear.bias.data.fill_(0)

    def forward(self, state_emb, next_state_emb):
        x = torch.cat([state_emb, next_state_emb], dim=1)
        x = F.leaky_relu(self.hidden(x))
        return self.out_linear(x)


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class LSTMCellLayerNorm(nn.Module):
    """
    A lstm cell that layer norms the cell state
    https://github.com/seba-1511/lstms.pth/blob/master/lstms/lstm.py for reference.
    Original License Apache 2.0
    """
    def __init__(self, input_size, hidden_size, bias=True, forget_bias=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        if bias:
            self.ih.bias.data.fill_(0)
            self.hh.bias.data.fill_(0)
            # forget bias init
            self.ih.bias.data[hidden_size:hidden_size * 2].fill_(forget_bias)
            self.hh.bias.data[hidden_size:hidden_size * 2].fill_(forget_bias)

        self.ln_cell = nn.LayerNorm(hidden_size)

    def forward(self, x, hidden):
        h, c = hidden

        # Linear mappings
        i2h = self.ih(x)
        h2h = self.hh(h)
        preact = i2h + h2h

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        c_t = self.ln_cell(c_t)
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t


class Residual2D(nn.Module):
    def __init__(self, nb_in_chan, nb_out_chan, stride=1):
        super(Residual2D, self).__init__()

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
