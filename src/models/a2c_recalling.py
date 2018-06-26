from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import norm_col_init, weights_init
from src.ltms.dnd import FreqPruningLTM
from torch.nn import LayerNorm
from torch.nn import init


class RecallingModel(torch.nn.Module):
    def __init__(self, nb_input_chan, nb_action, ltm_query_breadth, max_mem_len):
        super(RecallingModel, self).__init__()
        self.conv1 = nn.Conv2d(nb_input_chan, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm = LSTMCell(1024 + nb_action + 1, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, nb_action)

        self.ltm = FreqPruningLTM(512, 1024 + nb_action + 1, ltm_query_breadth, max_mem_len)
        self.cx_layer_norm = LayerNorm(512)
        self.env_layernorm = LayerNorm(1024 + nb_action + 1)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()  # TODO remove

    def forward(self, mode, *args):
        if mode == 'embed':
            xs_batch, actions_1hot, values = args
            state_embs = F.relu(self.maxp1(self.conv1(xs_batch)))
            state_embs = F.relu(self.maxp2(self.conv2(state_embs)))
            state_embs = F.relu(self.maxp3(self.conv3(state_embs)))
            state_embs = self.maxp4(self.conv4(state_embs))

            state_embs = state_embs.view(state_embs.size(0), -1)
            env_embs = torch.cat([state_embs, actions_1hot, values], dim=1)
            env_embs = self.env_layernorm(env_embs)
            return env_embs
        elif mode == 'recall':
            env_embeddings, hxs, cxs = args
            # Each item in the batch needs a unique cx and hx
            hx, cx = None, None
            hs, cs = [], []
            for x, hx, cx in zip(env_embeddings, hxs, cxs):
                h, c = self.lstm(x.unsqueeze(0), (hx, cx))
                hs.append(h)
                cs.append(c)

            if hx is None or cx is None:
                raise ValueError('Length must be >= 1: len(hxs) = {}, len(cxs): {}'.format(len(hxs), len(cxs)))

            # recall relevant memories
            memories, inds, weights = self.ltm(self.cx_layer_norm(torch.cat(cs, dim=0)))

            # process memories
            hx2s, cx2s = [], []
            for r, hx, cx in zip(memories, hs, cs):
                h, c = self.lstm(r.unsqueeze(0), (hx, cx))
                hx2s.append(h)
                cx2s.append(c)

            hx2s_batch = torch.cat(hx2s, dim=0)
            return self.critic_linear(hx2s_batch), self.actor_linear(hx2s_batch), hx2s, cx2s, inds, weights

        else:
            raise ValueError('Unrecognized mode ' + str(mode))


class LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(torch.zeros(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.zeros(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        init.orthogonal(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi, self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1
