import math

import torch
from src.utils import weights_init, norm_col_init
from torch import nn as nn
from torch.nn import functional as F, Linear, Softmax


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
        """
        LSTM Cell that layer normalizes the cell state.
        :param x: Tensor{B, C}
        :param hidden: A Tuple[Tensor{B, C}, Tensor{B, C}] of (previous output, cell state)
        :return:
        """
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


class MultiHeadSelfAttention(torch.nn.Module):
    """
    Multi-head Self Attention.
    Adapted from:
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_py.py
    Reference implementation (Tensorflow):
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L2674
    """
    def __init__(self, nb_embed, nb_qk_chan, nb_v_chan, nb_head, scale=False):
        super(MultiHeadSelfAttention, self).__init__()
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert nb_qk_chan % nb_head == 0
        self.register_buffer('b', torch.tril(torch.ones(nb_embed, nb_embed)).view(1, 1, nb_embed, nb_embed))
        self.nb_head = nb_head
        self.split_size = nb_qk_chan
        self.scale = scale
        self.qk_projection = Linear(nb_qk_chan, nb_qk_chan * 2)
        self.v_projection = Linear(nb_qk_chan, nb_v_chan)
        # self.layer_norm = LayerNorm([seq_len, nb_channel])

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        w = Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        # keep dims, but expand the last dim to be [head, chan // head]
        new_x_shape = x.size()[:-1] + (self.nb_head, x.size(-1) // self.nb_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            # batch, head, channel, attend
            return x.permute(0, 2, 3, 1)
        else:
            # batch, head, attend, channel
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        """
        :param x: A tensor with a shape of [batch, nb_embed, nb_channel]
        :return: A tensor with a shape of [batch, nb_embed, nb_channel]
        """
        qk = self.qk_projection(x)
        query, key = qk.split(self.split_size, dim=2)

        value = self.v_projection(x)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        a = self._attn(query, key, value)
        return self.merge_heads(a)


class RMCCell(torch.nn.Module):
    """
    Strict implementation a Relational Memory Core.

    Paper: https://arxiv.org/pdf/1806.01822.pdf
    Reference implementation: https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
    """
    def __init__(
        self,
        nb_input_embed,
        nb_memory_embed,
        nb_channel,
        nb_head=1,
        nb_block=1,
        nb_mlp=2,
        input_bias=0,
        forget_bias=1
    ):
        super(RMCCell, self).__init__()
        self._mem_slots = nb_memory_embed
        self._head_size = nb_channel
        self._num_heads = nb_head
        self._nb_block = nb_block
        self._nb_total_mem_chan = nb_channel * nb_head

        self._input_bias = input_bias
        self._forget_bias = forget_bias

        self.input_linear = Linear(nb_channel, self._nb_total_mem_chan)
        self.ih = Linear(nb_channel, 2 * self._nb_total_mem_chan)
        self.hh = Linear(nb_channel, 2 * self._nb_total_mem_chan)

        self.ih.bias.data.fill_(0)
        self.hh.bias.data.fill_(0)
        # forget bias init
        self.attention = MultiHeadSelfAttention(nb_input_embed + nb_memory_embed, nb_channel, nb_channel, 1, scale=True)
        self.mlp = torch.nn.ModuleList(
            [Linear(self._nb_total_mem_chan, self._nb_total_mem_chan) for _ in range(nb_mlp)]
        )
        self.ln1 = torch.nn.LayerNorm([nb_input_embed + nb_memory_embed, self._nb_total_mem_chan])
        self.ln2 = torch.nn.LayerNorm([nb_input_embed + nb_memory_embed, self._nb_total_mem_chan])

    def _attend(self, memory):
        for _ in range(self._nb_block):
            attended_mem = self.attention(memory)
            memory = self.ln1(memory + attended_mem)  # skip connection
            mlp_mem = memory
            for layer in self.mlp:
                mlp_mem = F.relu(layer(mlp_mem))
            memory = self.ln2(mlp_mem + memory)  # skip connection
        return memory

    def forward(self, input, prev_memory):
        """
        B: Batch length
        E: Embeddings
        C: Channels

        Type{Shape}[Contents]
        :param input: Tensor{B, Ei, Ci}
        :param prev_memory: Tensor{B, Em, Cm}
        :return:
        """
        # project the input channels to match memory channels
        input = self.input_linear(input)  # Tensor{B, Ei, Cm}

        memory_plus_input = torch.cat([prev_memory, input], dim=1)  # Tensor{B, Ei + Em, Cm}
        next_memory = self._attend(memory_plus_input)
        next_memory = next_memory[:, :-prev_memory.size(1), :]

        # calc gates
        i2h = self.ih(input)
        h2h = self.hh(prev_memory.tanh())
        preact = i2h + h2h
        input_gate, forget_gate = torch.chunk(preact, 2, dim=2)
        input_gate = (input_gate + self._input_bias).sigmoid()
        forget_gate = (forget_gate + self._forget_bias).sigmoid()

        # apply gates
        next_memory = input_gate * next_memory.tanh()
        next_memory += forget_gate * prev_memory

        return next_memory
