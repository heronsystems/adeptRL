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
import math

import torch
from torch import nn
from torch.nn import Linear, Softmax, functional as F


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
        self.register_buffer(
            'b',
            torch.tril(torch.ones(nb_embed,
                                  nb_embed)).view(1, 1, nb_embed, nb_embed)
        )
        self.nb_head = nb_head
        self.split_size = nb_qk_chan
        self.scale = scale
        self.qk_projection = Linear(nb_qk_chan, nb_qk_chan * 2)
        self.v_projection = Linear(nb_qk_chan, nb_v_chan)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1e9 * (
            1 - self.b
        )  # TF implem method: mask_attn_weights
        w = Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1), )
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
        self.attention = MultiHeadSelfAttention(
            nb_input_embed + nb_memory_embed,
            nb_channel,
            nb_channel,
            1,
            scale=True
        )
        self.mlp = torch.nn.ModuleList(
            [
                Linear(self._nb_total_mem_chan, self._nb_total_mem_chan)
                for _ in range(nb_mlp)
            ]
        )
        self.ln1 = torch.nn.LayerNorm(
            [nb_input_embed + nb_memory_embed, self._nb_total_mem_chan]
        )
        self.ln2 = torch.nn.LayerNorm(
            [nb_input_embed + nb_memory_embed, self._nb_total_mem_chan]
        )

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

        memory_plus_input = torch.cat(
            [prev_memory, input], dim=1
        )  # Tensor{B, Ei + Em, Cm}
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


class RelationalMHDPA(nn.Module):
    """
    Multi-head dot product attention.
    Adapted from:
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_py.py
    Reference implementation (Tensorflow):
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L2674
    """

    def __init__(self, height, width, nb_channel, nb_head, scale=False):
        super(RelationalMHDPA, self).__init__()
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert nb_channel % nb_head == 0
        seq_len = height * width
        self.register_buffer(
            'b',
            torch.tril(torch.ones(seq_len,
                                  seq_len)).view(1, 1, seq_len, seq_len)
        )
        self.nb_head = nb_head
        self.split_size = nb_channel
        self.scale = scale
        self.projection = nn.Linear(nb_channel, nb_channel * 3)
        self.mlp = nn.Linear(nb_channel, nb_channel)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        print('q', q.shape, 'k', k.shape, 'v', v.shape, 'w', w.shape)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1e9 * (
            1 - self.b
        )  # TF implem method: mask_attn_weights
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1), )
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
        :param x: A tensor with a shape of [batch, seq_len, nb_channel]
        :return: A tensor with a shape of [batch, seq_len, nb_channel]
        """
        size_out = x.size()[:-1] + (self.split_size * 3, )
        x = self.projection(x.view(-1, x.size(-1)))
        x = x.view(*size_out)

        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        a = self._attn(query, key, value)
        e = self.merge_heads(a)

        return self.mlp(e)

    def get_parameter_names(self, layer):
        return [
            'Proj{}_W'.format(layer),
            'Proj{}_b'.format(layer),
            'MLP{}_W'.format(layer),
            'MLP{}_b'.format(layer),
        ]
