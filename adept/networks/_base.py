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
import abc

import torch
from adept.utils.requires_args import RequiresArgs


class BaseNetwork(torch.nn.Module):
    @classmethod
    @abc.abstractmethod
    def from_args(
        cls,
        args,
        observation_space,
        headname_to_output_shape,
        network_registry
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def new_internals(self, device):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, obsname_to_obs, internals):
        raise NotImplementedError


class NetworkModule(BaseNetwork, RequiresArgs, metaclass=abc.ABCMeta):
    pass


class NetworkHead(torch.nn.Module):
    def __init__(self, nb_channel, head_dict):
        super().__init__()

        self.heads = torch.nn.ModuleList()
        # Must be sorted for mpi methods so that the creation order is
        # deterministic
        for head_name in sorted(head_dict.keys()):
            head_size = head_dict[head_name]
            self.heads.add_module(
                head_name, torch.nn.Linear(nb_channel, head_size)
            )

    def forward(self, embedding, internals):
        return {
            name: module(embedding)
            for name, module in self.heads.named_children()
        }, internals


class NetworkBody(torch.nn.Module, RequiresArgs, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_args(cls, nb_input_channel, nb_out_channel, args):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def new_internals(self, device):
        raise NotImplementedError


class NetworkJunction(torch.nn.Module):
    def __init__(self, obsname_to_inputnet):
        super().__init__()

        nb_output_channel = 0
        self.input_nets = torch.nn.ModuleList()
        for name, input_net in obsname_to_inputnet.items():
            nb_output_channel += input_net.nb_output_channel
            self.input_nets.add_module(name, input_net)
        self.nb_output_channel = nb_output_channel

    def forward(self, obs_dict):
        """
        :param obs_dict: Dict[str, Tensor] mapping the pathname to observation
        Tensors
        :return: a Tensor embedding
        """
        embeddings = []
        for name, input_net in self.input_nets.named_children():
            embeddings.append(input_net.forward(obs_dict[name]))
        return torch.cat(embeddings, dim=1)
