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
            headname_to_output_shape
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def new_internals(self, device):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, name_to_obs, internals):
        raise NotImplementedError


class NetworkModule(BaseNetwork, RequiresArgs, metaclass=abc.ABCMeta):
    pass


class NetworkHead(torch.nn.Module):
    def __init__(self, nb_channel, head_dict):
        super().__init__()

        self.heads = torch.nn.ModuleList()
        # Must be sorted for mpi methods so that the creation order is deterministic
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


class ModularNetwork(BaseNetwork, metaclass=abc.ABCMeta):
    def __init__(self, junc, body, head):
        super().__init__()
        self.junc = junc
        self.body = body
        self.head = head

    @classmethod
    def from_args(
        cls,
        args,
        observation_space,
        headname_to_output_shape
    ):
        pass  #TODO

    def forward(self, obs_dict, internals):
        embedding = self.junc.forward(obs_dict)
        pre_result, internals = self.body.forward(embedding, internals)
        result, internals = self.head.forward(pre_result, internals)
        return result, internals

    def new_internals(self, device):
        return self.body.new_internals(device)


class NetworkBody(torch.nn.Module, RequiresArgs, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_args(cls, nb_input_channel, nb_out_channel, args):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nb_output_channel(self):
        raise NotImplementedError

    @abc.abstractmethod
    def new_internals(self, device):
        raise NotImplementedError


class NetworkJunction(torch.nn.Module):
    def __init__(self, obs_name_to_input_net):
        super().__init__()

        nb_output_channel = 0
        self.pathways = torch.nn.ModuleList()
        for name, pathway in obs_name_to_input_net.items():
            nb_output_channel += pathway.nb_output_channel
            self.pathways.add_module(name, pathway)
        self.nb_output_channel = nb_output_channel

    def forward(self, obs_dict):
        """
        :param obs_dict: Dict[str, Tensor] mapping the pathname to observation
        Tensors
        :return: a Tensor embedding
        """
        embeddings = []
        for name, pathway in self.pathways.named_children():
            embeddings.append(pathway.forward(obs_dict[name]))
        return torch.cat(embeddings, dim=1)


class InputNetwork(torch.nn.Module, RequiresArgs, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def nb_output_channel(self):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_args(cls, nb_in_channel, args):
        raise NotImplementedError
