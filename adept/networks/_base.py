import abc

from gym import spaces
from torch.nn import Module
import torch


class NetworkInterface(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def new_internals(self, device):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, obs_dict, internals):
        raise NotImplementedError

    @staticmethod
    def stack_internals(internals):
        return {key: torch.stack(internal) for key, internal in internals.items()}

    @staticmethod
    def unstack_internals(stacked_internals):
        return {key: list(torch.unbind(stacked_internal)) for key, stacked_internal in stacked_internals.items()}


class NetworkHead(Module):
    def __init__(self, nb_channel, head_dict):
        super().__init__()

        self.heads = torch.nn.ModuleList()
        # Must be sorted for mpi methods so that the creation order is deterministic
        for head_name in sorted(head_dict.keys()):
            head_size = head_dict[head_name]
            self.heads.add_module(head_name, torch.nn.Linear(nb_channel, head_size))

    def forward(self, embedding, internals):
        return {name: module(embedding) for name, module in self.heads.named_children()}, internals


class ModularNetwork(NetworkInterface, metaclass=abc.ABCMeta):
    def __init__(self, trunk, body, head):
        super().__init__()
        self.trunk = trunk
        self.body = body
        self.head = head

    def forward(self, obs_dict, internals):
        embedding = self.trunk.forward(obs_dict)
        pre_result, internals = self.body.forward(embedding, internals)
        result, internals = self.head.forward(pre_result, internals)
        return result, internals

    def new_internals(self, device):
        return self.body.new_internals(device)


class NetworkBody(Module, metaclass=abc.ABCMeta):
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


class NetworkTrunk(Module):
    def __init__(self, pathways_by_name):
        super().__init__()

        nb_output_channel = 0
        self.pathways = torch.nn.ModuleList()
        for name, pathway in pathways_by_name.items():
            nb_output_channel += pathway.nb_output_channel
            self.pathways.add_module(name, pathway)
        self.nb_output_channel = nb_output_channel

    def forward(self, obs_dict):
        """
        :param obs_dict: a Dict[str: Tensor] mapping the pathname to observation Tensors
        :return: a Tensor embedding
        """
        embeddings = []
        for name, pathway in self.pathways.named_children():
            embeddings.append(pathway.forward(obs_dict[name]))
        return torch.cat(embeddings, dim=1)


class InputNetwork(torch.nn.Module, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def nb_output_channel(self):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_args(cls, nb_in_channel, args):
        raise NotImplementedError
