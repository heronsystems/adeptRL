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
import os

from adept.networks import VISION_NETWORKS, DISCRETE_NETWORKS, NETWORK_BODIES
from adept.networks._base import NetworkTrunk, ModularNetwork, NetworkHead


def make_network(
    observation_space,
    network_head_shapes,
    args,
    chw_networks=VISION_NETWORKS,
    c_networks=DISCRETE_NETWORKS,
    network_bodies=NETWORK_BODIES,
    embedding_size=512
):
    pathways_by_name = {}
    nbr = observation_space.names_by_rank
    ebn = observation_space.entries_by_name
    for rank, names in nbr.items():
        for name in names:
            if rank == 1:
                pathways_by_name[name] = c_networks[args.net1d]\
                    .from_args(ebn[name].shape, args)
            elif rank == 2:
                raise NotImplementedError('Rank 2 inputs not implemented')
            elif rank == 3:
                pathways_by_name[name] = chw_networks[args.net3d]\
                    .from_args(ebn[name].shape, args)
            elif rank == 4:
                raise NotImplementedError('Rank 4 inputs not implemented')
            else:
                raise NotImplementedError(
                    'Rank {} inputs not implemented'.format(rank)
                )

    trunk = NetworkTrunk(pathways_by_name)
    body = network_bodies[args.netbody].from_args(
        trunk.nb_output_channel, embedding_size, args
    )
    head = NetworkHead(body.nb_output_channel, network_head_shapes)
    network = ModularNetwork(trunk, body, head)
    return network


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_bool_str(bool_str):
    """
    Convert string to boolean.

    :param bool_str: str
    :return: Bool
    """
    if bool_str.lower() == 'false':
        return False
    elif bool_str.lower() == 'true':
        return True
    else:
        raise ValueError('Unable to parse "{}"'.format(bool_str))


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_list_str(list_str, item_type):
    items = list_str.split(',')
    return [item_type(item) for item in items]


def parse_none(none_str):
    if none_str == 'None':
        return None
    else:
        return none_str


class LogDirHelper:
    def __init__(self, log_id_path):
        """
        :param log_id_path: str Path to Log ID
        """
        self._log_id_path = log_id_path

    def epochs(self):
        return [
            int(epoch)
            for epoch in os.listdir(self._log_id_path)
            if os.path.isdir(os.path.join(self._log_id_path, epoch))
        ]

    def latest_epoch(self):
        return max(self.epochs())

    def latest_epoch_path(self):
        return os.path.join(self._log_id_path, str(self.latest_epoch()))

    def latest_network_path(self):
        network_file = [
            f for f in os.listdir(self.latest_epoch_path())
            if ('model' in f)
        ][0]
        return os.path.join(self.latest_epoch_path(), network_file)

    def latest_optim_path(self):
        optim_file = [
            f for f in os.listdir(self.latest_epoch_path())
            if ('optim' in f)
        ][0]
        return os.path.join(self.latest_epoch_path(), optim_file)

    def epoch_path_at_epoch(self, epoch):
        return os.path.join(self._log_id_path, str(epoch))

    def network_path_at_epoch(self, epoch):
        epoch_path = self.epoch_path_at_epoch(epoch)
        network_file = [
            f for f in os.listdir(epoch_path)
            if ('model' in f)
        ][0]
        return os.path.join(epoch_path, network_file)

    def optim_path_at_epoch(self, epoch):
        epoch_path = self.epoch_path_at_epoch(epoch)
        optim_file = [
            f for f in os.listdir(epoch_path)
            if ('optim' in f)
        ][0]
        return os.path.join(epoch_path, optim_file)

    def timestamp(self):
        splits = self._log_id_path.split('_')
        timestamp = splits[-2] + '_' + splits[-1]
        return timestamp

    def args_file_path(self):
        return os.path.join(self._log_id_path, 'args.json')
