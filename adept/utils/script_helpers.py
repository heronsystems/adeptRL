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


def parse_list_str(list_str, item_type):
    items = list_str.split(',')
    return [item_type(item) for item in items]


def parse_none(none_str):
    if none_str == 'None':
        return None
    else:
        return none_str


def parse_path(rel_path):
    """
    :param rel_path: (str) relative path
    :return: (str) absolute path
    """
    return os.path.abspath(rel_path)


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
               and 'rank' not in os.path.join(self._log_id_path, epoch)
        ]

    def latest_epoch(self):
        epochs = self.epochs()
        return max(epochs) if epochs else 0

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
