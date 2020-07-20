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
import json
import os
from time import sleep

from adept.utils.util import DotDict


def parse_bool_str(bool_str):
    """
    Convert string to boolean.

    :param bool_str: str
    :return: Bool
    """
    if bool_str.lower() == "false":
        return False
    elif bool_str.lower() == "true":
        return True
    else:
        raise ValueError('Unable to parse "{}"'.format(bool_str))


def parse_list_str(list_str, item_type):
    items = list_str.split(",")
    return [item_type(item) for item in items]


def parse_none(none_str):
    if none_str == "None":
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
        epochs = []
        for item in os.listdir(self._log_id_path):
            item_path = os.path.join(self._log_id_path, item)
            if os.path.isdir(item_path):
                if item.isnumeric():
                    item_int = int(item)
                    if item_int >= 0:
                        epochs.append(item_int)
        return list(sorted(epochs))

    def latest_epoch(self):
        epochs = self.epochs()
        return max(epochs) if epochs else 0

    def latest_epoch_path(self):
        return os.path.join(self._log_id_path, str(self.latest_epoch()))

    def latest_network_path(self):
        network_file = [
            f for f in os.listdir(self.latest_epoch_path()) if ("model" in f)
        ][0]
        return os.path.join(self.latest_epoch_path(), network_file)

    def latest_optim_path(self):
        optim_file = [
            f for f in os.listdir(self.latest_epoch_path()) if ("optim" in f)
        ][0]
        return os.path.join(self.latest_epoch_path(), optim_file)

    def epoch_path_at_epoch(self, epoch):
        return os.path.join(self._log_id_path, str(epoch))

    def network_path_at_epoch(self, epoch, num_tries=1, retry_delay=3):
        """Find network path at epoch

        Parameters
        ----------
        epoch : int
            epoch to find network path for
        num_tries: int, optional
            number of tries to do, by default 1 (no retries)
        retry_delay : int, optional
            delay between retry attempts, by default 3 (seconds)

        Returns
        -------
        str
            path to network file
        """
        assert num_tries, "num_tries must be greater than 0"

        epoch_path = self.epoch_path_at_epoch(epoch)

        for try_idx in range(num_tries):
            if try_idx > 0:
                sleep(retry_delay)

            network_files = [f for f in os.listdir(epoch_path) if ("model" in f)]

            if len(network_files):
                break
        else:
            raise AssertionError(
                "No network files found at epoch {epoch} for {self._log_id_path} after {num_tries} tries"
            )

        assert len(network_files) <= 1, (
            "More than one network paths at epoch {epoch}, "
            "maybe you want network_paths_at_epoch()"
        )

        network_file = network_files[0]
        return os.path.join(epoch_path, network_file)

    def network_paths_at_epoch(self, epoch):
        epoch_path = self.epoch_path_at_epoch(epoch)
        return [
            os.path.join(epoch_path, f)
            for f in os.listdir(epoch_path)
            if ("model" in f)
        ]

    def optim_path_at_epoch(self, epoch):
        epoch_path = self.epoch_path_at_epoch(epoch)
        optim_file = [f for f in os.listdir(epoch_path) if ("optim" in f)][0]
        return os.path.join(epoch_path, optim_file)

    def timestamp(self):
        splits = self._log_id_path.split("_")
        timestamp = splits[-2] + "_" + splits[-1]
        return timestamp

    def args_file_path(self):
        return os.path.join(self._log_id_path, "args.json")

    def load_args(self):
        with open(self.args_file_path()) as args_file:
            return DotDict(json.load(args_file))

    def eval_path(self):
        return os.path.join(self._log_id_path, "eval.csv")
