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
from collections import namedtuple
from time import time

import torch

from adept.utils.util import HeapQueue


class ModelSaver:
    BufferEntry = namedtuple(
        "BufferEntry", ["reward", "priority", "network", "optimizer"]
    )

    def __init__(self, nb_top_model, log_id_dir):
        self.nb_top_model = nb_top_model
        self._buffer = HeapQueue(nb_top_model)
        self._log_id_dir = log_id_dir

    def append_if_better(self, reward, network, optimizer):
        self._buffer.push(
            self.BufferEntry(
                reward, time(), network.state_dict(), optimizer.state_dict()
            )
        )

    def write_state_dicts(self, epoch_id):
        save_dir = os.path.join(self._log_id_dir, str(epoch_id))
        if len(self._buffer) > 0:
            os.makedirs(save_dir)
        for j, buff_entry in enumerate(self._buffer.flush()):
            torch.save(
                buff_entry.network,
                os.path.join(
                    save_dir,
                    "model_{}_{}.pth".format(j + 1, int(buff_entry.reward)),
                ),
            )
            torch.save(
                buff_entry.optimizer,
                os.path.join(
                    save_dir,
                    "optimizer_{}_{}.pth".format(j + 1, int(buff_entry.reward)),
                ),
            )


class SimpleModelSaver:
    def __init__(self, log_id_dir):
        self._log_id_dir = log_id_dir

    def save_state_dicts(self, network, step_count, optimizer=None):
        save_dir = os.path.join(self._log_id_dir, str(step_count))
        os.makedirs(save_dir)
        torch.save(
            network.state_dict(),
            os.path.join(save_dir, "model_{}.pth".format(step_count)),
        )
        if optimizer is not None:
            torch.save(
                optimizer.state_dict(),
                os.path.join(save_dir, "optimizer_{}.pth".format(step_count)),
            )
