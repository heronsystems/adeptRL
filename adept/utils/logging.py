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
import logging
import os
from collections import namedtuple
from datetime import datetime
from time import time

import torch

from adept.globals import VERSION
from adept.utils.util import HeapQueue


def print_ascii_logo():
    version_len = len(VERSION)
    print(
        """
                     __           __
          ____ _____/ /__  ____  / /_
         / __ `/ __  / _ \/ __ \/ __/
        / /_/ / /_/ /  __/ /_/ / /_
        \__,_/\__,_/\___/ .___/\__/
                       /_/           """ + '\n' +
        '                                     ' [:-(version_len + 2)] +
        'v{} '.format(VERSION)
    )


def make_log_id(tag, mode_name, agent_name, network_name):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if tag:
        log_id = '_'.join([tag, mode_name, agent_name, network_name, timestamp])
    else:
        log_id = '_'.join([mode_name, agent_name, network_name, timestamp])
    return log_id


def make_log_id_from_timestamp(
    tag, mode_name, agent_name, network_name, timestamp
):
    if tag:
        log_id = '_'.join([tag, mode_name, agent_name, network_name, timestamp])
    else:
        log_id = '_'.join([mode_name, agent_name, network_name, timestamp])
    return log_id


def make_logger(logger_name, log_file):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s  [%(levelname)s] %(message)s')
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


def log_args(logger, args):
    args = args if isinstance(args, dict) else vars(args)
    for k, v in args.items():
        logger.info('{}: {}'.format(k, v))


def write_args_file(log_id_dir, args):
    args = args if isinstance(args, dict) else vars(args)
    with open(os.path.join(log_id_dir, 'args.json'), 'w') as args_file:
        json.dump(args, args_file, indent=4, sort_keys=True)


class ModelSaver:
    BufferEntry = namedtuple(
        'BufferEntry', ['reward', 'priority', 'network', 'optimizer']
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
                    'model_{}_{}.pth'.format(j + 1, int(buff_entry.reward))
                )
            )
            torch.save(
                buff_entry.optimizer,
                os.path.join(
                    save_dir,
                    'optimizer_{}_{}.pth'.format(j + 1, int(buff_entry.reward))
                )
            )


class SimpleModelSaver:
    def __init__(self, log_id_dir):
        self._log_id_dir = log_id_dir

    def save_state_dicts(self, network, step_count, optimizer=None):
        save_dir = os.path.join(self._log_id_dir, str(step_count))
        os.makedirs(save_dir)
        torch.save(
            network.state_dict(),
            os.path.join(save_dir, 'model_{}.pth'.format(step_count))
        )
        if optimizer is not None:
            torch.save(
                optimizer.state_dict(),
                os.path.join(save_dir, 'optimizer_{}.pth'.format(step_count))
            )
