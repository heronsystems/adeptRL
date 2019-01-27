#!/usr/bin/env python
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
"""
Distributed worker script. Called from launcher (distributed.py).
"""
import os

import torch
import torch.distributed as dist

from adept.agents.agent_registry import AgentRegistry
from adept.environments.env_registry import EnvRegistry
from adept.networks.network_registry import NetworkRegistry
from adept.utils.script_helpers import (
    count_parameters, parse_none, LogDirHelper, parse_path, parse_bool_str
)
from adept.utils.util import DotDict


WORLD_SIZE = int(os.environ['WORLD_SIZE'])
GLOBAL_RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-id-dir', required=True)
    parser.add_argument('--local_rank', required=True)
    args = parser.parse_args()
    return args


def main(
    args,
    agent_registry=AgentRegistry(),
    env_registry=EnvRegistry(),
    net_registry=NetworkRegistry()
):
    """
    Run distributed training.

    :param args: Dict[str, Any]
    :param agent_registry: AgentRegistry
    :param env_registry: EnvRegistry
    :param net_registry: NetworkRegistry
    :return:
    """
    dist.init_process_group(
        backend='nccl',
        init_method='file:///tmp/adept_init',
        world_size=WORLD_SIZE,
        rank=LOCAL_RANK
    )

    torch.cuda.set_device(LOCAL_RANK)
    print(dist.get_rank())
    print(dist.get_world_size())
    a = torch.tensor(5.).cuda()

    dist.all_reduce_multigpu([a])

    print(a)


if __name__ == '__main__':
    main(parse_args())
