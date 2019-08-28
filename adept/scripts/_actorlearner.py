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
import argparse
import json
import os

import torch
import torch.distributed as dist

from adept.container import Init
from adept.registry import REGISTRY as R
from adept.manager.subproc_env_manager import SubProcEnvManager
from adept.network.modular_network import ModularNetwork
from adept.utils.script_helpers import (
    LogDirHelper
)
from adept.utils.util import DotDict

MODE = 'ActorLearner'
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
GLOBAL_RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
NB_NODE = int(os.environ['NB_NODE'])

# hack to use argparse for SC2
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['local.py'])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-id-dir', required=True)
    parser.add_argument(
        '--resume', type=str2bool, nargs='?', const=True, default=False
    )
    parser.add_argument('--load-network', default=None)
    parser.add_argument('--load-optim', default=None)
    parser.add_argument('--initial-step-count', type=int, default=0)
    args = parser.parse_args()
    return args


def main(local_args,):
    """
    Run local training.

    :param args: Dict[str, Any]
    :return:
    """
    # host needs to broadcast timestamp so all procs create the same log dir
    log_id_dir = local_args.log_id_dir
    initial_step_count = local_args.initial_step_count

    R.load_extern_classes(log_id_dir)
    logger = Init.setup_logger(
        MODE + str(LOCAL_RANK),
        log_id_dir,
        'train{}'.format(GLOBAL_RANK)
    )

    helper = LogDirHelper(log_id_dir)
    with open(helper.args_file_path(), 'r') as args_file:
        args = DotDict(json.load(args_file))

    if local_args.resume:
        args = DotDict({**args, **vars(local_args)})

    dist.init_process_group(
        backend='gloo',
        world_size=WORLD_SIZE,
        rank=LOCAL_RANK
    )
    logger.info('Rank {} initialized.'.format(GLOBAL_RANK))

    if LOCAL_RANK == 0:
        container = ImpalaHost(
            # TODO
        )
    else:
        container = ImpalaWorker(
            # TODO
        )

    container.run(args.nb_step, initial_count=initial_step_count)
    env.close()


if __name__ == '__main__':
    main(parse_args())
