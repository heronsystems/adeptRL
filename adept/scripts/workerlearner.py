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
             __           __
  ____ _____/ /__  ____  / /_
 / __ `/ __  / _ \/ __ \/ __/
/ /_/ / /_/ /  __/ /_/ / /_
\__,_/\__,_/\___/ .___/\__/
               /_/

Worker Learner Mode

Train an agent using multiple servers and GPUs via Workers and Learners.

Usage:
    actorlearner [options]
    actorlearner (-h | --help)

Distributed Options:
    --nb-learner <int>       Number of distributed nodes [default: 1]
    --nb-worker <int>        ID of the node for multi-node training [default: 0]
    --ray-addr <str>         Master node (rank 0's) address [default: 127.0.0.1]
    --ray-port <int>         Master node (rank 0's) comm port [default: 6008]
    --nccl-addr <str>        IP address of NCCL host [default: 127.0.0.1]
    --nccl-port <int>        Port to use for NCCL comms [default: 6009]

Topology Options:
    --actor-host <str>        Name of host actor [default: ImpalaHostActor]
    --actor-worker <str>      Name of worker actor [default: ImpalaWorkerActor]
    --learner <str>           Name of learner [default: ImpalaLearner]
    --exp <str>               Name of host experience cache [default: Rollout]
    --nb-learn-batch <int>    Number of worker batches to learn on [default: 2]

Environment Options:
    --env <str>               Environment name [default: PongNoFrameskip-v4]
    --rwd-norm <str>          Reward normalizer name [default: Clip]

Script Options:
    --gpu-ids <ids>         Comma-separated CUDA IDs [default: 0,1]
    --nb-env <int>          Number of env per Tower [default: 16]
    --seed <int>            Seed for random variables [default: 0]
    --nb-step <int>         Number of steps to train for [default: 10e6]
    --load-network <path>   Path to network file
    --load-optim <path>     Path to optimizer file
    --resume <path>         Resume training from log ID .../<logdir>/<env>/<log-id>/
    --config <path>         Use a JSON config file for arguments
    --eval                  Run an evaluation after training
    --prompt                Prompt to modify arguments

Network Options:
    --net1d <str>           Network to use for 1d input [default: Identity1D]
    --net2d <str>           Network to use for 2d input [default: Identity2D]
    --net3d <str>           Network to use for 3d input [default: FourConv]
    --net4d <str>           Network to use for 4d input [default: Identity4D]
    --netbody <str>         Network to use on merged inputs [default: LSTM]
    --head1d <str>          Network to use for 1d output [default: Identity1D]
    --head2d <str>          Network to use for 2d output [default: Identity2D]
    --head3d <str>          Network to use for 3d output [default: Identity3D]
    --head4d <str>          Network to use for 4d output [default: Identity4D]
    --custom-network        Name of custom network class

Optimizer Options:
    --lr <float>               Learning rate [default: 0.0007]

Logging Options:
    --tag <str>                Name your run [default: None]
    --logdir <path>            Path to logging directory [default: /tmp/adept_logs/]
    --epoch-len <int>          Save a model every <int> frames [default: 1e6]
    --summary-freq <int>       Tensorboard summary frequency [default: 10]

Troubleshooting Options:
    --profile                 Profile this script
"""
import os
import subprocess
import sys
from itertools import chain

import ray

from adept.container import Init
from adept.utils.script_helpers import (
    parse_list_str, parse_path, parse_none, LogDirHelper
)
from adept.utils.util import DotDict
from adept.registry import REGISTRY as R
from adept.container import Worker, Learner

MODE = 'WorkerLearner'


def parse_args():
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.strip('--').replace('-', '_'): v for k, v in args.items()}
    del args['h']
    del args['help']
    args = DotDict(args)

    args.nb_learner = int(args.nb_learner)
    args.nb_worker = int(args.nb_worker)
    args.ray_port = int(args.ray_port)
    args.nccl_port = int(args.nccl_port)

    # Ignore other args if resuming
    if args.resume:
        args.resume = parse_path(args.resume)
        return args

    if args.config:
        args.config = parse_path(args.config)

    # Container Options
    args.logdir = parse_path(args.logdir)
    args.gpu_ids = parse_list_str(args.gpu_ids, int)
    args.nb_env = int(args.nb_env)
    args.seed = int(args.seed)
    args.nb_step = int(float(args.nb_step))
    args.nb_learn_batch = int(args.nb_learn_batch)

    # Logging Options
    args.tag = parse_none(args.tag)
    args.summary_freq = int(args.summary_freq)
    args.lr = float(args.lr)
    args.epoch_len = int(float(args.epoch_len))

    # Troubleshooting Options
    args.profile = bool(args.profile)
    return args


def flatten(items):
    return chain.from_iterable(items)


def main(args):
    """
    Run impala training.

    :param args: Dict[str, Any]
    :return:
    """
    args = DotDict(args)

    args, log_id_dir, initial_step, logger = Init.main(MODE, args)
    R.save_extern_classes(log_id_dir)

    ray.init(address='{}:{}'.format(args.ray_addr, args.ray_port))
    learner_ranks = list(range(args.nb_learner))
    worker_ranks = list(range(args.nb_learner, args.nb_learner + args.nb_worker))

    actor_args = {k: v for k, v in args.items()}

    # instantiate as many hosts/workers as requested
    learners = [
        ray.remote(num_gpus=1)(Learner).remote(
            actor_args,
            rank, learner_ranks, worker_ranks
        )
        for rank in learner_ranks
    ]
    workers = {
        rank: ray.remote(num_gpus=0.25)(Worker).remote(
            actor_args,
            rank, learner_ranks, worker_ranks
        )
        for rank in worker_ranks
    }

    # all workers compute exp
    undone_ranks = [worker.step.remote() for worker in workers.values()]

    for step in range(args.nb_step):
        l_steps = []
        all_w_ranks = []
        for l_rank, learner in enumerate(learners):
            # host wait for n to be ready
            w_ranks, undone_ranks = ray.wait(undone_ranks, num_returns=args.nb_learn_batch)
            w_ranks = [ray.get(rank) for rank in w_ranks]
            learner.sync_exps.remote(w_ranks)
            for rank in w_ranks:
                workers[rank].sync_exp.remote(l_rank)
            # when ready, merge batch and learn
            l_steps.append(learner.step.remote(step, w_ranks))
            all_w_ranks.append(w_ranks)

        # sync learns
        # wait for all learners to step and sync
        [ray.get(l_step) for l_step in l_steps]

        # sync networks
        for w_ranks, l_rank, learner in zip(
                all_w_ranks,
                range(len(learners)),
                learners
        ):
            learner.sync_network.remote(w_ranks)
            for rank in w_ranks:
                workers[rank].sync_network.remote(l_rank)

        # unblock workers
        for rank in flatten(all_w_ranks):
            undone_ranks.append(workers[rank].step.remote())

    if args.eval:
        from adept.scripts.evaluate import main
        eval_args = {
            'log_id_dir': log_id_dir,
            'gpu_id': 0,
            'nb_episode': 30,
        }
        if args.custom_network:
            eval_args['custom_network'] = args.custom_network
        main(eval_args)


if __name__ == '__main__':
    main(parse_args())
