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

Ray Mode

Train an agent with rollout workers distributed and run by ray.

Usage:
    ray [options]
    ray --resume <path>
    ray (-h | --help)

Distributed Options:
    --nb-learners <int>         Number of distributed learners [default: 2]
    --nb-workers <int>          Number of distributed workers per learner [default: 2]
    --colocate-workers <bool>   Colocate workers with actors [default: False]
    --ray-addr <str>            Ray head node address, None for local [default: None]
    --nccl-timeout <int>        Seconds to wait for any NCCL op before timeout [default: 30]

Topology Options:
    --worker-type <str>          Name of distributed workers [default: RolloutWorker]
    --actor-worker <str>         Name of worker actor [default: ImpalaWorkerActor]
    --learner <str>              Name of learner [default: ImpalaLearner]
    --exp-worker <str>           Name of worker experience cache [default: ImpalaRollout]
    --nb-env <int>               Number of env per worker [default: 32]
    --worker-rollout-len <int>   Number of steps to include in a worker rollout [default: 20]
    --worker-cpu-alloc <int>     Number of cpus for each rollout worker [default: 32]
    --worker-gpu-alloc <float>   Number of gpus for each rollout worker [default: 0.25]
    --learner-cpu-alloc <int>     Number of cpus for each learner [default: 1]
    --learner-gpu-alloc <float>   Number of gpus for each learner [default: 1]
    --nb-rollouts-in-batch <int>  Number of rollouts per batch [default: 2]
    --rollout-queue-size <int>   Max length of rollout queue before blocking [default: 4]

Environment Options:
    --env <str>               Environment name [default: PongNoFrameskip-v4]
    --rwd-norm <str>        Reward normalizer name [default: Clip]

Script Options:
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
import ray

from adept.container import Init
from adept.experimental.ray_nccl_container import RayContainer, RayPeerLearnerContainer
from adept.experimental.rollout_worker import RolloutWorker
from adept.utils.script_helpers import (
    parse_list_str, parse_path, parse_none, LogDirHelper, parse_bool_str
)
from adept.utils.util import DotDict

MODE = 'Ray'


def parse_args():
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.strip('--').replace('-', '_'): v for k, v in args.items()}
    del args['h']
    del args['help']
    args = DotDict(args)

    # Ignore other args if resuming
    if args.resume:
        args.resume = parse_path(args.resume)
        return args

    if args.config:
        args.config = parse_path(args.config)

    args.logdir = parse_path(args.logdir)
    args.nb_env = int(args.nb_env)
    args.seed = int(args.seed)
    args.nb_step = int(float(args.nb_step))
    args.tag = parse_none(args.tag)
    args.summary_freq = int(args.summary_freq)
    args.lr = float(args.lr)
    args.epoch_len = int(float(args.epoch_len))
    args.profile = bool(args.profile)

    args.ray_addr = parse_none(args.ray_addr)
    args.nb_learners = int(args.nb_learners)
    args.nb_workers = int(args.nb_workers)
    args.colocate_workers = parse_bool_str(args.colocate_workers)
    args.learner_cpu_alloc = int(args.learner_cpu_alloc)
    args.learner_gpu_alloc = float(args.learner_gpu_alloc)
    args.worker_cpu_alloc = int(args.worker_cpu_alloc)
    args.worker_gpu_alloc = float(args.worker_gpu_alloc)
    args.worker_rollout_len = int(args.worker_rollout_len)
    args.nccl_timeout = int(args.nccl_timeout)

    args.nb_rollouts_in_batch = int(args.nb_rollouts_in_batch)
    args.rollout_queue_size = int(args.rollout_queue_size)

    # arg checking
    assert args.nb_rollouts_in_batch <= args.nb_workers, 'WARNING: nb_rollouts_in_batch must be <= nb_workers. Got {} <= {}' \
           .format(args.nb_rollouts_in_batch, args.nb_workers)
    return args


def main(args):
    """
    Run ray training.
    :param args: Dict[str, Any]
    :return:
    """
    args, log_id_dir, initial_step, logger = Init.main(MODE, args)

    # start ray
    if args.ray_addr is not None:
        ray.init(address=args.ray_addr)
        print('Using Ray on a cluster. Head node address: {}'.format(args.ray_addr))
    else:
        print('Using Ray on a single machine.')
        ray.init()

    # create a main learner which logs summaries and saves weights
    main_learner_cls = RayContainer.as_remote(num_cpus=args.learner_cpu_alloc,
                                              num_gpus=args.learner_gpu_alloc)
    main_learner = main_learner_cls.remote(
        args, logger, log_id_dir, initial_step, rank=0
    )

    # if multiple learners setup nccl
    if args.nb_learners > 1:
        # create N peer learners
        peer_learners = []
        for p_ind in range(args.nb_learners - 1):
            # TODO: learner GPU alloc from args
            remote_cls = RayPeerLearnerContainer.as_remote(num_cpus=args.learner_cpu_alloc,
                                                           num_gpus=args.learner_gpu_alloc)
            # init
            remote = remote_cls.remote(args, initial_step, rank=p_ind + 1)
            peer_learners.append(remote)

        # figure out main learner node ip
        nccl_addr, nccl_ip, nccl_port = ray.get(main_learner._rank0_nccl_port_init.remote())

        # setup all nccls
        nccl_inits = [main_learner._nccl_init.remote(nccl_addr, nccl_ip, nccl_port)]
        nccl_inits.extend([p._nccl_init.remote(nccl_addr, nccl_ip, nccl_port) for p in peer_learners])
        # wait for all
        ray.get(nccl_inits)
        print('NCCL initialized')

        # have all sync parameters
        [f._sync_peer_parameters.remote() for f in peer_learners]
        main_learner._sync_peer_parameters.remote()

        # main_learner must be first index
        learners = [main_learner] + peer_learners

    # create workers
    # TODO: actually lookup from registry
    workers = [RolloutWorker.as_remote(num_cpus=args.worker_cpu_alloc,
                                       num_gpus=args.worker_gpu_alloc)
                    .remote(args, initial_step, w_ind)
                    for w_ind in range(args.nb_workers)]

    # synchronize worker variables
    main_learner.synchronize_worker_parameters.remote(workers, initial_step, blocking=True)

    if args.profile:
        # TODO: for profiling main learner shouldn't be remote
        # or run method must implement pyinstrument profiling
        raise NotImplementedError('TODO')

    try:
        # startup the run method of all containers
        runs = [f.run.remote(workers) for f in learners]
        done_training = ray.wait(runs)

    finally:
        # if args.profile:
            # profiler.stop()
            # print(profiler.output_text(unicode=True, color=True))

        closes = [f.close.remote() for f in learners]

    if args.eval:
        import subprocess
        command = [
            'python',
            '-m',
            'adept.scripts.evaluate',
            '--log-id-dir',
            log_id_dir,
            '--gpu-id',
            str(args.gpu_id),
            '--nb-episode',
            str(30)
        ]
        if args.custom_network:
            command += [
                '--custom-network',
                args.custom_network
            ]
        exit(subprocess.call(command, env=os.environ))


if __name__ == '__main__':
    main(parse_args())

