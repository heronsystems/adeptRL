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

Distributed Mode

Train an agent with multiple GPUs locally or on a cluster.
More info:
* https://pytorch.org/docs/stable/distributed.html
* https://pytorch.org/tutorials/intermediate/dist_tuto.html

Usage:
    distrib [options]
    distrib --resume <path>
    distrib (-h | --help)

Distributed Options:
    --nb-node <int>         Number of distributed nodes [default: 1]
    --node-rank <int>       ID of the node for multi-node training [default: 0]
    --nb-proc <int>         Number of processes per node [default: 2]
    --master-addr <str>     Master node (rank 0's) address [default: 127.0.0.1]
    --master-port <int>     Master node (rank 0's) comm port [default: 29500]

Agent Options:
    --agent <str>           Name of agent class [default: ActorCritic]

Environment Options:
    --env <str>             Environment name [default: PongNoFrameskip-v4]
    --rwd-norm <str>        Reward normalizer name [default: Clip]

Script Options:
    --nb-env <int>          Number of parallel env [default: 32]
    --seed <int>            Seed for random variables [default: 0]
    --nb-step <int>         Number of steps to train for [default: 10e6]
    --load-network <path>   Path to network file
    --load-optim <path>     Path to optimizer file
    --resume <path>         Resume training from log ID .../<logdir>/<env>/<log-id>/
    --eval                  Run an evaluation after training
    -y, --use-defaults      Skip prompts, use defaults

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
    --custom-network <str>  Name of custom network class

Optimizer Options:
    --lr <float>            Learning rate [default: 0.0007]

Logging Options:
    --tag <str>             Name your run [default: None]
    --logdir <path>         Path to logging directory [default: /tmp/adept_logs/]
    --epoch-len <int>       Save a model every <int> frames [default: 1e6]
    --nb-eval-env <int>     Evaluate agent in a separate thread [default: 0]
    --summary-freq <int>    Tensorboard summary frequency [default: 10]

Troubleshooting Options:
    --profile               Profile this script
"""
import os
import subprocess
import sys

from adept.container import Init
from adept.utils.script_helpers import parse_path, parse_none
from adept.utils.util import DotDict

MODE = 'Distrib'


def parse_args():
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.strip('--').replace('-', '_'): v for k, v in args.items()}
    del args['h']
    del args['help']

    args = DotDict(args)

    args.nb_node = int(args.nb_node)
    args.node_rank = int(args.node_rank)
    args.nb_proc = int(args.nb_proc)
    args.master_port = int(args.master_port)

    if args.resume:
        args.resume = parse_path(args.resume)
        return args

    args.logdir = parse_path(args.logdir)
    args.nb_env = int(args.nb_env)
    args.seed = int(args.seed)
    args.nb_step = int(float(args.nb_step))
    args.tag = parse_none(args.tag)
    args.nb_eval_env = int(args.nb_eval_env)
    args.summary_freq = int(args.summary_freq)
    args.lr = float(args.lr)
    args.epoch_len = int(float(args.epoch_len))
    args.profile = bool(args.profile)
    return args


def main(args):
    """
    Run distributed training.
    :param args: Dict[str, Any]
    :return:
    """
    args = DotDict(args)

    dist_world_size = args.nb_proc * args.nb_node

    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    if args.resume:
        args, log_id_dir, initial_step = Init.from_resume(MODE, args)
    elif args.use_defaults:
        args, log_id_dir, initial_step = Init.from_defaults(MODE, args)
    else:
        args, log_id_dir, initial_step = Init.from_prompt(MODE, args)

    Init.print_ascii_logo()
    Init.make_log_dirs(log_id_dir)
    Init.write_args_file(log_id_dir, args)
    logger = Init.setup_logger(MODE, log_id_dir)
    Init.log_args(logger, args)

    processes = []

    for local_rank in range(0, args.nb_proc):
        # each process's rank
        dist_rank = args.nb_proc * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        if not args.resume:
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "adept.scripts._distrib",
                "--log-id-dir={}".format(log_id_dir)
            ]
        else:
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "adept.scripts._distrib",
                "--log-id-dir={}".format(log_id_dir),
                "--resume={}".format(True),
                "--load-network={}".format(args.load_network),
                "--load-optim={}".format(args.load_optim),
                "--initial-step-count={}".format(initial_step)
            ]
        if args.custom_network:
            cmd += [
                '--custom-network',
                args.custom_network
            ]

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()

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
