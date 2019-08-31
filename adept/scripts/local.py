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

Local Mode

Train an agent with a single GPU.

Usage:
    local [options]
    local --resume <path>
    local (-h | --help)

Agent Options:
    --agent <str>           Name of agent class [default: ActorCritic]

Environment Options:
    --env <str>             Environment name [default: PongNoFrameskip-v4]
    --rwd-norm <str>        Reward normalizer name [default: Clip]

Script Options:
    --gpu-id <int>          CUDA device ID of GPU [default: 0]
    --nb-env <int>          Number of parallel env [default: 64]
    --seed <int>            Seed for random variables [default: 0]
    --nb-step <int>         Number of steps to train for [default: 10e6]
    --load-network <path>   Path to network file (for pretrained weights)
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

from absl import flags

from adept.container import Init, Local
from adept.utils.script_helpers import (
    parse_none, parse_path
)
from adept.utils.util import DotDict

# hack to use bypass pysc2 flags
FLAGS = flags.FLAGS
FLAGS(['local.py'])

MODE = 'Local'


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
    args.gpu_id = int(args.gpu_id)
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
    Run local training.
    :param args: Dict[str, Any]
    :return:
    """
    args, log_id_dir, initial_step, logger = Init.main(MODE, args)

    container = Local(args, logger, log_id_dir, initial_step)

    if args.profile:
        try:
            from pyinstrument import Profiler
        except:
            raise ImportError('You must install pyinstrument to use profiling.')
        container.nb_step = 10e3
        profiler = Profiler()
        profiler.start()

    try:
        container.run()
    finally:
        if args.profile:
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))
        container.close()

    if args.eval:
        import subprocess
        command = [
            'python',
            '-m',
            'adept.scripts.evaluate',
            '--log-id-dir',
            container.log_id_dir,
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
