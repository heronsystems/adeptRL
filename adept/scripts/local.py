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
import json
import os

import torch
from absl import flags
from tensorboardX import SummaryWriter

from adept.registry import REGISTRY
from adept.container import Local
from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.utils.logging import (
    make_log_id, make_logger, print_ascii_logo, log_args, write_args_file,
    SimpleModelSaver
)
from adept.utils.script_helpers import (
    count_parameters, parse_none, LogDirHelper, parse_path
)
from adept.utils.util import DotDict

# hack to use bypass pysc2 flags
FLAGS = flags.FLAGS
FLAGS(['local.py'])


def parse_args():
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.strip('--').replace('-', '_'): v for k, v in args.items()}
    del args['h']
    del args['help']
    args = DotDict(args)

    # Ignore other args if resuming
    if args.resume:
        return args

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
    args = DotDict(args)
    initial_step_count = 0
    if args.resume:
        log_dir_helper = LogDirHelper(args.resume)

        with open(log_dir_helper.args_file_path(), 'r') as args_file:
            args = DotDict(json.load(args_file))

        args.load_network = log_dir_helper.latest_network_path()
        args.load_optim = log_dir_helper.latest_optim_path()
        initial_step_count = log_dir_helper.latest_epoch()

        log_id = make_log_id(
            args.tag, 'Local', args.agent, args.net3d + args.netbody,
            timestamp=log_dir_helper.timestamp()
        )
    else:
        if args.use_defaults:
            agent_args = REGISTRY.lookup_agent(args.agent).args
            env_args = REGISTRY.lookup_env_class(args.env).args
            rwdnorm_args = REGISTRY.lookup_reward_normalizer(args.rwd_norm).args
            if args.custom_network:
                net_args = REGISTRY.lookup_network(
                    args.custom_network).args
            else:
                net_args = REGISTRY.lookup_modular_args(args)
        else:
            agent_args = REGISTRY.lookup_agent(args.agent).prompt()
            env_args = REGISTRY.lookup_env(args.env).prompt()
            rwdnorm_args = REGISTRY.lookup_reward_normalizer(
                args.rwd_norm).prompt()
            if args.custom_network:
                net_args = REGISTRY.lookup_network(
                    args.custom_network).prompt()
            else:
                net_args = REGISTRY.prompt_modular_args(args)
        args = DotDict({
            **args, **agent_args, **env_args, **rwdnorm_args, **net_args
        })

        # construct logging objects
        log_id = make_log_id(
            args.tag, 'Local', args.agent, args.net3d + args.netbody
        )

    log_id_dir = os.path.join(args.logdir, args.env, log_id)

    os.makedirs(log_id_dir, exist_ok=True)
    logger = make_logger('Local', os.path.join(log_id_dir, 'train_log.txt'))
    summary_writer = SummaryWriter(log_id_dir)
    saver = SimpleModelSaver(log_id_dir)
    print_ascii_logo()
    log_args(logger, args)
    write_args_file(log_id_dir, args)

    # construct env
    env = SubProcEnvManager.from_args(args, registry=REGISTRY)

    # construct network
    torch.manual_seed(args.seed)
    output_space = REGISTRY.lookup_output_space(
        args.agent, env.action_space
    )
    if args.custom_network:
        network = REGISTRY.lookup_network(args.custom_network).from_args(
            args,
            env.observation_space,
            output_space,
            env.gpu_preprocessor,
            REGISTRY
        )
    else:
        network = ModularNetwork.from_args(
            args,
            env.observation_space,
            output_space,
            env.gpu_preprocessor,
            REGISTRY
        )

    # possibly load network
    if args.load_network:
        network.load_state_dict(
            torch.load(
                args.load_network, map_location=lambda storage, loc: storage
            )
        )
        logger.info('Reloaded network from {}'.format(args.load_network))

    device = torch.device(
        "cuda:{}".format(args.gpu_id)
        if (torch.cuda.is_available() and args.gpu_id >= 0)
        else "cpu"
    )
    torch.backends.cudnn.benchmark = True

    # construct agent
    agent = REGISTRY.lookup_agent(args.agent).from_args(
        args,
        REGISTRY.lookup_reward_normalizer(args.rwd_norm).from_args(args),
        env.action_space
    )

    # Construct the Container
    def make_optimizer(params):
        opt = torch.optim.RMSprop(
            params, lr=args.lr, eps=1e-5, alpha=0.99
        )
        if args.load_optim:
            opt.load_state_dict(
                torch.load(
                    args.load_optim,
                    map_location=lambda storage, loc: storage
                )
            )
            logger.info("Reloaded optimizer from {}".format(args.load_optim))
        return opt

    logger.info('Network Parameter Count: {}'.format(count_parameters(network)))
    container = Local(
        agent, env, network, make_optimizer, args.epoch_len, args.nb_env, logger,
        summary_writer, args.summary_freq, saver, device
    )

    # Run the container
    if args.profile:
        try:
            from pyinstrument import Profiler
        except:
            raise ImportError('You must install pyinstrument to use profiling.')
        profiler = Profiler()
        profiler.start()
        container.run(10e3)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
    else:
        container.run(args.nb_step, initial_count=initial_step_count)
    env.close()

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
            str(args.nb_env)
        ]
        if args.custom_network:
            command += [
                '--custom-network',
                args.custom_network
            ]
        exit(subprocess.call(command, env=os.environ))


if __name__ == '__main__':
    main(parse_args())
