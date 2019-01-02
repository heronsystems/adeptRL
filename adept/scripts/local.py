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
    local (-h | --help)

Agent Options:
    --agent <str>           Name of agent class [default: ActorCritic]

Environment Options:
    --env <str>             Environment name [default: PongNoFrameskip-v4]

Network Options:
    --net1d <str>           Network to use for 1d input [default: Identity]
    --net2d <str>           Network to use for 2d input [default: Identity]
    --net3d <str>           Network to use for 3d input [default: FourConv]
    --net4d <str>           Network to use for 4d input [default: Identity]
    --netjunc <str>         Network junction to merge inputs [default: TODO]
    --netbody <str>         Network to use on merged inputs [default: LSTM]
    --load-network <path>   Path to network to load

Optimizer Options:
    --lr <float>            Learning rate [default: 0.0007]

Container Options:
    --gpu-id <id>           CUDA device ID of GPU [default: 0]
    --nb-env <int>          Number of parallel environments [default: 64]
    --seed <int>            Seed for random variables [default: 0]
    --nb-train-frame <int>  Number of frames to train on [default: 10e6]

Logging Options:
    --tag <str>             Name your run [default: None]
    --logdir <path>         Path to logging directory [default: /tmp/adept_logs/]
    --epoch-len <int>       Save a model every <int> frames [default: 1e6]
    --nb-eval-env <int>     Evaluate agent in a separate thread [default: 0]
    --summary-freq <int>    Tensorboard summary frequency [default: 10]

Troubleshooting Options:
    --debug
    --profile
"""


import os
from copy import deepcopy

import torch
from absl import flags
from tensorboardX import SummaryWriter

from adept.containers import Local, EvaluationThread
from adept.environments import SubProcEnvManager
from adept.registries.environment import EnvPluginRegistry
from adept.utils.logging import (
    make_log_id, make_logger, print_ascii_logo, log_args, write_args_file,
    SimpleModelSaver
)
from adept.utils.script_helpers import (
    make_agent, make_network, get_head_shapes, count_parameters
)

# hack to use bypass pysc2 flags
FLAGS = flags.FLAGS
FLAGS(['local.py'])


def parse_args():
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.strip('--').replace('-', '_'): v for k, v in args.items()}
    args = DotDict(args)
    args.gpu_id = int(args.gpu_id)
    args.nb_env = int(args.nb_env)
    args.seed = int(args.seed)
    args.nb_train_frame = int(float(args.nb_train_frame))
    args.nb_eval_env = int(args.nb_eval_env)
    args.summary_freq = int(args.summary_freq)
    args.lr = float(args.lr)
    args.epoch_len = int(float(args.epoch_len))
    return args


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main(args, env_registry=EnvPluginRegistry()):
    """
    :param args: Dict[str, Any]
    :param env_registry: EnvPluginRegistry
    :return:
    """
    args = DotDict(args)
    env_args = env_registry.lookup_env_class(args.env).prompt()
    args = DotDict({**args, **env_args})

    # construct logging objects
    print_ascii_logo()
    log_id = make_log_id(
        args.tag, 'Local', args.agent, args.net3d + args.netbody
    )
    log_id_dir = os.path.join(args.logdir, args.env, log_id)

    os.makedirs(log_id_dir)
    logger = make_logger('Local', os.path.join(log_id_dir, 'train_log.txt'))
    summary_writer = SummaryWriter(log_id_dir)
    saver = SimpleModelSaver(log_id_dir)

    log_args(logger, args)
    write_args_file(log_id_dir, args)

    # construct env
    env = SubProcEnvManager.from_args(args, registry=env_registry)

    # construct network
    torch.manual_seed(args.seed)
    network_head_shapes = get_head_shapes(env.action_space, args.agent)
    network = make_network(env.observation_space, network_head_shapes, args)
    # possibly load network
    initial_step_count = 0
    if args.load_network:
        network.load_state_dict(
            torch.load(
                args.load_network, map_location=lambda storage, loc: storage
            )
        )
        # get step count from network file
        epoch_dir = os.path.split(args.load_network)[0]
        initial_step_count = int(os.path.split(epoch_dir)[-1])
        print('Reloaded network from {}'.format(args.load_network))
    logger.info('Network Parameter Count: {}'.format(count_parameters(network)))

    # construct agent
    device = torch.device(
        "cuda:{}".format(args.gpu_id)
        if (torch.cuda.is_available() and args.gpu_id >= 0)
        else "cpu"
    )
    torch.backends.cudnn.benchmark = True
    agent = make_agent(
        network, device, env.gpu_preprocessor, env.engine, env.action_space,
        args
    )

    # Construct the Container
    def make_optimizer(params):
        opt = torch.optim.RMSprop(
            params, lr=args.lr, eps=1e-5, alpha=0.99
        )
        if args.load_optimizer:
            opt.load_state_dict(
                torch.load(
                    args.load_optimizer,
                    map_location=lambda storage, loc: storage
                )
            )
        return opt

    container = Local(
        agent, env, make_optimizer, args.epoch_len, args.nb_env, logger,
        summary_writer, args.summary_frequency, saver
    )

    # if running an eval thread create eval env, agent, & logger
    if args.nb_eval_env > 0:
        # replace args num envs & seed
        eval_args = deepcopy(args)

        # env and agent
        eval_args.nb_env = args.nb_eval_env
        eval_env = SubProcEnvManager.from_args(
            eval_args, seed=args.seed + args.nb_env, registry=env_registry
        )
        eval_net = make_network(
            eval_env.observation_space, network_head_shapes, eval_args
        )
        eval_agent = make_agent(
            eval_net, device, eval_env.gpu_preprocessor, eval_env.engine,
            env.action_space, eval_args
        )
        eval_net.load_state_dict(network.state_dict())

        # logger
        eval_logger = make_logger(
            'LocalEval', os.path.join(log_id_dir, 'eval_log.txt')
        )

        evaluation_container = EvaluationThread(
            network,
            eval_agent,
            eval_env,
            args.nb_eval_env,
            eval_logger,
            summary_writer,
            args.eval_step_rate,
            # wire local containers step count into eval
            override_step_count_fn=lambda: container.local_step_count
        )
        evaluation_container.start()

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
        container.run(args.nb_train_frame, initial_count=initial_step_count)
    env.close()

    if args.nb_eval_env > 0:
        evaluation_container.stop()
        eval_env.close()


if __name__ == '__main__':
    main(parse_args())
    # import argparse
    # from adept.utils.script_helpers import add_base_args
    #
    # base_parser = argparse.ArgumentParser(description='AdeptRL Local Mode')
    #
    # def add_args(parser):
    #     parser = parser.add_argument_group('Local Mode Args')
    #     parser.add_argument(
    #         '--gpu-id',
    #         type=int,
    #         default=0,
    #         help='Which GPU to use for training (default: 0)'
    #     )
    #     parser.add_argument(
    #         '--nb-eval-env',
    #         default=1,
    #         type=int,
    #         help=
    #         'Number of eval environments to run [in a separate thread] each with a different seed. '
    #         'Creates a copy of the network. Disable by setting to 0. (default: 1)'
    #     )
    #     parser.add_argument(
    #         '--eval-step-rate',
    #         default=0,
    #         type=int,
    #         help=
    #         'Number of eval steps allowed to run per second decreasing this amount can improve training speed. 0 is unlimited (default: 0)'
    #     )
    #
    # add_base_args(base_parser, add_args)
    # args = base_parser.parse_args()
    #
    # if args.debug:
    #     args.nb_env = 3
    #     args.log_dir = '/tmp/'
    #
    # args.mode_name = 'Local'
    # main(args)
