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

Script Options:
    --gpu-id <int>          CUDA device ID of GPU [default: 0]
    --nb-env <int>          Number of parallel environments [default: 64]
    --seed <int>            Seed for random variables [default: 0]
    --nb-train-frame <int>  Number of frames to train on [default: 10e6]
    --load-network <path>   Path to network file
    --load-optim <path>     Path to optimizer file
    --resume <path>         Resume training from log ID .../<logdir>/<env>/<log-id>/
    --eval                  Run an evaluation after training
    -y, --use-defaults      Skip prompts, use defaults

Network Options:
    --net1d <str>           Network to use for 1d input [default: Identity]
    --net2d <str>           Network to use for 2d input [default: Identity]
    --net3d <str>           Network to use for 3d input [default: FourConv]
    --net4d <str>           Network to use for 4d input [default: Identity]
    --netjunc <str>         Network junction to merge inputs [default: TODO]
    --netbody <str>         Network to use on merged inputs [default: LSTM]
    --custom-network        Name of custom network class

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

from adept.agents.agent_registry import AgentRegistry
from adept.containers import Local, EvaluationThread
from adept.environments import SubProcEnvManager
from adept.environments.env_registry import EnvModuleRegistry
from adept.networks.network_registry import NetworkRegistry
from adept.utils.logging import (
    make_log_id, make_logger, print_ascii_logo, log_args, write_args_file,
    SimpleModelSaver
)
from adept.utils.script_helpers import (
    make_network, count_parameters,
    parse_bool_str, parse_none, LogDirHelper
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

    args.gpu_id = int(args.gpu_id)
    args.nb_env = int(args.nb_env)
    args.seed = int(args.seed)
    args.nb_train_frame = int(float(args.nb_train_frame))
    args.tag = parse_none(args.tag)
    args.nb_eval_env = int(args.nb_eval_env)
    args.summary_freq = int(args.summary_freq)
    args.lr = float(args.lr)
    args.epoch_len = int(float(args.epoch_len))
    args.profile = bool(args.profile)
    return args


def main(
    args,
    agent_registry=AgentRegistry(),
    env_registry=EnvModuleRegistry(),
    net_registry=NetworkRegistry()
):
    """
    Run local training.

    :param args: Dict[str, Any]
    :param agent_registry: AgentRegistry
    :param env_registry: EnvModuleRegistry
    :param net_registry: NetworkRegistry
    :return:
    """

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
        args = DotDict(args)
        if args.use_defaults:
            agent_args = agent_registry.lookup_agent(args.agent).args
            env_args = env_registry.lookup_env_class(args.env).args
            if args.custom_network:
                net_args = net_registry.lookup_custom_net(args.net).args
            else:
                net_args = net_registry.lookup_modular_args(
                    args.net1d, args.net2d, args.net3d, args.net4d,
                    args.netjunc, args.netbody
                )
        else:
            agent_args = agent_registry.lookup_agent(args.agent).prompt()
            env_args = env_registry.lookup_env_class(args.env).prompt()
            if args.custom_network:
                net_args = net_registry.lookup_custom_net(args.net).prompt()
            else:
                net_args = net_registry.prompt_modular_args(
                    args.net1d, args.net2d, args.net3d, args.net4d,
                    args.netjunc, args.netbody
                )
        args = DotDict({**args, **agent_args, **env_args, **net_args})

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
    env = SubProcEnvManager.from_args(args, registry=env_registry)

    # construct network
    torch.manual_seed(args.seed)
    network = make_network(
        env.observation_space,
        agent_registry.lookup_output_shape(args.agent, env.action_space),
        args
    )
    # possibly load network
    if args.load_network:
        network.load_state_dict(
            torch.load(
                args.load_network, map_location=lambda storage, loc: storage
            )
        )
        logger.info('Reloaded network from {}'.format(args.load_network))
    logger.info('Network Parameter Count: {}'.format(count_parameters(network)))

    # construct agent
    device = torch.device(
        "cuda:{}".format(args.gpu_id)
        if (torch.cuda.is_available() and args.gpu_id >= 0)
        else "cpu"
    )
    torch.backends.cudnn.benchmark = True
    agent = agent_registry.lookup_agent(args.agent).from_args(
        args,
        network,
        device,
        env_registry.lookup_reward_normalizer(args.env),
        env.gpu_preprocessor,
        env.engine,
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

    container = Local(
        agent, env, make_optimizer, args.epoch_len, args.nb_env, logger,
        summary_writer, args.summary_freq, saver
    )

    # if running an eval thread create eval env, agent, & logger
    if args.nb_eval_env > 0:

        # env and agent
        eval_env = SubProcEnvManager.from_args(
            args,
            seed=args.seed + args.nb_env,
            nb_env=args.nb_eval_env,
            registry=env_registry
        )
        eval_net = make_network(
            eval_env.observation_space,
            agent_registry.lookup_output_shape(args.agent, env.action_space),
            args
        )
        eval_agent = agent_registry.lookup_agent(args.agent).from_args(
            args,
            eval_net,
            device,
            eval_env.gpu_preprocessor,
            eval_env.engine,
            env.action_space
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

    if args.eval:
        import subprocess
        exit(subprocess.call([
            'python',
            '-m',
            'adept.scripts.evaluate',
            '--log-id-dir',
            log_id_dir,
            '--gpu-id',
            str(args.gpu_id)
        ], env=os.environ))


if __name__ == '__main__':
    main(parse_args())
