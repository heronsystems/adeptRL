#!python
"""
Copyright (C) 2018 Heron Systems, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os

import torch
from absl import flags
from copy import deepcopy

from adept.containers import Local, EvaluationThread
from adept.utils.script_helpers import make_agent, make_network, make_env, get_head_shapes, count_parameters
from adept.utils.logging import make_log_id, make_logger, print_ascii_logo, log_args, write_args_file, SimpleModelSaver
from tensorboardX import SummaryWriter

# hack to use argparse for SC2
FLAGS = flags.FLAGS
FLAGS(['local.py'])


def main(args):
    # construct logging objects
    print_ascii_logo()
    log_id = make_log_id(args.tag, args.mode_name, args.agent, args.vision_network + args.network_body)
    log_id_dir = os.path.join(args.log_dir, args.env_id, log_id)

    os.makedirs(log_id_dir)
    logger = make_logger('Local', os.path.join(log_id_dir, 'train_log.txt'))
    summary_writer = SummaryWriter(log_id_dir)
    saver = SimpleModelSaver(log_id_dir)

    log_args(logger, args)
    write_args_file(log_id_dir, args)

    # construct env
    env = make_env(args, args.seed)

    # construct network
    torch.manual_seed(args.seed)
    network_head_shapes = get_head_shapes(env.action_space, env.engine, args.agent)
    network = make_network(env.observation_space, network_head_shapes, args)
    logger.info('Network Parameter Count: {}'.format(count_parameters(network)))

    # construct agent
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    agent = make_agent(network, device, env.engine, env.gpu_preprocessor, args)

    # Construct the Container
    def make_optimizer(params):
        opt = torch.optim.RMSprop(params, lr=args.learning_rate, eps=1e-5, alpha=0.99)
        return opt

    container = Local(
        agent,
        env,
        make_optimizer,
        args.epoch_len,
        args.nb_env,
        logger,
        summary_writer,
        args.summary_frequency,
        saver
    )

    # if running an eval thread create eval env, agent, & logger
    if args.nb_eval_env > 0:
        # replace args num envs & seed
        eval_args = deepcopy(args)
        eval_args.seed = args.seed + args.nb_env

        # env and agent
        eval_args.nb_env = args.nb_eval_env
        eval_env = make_env(eval_args, eval_args.seed)
        eval_net = make_network(eval_env.observation_space, network_head_shapes, eval_args)
        eval_agent = make_agent(eval_net, device, eval_env.engine, eval_env.gpu_preprocessor, eval_args)
        eval_net.load_state_dict(network.state_dict())

        # logger
        eval_logger = make_logger('LocalEval', os.path.join(log_id_dir, 'eval_log.txt'))

        evaluation_container = EvaluationThread(
            network,
            eval_agent,
            eval_env,
            args.nb_eval_env,
            eval_logger,
            summary_writer,
            args.eval_step_rate,
            override_step_count_fn=lambda: container.local_step_count  # wire local containers step count into eval
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
        container.run(args.max_train_steps)
    env.close()

    if args.nb_eval_env > 0:
        evaluation_container.stop()
        eval_env.close()


if __name__ == '__main__':
    import argparse
    from adept.utils.script_helpers import add_base_args, parse_bool

    parser = argparse.ArgumentParser(description='AdeptRL Local Mode')
    parser = add_base_args(parser)
    parser.add_argument('--gpu-id', type=int, default=0, help='Which GPU to use for training (default: 0)')
    parser.add_argument(
        '-vn', '--vision-network', default='FourConv',
        help='name of preset network (default: FourConv)'
    )
    parser.add_argument(
        '-dn', '--discrete-network', default='Identity',
    )
    parser.add_argument(
        '-nb', '--network-body', default='LSTM',
    )
    parser.add_argument(
        '--agent', default='ActorCritic',
        help='name of preset agent (default: ActorCritic)'
    )
    parser.add_argument(
        '--nb-eval-env', default=1, type=int,
        help='Number of eval environments to run [in a separate thread] each with a different seed. '
             'Creates a copy of the network. Disable by setting to 0. (default: 1)'
    )
    parser.add_argument(
        '--eval-step-rate', default=0, type=int,
        help='Number of eval steps allowed to run per second decreasing this amount can improve training speed. 0 to disable (default: 0)'
    )
    parser.add_argument(
        '--profile', type=parse_bool, nargs='?', const=True, default=False,
        help='displays profiling tree after 10e3 steps (default: False)'
    )
    parser.add_argument(
        '--debug', type=parse_bool, nargs='?', const=True, default=False,
        help='debug mode sends the logs to /tmp/ and overrides number of workers to 3 (default: False)'
    )

    args = parser.parse_args()

    if args.debug:
        args.nb_env = 3
        args.log_dir = '/tmp/'

    args.mode_name = 'Local'
    main(args)
