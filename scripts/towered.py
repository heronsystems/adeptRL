import os
import torch
from absl import flags
from mpi4py import MPI as mpi
from tensorboardX import SummaryWriter

from adept.containers import ToweredHost, ToweredWorker
from adept.utils.logging import make_log_id_from_timestamp, make_logger, print_ascii_logo, log_args, write_args_file
from adept.utils.script_helpers import make_agent, make_network, make_env, agent_output_shape
from datetime import datetime


# hack to use argparse for SC2
FLAGS = flags.FLAGS
FLAGS(['local.py'])


def main(args):
    # mpi comm, rank, and size
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # host needs to broadcast timestamp so all procs create the same log dir
    if rank == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        timestamp = None
    timestamp = comm.bcast(timestamp, root=0)

    log_id = make_log_id_from_timestamp(args.mode_name, args.agent, args.network, timestamp)
    log_id_dir = os.path.join(args.log_dir, args.env_id, log_id)

    # host needs to make dir so other procs can access
    if rank == 0:
        print_ascii_logo()
        os.makedirs(log_id_dir)
    comm.Barrier()

    # construct env
    seed = args.seed if rank == 0 else args.seed * (args.nb_env * (rank - 1))  # unique seed per process
    env = make_env(args, seed)

    # construct network
    torch.manual_seed(args.seed)
    network_head_shapes = agent_output_shape(env.action_space, env.engine, args)
    network = make_network(env.observation_space, network_head_shapes, args)

    # sync network params
    if rank == 0:
        for v in network.parameters():
            comm.Bcast(v.detach().cpu().numpy(), root=0)
        print('Root variables synced')
    else:
        # can just use the numpy buffers
        variables = [v.detach().cpu().numpy() for v in network.parameters()]
        for v in variables:
            comm.Bcast(v, root=0)
        for shared_v, model_v in zip(variables, network.parameters()):
            model_v.data.copy_(torch.from_numpy(shared_v), non_blocking=True)
        print('{} variables synced'.format(rank))

    # host is rank 0
    if rank != 0:
        # construct logger
        logger = make_logger(
            'ToweredWorker{}'.format(rank),
            os.path.join(log_id_dir, 'train_log_rank{}.txt'.format(rank))
        )
        summary_writer = SummaryWriter(os.path.join(log_id_dir, 'rank{}'.format(rank)))

        # construct agent
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        agent = make_agent(network, device, env.engine, args)

        # construct container
        container = ToweredWorker(agent, env, args.nb_env, logger, summary_writer, args.summary_frequency)

        # Run the container
        try:
            container.run()
        finally:
            env.close()
    # host
    else:
        logger = make_logger('ToweredHost', os.path.join(log_id_dir, 'train_log_rank{}.txt'.format(rank)))
        log_args(logger, args)
        write_args_file(log_id_dir, args)

        # no need for the env anymore
        env.close()

        # Construct the optimizer
        def make_optimizer(params):
            opt = torch.optim.RMSprop(params, lr=args.learning_rate, eps=1e-5, alpha=0.99)
            return opt

        container = ToweredHost(comm, size - 1, network, make_optimizer, logger)

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


if __name__ == '__main__':
    import argparse
    from adept.utils.script_helpers import add_base_args, parse_bool

    parser = argparse.ArgumentParser(description='AdeptRL Towered Mode')
    parser = add_base_args(parser)
    # TODO accept multiple gpu ids
    parser.add_argument('--gpu-id', type=int, default=0, help='Which GPU to use for training (default: 0)')
    parser.add_argument(
        '-net', '--network', default='FourConvLSTM',
        help='name of preset network (default: FourConvLSTM)'
    )
    parser.add_argument(
        '--profile', type=parse_bool, nargs='?', const=True, default=False,
        help='displays profiling tree after 10e3 steps (default: False)'
    )
    parser.add_argument(
        '--agent', default='ActorCritic',
        help='name of preset agent (default: ActorCritic)'
    )
    parser.add_argument(
        '--debug', type=parse_bool, nargs='?', const=True, default=False,
        help='debug mode sends the logs to /tmp/ and overrides number of workers to 3 (default: False)'
    )
    args = parser.parse_args()

    if args.debug:
        args.nb_env = 3
        args.log_dir = '/tmp/'
    args.mode_name = 'Towered'

    main(args)
