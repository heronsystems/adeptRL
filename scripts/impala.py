import os
from mpi4py import MPI as mpi

# mpi comm, rank, and size
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import torch
from absl import flags
from adept.containers import ImpalaHost, ImpalaWorker
from adept.utils.script_helpers import make_agent, make_network, make_env, agent_output_shape
from adept.utils.logging import make_log_id_from_timestamp, make_logger, print_ascii_logo, log_args
from tensorboardX import SummaryWriter
from datetime import datetime


# hack to use argparse for SC2
FLAGS = flags.FLAGS
FLAGS(['local.py'])


def main(args):
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

    # construct agent
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn = True
    # disable cudnn for dynamic batches
    if rank == 0 and args.max_dynamic_batch > 0:
        cudnn = False

    torch.backends.cudnn.benchmark = cudnn
    agent = make_agent(network, device, env.engine, args)

    # workers
    if rank != 0:
        logger = make_logger('ImpalaWorker{}'.format(rank), os.path.join(log_id_dir, 'train_log{}.txt'.format(rank)))
        summary_writer = SummaryWriter(os.path.join(log_id_dir, str(rank)))
        container = ImpalaWorker(agent, env, args.nb_env, logger, summary_writer, use_local_buffers=args.use_local_buffers)

        # Run the container
        if args.profile:
            try:
                from pyinstrument import Profiler
            except:
                raise ImportError('You must install pyinstrument to use profiling.')
            profiler = Profiler()
            profiler.start()
            container.run()
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))
        else:
            container.run()
        env.close()
    # host
    else:
        logger = make_logger('ImpalaHost', os.path.join(log_id_dir, 'train_log{}.txt'.format(rank)))
        summary_writer = SummaryWriter(os.path.join(log_id_dir, str(rank)))
        log_args(logger, args)

        # no need for the env anymore
        env.close()

        # Construct the optimizer
        def make_optimizer(params):
            opt = torch.optim.RMSprop(params, lr=args.learning_rate, eps=1e-5, alpha=0.99)
            return opt

        container = ImpalaHost(agent, comm, make_optimizer, summary_writer, args.summary_frequency, args.host_training_info_interval, use_local_buffers=args.use_local_buffers)

        # Run the container
        if args.profile:
            try:
                from pyinstrument import Profiler
            except:
                raise ImportError('You must install pyinstrument to use profiling.')
            profiler = Profiler()
            profiler.start()
            if args.max_dynamic_batch > 0:
                container.run(args.max_dynamic_batch, args.max_queue_length, args.max_train_steps, dynamic=True)
            else:
                container.run(args.num_rollouts_in_batch, args.max_queue_length, args.max_train_steps)
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))
        else:
            if args.max_dynamic_batch > 0:
                container.run(args.max_dynamic_batch, args.max_queue_length, args.max_train_steps, dynamic=True)
            else:
                container.run(args.num_rollouts_in_batch, args.max_queue_length, args.max_train_steps)


if __name__ == '__main__':
    import argparse
    from adept.utils.script_helpers import add_base_args, parse_bool

    parser = argparse.ArgumentParser(description='AdeptRL IMPALA Mode')
    parser = add_base_args(parser)
    parser.add_argument('--gpu-id', type=int, default=0, help='Which GPU to use for training (default: 0)')
    parser.add_argument(
        '-net', '--network', default='FourConvLSTM',
        help='name of preset network (default: FourConvLSTM)'
    )
    parser.add_argument(
        '--agent', default='ActorCriticVtrace',
        help='name of preset agent (default: ActorCriticVtrace)'
    )
    parser.add_argument(
        '--profile', type=parse_bool, nargs='?', const=True, default=False,
        help='displays profiling tree after 10e3 steps (default: False)'
    )
    parser.add_argument(
        '--debug', type=parse_bool, nargs='?', const=True, default=False,
        help='debug mode sends the logs to /tmp/ and overrides number of workers to 3 (default: False)'
    )
    parser.add_argument(
        '--max-queue-length', type=int, default=(size - 1) * 2,
        help='Maximum rollout queue length. If above the max, workers will wait to append (default: (size - 1) * 2)'
    )
    parser.add_argument(
        '--num-rollouts-in-batch', type=int, default=(size - 1),
        help='The batch size in rollouts (so total batch is this number * nb_env * seq_len). '
        + 'Not compatible with --dynamic-batch (default: (size - 1))'
    )
    parser.add_argument(
        '--max-dynamic-batch', type=int, nargs='?', const=True, default=0,
        help='When > 0 uses dynamic batching (disables cudnn and --num-rollouts-in-batch). '
        + 'Limits the maximum rollouts in the batch to limit GPU memory usage. (default: 0 (False))'
    )
    parser.add_argument(
        '--host-training-info-interval', type=int, nargs='?', const=True, default=100,
        help='The number of training steps before the host writes an info summary. (default: 100)'
    )
    parser.add_argument(
        '--use-local-buffers', type=parse_bool, nargs='?', const=True, default=False,
        help='If true all workers use their local network buffers (for batch norm: mean & var are not shared) (default: False)'
    )
    args = parser.parse_args()

    if args.debug:
        args.nb_env = 3
        args.log_dir = '/tmp/'

    args.mode_name = 'IMPALA'
    main(args)
