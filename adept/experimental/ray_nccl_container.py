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
import os
from time import time
import datetime

import numpy as np
import ray
from ray.experimental.sgd.utils import find_free_port
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from adept.container.base import Container
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.experimental.rollout_queuer import RolloutQueuerAsync
from adept.experimental.rollout_worker import RolloutWorker
from adept.utils import dtensor_to_dev, listd_to_dlist
from adept.utils.logging import SimpleModelSaver


class NCCLOptimizer:
    def __init__(self, optimizer_fn, network, world_size, param_sync_rate=1000):
        self.network = network
        self.optimizer = optimizer_fn(self.network.parameters())
        self.param_sync_rate = param_sync_rate
        self._opt_count = 0
        self.process_group = None
        self.world_size = world_size

    def set_process_group(self, pg):
        self.process_group = pg

    def step(self):
        handles = []
        for param in self.network.parameters():
            handles.append(
            self.process_group.allreduce(param.grad))
        for handle in handles:
            handle.wait()
        for param in self.network.parameters():
            param.grad.mul_(1. / self.world_size)
        self.optimizer.step()
        self._opt_count += 1

        # sync params every once in a while to reduce numerical errors
        if self._opt_count % self.param_sync_rate == 0:
            self.sync_parameters()
            self.sync_buffers()

    def sync_parameters(self):
        for param in self.network.parameters():
            self.process_group.allreduce(param.data)
            param.data.mul_(1. / self.world_size)

    def sync_buffers(self):
        for b in self.network.buffers():
            self.process_group.allreduce(b.data)
            b.data.mul_(1. / self.world_size)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, d):
        return self.optimizer.load_state_dict(d)



class RayContainer(Container):
    def __init__(
            self,
            args,
            logger,
            log_id_dir,
            initial_step_count,
            rank=0,
    ):
        # Ray can only be started once 
        if rank == 0:
            if args.ray_addr is not None:
                ray.init(address=args.ray_addr)
                print('Using Ray on a cluster. Head node address: {}'.format(args.ray_addr))
            else:
                print('Using Ray on a single machine.')
                ray.init()

        # ARGS TO STATE VARS
        self._args = args
        self.nb_learners = args.nb_learners
        self.nb_workers = args.nb_workers
        self.colocate_workers = args.colocate_workers
        self.rank = rank
        self.nb_step = args.nb_step
        self.nb_env = args.nb_env
        self.initial_step_count = initial_step_count
        self.epoch_len = args.epoch_len
        self.summary_freq = args.summary_freq
        self.nb_rollouts_in_batch = args.nb_rollouts_in_batch
        self.rollout_queue_size = args.rollout_queue_size
        self.worker_rollout_len = args.worker_rollout_len
        self.worker_cpu_alloc = args.worker_cpu_alloc
        self.worker_gpu_alloc = args.worker_gpu_alloc
        # can be none if rank != 0
        self.logger = logger
        self.log_id_dir = log_id_dir

        # ENV
        env_cls = REGISTRY.lookup_env(args.env)
        env = env_cls.from_args_curry(args, 0)()
        env_action_space, env_observation_space, env_gpu_preprocessor = \
            env.action_space, env.observation_space, env.gpu_preprocessor
        env.close()

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device("cuda:{}".format(0))  # ray handles gpus
        torch.backends.cudnn.benchmark = True
        output_space = REGISTRY.lookup_output_space(
            args.actor_worker, env_action_space)
        if args.custom_network:
            net_cls = REGISTRY.lookup_network(args.custom_network)
        else:
            net_cls = ModularNetwork
        net = net_cls.from_args(
            args,
            env_observation_space,
            output_space,
            env_gpu_preprocessor,
            REGISTRY
        )
        self.network = net.to(device)
        # TODO: this is a hack, remove once queuer puts rollouts on the correct device
        self.network.device = device
        self.device = device
        self.network.train()

        # OPTIMIZER
        def optim_fn(x):
            return torch.optim.RMSprop(x, lr=args.lr, eps=1e-5, alpha=0.99)
        if args.nb_learners > 1:
            self.optimizer = NCCLOptimizer(optim_fn, self.network, self.nb_learners)
        else:
            self.optimizer = optim_fn(self.network.parameters())

        # LEARNER
        learner = REGISTRY.lookup_learner(args.learner).from_args(args)
        self.learner = learner

        # Rank 0 setup, load network/optimizer and create SummaryWriter/Saver
        if rank == 0:
            if args.load_network:
                self.network = self.load_network(self.network, args.load_network)
                logger.info('Reloaded network from {}'.format(args.load_network))
            if args.load_optim:
                self.optimizer = self.load_optim(self.optimizer, args.load_optim)
                logger.info('Reloaded optimizer from {}'.format(args.load_optim))

            logger.info('Network parameters: ' + str(self.count_parameters(net)))
            self.summary_writer = SummaryWriter(log_id_dir)
            self.saver = SimpleModelSaver(log_id_dir)

    def run(self):
        # setup peers
        if self.rank == 0 and self.nb_learners > 1:
            # create peer containers
            self.peer_learners = self._init_peer_learners()

            # tell them to connect to nccl and sync parameters
            self._init_peer_nccl(self.peer_learners)
            print('All NCCL peers initialized')

            # sync nccl, must start others first otherwise deadlock
            [f._sync_peer_parameters.remote() for f in self.peer_learners]
            self._sync_peer_parameters()

            # startup the run method of peer containers
            [f.run.remote() for f in self.peer_learners]

        if self.nb_learners > 1:
            # set optimizer process_group
            self.optimizer.set_process_group(self.process_group)

        # peers are all synced, now create workers
        # TODO: this is a hack, probably a better way to do it
        if self.colocate_workers:
            resources = {"Agent{}_Colocate".format(self.rank): (1/self.nb_workers) - 0.1}
        else:
            resources = {}

        # TODO: actually lookup from registry
        self.workers = [RolloutWorker.as_remote(num_cpus=self.worker_cpu_alloc,
                                                num_gpus=self.worker_gpu_alloc,
                                                resources=resources)
                        .remote(self._args, self.initial_step_count, w_ind + (self.rank * self.nb_workers))
                        for w_ind in range(self.nb_workers)]

        # synchronize worker variables
        self.synchronize_worker_parameters(self.initial_step_count, blocking=True)

        # setup queuer
        self.rollout_queuer = RolloutQueuerAsync(self.workers, self.nb_rollouts_in_batch, self.rollout_queue_size)
        self.rollout_queuer.start()

        # initial setup
        global_step_count = self.initial_step_count
        next_save = self.init_next_save(self.initial_step_count, self.epoch_len)
        prev_step_t = time()
        ep_rewards = torch.zeros(self.nb_env)
        start_time = time()

        # loop until total number steps
        print('{} starting training'.format(self.rank))
        while not self.done(global_step_count):
            # Learn
            batch, terminal_rewards = self.rollout_queuer.get()

            loss_dict, metric_dict = self.learner.compute_loss(
                self.network, batch
            )
            total_loss = torch.sum(
                torch.stack(tuple(loss for loss in loss_dict.values()))
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Perform state updates
            global_step_count += self.nb_env * self.nb_rollouts_in_batch * self.worker_rollout_len * self.nb_learners

            # send parameters to workers async
            self.synchronize_worker_parameters(global_step_count)

            # if rank 0 write summaries and save
            if self.rank == 0:
                # possible save
                if global_step_count >= next_save:
                    self.saver.save_state_dicts(
                        self.network, global_step_count, self.optimizer
                    )
                    next_save += self.epoch_len

                # write reward summaries
                if any(terminal_rewards):
                    terminal_rewards = list(filter(lambda x: x is not None, terminal_rewards))
                    self.summary_writer.add_scalar(
                        'reward', np.mean(terminal_rewards), global_step_count
                    )

            # write summaries
            cur_step_t = time()
            if cur_step_t - prev_step_t > self.summary_freq:
                print('Metrics:', self.rollout_queuer.metrics())
                if self.rank == 0:
                    self.write_summaries(
                        self.summary_writer, global_step_count, total_loss,
                        loss_dict, metric_dict, self.network.named_parameters()
                    )
                prev_step_t = cur_step_t
        print('{} stopped training'.format(self.rank))

    def done(self, global_step_count):
        return global_step_count >= self.nb_step

    def close(self):
        self.rollout_queuer.stop()
        if self.rank == 0 and self.nb_learners > 1:
            for l in self.peer_learners:
                ray.get(l.close.remote())

    def get_parameters(self):
        params = [p.cpu() for p in self.network.parameters()]
        params.extend([b.cpu() for b in self.network.buffers()])
        return params

    def synchronize_worker_parameters(self, global_step_count=0, blocking=False):
        parameters = self.get_parameters()
        futures = [w.set_weights.remote(parameters) for w in self.workers]

        if global_step_count != 0:
            futures.extend([w.set_global_step.remote(global_step_count) for w in self.workers])

        if blocking:
            ray.get(futures)

    def _rank0_nccl_port_init(self):
        ip = ray.services.get_node_ip_address()
        port = find_free_port()
        nccl_addr = "tcp://{ip}:{port}".format(ip=ip, port=port)
        return nccl_addr, ip, port

    def _nccl_init(self):
        print('Rank {} calling init_process_group.'.format(self.rank))
        # from https://github.com/pytorch/pytorch/blob/master/test/simulate_nccl_errors.py
        store = dist.TCPStore(self.nccl_ip, self.nccl_port, self.nb_learners, self.rank == 0)
        process_group = dist.ProcessGroupNCCL(store, self.rank, self.nb_learners)
        print('Rank {} initialized process group.'.format(self.rank))
        process_group.barrier()
        print('Rank {} process group barrier finished.'.format(self.rank))
        self.process_group = process_group

    def _init_peer_learners(self):
        # create N peer learners
        self.nccl_addr, self.nccl_ip, self.nccl_port = self._rank0_nccl_port_init()

        peer_learners = []
        for p_ind in range(self.nb_learners - 1):
            # TODO: this is a hack
            if self.colocate_workers:
                resources = {"Agent{}_Colocate".format(p_ind + 1): 0.1}
            else:
                resources = {}
            remote_cls = RayPeerLearnerContainer.as_remote(num_cpus=1,
                                                           # TODO: learner GPU alloc from args
                                                           num_gpus=0.25,
                                                           resources=resources)
            # init
            remote = remote_cls.remote(self._args, self.initial_step_count, rank=p_ind + 1,
                                       nccl_addr=self.nccl_addr, nccl_ip=self.nccl_ip, nccl_port=self.nccl_port)
            peer_learners.append(remote)

        # wait for all peer learners to initialize?
        return peer_learners

    def _init_peer_nccl(self, peers):
        # start init for all peers
        nccl_inits = [p._nccl_init.remote() for p in peers]
        # join nccl
        self._nccl_init()
        # wait for all
        ray.get(nccl_inits)

    def _sync_peer_parameters(self):
        print('Rank {} syncing parameters.'.format(self.rank))
        self.process_group.barrier()
        for p in self.network.parameters():
            self.process_group.allreduce(p.data)
            p.data = p.data / self.nb_learners
        print('Rank {} parameters synced.'.format(self.rank))


class RayPeerLearnerContainer(RayContainer):
    @classmethod
    def as_remote(cls,
                  num_cpus=None,
                  num_gpus=None,
                  memory=None,
                  object_store_memory=None,
                  resources=None):
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources)(cls)

    def __init__(
            self,
            args,
            initial_step_count,
            rank,
            nccl_addr,
            nccl_ip,
            nccl_port,
    ):
        # no logger for peer learners
        super().__init__(args, None, None, initial_step_count, rank)
        self.nccl_addr = nccl_addr
        self.nccl_ip = nccl_ip
        self.nccl_port = nccl_port
        self._should_stop = False

    def done(self, global_step_count):
        return self._should_stop

    def close(self):
        # This is called from the host so it's run in a ray thread
        self._should_stop = True

