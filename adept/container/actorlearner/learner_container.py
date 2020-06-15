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
from ray.util.sgd.utils import find_free_port
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from adept.container.base import Container, NCCLOptimizer
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.container.actorlearner.rollout_queuer import RolloutQueuerAsync
from adept.utils import dtensor_to_dev, listd_to_dlist
from adept.utils.logging import SimpleModelSaver


class ActorLearnerHost(Container):
    @classmethod
    def as_remote(
        cls,
        num_cpus=None,
        num_gpus=None,
        memory=None,
        object_store_memory=None,
        resources=None,
    ):
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    def __init__(
        self, args, log_id_dir, initial_step_count, rank=0,
    ):
        # ARGS TO STATE VARS
        self._args = args
        self.nb_learners = args.nb_learners
        self.nb_workers = args.nb_workers
        self.rank = rank
        self.nb_step = args.nb_step
        self.nb_env = args.nb_env
        self.initial_step_count = initial_step_count
        self.epoch_len = args.epoch_len
        self.summary_freq = args.summary_freq
        self.nb_learn_batch = args.nb_learn_batch
        self.rollout_queue_size = args.rollout_queue_size
        # can be none if rank != 0
        self.log_id_dir = log_id_dir

        # load saved registry classes
        REGISTRY.load_extern_classes(log_id_dir)

        # ENV (temporary)
        env_cls = REGISTRY.lookup_env(args.env)
        env = env_cls.from_args(args, 0)
        env_action_space, env_observation_space, env_gpu_preprocessor = (
            env.action_space,
            env.observation_space,
            env.gpu_preprocessor,
        )
        env.close()

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device("cuda")  # ray handles gpus
        torch.backends.cudnn.benchmark = True
        output_space = REGISTRY.lookup_output_space(
            args.actor_worker, env_action_space
        )
        if args.custom_network:
            net_cls = REGISTRY.lookup_network(args.custom_network)
        else:
            net_cls = ModularNetwork
        net = net_cls.from_args(
            args,
            env_observation_space,
            output_space,
            env_gpu_preprocessor,
            REGISTRY,
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
            self.optimizer = NCCLOptimizer(
                optim_fn, self.network, self.nb_learners
            )
        else:
            self.optimizer = optim_fn(self.network.parameters())

        # LEARNER / EXP
        rwd_norm = REGISTRY.lookup_reward_normalizer(args.rwd_norm).from_args(
            args
        )
        actor_cls = REGISTRY.lookup_actor(args.actor_host)
        builder = actor_cls.exp_spec_builder(
            env.observation_space,
            env.action_space,
            net.internal_space(),
            args.nb_env * args.nb_learn_batch,
        )
        w_builder = REGISTRY.lookup_actor(args.actor_worker).exp_spec_builder(
            env.observation_space,
            env.action_space,
            net.internal_space(),
            args.nb_env,
        )
        actor = actor_cls.from_args(args, env.action_space)
        learner = REGISTRY.lookup_learner(args.learner).from_args(
            args, rwd_norm
        )

        exp_cls = REGISTRY.lookup_exp(args.exp).from_args(args, builder)

        self.actor = actor
        self.learner = learner
        self.exp = exp_cls.from_args(args, builder).to(device)

        # Rank 0 setup, load network/optimizer and create SummaryWriter/Saver
        if rank == 0:
            if args.load_network:
                self.network = self.load_network(
                    self.network, args.load_network
                )
                print("Reloaded network from {}".format(args.load_network))
            if args.load_optim:
                self.optimizer = self.load_optim(
                    self.optimizer, args.load_optim
                )
                print("Reloaded optimizer from {}".format(args.load_optim))

            print("Network parameters: " + str(self.count_parameters(net)))
            self.summary_writer = SummaryWriter(log_id_dir)
            self.saver = SimpleModelSaver(log_id_dir)

    def run(self, workers, profile=False):
        if profile:
            try:
                from pyinstrument import Profiler
            except:
                raise ImportError(
                    "You must install pyinstrument to use profiling."
                )
            profiler = Profiler()
            profiler.start()

        # setup queuer
        rollout_queuer = RolloutQueuerAsync(
            workers, self.nb_learn_batch, self.rollout_queue_size
        )
        rollout_queuer.start()

        # initial setup
        global_step_count = self.initial_step_count
        next_save = self.init_next_save(self.initial_step_count, self.epoch_len)
        prev_step_t = time()
        ep_rewards = torch.zeros(self.nb_env)
        start_time = time()

        # loop until total number steps
        print("{} starting training".format(self.rank))
        while not self.done(global_step_count):
            self.exp.clear()
            # Get batch from queue
            rollouts, terminal_rewards, terminal_infos = rollout_queuer.get()

            # Iterate forward on batch
            self.exp.write_exps(rollouts)
            # keep a copy of terminals on the cpu it's faster
            rollout_terminals = torch.stack(self.exp["terminals"]).numpy()
            self.exp.to(self.device)
            r = self.exp.read()
            internals = {k: ts[0].unbind(0) for k, ts in r.internals.items()}
            for obs, rewards, terminals in zip(
                r.observations, r.rewards, rollout_terminals
            ):
                _, h_exp, internals = self.actor.act(
                    self.network, obs, internals
                )
                self.exp.write_actor(h_exp, no_env=True)

                # where returns a single element tuple with the indexes
                terminal_inds = np.where(terminals)[0]
                for i in terminal_inds:
                    for k, v in self.network.new_internals(self.device).items():
                        internals[k][i] = v

            # compute loss
            loss_dict, metric_dict = self.learner.learn_step(
                self.network, self.exp.read(), r.next_observation, internals
            )
            total_loss = torch.sum(
                torch.stack(tuple(loss for loss in loss_dict.values()))
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Perform state updates
            global_step_count += (
                self.nb_env
                * self.nb_learn_batch
                * len(r.terminals)
                * self.nb_learners
            )

            # if rank 0 write summaries and save
            # and send parameters to workers async
            if self.rank == 0:
                # TODO: this could be parallelized, chunk by nb learners
                self.synchronize_worker_parameters(workers, global_step_count)

                # possible save
                if global_step_count >= next_save:
                    self.saver.save_state_dicts(
                        self.network, global_step_count, self.optimizer
                    )
                    next_save += self.epoch_len

                # write reward summaries
                if any(terminal_rewards):
                    terminal_rewards = list(
                        filter(lambda x: x is not None, terminal_rewards)
                    )
                    self.summary_writer.add_scalar(
                        "reward", np.mean(terminal_rewards), global_step_count
                    )

                # write infos
                if any(terminal_infos):
                    terminal_infos = list(
                        filter(lambda x: x is not None, terminal_infos)
                    )
                    float_keys = [
                        k
                        for k, v in terminal_infos[0].items()
                        if type(v) == float
                    ]
                    terminal_infos_dlist = listd_to_dlist(terminal_infos)
                    for k in float_keys:
                        self.summary_writer.add_scalar(
                            f"info/{k}",
                            np.mean(terminal_infos_dlist[k]),
                            global_step_count,
                        )

            # write summaries
            cur_step_t = time()
            if cur_step_t - prev_step_t > self.summary_freq:
                print(
                    "Rank {} Metrics:".format(self.rank),
                    rollout_queuer.metrics(),
                )
                if self.rank == 0:
                    self.write_summaries(
                        self.summary_writer,
                        global_step_count,
                        total_loss,
                        loss_dict,
                        metric_dict,
                        self.network.named_parameters(),
                    )
                prev_step_t = cur_step_t

        rollout_queuer.close()
        print("{} stopped training".format(self.rank))
        if profile:
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))

    def done(self, global_step_count):
        return global_step_count >= self.nb_step

    def close(self):
        pass

    def get_parameters(self):
        params = [p.cpu() for p in self.network.parameters()]
        return params

    def synchronize_worker_parameters(
        self, workers, global_step_count=0, blocking=False
    ):
        parameters = self.get_parameters()
        futures = [w.set_weights.remote(parameters) for w in workers]

        if global_step_count != 0:
            futures.extend(
                [w.set_global_step.remote(global_step_count) for w in workers]
            )

        if blocking:
            ray.get(futures)

    def _rank0_nccl_port_init(self):
        ip = ray.services.get_node_ip_address()
        port = find_free_port()
        nccl_addr = "tcp://{ip}:{port}".format(ip=ip, port=port)
        return nccl_addr, ip, port

    def _nccl_init(self, nccl_addr, nccl_ip, nccl_port):
        self.nccl_ip, self.nccl_addr, self.nccl_port = (
            nccl_ip,
            nccl_addr,
            nccl_port,
        )
        print(
            "Rank {} calling init_process_group. Addr: {}".format(
                self.rank, nccl_addr
            )
        )
        # from https://github.com/pytorch/pytorch/blob/master/test/simulate_nccl_errors.py
        store = dist.TCPStore(
            self.nccl_ip, self.nccl_port, self.nb_learners, self.rank == 0
        )
        process_group = dist.ProcessGroupNCCL(
            store, self.rank, self.nb_learners
        )
        print("Rank {} initialized process group.".format(self.rank))
        process_group.barrier()
        print("Rank {} process group barrier finished.".format(self.rank))
        self.process_group = process_group
        # set optimizer process_group
        self.optimizer.set_process_group(self.process_group)

    def _sync_peer_parameters(self):
        print("Rank {} syncing parameters.".format(self.rank))
        self.process_group.barrier()
        for p in self.network.parameters():
            self.process_group.allreduce(p.data)
            p.data = p.data / self.nb_learners
        print("Rank {} parameters synced.".format(self.rank))
