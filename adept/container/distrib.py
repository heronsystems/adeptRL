# Copyright (C) 2020 Heron Systems, Inc.
import numpy as np
import os
import torch
import torch.distributed as dist
from time import time
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.utils import dtensor_to_dev, listd_to_dlist
from adept.utils.logging import SimpleModelSaver
from .base import Container
from .base.updater import Updater


class DistribUpdater(Updater):
    def __init__(
        self, optimizer, network, grad_norm_clip, world_sz, divide_grad
    ):
        super().__init__(optimizer, network, grad_norm_clip)
        self.world_sz = world_sz
        self.divide_grad = divide_grad
        self.grad_norm_clip = grad_norm_clip

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        dist.barrier()
        handles = []
        for param in self.network.parameters():
            handles.append(dist.all_reduce(param.grad, async_op=True))
        for handle in handles:
            handle.wait()
        if self.divide_grad:
            for param in self.network.parameters():
                param.grad.mul_(1.0 / self.world_sz)
        if self.grad_norm_clip:
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.grad_norm_clip
            )
        self.optimizer.step()


class DistribHost(Container):
    """
    DistribHost saves models and writes summaries. This is the only difference
    from the worker.
    """

    def __init__(
        self,
        args,
        logger,
        log_id_dir,
        initial_step_count,
        local_rank,
        global_rank,
        world_size,
    ):
        seed = (
            args.seed
            if global_rank == 0
            else args.seed + args.nb_env * global_rank
        )
        logger.info("Using {} for rank {} seed.".format(seed, global_rank))

        # ENV
        engine = REGISTRY.lookup_engine(args.env)
        env_cls = REGISTRY.lookup_env(args.env)
        mgr_cls = REGISTRY.lookup_manager(args.manager)
        env_mgr = mgr_cls.from_args(args, engine, env_cls, seed=seed)

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device("cuda:{}".format(local_rank))
        output_space = REGISTRY.lookup_output_space(
            args.agent, env_mgr.action_space
        )
        if args.custom_network:
            net_cls = REGISTRY.lookup_network(args.custom_network)
        else:
            net_cls = ModularNetwork
        net = net_cls.from_args(
            args,
            env_mgr.observation_space,
            output_space,
            env_mgr.gpu_preprocessor,
            REGISTRY,
        )
        logger.info("Network parameters: " + str(self.count_parameters(net)))

        def optim_fn(x):
            return torch.optim.RMSprop(x, lr=args.lr, eps=1e-5, alpha=0.99)

        # AGENT
        rwd_norm = REGISTRY.lookup_reward_normalizer(args.rwd_norm).from_args(
            args
        )
        agent_cls = REGISTRY.lookup_agent(args.agent)
        builder = agent_cls.exp_spec_builder(
            env_mgr.observation_space,
            env_mgr.action_space,
            net.internal_space(),
            env_mgr.nb_env,
        )
        agent = agent_cls.from_args(
            args, rwd_norm, env_mgr.action_space, builder
        )

        self.agent = agent
        self.nb_step = args.nb_step
        self.env_mgr = env_mgr
        self.nb_env = args.nb_env
        self.network = net.to(device)
        self.optimizer = optim_fn(self.network.parameters())
        self.device = device
        self.initial_step_count = initial_step_count
        self.log_id_dir = log_id_dir
        self.epoch_len = args.epoch_len
        self.summary_freq = args.summary_freq
        self.logger = logger
        self.summary_writer = SummaryWriter(
            os.path.join(log_id_dir, "rank{}".format(global_rank))
        )
        self.saver = SimpleModelSaver(log_id_dir)
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.updater = DistribUpdater(
            self.optimizer,
            self.network,
            args.grad_norm_clip,
            world_size,
            not args.no_divide,
        )

        if args.load_network:
            self.network = self.load_network(self.network, args.load_network)
            logger.info("Reloaded network from {}".format(args.load_network))
        if args.load_optim:
            self.optimizer = self.load_optim(self.optimizer, args.load_optim)
            logger.info("Reloaded optimizer from {}".format(args.load_optim))

        self.network.train()

    def run(self):
        local_step_count = global_step_count = self.initial_step_count
        next_save = self.init_next_save(self.initial_step_count, self.epoch_len)
        prev_step_t = time()
        ep_rewards = torch.zeros(self.nb_env)

        obs = dtensor_to_dev(self.env_mgr.reset(), self.device)
        internals = listd_to_dlist(
            [
                self.network.new_internals(self.device)
                for _ in range(self.nb_env)
            ]
        )
        start_time = time()
        while global_step_count < self.nb_step:
            actions, internals = self.agent.act(self.network, obs, internals)
            next_obs, rewards, terminals, infos = self.env_mgr.step(actions)
            next_obs = dtensor_to_dev(next_obs, self.device)

            self.agent.observe(
                obs,
                rewards.to(self.device).float(),
                terminals.to(self.device).float(),
                infos,
            )
            for i, terminal in enumerate(terminals):
                if terminal:
                    for k, v in self.network.new_internals(self.device).items():
                        internals[k][i] = v

            # Perform state updates
            local_step_count += self.nb_env
            global_step_count += self.nb_env * self.world_size
            ep_rewards += rewards.float()
            obs = next_obs

            term_rewards = []
            for i, terminal in enumerate(terminals):
                if terminal:
                    for k, v in self.network.new_internals(self.device).items():
                        internals[k][i] = v
                    term_rewards.append(ep_rewards[i].item())
                    ep_rewards[i].zero_()

            if term_rewards:
                term_reward = np.mean(term_rewards)
                delta_t = time() - start_time
                self.logger.info(
                    "RANK: {} "
                    "GLOBAL STEP: {} "
                    "REWARD: {} "
                    "GLOBAL STEP/S: {} "
                    "LOCAL STEP/S: {}".format(
                        self.global_rank,
                        global_step_count,
                        term_reward,
                        (global_step_count - self.initial_step_count) / delta_t,
                        (local_step_count - self.initial_step_count) / delta_t,
                    )
                )
                self.summary_writer.add_scalar(
                    "reward", term_reward, global_step_count
                )

            if global_step_count >= next_save:
                self.saver.save_state_dicts(
                    self.network, global_step_count, self.optimizer
                )
                next_save += self.epoch_len

            # Learn
            if self.agent.is_ready():
                loss_dict, metric_dict = self.agent.learn_step(
                    self.updater, self.network, next_obs, internals
                )
                total_loss = torch.sum(
                    torch.stack(tuple(loss for loss in loss_dict.values()))
                )

                self.agent.clear()
                for k, vs in internals.items():
                    internals[k] = [v.detach() for v in vs]

                # write summaries
                cur_step_t = time()
                if cur_step_t - prev_step_t > self.summary_freq:
                    self.write_summaries(
                        self.summary_writer,
                        global_step_count,
                        total_loss,
                        loss_dict,
                        metric_dict,
                        self.network.named_parameters(),
                    )
                    prev_step_t = cur_step_t

    def close(self):
        return self.env_mgr.close()


class DistribWorker(Container):
    """
    DistribWorker does all the same computation as a host process but does not
    save models or write tensorboard summaries.
    """

    def __init__(
        self,
        args,
        logger,
        log_id_dir,
        initial_step_count,
        local_rank,
        global_rank,
        world_size,
    ):
        seed = (
            args.seed
            if global_rank == 0
            else args.seed + args.nb_env * global_rank
        )
        logger.info("Using {} for rank {} seed.".format(seed, global_rank))

        # ENV
        engine = REGISTRY.lookup_engine(args.env)
        env_cls = REGISTRY.lookup_env(args.env)
        env_mgr = SubProcEnvManager.from_args(args, engine, env_cls, seed=seed)

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device("cuda:{}".format(local_rank))
        output_space = REGISTRY.lookup_output_space(
            args.agent, env_mgr.action_space
        )
        if args.custom_network:
            net_cls = REGISTRY.lookup_network(args.custom_network)
        else:
            net_cls = ModularNetwork
        net = net_cls.from_args(
            args,
            env_mgr.observation_space,
            output_space,
            env_mgr.gpu_preprocessor,
            REGISTRY,
        )

        def optim_fn(x):
            return torch.optim.RMSprop(x, lr=args.lr, eps=1e-5, alpha=0.99)

        # AGENT
        rwd_norm = REGISTRY.lookup_reward_normalizer(args.rwd_norm).from_args(
            args
        )
        agent_cls = REGISTRY.lookup_agent(args.agent)
        builder = agent_cls.exp_spec_builder(
            env_mgr.observation_space,
            env_mgr.action_space,
            net.internal_space(),
            env_mgr.nb_env,
        )
        agent = agent_cls.from_args(
            args, rwd_norm, env_mgr.action_space, builder
        )

        self.agent = agent
        self.nb_step = args.nb_step
        self.env_mgr = env_mgr
        self.nb_env = args.nb_env
        self.network = net.to(device)
        self.optimizer = optim_fn(self.network.parameters())
        self.device = device
        self.initial_step_count = initial_step_count
        self.log_id_dir = log_id_dir
        self.epoch_len = args.epoch_len
        self.summary_freq = args.summary_freq
        self.logger = logger
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.updater = DistribUpdater(
            self.optimizer,
            self.network,
            args.grad_norm_clip,
            world_size,
            not args.no_divide,
        )

        if args.load_network:
            self.network = self.load_network(self.network, args.load_network)
            logger.info("Reloaded network from {}".format(args.load_network))
        if args.load_optim:
            self.optimizer = self.load_optim(self.optimizer, args.load_optim)
            logger.info("Reloaded optimizer from {}".format(args.load_optim))

        self.network.train()

    def run(self):
        local_step_count = global_step_count = self.initial_step_count
        ep_rewards = torch.zeros(self.nb_env)

        obs = dtensor_to_dev(self.env_mgr.reset(), self.device)
        internals = listd_to_dlist(
            [
                self.network.new_internals(self.device)
                for _ in range(self.nb_env)
            ]
        )
        start_time = time()
        while global_step_count < self.nb_step:
            actions, internals = self.agent.act(self.network, obs, internals)
            next_obs, rewards, terminals, infos = self.env_mgr.step(actions)
            next_obs = dtensor_to_dev(next_obs, self.device)

            self.agent.observe(
                obs,
                rewards.to(self.device).float(),
                terminals.to(self.device).float(),
                infos,
            )
            for i, terminal in enumerate(terminals):
                if terminal:
                    for k, v in self.network.new_internals(self.device).items():
                        internals[k][i] = v

            # Perform state updates
            local_step_count += self.nb_env
            global_step_count += self.nb_env * self.world_size
            ep_rewards += rewards.float()
            obs = next_obs

            term_rewards = []
            for i, terminal in enumerate(terminals):
                if terminal:
                    for k, v in self.network.new_internals(self.device).items():
                        internals[k][i] = v
                    term_rewards.append(ep_rewards[i].item())
                    ep_rewards[i].zero_()

            if term_rewards:
                term_reward = np.mean(term_rewards)
                delta_t = time() - start_time
                self.logger.info(
                    "RANK: {} "
                    "GLOBAL STEP: {} "
                    "REWARD: {} "
                    "GLOBAL STEP/S: {} "
                    "LOCAL STEP/S: {}".format(
                        self.global_rank,
                        global_step_count,
                        term_reward,
                        (global_step_count - self.initial_step_count) / delta_t,
                        (local_step_count - self.initial_step_count) / delta_t,
                    )
                )

            # Learn
            if self.agent.is_ready():
                _, _ = self.agent.learn_step(
                    self.updater, self.network, next_obs, internals
                )

                self.agent.clear()
                for k, vs in internals.items():
                    internals[k] = [v.detach() for v in vs]

    def close(self):
        return self.env_mgr.close()
