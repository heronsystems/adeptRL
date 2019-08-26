from time import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.utils.logging import SimpleModelSaver
from adept.utils.util import dtensor_to_dev, listd_to_dlist

from .base import Container


class Local(Container):
    def __init__(
            self,
            args,
            logger,
            log_id_dir,
            initial_step_count
    ):
        # ENV
        engine = REGISTRY.lookup_engine(args.env)
        env_cls = REGISTRY.lookup_env(args.env)
        env_mgr = SubProcEnvManager.from_args(args, engine, env_cls)

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device(
            "cuda:{}".format(args.gpu_id)
            if (torch.cuda.is_available() and args.gpu_id >= 0)
            else "cpu"
        )
        output_space = REGISTRY.lookup_output_space(
            args.agent, env_mgr.action_space)
        if args.custom_network:
            net_cls = REGISTRY.lookup_network(args.custom_network)
        else:
            net_cls = ModularNetwork
        net = net_cls.from_args(
            args,
            env_mgr.observation_space,
            output_space,
            env_mgr.gpu_preprocessor,
            REGISTRY
        )

        def optim_fn(x):
            return torch.optim.RMSprop(x, lr=args.lr, eps=1e-5, alpha=0.99)

        # AGENT
        rwd_norm = REGISTRY.lookup_reward_normalizer(
            args.rwd_norm).from_args(args)
        agent = REGISTRY.lookup_agent(args.agent).from_args(
            args,
            rwd_norm,
            env_mgr.action_space
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
        self.summary_writer = SummaryWriter(log_id_dir)
        self.saver = SimpleModelSaver(log_id_dir)

        if args.load_network:
            self.network = self.load_network(self.network, args.load_network)
            logger.info('Reloaded network from {}'.format(args.load_network))
        if args.load_optim:
            self.optimizer = self.load_optim(self.optimizer, args.load_optim)
            logger.info('Reloaded optimizer from {}'.format(args.load_optim))

        self.network.train()

    def run(self):
        step_count = self.initial_step_count
        next_save = self.init_next_save(self.initial_step_count, self.epoch_len)
        prev_step_t = time()
        ep_rewards = torch.zeros(self.nb_env)

        next_obs = dtensor_to_dev(self.env_mgr.reset(), self.device)
        internals = listd_to_dlist([
            self.network.new_internals(self.device) for _ in
            range(self.nb_env)
        ])
        start_time = time()
        while step_count < self.nb_step:
            obs = next_obs
            actions, internals = self.agent.act(self.network, obs, internals)
            next_obs, rewards, terminals, infos = self.env_mgr.step(actions)
            next_obs = dtensor_to_dev(next_obs, self.device)

            self.agent.observe(
                obs,
                rewards.to(self.device),
                terminals.to(self.device),
                infos
            )

            # Perform state updates
            step_count += self.nb_env
            ep_rewards += rewards.float()

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
                    'STEP: {} REWARD: {} STEP/S: {}'.format(
                        step_count,
                        term_reward,
                        (step_count - self.initial_step_count) / delta_t
                    )
                )
                self.summary_writer.add_scalar(
                    'reward', term_reward, step_count
                )

            if step_count >= next_save:
                self.saver.save_state_dicts(
                    self.network, step_count, self.optimizer
                )
                next_save += self.epoch_len

            # Learn
            if self.agent.is_ready():
                loss_dict, metric_dict = self.agent.compute_loss(
                    self.network, next_obs, internals
                )
                total_loss = torch.sum(
                    torch.stack(tuple(loss for loss in loss_dict.values()))
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                self.agent.clear()
                for k, vs in internals.items():
                    internals[k] = [v.detach() for v in vs]

                # write summaries
                cur_step_t = time()
                if cur_step_t - prev_step_t > self.summary_freq:
                    self.write_summaries(
                        self.summary_writer, step_count, total_loss, loss_dict,
                        metric_dict, self.network.named_parameters()
                    )
                    prev_step_t = cur_step_t

    def close(self):
        return self.env_mgr.close()
