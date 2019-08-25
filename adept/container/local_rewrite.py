import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
from time import time
from torch.utils.tensorboard import SummaryWriter

from adept.globals import VERSION
from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.utils.logging import SimpleModelSaver
from adept.utils.script_helpers import LogDirHelper
from adept.utils.util import DotDict, dtensor_to_dev, listd_to_dlist


class Local:
    def __init__(
            self,
            agent,
            nb_step,
            env_mgr,
            nb_env,
            network,
            optimizer_fn,
            device,
            initial_step_count,
            log_id_dir,
            epoch_len,
            summary_freq,
            args,
            net_load_path=None,
            optim_load_path=None
    ):
        self.agent = agent
        self.nb_step = nb_step
        self.env_mgr = env_mgr
        self.nb_env = nb_env
        self.network = network.to(device)
        self.optimizer = optimizer_fn(self.network.parameters())
        self.device = device
        self.initial_step_count = initial_step_count
        self.log_id_dir = log_id_dir
        self.epoch_len = epoch_len
        self.summary_freq = summary_freq

        os.makedirs(log_id_dir, exist_ok=True)
        self.logger = self._setup_logger(
            logging.getLogger(self.__class__.__name__),
            os.path.join(log_id_dir, 'train_log.txt')
        )
        self.summary_writer = SummaryWriter(log_id_dir)
        self.saver = SimpleModelSaver(log_id_dir)

        if net_load_path:
            self.network = self.load_network_from_path(net_load_path)
        if optim_load_path:
            self.optimizer = self.load_optim_from_path(optim_load_path)

        self._print_ascii_logo()
        self._log_args(args)
        self._write_args_file(args)

    @classmethod
    def from_resume(cls, resume_path):
        log_dir_helper = LogDirHelper(resume_path)
        with open(log_dir_helper.args_file_path(), 'r') as args_file:
            args = DotDict(json.load(args_file))

        args.load_network = log_dir_helper.latest_network_path()
        args.load_optim = log_dir_helper.latest_optim_path()
        initial_step_count = log_dir_helper.latest_epoch()
        log_id = cls._make_log_id(
            args.tag, cls.__name__, args.agent, args.netbody,
            timestamp=log_dir_helper.timestamp()
        )
        return cls._from_args(args, log_id, initial_step_count)

    @classmethod
    def from_args(cls, args, log_id=None, initial_step_count=0):
        args = cls._parse_dynamic_args(args)
        return cls._from_args(args, log_id, initial_step_count)

    @classmethod
    def _from_args(cls, args, log_id, initial_step_count):
        if log_id is None:
            log_id = cls._make_log_id(
                args.tag, cls.__name__, args.agent, args.netbody
            )
        log_id_dir = os.path.join(args.logdir, args.env, log_id)

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
            net = REGISTRY.lookup_network(args.custom_network).from_args(
                args,
                env_mgr.action_space,
                output_space,
                env_mgr.gpu_preprocessor,
                REGISTRY
            )
        else:
            net = ModularNetwork.from_args(
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
        return cls(
            agent,
            args.nb_step,
            env_mgr,
            args.nb_env,
            net,
            optim_fn,
            device,
            initial_step_count,
            log_id_dir,
            args.epoch_len,
            args.summary_freq,
            args,
            args.load_network,
            args.load_optim
        )

    @staticmethod
    def _parse_dynamic_args(args):
        r = REGISTRY
        agent_cls = r.lookup_agent(args.agent)
        env_cls = r.lookup_env(args.env)
        rwdnorm_cls = r.lookup_reward_normalizer(args.rwd_norm)
        if args.use_defaults:
            agent_args = agent_cls.args
            env_args = env_cls.args
            rwdnorm_args = rwdnorm_cls.args
            if args.custom_network:
                net_args = r.lookup_network(args.custom_network).args
            else:
                net_args = r.lookup_modular_args(args)
        else:
            agent_args = agent_cls.prompt()
            env_args = env_cls.prompt()
            rwdnorm_args = rwdnorm_cls.prompt()
            if args.custom_network:
                net_args = r.lookup_network(args.custom_network).prompt()
            else:
                net_args = r.prompt_modular_args(args)
        return DotDict({
            **args, **agent_args, **env_args, **rwdnorm_args, **net_args
        })

    @staticmethod
    def _make_log_id(tag, mode_name, agent_name, network_name, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        parts = [mode_name, agent_name, network_name, timestamp]
        if tag:
            parts = [tag] + parts
        return '_'.join(parts)

    @staticmethod
    def _setup_logger(logger, log_file):
        logger.setLevel(logging.INFO)
        logger.propagate = False

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        fmt = logging.Formatter('%(message)s')
        sh.setFormatter(fmt)

        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s  [%(levelname)s] %(message)s')
        fh.setFormatter(fmt)

        logger.addHandler(sh)
        logger.addHandler(fh)

        return logger

    def load_network_from_path(self, path):
        self.network.load_state_dict(
            torch.load(
                path, map_location=lambda storage, loc: storage
            )
        )
        self.logger.info('Reloaded network from {}'.format(path))
        return self.network

    def load_optim_from_path(self, path):
        self.optimizer.load_state_dict(
                torch.load(
                    path,
                    map_location=lambda storage, loc: storage
                )
            )
        self.logger.info("Reloaded optimizer from {}".format(path))
        return self.optimizer

    def run(self):
        self.network.train()
        step_count = self.initial_step_count
        next_save = self._init_next_save()
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
            # Build rollout
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
                self.logger.info(
                    'STEP: {} REWARD: {} STEP/S: {}'.format(
                        step_count,
                        term_reward,
                        (step_count - self.initial_step_count) /
                        (time() - start_time)
                    )
                )
                self.summary_writer.add_scalar('reward', term_reward, step_count)

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
                    self._write_summaries(
                        step_count, total_loss, loss_dict, metric_dict,
                    )
                    prev_step_t = cur_step_t

    def close(self):
        self.env_mgr.close()

    def _init_next_save(self):
        next_save = 0
        if self.initial_step_count > 0:
            while next_save <= self.initial_step_count:
                next_save += self.epoch_len
        return next_save

    def _log_args(self, args):
        args = args if isinstance(args, dict) else vars(args)
        for k in sorted(args):
            self.logger.info('{}: {}'.format(k, args[k]))

    def _write_args_file(self, args):
        args = args if isinstance(args, dict) else vars(args)
        with open(os.path.join(self.log_id_dir, 'args.json'), 'w') as args_file:
            json.dump(args, args_file, indent=4, sort_keys=True)

    @staticmethod
    def _print_ascii_logo():
        version_len = len(VERSION)
        print(
            """
                         __           __
              ____ _____/ /__  ____  / /_
             / __ `/ __  / _ \/ __ \/ __/
            / /_/ / /_/ /  __/ /_/ / /_
            \__,_/\__,_/\___/ .___/\__/
                           /_/           """ + '\n' +
            '                                     '[:-(version_len + 2)] +
            'v{} '.format(VERSION)
        )

    def _write_summaries(
            self, step_count, total_loss, loss_dict, metric_dict,
    ):
        writer = self.summary_writer
        writer.add_scalar(
            'loss/total_loss', total_loss.item(), step_count
        )
        for l_name, loss in loss_dict.items():
            writer.add_scalar('loss/' + l_name, loss.item(), step_count)
        for m_name, metric in metric_dict.items():
            writer.add_scalar('metric/' + m_name, metric.item(), step_count)
        for p_name, param in self.network.named_parameters():
            p_name = p_name.replace('.', '/')
            writer.add_scalar(p_name, torch.norm(param).item(), step_count)
            if param.grad is not None:
                writer.add_scalar(
                    p_name + '.grad',
                    torch.norm(param.grad).item(), step_count
                )
