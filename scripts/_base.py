from __future__ import division

import os
from time import time

import numpy as np
import torch
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from src.environments.atari import make_atari_env
from src.utils import from_numpy, read_config, CircularBuffer
from tensorboardX import SummaryWriter


class BaseTrainingLoop:
    def __init__(self, args, frame_stack, rollout_cache):
        # constants
        self.args = args
        self.frame_stack = frame_stack
        self.rollout_cache = rollout_cache

        self.log_path = os.path.join(args.log_dir, args.name, args.env)
        self.writer = SummaryWriter(self.log_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.episode_reward_buff = torch.zeros(args.workers)

        # setup
        self.envs = self._setup_envs()
        self.model = self._setup_model()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=args.lr, eps=1e-5, alpha=0.99)
        self.states_batch = self.envs.reset()  # todo: _to_device(), _from_device()

    def _setup_model(self):
        """

        :return: model
        """
        raise NotImplementedError

    def _forward_step(self):
        """

        :return: rewards_unclipped, dones, infos
        """
        raise NotImplementedError

    def _unroll(self):
        """
        :return: tuple(dictionary of losses, dictionary of metrics)
        """
        raise NotImplementedError

    def _weight_losses(self, loss_dict):
        raise NotImplementedError

    def _after_backwards(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.rollout_cache.clear()

    def _setup_envs(self):
        args = self.args

        setup_json = read_config(args.env_config)
        env_conf = setup_json["Default"]
        for i in setup_json.keys():
            if i in args.env:
                env_conf = setup_json[i]

        envs = SubprocVecEnv(
            [make_atari_env(args.env, env_conf, args, args.seed + i, self.frame_stack) for i in range(args.workers)]
        )
        return envs

    def _save_top_models(self, top_models):
        for i, state_dict in enumerate(top_models):
            torch.save(state_dict, os.path.join(self.log_path, 'model{}.dat'.format(i + 1)))

    def _write_summaries(self, total_loss, loss_dict, metric_dict, step_count):
        writer = self.writer

        writer.add_scalar('macro_loss/total_loss', total_loss.item(), step_count)
        for l_name, loss in loss_dict.items():
            writer.add_scalar('loss/' + l_name, loss.item(), step_count)
        for m_name, metric in metric_dict.items():
            writer.add_scalar('metric/' + m_name, metric.item(), step_count)
        for p_name, param in self.model.named_parameters():
            p_name = p_name.replace('.', '/')
            writer.add_scalar(p_name, torch.norm(param).item(), step_count)
            if param.grad is not None:
                writer.add_scalar(p_name + '.grad', torch.norm(param.grad).item(), step_count)

    def _incr_ep_rewards(self, rewards_unclipped, dones, infos):
        self.episode_reward_buff += torch.tensor(rewards_unclipped).float()
        ep_rewards = []
        for ep_reward, done, info in zip(self.episode_reward_buff, dones, infos):
            if done and info:
                ep_rewards.append(ep_reward.item())
                ep_reward.zero_()
        return ep_rewards

    def _states_to_device(self, state):
        return from_numpy(state, self.device)

    def run(self):
        args = self.args
        writer = self.writer
        model = self.model

        top_3_models = CircularBuffer(3)
        step_count = 0
        high_score = 0
        start_time = time()
        while True:
            for step in range(args.num_steps + 1):
                step_count += args.workers

                # forward step
                rewards_unclipped, dones, infos = self._forward_step()

                # logging and model saving
                ep_rewards = self._incr_ep_rewards(rewards_unclipped, dones, infos)
                if ep_rewards:
                    reward_result = np.mean(ep_rewards)
                    writer.add_scalar('reward', reward_result, step_count)
                    print('step', step_count, 'reward', reward_result, 'sps', step_count / (time() - start_time))
                    if reward_result >= high_score and args.save_max:
                        top_3_models.append(model.state_dict())
                        high_score = reward_result

            losses_dict, metrics_dict = self._unroll()
            total_loss = self._weight_losses(losses_dict)
            total_loss.backward()

            # logging
            if step_count % 100 == 0:
                self._write_summaries(total_loss, losses_dict, metrics_dict, step_count)

            # clean up
            self._after_backwards()

            # check if done training
            if step_count > args.max_steps:
                self._save_top_models(top_3_models)
                break
