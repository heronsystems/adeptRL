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
import abc
import time

import numpy as np

from adept.environments import SubProcEnvManager
from adept.networks.modular_network import ModularNetwork
from adept.utils.script_helpers import LogDirHelper
from ._base import CountsRewards
import torch


class EvalContainer:
    def __init__(
            self,
            log_id_dir,
            gpu_id,
            nb_episode,
            seed,
            agent_registry,
            env_registry,
            net_registry
    ):
        """
        :param log_id_dir:
        :param gpu_id:
        :param nb_episode:
        :param seed:
        :param agent_registry:
        :param env_registry:
        :param net_registry:
        """
        self.log_dir_helper = log_dir_helper = LogDirHelper(log_id_dir)
        self.train_args = train_args = log_dir_helper.load_args()
        self.device = device = self._device_from_gpu_id(gpu_id)

        self.env_mgr = env_mgr = SubProcEnvManager.from_args(
            self.train_args,
            seed=seed,
            nb_env=nb_episode,
            registry=env_registry
        )

        if 'agent' in train_args:
            name = train_args.agent
        else:
            name = train_args.train_actor

        output_space = agent_registry.lookup_output_space(
            name, env_mgr.action_space
        )
        eval_actor_cls = agent_registry.lookup_eval_actor(name)
        self.actor = eval_actor_cls.from_args(
            env_mgr.action_space,
            eval_actor_cls.prompt()
        )

        self.network = self._init_network(
            train_args,
            env_mgr.observation_space,
            env_mgr.gpu_preprocessor,
            output_space,
            net_registry
        ).to(device)
        self.network.eval()

    @staticmethod
    def _device_from_gpu_id(gpu_id):
        return torch.device(
            "cuda:{}".format(gpu_id)
            if (torch.cuda.is_available() and gpu_id >= 0)
            else "cpu"
        )

    @staticmethod
    def _init_network(
            train_args,
            obs_space,
            gpu_preprocessor,
            output_space,
            net_reg
    ):
        if train_args.custom_network:
            return net_reg.lookup_custom_net(
                train_args.custom_network
            ).from_args(
                train_args,
                obs_space,
                output_space,
                net_reg
            )
        else:
            return ModularNetwork.from_args(
                train_args,
                obs_space,
                output_space,
                gpu_preprocessor,
                net_reg
            )

    def run(self):
        results = []
        selected_models = []
        for epoch_id in self.log_dir_helper.epochs():
            for net_paths in self.log_dir_helper.network_paths_at_epoch(epoch_id):

                best_mean = -float('inf')
                best_std_dev = 0.
                selected_model = None
                episode_reward_buffer = torch.zeros(self.env_mgr.nb_env)
                for net_path in net_paths:
                    self.network.load_state_dict(
                        torch.load(
                            net_path,
                            map_location=lambda storage, loc: storage
                        )
                    )
                    episode_complete_statuses = [
                        False for _ in range(self.env_mgr.nb_env)
                    ]

                    next_obs = self.env_mgr.reset()
                    while not all(episode_complete_statuses):
                        obs = next_obs
                        actions = self.actor.act(obs)
                        next_obs, rewards, terminals, infos = \
                            self.env_mgr.step(actions)

                        for i, terminal in enumerate(terminals):
                            if terminal:
                                for k, v in self.network.new_internals(self.device).items():
                                    internals[k][i] = v

                        for i in range(self.env_mgr.nb_env):
                            if terminals[i] and infos[i]:
                                episode_complete_statuses[i] = True
                        self.update_buffers(rewards, terminals, infos)

                    reward_buffer = self.episode_reward_buffer.numpy()


class EvalBase(metaclass=abc.ABCMeta):
    def __init__(self, agent, device, environment):
        self._agent = agent
        self._device = device
        self._environment = environment

    @property
    def environment(self):
        return self._environment

    @property
    def agent(self):
        return self._agent

    @property
    def device(self):
        return self._device


class ReplayGenerator(EvalBase):
    """
    Generates replays of agent interacting with SC2 environment.
    """

    def run(self):
        next_obs = self.environment.reset()
        while True:
            obs = next_obs
            actions = self.agent.act_eval(obs)
            next_obs, rewards, terminals, infos = self.environment.step(actions)
            self.agent.reset_internals(terminals)


class AtariRenderer(EvalBase):
    """
    Renders agent interacting with Atari environment.
    """

    def run(self):
        next_obs = self.environment.reset()
        while True:
            time.sleep(1. / 60.)
            self.environment.render()
            obs = next_obs
            actions = self.agent.act_eval(obs)
            next_obs, rewards, terminals, infos = self.environment.step(actions)
            self.agent.reset_internals(terminals)


class Evaluation(EvalBase, CountsRewards):
    def __init__(self, agent, device, environment):
        super().__init__(agent, device, environment)
        self._episode_count = 0
        self.episode_complete_statuses = [False for _ in range(self.nb_env)]

    @property
    def nb_env(self):
        return self._environment.nb_env

    def run(self):
        """
        Run the evaluation. Terminates once each environment has returned a
        score. Averages scores to produce final eval score.

        :return: Tuple[int, int] (mean score, standard deviation)
        """
        next_obs = self.environment.reset()
        while not all(self.episode_complete_statuses):
            obs = next_obs
            actions = self.agent.act_eval(obs)
            next_obs, rewards, terminals, infos = self.environment.step(actions)

            self.agent.reset_internals(terminals)
            self.update_buffers(rewards, terminals, infos)

        reward_buffer = self.episode_reward_buffer.numpy()
        return (
            np.mean(reward_buffer),
            np.std(reward_buffer)
        )

    def update_buffers(self, rewards, terminals, infos):
        """
        Override the reward buffer update rule. Each environment instance will
        only contribute one reward towards the averaged eval score.

        :param rewards: List[float]
        :param terminals: List[bool]
        :param infos: List[Dict[str, Any]]
        :return: None
        """
        for i in range(len(rewards)):
            if self.episode_complete_statuses[i]:
                continue
            elif terminals[i] and infos[i]:
                self.episode_reward_buffer[i] += rewards[i]
                self.episode_complete_statuses[i] = True
            else:
                self.episode_reward_buffer[i] += rewards[i]
        return
