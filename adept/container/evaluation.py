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
import os
import time

import torch

from adept.registry import REGISTRY
from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.utils.script_helpers import LogDirHelper
from adept.utils.util import listd_to_dlist, dtensor_to_dev


class EvalContainer:
    def __init__(
            self,
            logger,
            log_id_dir,
            gpu_id,
            nb_episode,
            seed
    ):
        """
        :param log_id_dir:
        :param gpu_id:
        :param nb_episode:
        :param seed:
        """
        self.log_dir_helper = log_dir_helper = LogDirHelper(log_id_dir)
        self.train_args = train_args = log_dir_helper.load_args()
        self.device = device = self._device_from_gpu_id(gpu_id)
        self.logger = logger

        engine = REGISTRY.lookup_engine(train_args.env)
        env_cls = REGISTRY.lookup_env(train_args.env)
        self.env_mgr = env_mgr = SubProcEnvManager.from_args(
            self.train_args,
            engine,
            env_cls,
            seed=seed,
            nb_env=nb_episode
        )

        output_space = REGISTRY.lookup_output_space(
            train_args.agent, env_mgr.action_space
        )
        eval_actor_cls = REGISTRY.lookup_eval_actor(train_args.agent)
        self.actor = eval_actor_cls.from_args(
            eval_actor_cls.prompt(),
            env_mgr.action_space
        )

        self.network = self._init_network(
            train_args,
            env_mgr.observation_space,
            env_mgr.gpu_preprocessor,
            output_space,
            REGISTRY
        ).to(device)

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
            net_cls = net_reg.lookup_network(train_args.custom_network)
        else:
            net_cls = ModularNetwork

        return net_cls.from_args(
            train_args,
            obs_space,
            output_space,
            gpu_preprocessor,
            net_reg
        )

    def run(self):
        nb_env = self.env_mgr.nb_env
        best_epoch_id = None
        overall_mean = -float('inf')
        for epoch_id in self.log_dir_helper.epochs():
            best_mean = -float('inf')
            best_std = None
            selected_model = None
            reward_buf = torch.zeros(nb_env)
            for net_path in self.log_dir_helper.network_paths_at_epoch(epoch_id):
                self.network.load_state_dict(
                    torch.load(
                        net_path,
                        map_location=lambda storage, loc: storage
                    )
                )
                self.network.eval()

                internals = listd_to_dlist([
                    self.network.new_internals(self.device) for _ in
                    range(nb_env)
                ])
                episode_completes = [False for _ in range(nb_env)]
                next_obs = dtensor_to_dev(self.env_mgr.reset(), self.device)

                while not all(episode_completes):
                    obs = next_obs
                    with torch.no_grad():
                        actions, _, internals = self.actor.act(self.network, obs, internals)
                    next_obs, rewards, terminals, infos = self.env_mgr.step(actions)
                    next_obs = dtensor_to_dev(next_obs, self.device)

                    for i in range(self.env_mgr.nb_env):
                        if episode_completes[i]:
                            continue
                        elif terminals[i]:
                            reward_buf[i] += rewards[i]
                            episode_completes[i] = True
                        else:
                            reward_buf[i] += rewards[i]

                mean = reward_buf.mean().item()
                std = reward_buf.std().item()

                if mean >= best_mean:
                    best_mean = mean
                    best_std = std
                    selected_model = os.path.split(net_path)[-1]

            self.logger.info(
                f'EPOCH_ID: {epoch_id} '
                f'MEAN_REWARD: {best_mean} '
                f'STD_DEV: {best_std} '
                f'SELECTED_MODEL: {selected_model}'
            )
            with open(self.log_dir_helper.eval_path(), 'a') as eval_f:
                eval_f.write(f'{epoch_id},'
                             f'{best_mean},'
                             f'{best_std},'
                             f'{selected_model}\n')

            if best_mean >= overall_mean:
                best_epoch_id = epoch_id
                overall_mean = best_mean
        print(f'*** EPOCH_ID: {best_epoch_id} MEAN_REWARD: {overall_mean} ***')

    def close(self):
        self.env_mgr.close()


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
