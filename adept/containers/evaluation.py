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

from ._base import CountsRewards


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
