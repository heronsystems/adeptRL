"""
Copyright (C) 2018 Heron Systems, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import abc
from ._base import HasAgent, CountsRewards
import numpy as np
import time


class EvalBase(HasAgent, abc.ABC):
    def __init__(self, agent, env, device):
        self._agent = agent
        self._environment = env
        self._device = device

    @property
    def agent(self):
        return self._agent

    @property
    def environment(self):
        return self._environment

    @property
    def device(self):
        return self._device


class ReplayGenerator(EvalBase):
    """
    Meant for SC2
    """
    def run(self):
        next_obs = self.environment.reset()
        while True:
            obs = next_obs
            actions = self.agent.act_eval(obs)
            next_obs, rewards, terminals, infos = self.environment.step(actions)
            self.agent.reset_internals(terminals)


class Renderer(EvalBase):
    """
    Atari Only
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
    def __init__(self, agent, env, device, nb_env):
        super().__init__(agent, env, device)
        self._nb_env = nb_env
        self._episode_count = 0

    @property
    def nb_env(self):
        return self._nb_env

    def run(self, nb_episode):
        next_obs = self.environment.reset()
        results = []
        while len(results) < nb_episode:
            obs = next_obs
            actions = self.agent.act_eval(obs)
            next_obs, rewards, terminals, infos = self.environment.step(actions)

            self.agent.reset_internals(terminals)
            episode_rewards = self.update_buffers(rewards, terminals, infos)
            for reward in episode_rewards:
                self._episode_count += 1
                results.append(reward)
                if len(results) == nb_episode:
                    break
        return np.mean(results), np.std(results)
