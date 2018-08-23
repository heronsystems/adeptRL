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

from adept.environments import Engines
from ._base import HasAgent, CountsRewards
import numpy as np
import time


class EvalBase(HasAgent, abc.ABC):
    def __init__(self, agent, env_fn, device, seed):
        self._agent = agent
        self._environment = env_fn(seed)
        self._device = device

    @property
    def agent(self):
        return self._agent

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, new_env):
        self._environment = new_env

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
    def __init__(self, agent, env_fn, device, seed, render):
        super().__init__(agent, env_fn, device, seed)
        self._episode_count = 0
        self.seed = seed
        self.env_fn = env_fn
        self.render = render

    @property
    def nb_env(self):
        return 1

    def run(self, nb_episode):
        next_obs = self.environment.reset()
        results = []
        while len(results) < nb_episode:
            obs = next_obs
            if self.render and self.environment.engine == Engines.GYM:
                self.environment.render()
            actions = self.agent.act_eval(obs)
            next_obs, rewards, terminals, infos = self.environment.step(actions)

            self.agent.reset_internals(terminals)
            episode_rewards, _ = self.update_buffers(rewards, terminals, infos)
            for reward in episode_rewards:
                # remake and reseed env when episode finishes
                self.environment.close()
                self.seed += 1
                self.environment = self.env_fn(self.seed)
                next_obs = self.environment.reset()
                self._episode_count += 1
                results.append(reward)
                if len(results) == nb_episode:
                    break
        return np.mean(results), np.std(results)
