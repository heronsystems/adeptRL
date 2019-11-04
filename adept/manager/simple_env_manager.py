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
import numpy as np
import torch

from adept.utils import listd_to_dlist
from adept.utils.util import dlist_to_listd
from .base.manager_module import EnvManagerModule


class SimpleEnvManager(EnvManagerModule):
    """
    Manages multiple env in the same process. This is slower than a
    SubProcEnvManager but allows debugging.
    """

    args = {}

    def __init__(self, env_fns, engine):
        super(SimpleEnvManager, self).__init__(env_fns, engine)
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._cpu_preprocessor = env.cpu_preprocessor
        self._gpu_preprocessor = env.gpu_preprocessor

        self.buf_obs = [None for _ in range(self.nb_env)]
        self.buf_dones = [None for _ in range(self.nb_env)]
        self.buf_rews = [None for _ in range(self.nb_env)]
        self.buf_infos = [None for _ in range(self.nb_env)]
        self.actions = None

    @property
    def cpu_preprocessor(self):
        return self._cpu_preprocessor

    @property
    def gpu_preprocessor(self):
        return self._gpu_preprocessor

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = dlist_to_listd(actions)

    def step_wait(self):
        obs = []
        for e in range(self.nb_env):
            ob, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = \
                self.envs[e].step(self.actions[e])
            if self.buf_dones[e]:
                ob = self.envs[e].reset()
            obs.append(ob)
        obs = listd_to_dlist(obs)
        new_obs = {}
        for k, v in dummy_handle_ob(obs).items():
            if self._is_tensor_key(k):
                new_obs[k] = torch.stack(v)
            else:
                new_obs[k] = v
        self.buf_obs = new_obs

        return self.buf_obs, torch.tensor(self.buf_rews), torch.tensor(self.buf_dones), self.buf_infos

    def reset(self):
        obs = []
        for e in range(self.nb_env):
            ob = self.envs[e].reset()
            obs.append(ob)
        obs = listd_to_dlist(obs)
        new_obs = {}
        for k, v in dummy_handle_ob(obs).items():
            if self._is_tensor_key(k):
                new_obs[k] = torch.stack(v)
            else:
                new_obs[k] = v
        self.buf_obs = new_obs
        return self.buf_obs

    def close(self):
        return [e.close() for e in self.envs]

    def render(self, mode='human'):
        return [e.render(mode=mode) for e in self.envs]

    def _is_tensor_key(self, key):
        return None not in self.cpu_preprocessor.observation_space[key]


def dummy_handle_ob(ob):
    new_ob = {}
    for k, v in ob.items():
        if isinstance(v, np.ndarray):
            new_ob[k] = torch.from_numpy(v)
        else:
            new_ob[k] = v
    return new_ob
