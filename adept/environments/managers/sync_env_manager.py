import torch

from adept.environments._base import AdeptEnv
from adept.environments.managers.parallel_env_manager import dummy_handle_ob
from adept.utils import listd_to_dlist


class DebugEnvManager(AdeptEnv):
    """
    Modified.
    MIT License
    Copyright (c) 2017 OpenAI (http://openai.com)
    """
    def __init__(self, env_fns, engine):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.engine = engine
        self._observation_space, self._action_space = env.observation_space, env.action_space
        self._cpu_preprocessor, self._gpu_preprocessor = env.cpu_preprocessor, env.gpu_preprocessor

        self.nb_env = len(env_fns)
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
        self.actions = actions

    def step_wait(self):
        obs = []
        for e in range(self.nb_env):
            ob, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(self.actions[e])
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

        return self.buf_obs, self.buf_rews, self.buf_dones, self.buf_infos

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
        ebn = self.cpu_preprocessor.observation_space.entries_by_name
        return None not in ebn[key].shape