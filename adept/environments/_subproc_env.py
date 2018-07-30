from collections import OrderedDict

import numpy as np
import torch
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from gym import spaces
from torch import multiprocessing as mp

from adept.utils.util import listd_to_dlist


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()

    if isinstance(env.observation_space, spaces.Box):
        shared_memory = {'obs': torch.FloatTensor(*env.observation_space.shape)}
    elif isinstance(env.observation_space, spaces.Dict):
        shared_memory = {}
        for k, space in env.observation_space.spaces.items():
            if isinstance(space, spaces.Box):
                shared_memory[k] = torch.FloatTensor(*space.shape).share_memory_()
            elif isinstance(space, spaces.Discrete):
                shared_memory[k] = torch.FloatTensor(space.n).share_memory_()
            else:
                raise NotImplementedError()
    else:
        raise NotImplementedError()

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            ob = handle_ob(ob, shared_memory)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            ob = handle_ob(ob, shared_memory)
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            ob = handle_ob(ob, shared_memory)
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_shared_memory':
            remote.send(shared_memory)
        else:
            raise NotImplementedError


def handle_ob(ob, shared_memory):
    non_shared = {}
    for k, v in ob.items():
        if isinstance(v, torch.Tensor):
            shared_memory[k].copy_(v)
        elif isinstance(v, np.ndarray):
            shared_memory[k].copy_(torch.from_numpy(v))
        else:
            non_shared[k] = v
    return non_shared


class SubProcEnv(VecEnv):
    def __init__(self, env_fns, engine):
        """
        envs: list of gym environments to run in subprocesses
        """
        # TODO: sharing cuda tensors requires spawn or forkserver but these do not work with mpi
        # mp.set_start_method('spawn')
        self.engine = engine

        self.waiting = False
        self.closed = False
        self.nb_env = len(env_fns)

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.nb_env)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        shared_memories = []
        for remote in self.remotes:
            remote.send(('get_shared_memory', None))
            shared_memories.append(remote.recv())
        self.shared_memories = listd_to_dlist(shared_memories)

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        obs = listd_to_dlist(obs)
        obs = {**obs, **self.shared_memories}
        return obs, rews, dones, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = listd_to_dlist([remote.recv() for remote in self.remotes])
        obs = {**obs, **self.shared_memories}
        return obs

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns, engine):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.engine = engine
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        shapes, dtypes = {}, {}
        self.keys = []
        obs_space = env.observation_space

        if isinstance(obs_space, spaces.Dict):
            assert isinstance(obs_space.spaces, OrderedDict)
            subspaces = obs_space.spaces
        else:
            subspaces = {None: obs_space}

        for key, box in subspaces.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = [None for _ in range(self.num_envs)]
        self.buf_dones = [None for _ in range(self.num_envs)]
        self.buf_rews = [None for _ in range(self.num_envs)]
        self.buf_infos = [None for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for e in range(self.num_envs):
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(self.actions[e])
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self.buf_obs[e] = obs
        return listd_to_dlist(self.buf_obs), self.buf_rews, self.buf_dones, self.buf_infos

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self.buf_obs[e] = obs
        return listd_to_dlist(self.buf_obs)

    def close(self):
        return

    def render(self, mode='human'):
        return [e.render(mode=mode) for e in self.envs]
