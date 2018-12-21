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

import pickle

import cloudpickle
import numpy as np
import torch
from torch import multiprocessing as mp

from adept.environments.managers._manager import AdeptEnvManager
from adept.utils.util import listd_to_dlist, dlist_to_listd


class SubProcEnvManager(AdeptEnvManager):
    """
    Modified.
    MIT License
    Copyright (c) 2017 OpenAI (http://openai.com)
    """
    def __init__(self, env_fns, engine):
        super(SubProcEnvManager, self).__init__(env_fns, engine)
        # TODO: sharing cuda tensors requires spawn or forkserver but these do not work with mpi
        # mp.set_start_method('spawn')

        self.waiting = False
        self.closed = False

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.nb_env)])
        self.ps = [
            mp.Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn))
            ) for (work_remote, remote, env_fn) in zip(self.work_remotes,
                                                       self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self._observation_space, self._action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_processors', None))
        self._cpu_preprocessor, self._gpu_preprocessor = self.remotes[0].recv()

        shared_memories = []
        for remote in self.remotes:
            remote.send(('get_shared_memory', None))
            shared_memories.append(remote.recv())
        self.shared_memories = listd_to_dlist(shared_memories)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def cpu_preprocessor(self):
        return self._cpu_preprocessor

    @property
    def gpu_preprocessor(self):
        return self._gpu_preprocessor

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        action_dicts = dlist_to_listd(actions)
        for remote, action in zip(self.remotes, action_dicts):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        obs = listd_to_dlist(obs)
        shared_mems = {k: torch.stack(v) for k, v in self.shared_memories.items()}
        obs = {**obs, **shared_mems}
        return obs, rews, dones, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = listd_to_dlist([remote.recv() for remote in self.remotes])
        shared_mems = {k: torch.stack(v) for k, v in self.shared_memories.items()}
        obs = {**obs, **shared_mems}
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


def dummy_handle_ob(ob):
    new_ob = {}
    for k, v in ob.items():
        if isinstance(v, np.ndarray):
            new_ob[k] = torch.from_numpy(v)
        else:
            new_ob[k] = v
    return new_ob


def worker(remote, parent_remote, env_fn_wrapper):
    """
    Modified.
    MIT License
    Copyright (c) 2017 OpenAI (http://openai.com)
    """
    parent_remote.close()
    env = env_fn_wrapper.x()

    ebn = env.cpu_preprocessor.observation_space.entries_by_name
    shared_memory = {}
    for name, entry in ebn.items():
        if None not in entry.shape:
            if entry.dtype == np.uint8:
                tensor = torch.ByteTensor(*entry.shape)
            elif entry.dtype == np.int16 or entry.dtype == np.int32 or entry.dtype == np.int64:
                tensor = torch.LongTensor(*entry.shape)
            # TODO: support more datatypes
            else:
                tensor = torch.FloatTensor(*entry.shape)
            shared_memory[name] = tensor

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
        elif cmd == 'get_processors':
            remote.send((env.cpu_preprocessor, env.gpu_preprocessor))
        elif cmd == 'get_shared_memory':
            remote.send(shared_memory)
        else:
            raise NotImplementedError


def handle_ob(ob, shared_memory):
    non_shared = {}
    for k, v in ob.items():
        if isinstance(v, torch.Tensor):
            shared_memory[k].copy_(v)
        else:
            non_shared[k] = v
    return non_shared


class CloudpickleWrapper(object):
    """
    Modified.
    MIT License
    Copyright (c) 2017 OpenAI (http://openai.com)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)