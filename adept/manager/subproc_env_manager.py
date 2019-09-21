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
import os
import pickle

import cloudpickle
import numpy as np
import torch
from torch import multiprocessing as mp

from adept.manager._manager import EnvManager
from adept.utils.util import listd_to_dlist, dlist_to_listd

import zmq
import json

ZMQ_CONNECT_METHOD = 'ipc'


class SubProcEnvManager(EnvManager):
    """
    Modified.
    MIT License
    Copyright (c) 2017 OpenAI (http://openai.com)
    """
    def __init__(self, env_fns, engine):
        super(SubProcEnvManager, self).__init__(env_fns, engine)
        self.waiting = False
        self.closed = False
        self.processes = []

        self._zmq_context = zmq.Context()
        self._zmq_ports = []
        self._zmq_sockets = []

        # make a temporary env to get stuff
        dummy = env_fns[0]()
        self._observation_space = dummy.observation_space
        self._action_space = dummy.action_space
        self._cpu_preprocessor = dummy.cpu_preprocessor
        self._gpu_preprocessor = dummy.gpu_preprocessor
        dummy.close()

        # iterate envs to get torch shared memory through pipe then close it
        shared_memories = []

        for w_ind in range(self.nb_env):
            pipe, w_pipe = mp.Pipe()
            socket, port = zmq_robust_bind_socket(self._zmq_context)

            process = mp.Process(target=worker, args=(
                w_pipe, pipe, port, CloudpickleWrapper(env_fns[w_ind])
            ))
            process.daemon = True
            process.start()
            self.processes.append(process)

            self._zmq_sockets.append(socket)

            pipe.send(('get_shared_memory', None))
            shared_memories.append(pipe.recv())

            # switch to zmq socket and close pipes
            pipe.send(('switch_zmq', None))
            pipe.close()
            w_pipe.close()

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
        # zmq send
        for socket, action in zip(self._zmq_sockets, action_dicts):
            msg = json.dumps({k: int(v) for k, v in action.items()})
            socket.send(msg.encode(), zmq.NOBLOCK, copy=False, track=False)

        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self._zmq_sockets]
        results = [json.loads(res.decode()) for res in results]
        obs, rews, dones, infos = zip(*results)

        self.waiting = False
        obs = listd_to_dlist(obs)
        shared_mems = {k: torch.stack(v) for k, v in self.shared_memories.items()}
        obs = {**obs, **shared_mems}
        return obs, torch.tensor(rews), torch.tensor(dones), infos

    def reset(self):
        for socket in self._zmq_sockets:
            socket.send('reset'.encode())
        obs = listd_to_dlist([json.loads(remote.recv().decode()) for remote in self._zmq_sockets])
        shared_mems = {k: torch.stack(v) for k, v in self.shared_memories.items()}
        obs = {**obs, **shared_mems}
        return obs

    def reset_task(self):
        for socket in self._zmq_sockets:
            socket.send('reset_task'.encode())
        return [json.loads(remote.recv().decode()) for remote in self._zmq_sockets]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self._zmq_sockets:
                remote.recv()
        for socket in self._zmq_sockets:
            socket.send('close'.encode())
        for p in self.processes:
            p.join()
        self.closed = True


def worker(remote, parent_remote, port, env_fn_wrapper):
    """
    Modified.
    MIT License
    Copyright (c) 2017 OpenAI (http://openai.com)
    """
    parent_remote.close()
    env = env_fn_wrapper.x()

    shared_memory = {}
    dtypes = env.cpu_preprocessor.observation_dtypes
    for name, shape in env.cpu_preprocessor.observation_space.items():
        if shape is not None:
            if dtypes is None:
                tensor = torch.FloatTensor(*shape)
            else:
                tensor = torch.zeros(*shape, dtype=dtypes[name])
            shared_memory[name] = tensor

    # initial python pipe setup
    python_pipe = True
    while python_pipe:
        cmd, _ = remote.recv()
        if cmd == 'get_shared_memory':
            remote.send(shared_memory)
        elif cmd == 'switch_zmq':
            # close python pipes
            remote.close()
            python_pipe = False
        else:
            raise NotImplementedError

    # zmq setup
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    if ZMQ_CONNECT_METHOD == 'tcp':
        socket.connect("tcp://localhost:{}".format(port))
    if ZMQ_CONNECT_METHOD == 'ipc':
        socket.connect("ipc:///tmp/adeptzmq/{}".format(port))

    running = True
    while running:
        socket_data = socket.recv()
        socket_parsed = socket_data.decode()

        # commands that aren't action dictionaries
        if socket_parsed == 'reset':
            ob = env.reset()
            ob = handle_ob(ob, shared_memory)
            # only the non-shared obs are returned here
            socket.send(json.dumps(ob).encode(), zmq.NOBLOCK, copy=False, track=False)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            ob = handle_ob(ob, shared_memory)
            # only the non-shared obs are returned here
            socket.send(json.dumps(ob).encode(), zmq.NOBLOCK, copy=False, track=False)
        elif socket_parsed == 'close':
            env.close()
            running = False
        # else action dictionary
        else:
            action_dictionary = json.loads(socket_parsed)
            ob, reward, done, info = env.step(action_dictionary)
            if done:
                ob = env.reset()
            ob = handle_ob(ob, shared_memory)
            # only the non-shared obs are returned here
            msg = json.dumps((ob, reward, done, info))
            socket.send(msg.encode(), zmq.NOBLOCK, copy=False, track=False)


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


def zmq_robust_bind_socket(zmq_context):
    try_count = 0
    while try_count < 3:
        try:
            socket = zmq_context.socket(zmq.PAIR)
            port = np.random.randint(5000, 30000)
            if ZMQ_CONNECT_METHOD == 'tcp':
                socket.bind("tcp://*:{}".format(port))
            if ZMQ_CONNECT_METHOD == 'ipc':
                os.makedirs('/tmp/adeptzmq/', exist_ok=True)
                socket.bind("ipc:///tmp/adeptzmq/{}".format(port))
        except zmq.error.ZMQError as e:
            try_count += 1
            socket = None
            last_error = e
            continue
        break
    if socket is None:
        raise Exception("ZMQ couldn't bind socket after 3 tries. {}".format(last_error))
    return socket, port
