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
import zmq
from torch import multiprocessing as mp

from adept.utils.util import listd_to_dlist, dlist_to_listd
from .base.manager_module import EnvManagerModule

ZMQ_CONNECT_METHOD = "tcp"


class WorkerError(BaseException):
    pass


class SubProcEnvManager(EnvManagerModule):
    """Subprocess Environment Manager

    This class uses torch multiprocessing to establish ZMQ connections to the
    environment subprocesses. Once ZMQ connections are established, the multiprocessing
    pipes are closed and ZMQ is used.

    This class is responsible for two main jobs:
        1) Aggregating the environment observations into a batch for learning.
        2) Unbundling a batch of agent actions and distributing them out to
         the environment subprocesses.

    Observation tensors are communicated via torch shared memory. Non-tensors
    are shared via pickling.

    Actions are serialized via pickling.
    """
    args = {}

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

            process = mp.Process(
                target=worker,
                args=(w_pipe, pipe, port, CloudpickleWrapper(env_fns[w_ind])),
            )
            process.daemon = True
            process.start()
            self.processes.append(process)

            self._zmq_sockets.append(socket)

            pipe.send(("get_shared_memory", None))
            shared_memories.append(pipe.recv())

            # switch to zmq socket and close pipes
            pipe.send(("switch_zmq", None))
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
        """Submits actions to the environment processes

        Parameters
        ----------
        actions : dict[str, torch.Tensor]
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        action_dicts = dlist_to_listd(actions)
        # zmq send
        for socket, action in zip(self._zmq_sockets, action_dicts):
            msg = pickle.dumps(action)
            socket.send(msg, zmq.NOBLOCK, copy=False, track=False)

        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self._zmq_sockets]
        self.waiting = False

        # check for errors and parse
        # self._check_for_errors(results)
        transition = [pickle.loads(res) for res in results]
        obs, rews, dones, infos = zip(*transition)

        obs = listd_to_dlist(obs)
        shared_mems = {
            k: torch.stack(v) for k, v in self.shared_memories.items()
        }
        obs = {**obs, **shared_mems}
        return obs, torch.tensor(rews), torch.tensor(dones), infos

    def reset(self):
        """Tell all subprocess environments to reset to initial state.

        Returns
        -------
        obs : dict[str, torch.Tensor]
            Observation
        """
        for socket in self._zmq_sockets:
            socket.send(pickle.dumps("reset"))
        obs = listd_to_dlist(
            [pickle.loads(remote.recv()) for remote in self._zmq_sockets]
        )
        shared_mems = {
            k: torch.stack(v) for k, v in self.shared_memories.items()
        }
        obs = {**obs, **shared_mems}
        return obs

    def reset_task(self):
        for socket in self._zmq_sockets:
            socket.send(pickle.dumps("reset_task"))
        return [pickle.loads(remote.recv()) for remote in self._zmq_sockets]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self._zmq_sockets:
                remote.recv()
        for socket in self._zmq_sockets:
            socket.send(pickle.dumps("close"))
        for p in self.processes:
            p.join()
        self.closed = True

    def _check_for_errors(self, results):
        errors = []
        del_inds = []
        # multiple workers can fail on the same step
        for i, r in enumerate(results):
            if r[:5] == b"error":
                del_inds.append(i)
                errors.append("Worker {} has an error {}".format(i, r))
        if len(errors) > 0:
            # have to delete from last to first, otherwise inds are invalid
            for d in reversed(sorted(del_inds)):
                # remove from open sockets
                del self._zmq_sockets[d]
            raise WorkerError(errors)


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
            if not dtypes:
                tensor = torch.FloatTensor(*shape)
            else:
                tensor = torch.zeros(*shape, dtype=dtypes[name])
            shared_memory[name] = tensor

    # initial python pipe setup
    python_pipe = True
    while python_pipe:
        cmd, _ = remote.recv()
        if cmd == "get_shared_memory":
            remote.send(shared_memory)
        elif cmd == "switch_zmq":
            # close python pipes
            remote.close()
            python_pipe = False
        else:
            raise NotImplementedError

    # zmq setup
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    if ZMQ_CONNECT_METHOD == "tcp":
        socket.connect("tcp://localhost:{}".format(port))
    if ZMQ_CONNECT_METHOD == "ipc":
        socket.connect("ipc:///tmp/adeptzmq/{}".format(port))

    running = True
    while running:
        try:
            socket_data = socket.recv()
            socket_parsed = pickle.loads(socket_data)

            # commands that aren't action dictionaries
            if socket_parsed == "reset":
                ob = env.reset()
                ob = handle_ob(ob, shared_memory)
                # only the non-shared obs are returned here
                socket.send(
                    pickle.dumps(ob), zmq.NOBLOCK, copy=False, track=False,
                )
            elif cmd == "reset_task":
                ob = env.reset_task()
                ob = handle_ob(ob, shared_memory)
                # only the non-shared obs are returned here
                socket.send(
                    pickle.dumps(ob), zmq.NOBLOCK, copy=False, track=False,
                )
            elif socket_parsed == "close":
                env.close()
                running = False
            # else action dictionary
            else:
                action_dictionary = socket_parsed
                ob, reward, done, info = env.step(action_dictionary)
                if done:
                    ob = env.reset()
                ob = handle_ob(ob, shared_memory)
                # only the non-shared obs are returned here
                msg = pickle.dumps((ob, reward, done, info))
                socket.send(msg, zmq.NOBLOCK, copy=False, track=False)
        except KeyboardInterrupt:
            pass
        # except Exception as e:
        #     running = False
        #     e_str = "{}: {}".format(type(e).__name__, e)
        #     print("Subprocess environment has an error", e_str)
        #     socket.send(
        #         pickle.dumps("error. {}".format(e_str)),
        #         zmq.NOBLOCK,
        #         copy=False,
        #         track=False,
        #     )


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
            if ZMQ_CONNECT_METHOD == "tcp":
                socket.bind("tcp://*:{}".format(port))
            if ZMQ_CONNECT_METHOD == "ipc":
                os.makedirs("/tmp/adeptzmq/", exist_ok=True)
                socket.bind("ipc:///tmp/adeptzmq/{}".format(port))
        except zmq.error.ZMQError as e:
            try_count += 1
            socket = None
            last_error = e
            continue
        break
    if socket is None:
        raise Exception(
            "ZMQ couldn't bind socket after 3 tries. {}".format(last_error)
        )
    return socket, port
