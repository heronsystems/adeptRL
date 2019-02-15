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
import time
import numpy as np
import torch
from ._base import MPIProc, HasAgent, HasEnvironment, LogsAndSummarizesRewards, WritesSummaries
from .mpi import MPIHelper, MpiMessages, ArrayFlattener
from adept.utils import listd_to_dlist
from collections import OrderedDict


class ImpalaHost(HasAgent, WritesSummaries, MPIProc):
    def __init__(
        self,
        agent,
        mpi_comm,
        make_optimizer,
        summary_writer,
        summary_frequency,
        saver,
        save_interval,
        summary_steps,
        rank,
        use_local_buffers=False
    ):
        """
        use_local_buffers: bool If true does not send the network's buffers to
        the workers (noisy batch norm)
        """
        self._agent = agent
        self._optimizer = make_optimizer(self.network.parameters())
        self._summary_writer = summary_writer
        self._summary_frequency = summary_frequency
        self._rank = rank
        self.saver = saver
        self.save_interval = save_interval
        self.comm = mpi_comm
        self.sent_parameters_count = 0
        self.summary_steps = summary_steps
        self.use_local_buffers = use_local_buffers
        self.variable_flattener = ArrayFlattener(self.get_parameter_shapes())

        # initial setup to get the sizes of the rollouts
        for w_ind in range(1, mpi_comm.size):
            # TODO: sanity check that all workers return the same size
            # Workers should send an ordered dict of {'name': shape, ...}
            self.rollout_sizes = self.comm.recv(
                source=w_ind, tag=MpiMessages.NAMES_AND_SHAPES
            )
            assert isinstance(self.rollout_sizes, OrderedDict)

        # create rollout flattener/unflattener
        self.sorted_keys = list(self.rollout_sizes.keys())
        self.rollout_flattener = ArrayFlattener(
            [v for v in self.rollout_sizes.values()] + [(1)]
        )  # +1 for timestep

    @property
    def agent(self):
        return self._agent

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def summary_writer(self):
        return self._summary_writer

    @property
    def summary_frequency(self):
        return self._summary_frequency

    def _batch_add_thread(
        self, listen_source, worker_wait_time, worker_timestep,
        max_items_in_queue
    ):
        """
            A thread that listens to a mpi worker and
            appends to the shared self.recieved_rollouts list
        """
        # iterative wait time to make sure the host doesn't get behind and not
        # be able to recover
        worker_wait_time.fill(0.0)
        # setup rollout buffers
        rollout_buffer = np.empty(self.rollout_flattener.total_size, np.float32)
        # timestep is a np array so the value is correctly shared
        worker_timestep.fill(0)

        # thread loop as long as master says we're not done
        while not self._threads_should_be_done:
            # possible wait if host is slow
            while len(
                self.received_rollouts
            ) > max_items_in_queue and not self._threads_should_be_done:
                # random wait time to ensure that threads can uniformly add
                # when max_queue size is reached
                how_long_to_wait = float(np.random.uniform(0.1, 1))
                time.sleep(how_long_to_wait)
                worker_wait_time += how_long_to_wait

            # break after possible waiting
            if self._threads_should_be_done:
                break

            # setup listener for rollout, Recv is blocking but okay on thread
            self.comm.Recv(
                rollout_buffer, source=listen_source, tag=MpiMessages.SEND
            )

            # rollout is in buffer, copy into individual parts
            unflattened_rollout = self.rollout_flattener.unflatten(
                rollout_buffer
            )
            recv_timestep = unflattened_rollout.pop()

            # to dict of tensors
            rollout = {
                # act on host must check terminal to determine if internals need to be reset, MUCH faster to do on cpu
                k: torch.from_numpy(v).to(self.agent.device) if k != 'terminals' else torch.from_numpy(v)
                for k, v in zip(
                    self.sorted_keys,
                    unflattened_rollout  # timestep has been popped
                )
            }
            self.received_rollouts.append(rollout)

            worker_timestep.fill(
                recv_timestep[0]
            )  # recv_timestep is a single value array

            # upon recieving batch, send ack
            self.comm.isend(
                self._threaded_global_step(),
                dest=listen_source,
                tag=MpiMessages.SEND_ACK
            )
        print('Thread listening for {} exiting'.format(listen_source))

    def _threaded_global_step(self):
        return sum(self._thread_steps)[0]  # numpy array

    def _saver_thread(self):
        next_save_step = self.save_interval
        try:
            while not self._threads_should_be_done:
                current_step = self._threaded_global_step()
                if current_step > next_save_step:
                    self.saver.save_state_dicts(
                        self.network,
                        int(current_step),
                        optimizer=self.optimizer
                    )
                    next_save_step += self.save_interval
                time.sleep(1)
        except Exception as e:
            print('Error saving', e)

        # final save
        self.saver.save_state_dicts(
            self.network,
            int(self._threaded_global_step()),
            optimizer=self.optimizer
        )

    def run(
        self,
        num_rollouts_in_batch,
        max_items_in_queue,
        max_steps=float('inf'),
        dynamic=False,
        min_dynamic_batch=0
    ):
        from threading import Thread
        size = self.comm.Get_size()

        # setup a shared rollout list across threads
        # python lists are GIL safe
        self.received_rollouts = []
        # shared done flag and steps
        self._threads_should_be_done = False
        self._thread_steps = [np.ones((1)) for i in range(1, size)]
        self._thread_wait_times = [np.zeros((1)) for i in range(1, size)]
        # start batch adding threads
        threads = [
            Thread(
                target=self._batch_add_thread,
                args=(
                    i,
                    self._thread_wait_times[i - 1],
                    self._thread_steps[i - 1],
                    max_items_in_queue,
                )
            ) for i in range(1, size)
        ]
        for t in threads:
            t.start()

        # start saver thread
        saver_thread = Thread(target=self._saver_thread)
        saver_thread.start()

        # this runs a "busy" loop assuming that loss.backward takes longer than
        # accumulating batches removing the need for the threads to notify of
        # new batches
        variable_buffer = np.empty(
            self.variable_flattener.total_size, np.float32
        )
        start_time = time.time()
        number_of_rollouts_waiting = 0
        while self._threaded_global_step() < max_steps:
            len_received_rollouts = len(self.received_rollouts)
            # rollouts waiting?
            if len_received_rollouts > 0:
                # dynamic requires at least one rollout
                if dynamic:
                    do_batch = len_received_rollouts >= min_dynamic_batch
                    # limit batch size to max of num_rollouts_in_batch
                    batch_slice = min(
                        (len_received_rollouts, num_rollouts_in_batch)
                    )
                else:
                    do_batch = len_received_rollouts >= num_rollouts_in_batch
                    batch_slice = num_rollouts_in_batch

                if do_batch:
                    try:
                        number_of_rollouts_waiting += len_received_rollouts

                        # pop everything from list starting first to last
                        popped_rollouts = self.received_rollouts[0:batch_slice]
                        # clear
                        self.received_rollouts = self.received_rollouts[
                            batch_slice:]

                        # convert list of dict to dict of list
                        rollouts = listd_to_dlist(popped_rollouts)

                        loss_dict, metric_dict = self.agent.compute_loss(
                            rollouts
                        )
                        total_loss = torch.sum(
                            torch.stack(
                                tuple(loss for loss in loss_dict.values())
                            )
                        )

                        self.optimizer.zero_grad()
                        total_loss.backward()
                        self.optimizer.step()

                        # write summaries
                        self.write_summaries(
                            total_loss, loss_dict, metric_dict,
                            self._threaded_global_step()
                        )

                        # send new variables
                        new_variables_flat = self.variable_flattener.flatten(
                            self.get_parameters_numpy(), buffer=variable_buffer
                        )
                        self.comm.Ibcast(new_variables_flat, root=0)
                        self.sent_parameters_count += 1

                        if self.sent_parameters_count % self.summary_steps == 0:
                            print('=' * 40)
                            print(
                                'Train Per Second', self.sent_parameters_count /
                                (time.time() - start_time)
                            )
                            print(
                                'Avg Queue Length', number_of_rollouts_waiting /
                                self.sent_parameters_count
                            )
                            print(
                                'Current Queue Length',
                                len(self.received_rollouts)
                            )
                            print(
                                'Thread wait times', [
                                    '{:.2f}'.format(x[0])
                                    for x in self._thread_wait_times
                                ]
                            )
                            print('=' * 40)
                    except Exception as e:
                        import traceback
                        print('Host error: {}'.format(e))
                        print(traceback.format_exc())
                        print('Trying to stop gracefully')
                        break

        # cleanup threads
        print('Host stopping threaded batch recvs')
        self._threads_should_be_done = True
        [t.join() for t in threads]

        # cleanup mpi
        self.stop_mpi_workers()

        # join saver after mpi workers are done to prevent infinite waiting
        saver_thread.join()

    def stop_mpi_workers(self):
        size = self.comm.Get_size()
        print('Host sending stop')
        for i in range(1, size):
            self.comm.isend(True, dest=i, tag=MpiMessages.STOP)
        print('Host waiting on recieving stops')
        stops = [
            self.comm.irecv(source=i, tag=MpiMessages.STOPPED)
            for i in range(1, size)
        ]
        threads_stoped = 0
        wait = [0] * (size - 1)
        while threads_stoped < size - 1:
            for w_ind, s in enumerate(stops):
                if s is not None:
                    worker = w_ind + 1
                    done = s.test()[0]
                    if done:
                        print('Host recieved {} done.'.format(worker))
                        stops[w_ind] = None
                        threads_stoped += 1
                    else:
                        wait[w_ind] += 1
                        if wait[w_ind] >= 5:
                            print(
                                'Waited 5 times for {}, skipping'.
                                format(worker)
                            )
                            threads_stoped += 1
                        print('Still waiting on {} to finish'.format(worker))
                        time.sleep(0.1)

        print('Host sees all threads as stopped.')

    def get_parameter_shapes(self):
        shapes = [tuple(x.shape) for x in self.network.parameters()]
        if not self.use_local_buffers:
            shapes.extend([tuple(x.shape) for x in self.network.buffers()])
        return shapes

    def get_parameters_numpy(self):
        params = [p.detach().cpu().numpy() for p in self.network.parameters()]
        if not self.use_local_buffers:
            params.extend([b.cpu().numpy() for b in self.network.buffers()])
        return params


class ImpalaWorker(HasAgent, HasEnvironment, LogsAndSummarizesRewards, MPIProc):
    def __init__(
        self,
        agent,
        environment,
        nb_env,
        logger,
        summary_writer,
        rank,
        use_local_buffers=False,
        max_parameter_skip=10,
        send_warning_time=float('inf'),
        recv_warning_time=0.1
    ):
        super().__init__()
        self._agent = agent
        self._environment = environment
        self._nb_env = nb_env
        self._logger = logger
        self._summary_writer = summary_writer
        self._rank = rank

        self.global_step = 0
        self.mpi_helper = None
        self.max_parameter_skip = max_parameter_skip
        self.send_warning_time = send_warning_time
        self.recv_warning_time = recv_warning_time
        self.use_local_buffers = use_local_buffers

    @property
    def agent(self):
        return self._agent

    @property
    def environment(self):
        return self._environment

    @property
    def nb_env(self):
        return self._nb_env

    @property
    def logger(self):
        return self._logger

    @property
    def summary_writer(self):
        return self._summary_writer

    def run(self, initial_count=0):
        self.local_step_count = initial_count
        next_obs = self.environment.reset()
        self._starting_internals = self.agent.internals
        while not self.should_stop():
            obs = next_obs
            # need to copy the obs so the rollout append works
            copied_obs = {k: [x.clone() for x in v] for k, v in obs.items()}
            # Build rollout
            if not self.use_local_buffers:
                # TODO: set only batch norm modules to eval
                self.network.eval()
            actions = self.agent.act(obs)
            next_internals = self.agent.internals
            next_obs, rewards, terminals, infos = self.environment.step(actions)
            self.agent.observe(copied_obs, rewards, terminals, infos)

            # Perform state updates
            terminal_rewards, terminal_infos = self.update_buffers(
                rewards, terminals, infos
            )
            self.log_episode_results(
                terminal_rewards,
                terminal_infos,
                self.global_step,
                self.local_step_count,
                self._rank,
                initial_step_count=initial_count
            )
            self.write_reward_summaries(terminal_rewards, self.global_step)

            # Learn
            if self.exp_cache.is_ready():
                self.learn(next_obs)
                self._starting_internals = next_internals
        self.close()

    def learn(self, next_obs):
        rollout = self.exp_cache.read()

        # compute flattened rollout cache size after the first training step
        if self.mpi_helper is None:
            # sorted keys to ensure deterministic order
            shapes = OrderedDict()
            for k in sorted(rollout._fields):
                if k != 'states':
                    # value
                    v = getattr(rollout, k)
                    # value is list of tensors
                    shapes[k] = (len(v), ) + v[0].shape

            # add internals shapes, sorted again
            for k in sorted(self._starting_internals.keys()):
                v = self._starting_internals[k]
                shapes['internals-' + k] = (len(v), ) + v[0].shape

            # add obs, sorted
            state_dlist = listd_to_dlist(rollout.states)
            for dict_key in sorted(state_dlist.keys()):
                shapes['rollout_obs-' + dict_key] = (len(state_dlist[dict_key]), ) + (len(state_dlist[dict_key][0]), ) + \
                                                    state_dlist[dict_key][0][0].shape

            # add next obs, sorted
            for k in sorted(next_obs.keys()):
                v = next_obs[k]
                shapes['next_obs-' + k] = (len(v), ) + v[0].shape
            parameter_shapes = self.get_parameter_shapes()
            self.mpi_helper = MPIHelper(
                shapes, parameter_shapes, 0, self.max_parameter_skip,
                self.send_warning_time, self.recv_warning_time, float('inf')
            )  # workers wait on send which waits on queue len so no warnings
            # for # of recv updates are needed

        # TODO: mpi_helper send should accept list of lists of tensors
        # then don't have to torch stack
        # combine rollout and internals
        sends = []
        for k in sorted(rollout._fields):
            if k != 'states':
                # value
                v = getattr(rollout, k)
                # value is list of tensors, or can be a single tensor, or numpy array (in the case of actions)
                if isinstance(v[0], np.ndarray):
                    v = [torch.from_numpy(x) for x in v]
                sends.append(torch.stack(v))

        # add internals shapes, sorted again
        for k in sorted(self._starting_internals.keys()):
            v = self._starting_internals[k]
            sends.append(torch.stack(v))

        # add obs, sorted
        state_dlist = listd_to_dlist(rollout.states)
        for dict_key in sorted(state_dlist.keys()):
            stacked = []
            for item in state_dlist[dict_key]:
                stacked.append(torch.stack(item))
            sends.append(torch.stack(stacked))

        # add next obs, sorted
        for k in sorted(next_obs.keys()):
            v = next_obs[k]
            sends.append(v)

        self.submit(sends)
        self.receive()

        self.exp_cache.clear()
        self.agent.detach_internals()

    def submit(self, rollout):
        """
            Submits rollout to a MPI host
        """
        host_info = self.mpi_helper.send(rollout, self.local_step_count)
        self.global_step = host_info if host_info is not None else 0

    def receive(self):
        """
            Receives parameters from MPI host
        """
        new_params = self.mpi_helper.receive_parameters()
        # new_params can be none if not waiting
        if new_params is not None:
            self.set_parameters(new_params)

    def set_parameters(self, parameters):
        local_params = list(self.network.parameters())
        if not self.use_local_buffers:
            local_params.extend([b for b in self.network.buffers()])

        for p, v in zip(local_params, parameters):
            p.data.copy_(v, non_blocking=True)

    def get_parameter_shapes(self):
        shapes = [tuple(x.shape) for x in self.network.parameters()]
        if not self.use_local_buffers:
            shapes.extend([tuple(x.shape) for x in self.network.buffers()])
        return shapes

    def close(self):
        self.mpi_helper.close()

    def should_stop(self):
        return False if self.mpi_helper is None else self.mpi_helper.should_stop(
        )
