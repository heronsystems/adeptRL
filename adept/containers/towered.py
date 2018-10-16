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
from ._base import (
    HasAgent, WritesSummaries, LogsAndSummarizesRewards, AppliesGrads, MPIProc, HasEnvironment
)
from .mpi import MPIHelper, MPIArraySend, MpiMessages, ArrayFlattener
import numpy as np
import torch
import time
from mpi4py import MPI as mpi
from collections import OrderedDict
from threading import Thread


class ToweredHost(AppliesGrads):
    def __init__(self, mpi_comm, num_grads_to_drop, network, make_optimizer, saver, save_interval, logger,
                 summary_steps=100):
        self._optimizer = make_optimizer(network.parameters())

        self.network = network
        self.logger = logger
        self.saver = saver
        self.save_interval = save_interval
        self._next_save_step = save_interval
        self.global_step = 0
        self.start_time = None
        self.comm = mpi_comm
        self.sent_parameters_count = 0
        self.recieved_grads_count = 0
        self.summary_steps = summary_steps
        self.num_grads_to_drop = num_grads_to_drop
        self.gradient_flattener = ArrayFlattener(
            [tuple(x.shape) for x in self.network.parameters()] + [(1)])  # +1 for timestep
        self.variable_flattener = ArrayFlattener([tuple(x.shape) for x in self.network.parameters()])

        # vars for sending network buffers
        if torch.__version__=='1.0.0a0+16b8075':
            self.mpi_buffer_sender = MPIArraySend(mpi_comm, [tuple(x.shape) for x in 
                                                         self.network.buffers()])
        else:
            self.mpi_buffer_sender = MPIArraySend(mpi_comm, [tuple(x.shape) for x in 
                                                         self.network._all_buffers()])

        # workers will send name and shapes of gradients just check the order is the same
        for w_ind in range(1, mpi_comm.size):
            # Workers should send an ordered dict of {'name': shape, ...}
            send_sizes = self.comm.recv(source=w_ind, tag=MpiMessages.NAMES_AND_SHAPES)
            assert isinstance(send_sizes, OrderedDict)
            for v_ind, (worker_v_size, host_v) in enumerate(zip(send_sizes.values(), self.network.parameters())):
                assert worker_v_size == tuple(host_v.shape)

    @property
    def optimizer(self):
        return self._optimizer

    def run(self, max_steps=float('inf'), initial_step_count=0):
        # start saving thread
        self._saver_should_be_done = False
        saver_thread = Thread(target=self._saver_thread, args=())
        saver_thread.start()

        variable_buffer = np.empty(self.variable_flattener.total_size, np.float32)

        # setup gradient buffers
        size = self.comm.Get_size()
        grad_buffers = [np.empty(self.gradient_flattener.total_size, np.float32) for i in
                        range(self.comm.Get_size() - 1)]
        received_grads = []
        timesteps = [0] * (self.comm.Get_size() - 1)
        reqs = []
        for i in range(1, size):
            reqs.append(self.comm.Irecv(grad_buffers[i - 1], source=i, tag=MpiMessages.SEND))

        # vars to make sure python doesn't gc
        batch_acks = [None] * (size - 1)
        self.start_time = time.time()
        workers_to_drop = set()
        workers_in_this_batch = []
        num_grads_to_keep = (size - 1) - self.num_grads_to_drop

        # vars for the next step to save/req buffers at
        while self.global_step < max_steps:
            # get batch workers
            while len(workers_in_this_batch) < num_grads_to_keep:
                status = mpi.Status()
                mpi.Request.Waitany(reqs, status)
                worker = status.source

                # grab timestep and send ack / setup new recv
                timesteps[worker - 1] = grad_buffers[worker - 1][-1]  # just want a single value
                self.global_step = sum(timesteps)
                # upon recieving batch, send ack
                batch_acks[worker - 1] = self.comm.isend((self.global_step), dest=worker,
                                                         tag=MpiMessages.SEND_ACK)
                # setup new receive
                reqs[worker - 1] = self.comm.Irecv(grad_buffers[worker - 1], source=worker, tag=MpiMessages.SEND)

                # if worker was dropped last round
                if worker in workers_to_drop:
                    # remove from drop list
                    workers_to_drop.remove(worker)
                else:  # not dropped grads are good
                    # gradient process
                    gradients = self.gradient_flattener.unflatten(grad_buffers[worker - 1])
                    received_grads.append(gradients[:-1])
                    workers_in_this_batch.append(worker)

            # any workers not in this batch are dropped next round
            for i in range(1, size):
                if i not in workers_in_this_batch:
                    workers_to_drop.add(i)

            # batch ready
            combined_grads = self.combine_gradients(received_grads)
            self.write_gradient_summaries(combined_grads, self.global_step)
            self.apply_gradients(combined_grads)
            new_variables_flat = self.variable_flattener.flatten(self.get_parameters_numpy(),
                                                                 buffer=variable_buffer)
            self.comm.Ibcast(new_variables_flat, root=0)

            self.sent_parameters_count += 1
            received_grads = []
            workers_in_this_batch = []

            if self.sent_parameters_count % self.summary_steps == 0:
                self.logger.info('train_frames: {} avg_train_fps: {}'.format(
                    self.global_step,
                    (self.global_step - initial_step_count) / (time.time() - self.start_time)
                ))

        # cleanup
        self._saver_should_be_done = True
        print('Host sending stop')
        for i in range(1, size):
            self.comm.isend(True, dest=i, tag=MpiMessages.STOP)
        print('Host waiting on receiving stops')
        stops = [self.comm.irecv(source=i, tag=MpiMessages.STOPPED) for i in range(1, size)]
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
                            print('Waited 5 times for {}, skipping'.format(worker))
                            threads_stoped += 1
                        print('Still waiting on {} to finish'.format(worker))
                        time.sleep(0.1)

        print('Host sees all threads as stopped.')

    def _saver_thread(self):
        num_workers = self.comm.Get_size() - 1
        # setup for receiving buffers
        if torch.__version__=='1.0.0a0+16b8075':
            buffer_shapes = [tuple(x.shape) for x in self.network.buffers()]
        else:
            buffer_shapes = [tuple(x.shape) for x in self.network._all_buffers()]

        buffer_flattener = ArrayFlattener(buffer_shapes)
        next_save_step = self.save_interval
        try:
            while not self._saver_should_be_done:
                current_step = self.global_step
                if current_step > next_save_step:
                    buffer_params = [buffer_flattener.create_buffer() for i in range(num_workers)]
                    statuses = []
                    # request buffers params from all workers
                    recv_comms = []
                    for i in range(num_workers):
                        self.comm.isend(True, dest=i, tag=MpiMessages.BUFFER_REQUEST)
                        recv_comms.append(self.comm.Irecv(buffer_params[i], source=i,
                                                          tag=MpiMessages.BUFFER_REQUEST))
                        statuses.append(mpi.Status())

                    # wait for all workers to send buffers
                    mpi.Request.Waitall(recv_comms, statuses)

                    # all buffers are filled
                    unflattened_buffer_params = None
                    for i in range(num_workers):
                        if unflattened_buffer_params is None:
                            unflattened_buffer_params = buffer_flattener.unflatten(buffer_params[i])
                        else:
                            for all_bp, bp in zip(unflattened_buffer_params,
                                                  buffer_flattener.unflatten(buffer_params[i])):
                                all_bp += bp

                    # can't divide here since numpy reduces from array to float on tensors with shape ()
                    all_buffer_params = [x for x in unflattened_buffer_params]
                    # set buffers
                    if torch.__version__=='1.0.0a0+16b8075':
                        for b, all_bp in zip(self.network.buffers(), all_buffer_params):
                            # mean over all workers
                            b.copy_(torch.from_numpy(all_bp)).div_(num_workers)
                    else:
                        for b, all_bp in zip(self.network._all_buffers(), all_buffer_params):
                            # mean over all workers
                            b.copy_(torch.from_numpy(all_bp)).div_(num_workers)

                    # finally save
                    self.saver.save_state_dicts(self.network, int(current_step), optimizer=self.optimizer)
                    next_save_step += self.save_interval
                time.sleep(1)
        except Exception as e:
            print('Error saving', e)

        # final save
        self.saver.save_state_dicts(self.network, int(self.global_step), optimizer=self.optimizer)

    def combine_gradients(self, gradients_list):
        gradients = []
        for i in range(len(gradients_list[0])):
            sum_grad = np.zeros_like(gradients_list[0][i])
            num_grads = len(gradients_list)
            for worker_ind in range(num_grads):
                sum_grad += gradients_list[worker_ind][i]
            gradients.append(torch.from_numpy(sum_grad) / num_grads)
        return gradients

    def write_gradient_summaries(self, gradients, timestep=0):
        # TODO: write grad summaries
        pass

    def get_parameters_numpy(self):
        return [p.detach().cpu().numpy() for p in self.network.parameters()]

    def apply_gradients(self, gradients):
        self.optimizer.zero_grad()

        for p, g in zip(self.network.parameters(), gradients):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)  # overwrite values in parameter grad without creating a new memory copy

        self.optimizer.step()


class ToweredWorker(HasAgent, HasEnvironment, WritesSummaries, LogsAndSummarizesRewards, MPIProc):
    def __init__(
            self,
            agent,
            environment,
            nb_env,
            logger,
            summary_writer,
            summary_frequency,
            max_parameter_skip=0,
            gradient_warning_time=0.1,
            variable_warning_time=0.1
    ):
        self._agent = agent
        self._environment = environment
        self._nb_env = nb_env
        self._logger = logger
        self._summary_writer = summary_writer
        self._summary_frequency = summary_frequency

        send_shapes = OrderedDict()
        for i, x in enumerate(self.network.parameters()):
            send_shapes[str(i)] = tuple(x.shape)
        recv_shapes = [tuple(x.shape) for x in self.network.parameters()]
        self._mpi_comm = mpi.COMM_WORLD
        self.mpi_helper = MPIHelper(
            send_shapes, recv_shapes, 0, max_parameter_skip, gradient_warning_time, variable_warning_time
        )
        self.mpi_buffer_request = self._create_mpi_buffer_request()
        if torch.__version__=='1.0.0a0+16b8075':
            self.mpi_buffer_sender = MPIArraySend(self._mpi_comm, [tuple(x.shape) for x in
                                                                self.network.buffers()])
        else:
            self.mpi_buffer_sender = MPIArraySend(self._mpi_comm, [tuple(x.shape) for x in
                                                                self.network._all_buffers()])

        self.global_step = 0

    @property
    def agent(self):
        return self._agent

    @property
    def environment(self):
        return self._environment

    @property
    def summary_writer(self):
        return self._summary_writer

    @property
    def summary_frequency(self):
        return self._summary_frequency

    @property
    def logger(self):
        return self._logger

    @property
    def nb_env(self):
        return self._nb_env

    def run(self, initial_count=0):
        next_obs = self.environment.reset()
        self.start_time = time.time()
        while not self.should_stop():
            obs = next_obs
            # Build rollout
            actions = self.agent.act(obs)
            next_obs, rewards, terminals, infos = self.environment.step(actions)
            self.agent.observe(obs, rewards, terminals, infos)

            # Perform state updates
            terminal_rewards, terminal_infos = self.update_buffers(rewards, terminals, infos)
            self.log_episode_results(terminal_rewards, terminal_infos, self.local_step_count, initial_count)
            self.write_reward_summaries(terminal_rewards, self.global_step)

            # Learn
            if self.exp_cache.is_ready():
                self.learn(next_obs)
        self.close()

    def learn(self, next_obs):
        loss_dict, metric_dict = self.agent.compute_loss(self.exp_cache.read(), next_obs)
        total_loss = torch.sum(torch.stack(tuple(loss for loss in loss_dict.values())))

        self.clear_gradients()
        total_loss.backward()
        self.submit()
        self.receive()

        self.exp_cache.clear()
        self.agent.detach_internals()

        # write summaries
        self.write_summaries(total_loss, loss_dict, metric_dict, self.global_step)

    def submit(self):
        """
            Submits gradients to a MPI host
        """
        gradients = self._get_gradients()
        host_info = self.mpi_helper.send(gradients, self.local_step_count)
        if host_info is not None:
            self.global_step = host_info
        else:
            self.global_step = 0
        # host decides when it wants pytorch buffers
        if self.mpi_buffer_request.test()[0]:
            if torch.__version__=='1.0.0a0+16b8075':
                buffer_list = [x.cpu().numpy() for x in self.network.buffers()]
            else:
                buffer_list = [x.cpu().numpy() for x in self.network._all_buffers()]
                
            self.mpi_buffer_sender.Isend(buffer_list, dest=0, tag=MpiMessages.BUFFER_REQUEST)
            self.mpi_buffer_request = self._create_mpi_buffer_request()

    def _create_mpi_buffer_request(self):
        return self._mpi_comm.irecv(source=0, tag=MpiMessages.BUFFER_REQUEST)

    def receive(self):
        """
            Receives parameters from MPI host
        """
        new_params = self.mpi_helper.receive_parameters()
        # new_params can be none if not waiting
        if new_params is not None:
            self.set_parameters(new_params)

    def set_parameters(self, parameters):
        for p, v in zip(self.network.parameters(), parameters):
            p.data.copy_(v, non_blocking=True)

    def _get_gradients(self):
        return [p.grad for p in self.network.parameters()]

    def close(self):
        self.mpi_helper.close()

    def should_stop(self):
        return self.mpi_helper.should_stop()

    def clear_gradients(self):
        for p in self.network.parameters():
            if p.grad is not None:
                p.grad.zero_()
