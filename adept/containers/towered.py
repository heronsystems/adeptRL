from ._base import (
    HasAgent, WritesSummaries, LogsAndSummarizesRewards, AppliesGrads, MPIProc, HasEnvironment
)
from .mpi import MPIHelper, MpiMessages, ArrayFlattener
import numpy as np
import torch
import time
from mpi4py import MPI as mpi
from collections import OrderedDict


class ToweredHost(AppliesGrads):
    def __init__(self, mpi_comm, num_grads_before_combine, network, make_optimizer, logger, summary_steps=100):
        self._optimizer = make_optimizer(network.parameters())

        self.network = network
        self.logger = logger
        self.global_step = 0
        self.start_time = None
        self.comm = mpi_comm
        self.sent_parameters_count = 0
        self.recieved_grads_count = 0
        self.summary_steps = summary_steps
        self.num_grads_before_combine = num_grads_before_combine
        self.gradient_flattener = ArrayFlattener([tuple(x.shape) for x in self.network.parameters()] + [(1)])  # +1 for timestep
        self.variable_flattener = ArrayFlattener([tuple(x.shape) for x in self.network.parameters()])

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
        variable_buffer = np.empty(self.variable_flattener.total_size, np.float32)

        # setup gradient buffers
        size = self.comm.Get_size()
        grad_buffers = [np.empty(self.gradient_flattener.total_size, np.float32) for i in range(self.comm.Get_size() - 1)]
        received_grads = []
        timesteps = [0] * (self.comm.Get_size() - 1)
        reqs = []
        for i in range(1, size):
            reqs.append(self.comm.Irecv(grad_buffers[i - 1], source=i, tag=MpiMessages.SEND))

        # vars to make sure python doesn't gc
        batch_acks = [None] * (size - 1)
        self.start_time = time.time()
        while self.global_step < max_steps:
            status = mpi.Status()
            mpi.Request.Waitany(reqs, status)
            worker = status.source

            # gradient process
            gradients = self.gradient_flattener.unflatten(grad_buffers[worker - 1])
            received_grads.append(gradients[:-1])
            timesteps[worker - 1] = gradients[-1][0]  # just want a single value
            self.global_step = sum(timesteps)

            # upon recieving batch, send ack
            batch_acks[worker - 1] = self.comm.isend(self.global_step, dest=worker, tag=MpiMessages.SEND_ACK)

            if len(received_grads) >= self.num_grads_before_combine:
                combined_grads = self.combine_gradients(received_grads)
                self.write_gradient_summaries(combined_grads, self.global_step)
                self.apply_gradients(combined_grads)
                new_variables_flat = self.variable_flattener.flatten(self.get_parameters_numpy(), buffer=variable_buffer)
                self.comm.Ibcast(new_variables_flat, root=0)
                self.sent_parameters_count += 1
                received_grads = []

                if self.sent_parameters_count % self.summary_steps == 0:
                    self.logger.info('train_frames: {} avg_train_fps: {}'.format(
                        self.global_step,
                        (self.global_step - initial_step_count) / (time.time() - self.start_time)
                    ))

            # setup new receive
            reqs[worker - 1] = self.comm.Irecv(grad_buffers[worker - 1], source=worker, tag=MpiMessages.SEND)

        # cleanup
        print('Host sending stop')
        for i in range(1, size):
            self.comm.isend(True, dest=i, tag=MpiMessages.STOP)
        print('Host waiting on recieving stops')
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
        self.mpi_helper = MPIHelper(
            send_shapes, recv_shapes, 0, max_parameter_skip, gradient_warning_time, variable_warning_time
        )
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
        self.global_step = self.mpi_helper.send(gradients, self.local_step_count)

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
