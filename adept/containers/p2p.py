from ._base import (
    HasAgent, WritesSummaries, LogsAndSummarizesRewards, MPIProc, HasEnvironment
)
from .mpi import MpiMessages, MPIArraySend, MPIArrayRecv, P2PBestProtocol
import torch
import time
from mpi4py import MPI as mpi


class P2PWorker(HasAgent, HasEnvironment, WritesSummaries, LogsAndSummarizesRewards, MPIProc):
    def __init__(
            self,
            agent,
            environment,
            make_optimizer,
            nb_env,
            logger,
            summary_writer,
            summary_frequency,
            shared_seed,
            synchronize_step_interval=None,
            share_optimizer_params=False
    ):
        self._agent = agent
        self._environment = environment
        self._optimizer = make_optimizer(self.network.parameters())
        self._nb_env = nb_env
        self._logger = logger
        self._summary_writer = summary_writer
        self._summary_frequency = summary_frequency
        self._synchronize_step_interval = synchronize_step_interval
        self._send_count = 0
        self._share_optimizer_params = share_optimizer_params
        self._mpi_comm = mpi.COMM_WORLD
        self._mpi_rank = mpi.COMM_WORLD.Get_rank()
        self._mpi_size = mpi.COMM_WORLD.Get_size()
        # These have to be created after the optimizer steps once so it's state exists
        self._mpi_send = None
        self._mpi_recv = None
        self.communication_protocol = P2PBestProtocol(mpi.COMM_WORLD, shared_seed)

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

    @property
    def optimizer(self):
        return self._optimizer

    def run(self, max_steps=float('inf'), initial_count=0):
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
            self.write_reward_summaries(terminal_rewards,
                                        self.local_step_count * self._mpi_size)  # an imperfect estimate of global step

            # Learn
            if self.exp_cache.is_ready():
                self.learn(next_obs)
        self.close()

    def learn(self, next_obs):
        loss_dict, metric_dict = self.agent.compute_loss(self.exp_cache.read(), next_obs)
        total_loss = torch.sum(torch.stack(tuple(loss for loss in loss_dict.values())))

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self._send_count += 1  # increment here since first step synchronization isn't needed
        if self._synchronize_step_interval is not None and self._send_count % self._synchronize_step_interval == 0:
            self.synchronize_parameters()
        else:
            self.submit()
            self.receive()

        self.exp_cache.clear()
        self.agent.detach_internals()

        # write summaries
        self.write_summaries(total_loss, loss_dict, metric_dict,
                             self.local_step_count * self._mpi_size)  # an imperfect estimate of global step

    def submit(self):
        """
            Sends parameters to a peer. Non-blocking
        """
        parameters = self._get_parameters()
        dest = self.communication_protocol.next_dest
        self.mpi_send.Isend(parameters, dest, MpiMessages.SEND)

    def receive(self):
        """
            Receives parameters from a peer. Blocking
        """
        source = self.communication_protocol.next_source
        # the tag we are listening for is a SEND from that node
        new_params = self.mpi_recv.Recv(source, MpiMessages.SEND)
        self.combine_parameters(new_params)

    def combine_parameters(self, parameters):
        if self._share_optimizer_params:
            params = parameters[0:len(parameters) // 2]
        else:
            params = parameters
        for p, v in zip(self.network.parameters(), params):
            p.data.add_(torch.from_numpy(v).to(self.agent.device))
            p.data.div_(2.0)

        if self._share_optimizer_params:
            optimizer_params = parameters[len(parameters) // 2:]
            for local_s, dist_s in zip(self._optimizer_state_list(), optimizer_params):
                local_s.data.add_(torch.from_numpy(dist_s).to(self.agent.device))
                local_s.data.div_(2.0)

    def _get_parameters(self, force_optimizer=False, force_buffers=False):
        params = [p.data.cpu().numpy() for p in self.network.parameters()]
        if self._share_optimizer_params or force_optimizer:
            params.extend([x.data.cpu().numpy() for x in self._optimizer_state_list()])
        if force_buffers:
            params.extend([x.data.cpu.numpy() for x in self.network._all_buffers()])
        return params

    def close(self):
        pass

    def mpi_shapes(self, force_optimizer=False, force_buffers=False):
        param_shapes = [tuple(x.shape) for x in self.network.parameters()]
        if self._share_optimizer_params or force_optimizer:
            param_shapes.extend([tuple(x.shape) for x in self._optimizer_state_list()])
        if force_buffers:
            param_shapes.extend([tuple(x.shape) for x in self.network._all_buffers()])
        return param_shapes

    def _optimizer_state_list(self):
        optimizer_states = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                for k in sorted(state.keys()):
                    if k != 'step':
                        optimizer_states.append(state[k])
        return optimizer_states

    @property
    def mpi_send(self):
        if self._mpi_send is None:
            self._mpi_send = MPIArraySend(mpi.COMM_WORLD, self.mpi_shapes())
        return self._mpi_send

    @property
    def mpi_recv(self):
        if self._mpi_recv is None:
            self._mpi_recv = MPIArrayRecv(mpi.COMM_WORLD, self.mpi_shapes())
        return self._mpi_recv

    def synchronize_parameters(self):
        # use send to get a buffer
        send_param_buffer = self.mpi_send.flattener.flatten(self._get_parameters())
        recv_param_buffer = self.mpi_send.flattener.create_buffer()
        self._mpi_comm.Allreduce(send_param_buffer, recv_param_buffer, op=mpi.SUM)
        parameters = self.mpi_send.flattener.unflatten(recv_param_buffer)

        # sync
        if self._share_optimizer_params:
            params = parameters[0:len(parameters) // 2]
        else:
            params = parameters
        for p, v in zip(self.network.parameters(), params):
            # allreduce sums the params, divide them by size to mean
            p.data.copy_(torch.from_numpy(v).to(self.agent.device))
            p.data.div_(self._mpi_size)

        if self._share_optimizer_params:
            optimizer_params = parameters[len(parameters) // 2:]
            for local_s, dist_s in zip(self._optimizer_state_list(), optimizer_params):
                local_s.data.copy_(torch.from_numpy(dist_s).to(self.agent.device))
                local_s.data.div_(self._mpi_size)
