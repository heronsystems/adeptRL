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
import time
from mpi4py import MPI as mpi
import torch
from .util import ArrayFlattener, VariableRecieverSingle, MpiMessages
from collections import OrderedDict


class MPIHelper:
    def __init__(
        self,
        send_names_shapes,
        recv_shapes,
        host_rank,
        max_recv_skip,
        send_warning_time=0.1,
        recv_warning_time=0.1,
        waiting_recv_warning=5
    ):
        self.send_names_shapes = send_names_shapes
        self.recv_shapes = recv_shapes
        self.host_rank = host_rank
        self.max_recv_skip = max_recv_skip
        self.mpi_comm = mpi.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()
        self.mpi_send_ack = None
        self.send_warning_time = send_warning_time
        self.recv_warning_time = recv_warning_time
        self.waiting_recv_warning = waiting_recv_warning

        # talk to host and send sizes and names
        assert isinstance(send_names_shapes, OrderedDict)
        self.mpi_comm.send(
            send_names_shapes,
            dest=self.host_rank,
            tag=MpiMessages.NAMES_AND_SHAPES
        )

        # create a flattener
        send_shapes = [v for k, v in send_names_shapes.items()]
        self.send_flattener = ArrayFlattener(
            send_shapes + [(1)]
        )  # +1 for timestep
        # have to use a class variable buffer otherwise python gc deletes the var
        self.send_buffer = np.empty(self.send_flattener.total_size, np.float32)
        self.variable_flattener = ArrayFlattener(recv_shapes)
        self.variable_receiver = VariableRecieverSingle(
            self.mpi_comm,
            np.empty(self.variable_flattener.total_size, np.float32),
            max_recv_skip,
            host_rank,
            warning_time=recv_warning_time,
            waiting_recv_warning=waiting_recv_warning
        )
        self.stop_command = self.mpi_comm.irecv(
            8, source=0, tag=MpiMessages.STOP
        )

    def send(self, list_of_tensors, timestep=0):
        host_info = None
        # wait for ack of last send
        # pull the tensors before waiting
        sends = [x.detach().cpu().numpy()
                 for x in list_of_tensors] + [np.asarray(timestep)]
        if self.mpi_send_ack is not None:
            st = time.time()
            host_info = self.mpi_send_ack.wait(
            )  # commonly a tuple with (global_step, others...)
            et = time.time()
            if et - st > self.send_warning_time:
                print(
                    '{} had to wait {} seconds for host to accept send'.format(
                        self.mpi_rank, et - st
                    )
                )

        self.send_flattener.flatten(sends, buffer=self.send_buffer)
        # send
        self.mpi_comm.Isend(
            self.send_buffer, dest=self.host_rank, tag=MpiMessages.SEND
        )

        # add a ack recieve to the queue
        self.mpi_send_ack = self.mpi_comm.irecv(
            source=self.host_rank, tag=MpiMessages.SEND_ACK
        )
        return host_info

    def receive_parameters(self):
        ret = self.variable_receiver.recieve()
        if ret is not None:
            return [
                torch.from_numpy(x)
                for x in self.variable_flattener.unflatten(ret)
            ]
        else:
            return None

    def close(self):
        self.mpi_send_ack.Cancel()
        self.mpi_comm.send(True, 0, MpiMessages.STOPPED)

    def should_stop(self):
        stop = self.stop_command.test()[0]
        return stop
