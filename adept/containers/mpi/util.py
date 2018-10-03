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
import time
import numpy as np
from enum import IntEnum


class MpiMessages(IntEnum):
    STOP = 1
    STOPPED = 2
    NAMES_AND_SHAPES = 3
    SEND = 20
    SEND_ACK = 25


class ArrayFlattener:
    def __init__(self, shapes, src_dtype=np.float32, dest_dtype=np.float32):
        self.shapes = shapes
        self.sizes = [int(np.prod(x))for x in shapes]
        self.total_size = np.sum([int(np.prod(x)) for x in self.shapes])
        self.src_dtype = src_dtype
        self.dest_dtype = dest_dtype

    def flatten(self, values, buffer=None):
        if buffer is None:
            buffer = self.create_buffer()
        curr_start = 0
        for ind, s in enumerate(self.shapes):
            val = values[ind]
            # reshape into buffer
            buffer[curr_start:curr_start+self.sizes[ind]] = val.ravel()
            curr_start += self.sizes[ind]
        return buffer

    def unflatten(self, value):
        values = []
        curr_start = 0
        for ind, (shape, size) in enumerate(zip(self.shapes, self.sizes)):
            val = np.empty(size, dtype=self.src_dtype)
            val[0:] = value[curr_start:curr_start+size]
            values.append(val.reshape(shape))
            curr_start += size
        return values

    def create_buffer(self):
        return np.empty(self.total_size, dtype=self.dest_dtype)


class MPIArraySend:
    def __init__(self, mpi_comm, shapes, dtype=np.float32):
        self.comm = mpi_comm
        self.dtype = dtype
        self.flattener = ArrayFlattener(shapes, dest_dtype=dtype)
        self._current_msg = None
        self._send_buffer = self.flattener.create_buffer()

    def Send(self, list_of_arrays, dest, tag):
        flattened = self.flattener.flatten(list_of_arrays)
        self.comm.Send(flattened, dest=dest, tag=tag)

    def Isend(self, list_of_arrays, dest, tag):
        self._send_buffer = self.flattener.flatten(list_of_arrays, buffer=self._send_buffer)
        self._current_msg = self.comm.Isend(self._send_buffer, dest=dest, tag=tag)

    def Wait(self):
        if self._current_msg is None:
            raise AttributeError("Not currently waiting on an Isend")
        else:
            self._current_msg.Wait()
            self._current_msg = None


class MPIArrayRecv:
    def __init__(self, mpi_comm, shapes, dtype=np.float32):
        self.comm = mpi_comm
        self.dtype = dtype
        self.flattener = ArrayFlattener(shapes, src_dtype=dtype)
        self._current_msg = None
        self._recv_buffer = self.flattener.create_buffer()

    def Recv(self, source, tag):
        self.comm.Recv(self._recv_buffer, source=source, tag=tag)
        return self.flattener.unflatten(self._recv_buffer)

    def Irecv(self, source, tag):
        self._current_msg = self.comm.Irecv(self._recv_buffer, source=source, tag=tag)

    def Wait(self):
        if self._current_msg is None:
            raise AttributeError("Not currently waiting on an Irecv")
        else:
            self._current_msg.Wait()
            self._current_msg = None
            return self.flattener.unflatten(self._recv_buffer)


class VariableRecieverSingle:
    def __init__(self, comm, variable_buffer, max_fail=5, root=0, warning_time=0.1, waiting_recv_warning=5):
        self.mpi_comm = comm
        self.variable_buffer = variable_buffer
        self.root = root
        self.comms = comm.Ibcast(self.variable_buffer, root=self.root)
        self.max_fail = max_fail
        self.curr_fail = 0
        self.warning_time = warning_time
        self.waiting_recv_warning = waiting_recv_warning

    def recieve(self):
        wait = False
        if self.curr_fail >= self.max_fail:
            wait = True

        # get the latest variables for all
        st = time.time()
        got_something = self.get_latest(self.variable_buffer, wait)
        et = time.time()
        if et - st > self.warning_time:
            print('WARNING {}: Long wait for variable updates. Waited {}'.format(
                self.mpi_comm.Get_rank(),
                et - st
            ))
        if not got_something:
            self.curr_fail += 1
            return None
        self.curr_fail = 0
        return self.variable_buffer

    def get_latest(self, v, wait=False):
        if not wait:
            done = self.comms.Test()
            if not done:
                return False
        else:
            self.comms.Wait()

        # so got something loop through queue
        number_of_recvs = 0
        while True:
            self.comms = self.mpi_comm.Ibcast(v, root=self.root)
            has_data = self.comms.Test()
            if not has_data:
                if number_of_recvs >= self.waiting_recv_warning:
                    print('WARNING {}: Worker had {} variable updates waiting. Worker is slow.'.format(
                        self.mpi_comm.Get_rank(),
                        number_of_recvs
                    ))

                return True  # we did get something
            else:
                number_of_recvs += 1
