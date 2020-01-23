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

class NCCLOptimizer:
    def __init__(self, network, optimizer, world_size, param_sync_rate=1000):
        self.network = network
        self.optimizer = optimizer
        self.param_sync_rate = param_sync_rate
        self._opt_count = 0
        self.process_group = None
        self.world_size = world_size

    def set_process_group(self, pg):
        self.process_group = pg

    def step(self):
        handles = []
        for param in self.network.parameters():
            handles.append(
            self.process_group.allreduce(param.grad))
        for handle in handles:
            handle.wait()
        for param in self.network.parameters():
            param.grad.mul_(1. / self.world_size)
        self.optimizer.step()
        self._opt_count += 1

        # sync params every once in a while to reduce numerical errors
        if self._opt_count % self.param_sync_rate == 0:
            self.sync_parameters()
            # can't just sync buffers, some are int and don't mean well
            # self.sync_buffers()

    def sync_parameters(self):
        for param in self.network.parameters():
            self.process_group.allreduce(param.data)
            param.data.mul_(1. / self.world_size)

    def sync_buffers(self):
        for b in self.network.buffers():
            self.process_group.allreduce(b.data)
            b.data.mul_(1. / self.world_size)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, d):
        return self.optimizer.load_state_dict(d)

