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
from adept.container.base.mixin import LogsRewards


class ALAHost:
    pass


class ALAWorker(LogsRewards):
    """
    Actor Learner Architecture worker.
    """

    def __init__(
        self,
        args,
        global_rank,
        world_size,
        device
    ):
        self._agent = agent
        self._environment = environment
        self._nb_env = nb_env
        self._logger = logger

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

    def run(self, nb_step, initial_step_count=None):
        pass
