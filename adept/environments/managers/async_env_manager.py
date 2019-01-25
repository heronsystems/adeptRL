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
import torch

from adept.environments.managers._manager import EnvManager
from adept.utils import listd_to_dlist
from concurrent.futures import ProcessPoolExecutor


class AsyncEnvManager(EnvManager):
    """
    Makes asynchronous calls to environments.
    """

    def __init__(self, env_fns, engine):
        super(AsyncEnvManager, self).__init__(env_fns, engine)
        self.envs = [fn() for fn in env_fns]
        self._executor = ProcessPoolExecutor()
        