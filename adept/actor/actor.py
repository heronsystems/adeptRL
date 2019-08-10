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
import abc

from adept.utils import listd_to_dlist
from adept.utils.requires_args import RequiresArgs

class ActorModule(RequiresArgs, metaclass=abc.ABCMeta):
    """
    An actor observes the environment and takes actions. It also extra info
    necessary for computing losses.
    """

    def __init__(
            self,
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            policy,
            nb_env
    ):
