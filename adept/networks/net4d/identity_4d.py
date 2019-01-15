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
from adept.networks.net4d.submodule_4d import SubModule4D


class Identity4D(SubModule4D):
    args = {}

    def __init__(self, input_shape, id):
        super().__init__(input_shape, id)

    @classmethod
    def from_args(cls, args, input_shape, id):
        return cls(input_shape, id)

    @property
    def _output_shape(self):
        return self.input_shape

    def _forward(self, input, internals, **kwargs):
        return input, {}

    def _new_internals(self):
        return {}
