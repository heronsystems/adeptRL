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
from .._base import InputNetwork


class DiscreteIdentity(InputNetwork):
    def __init__(self, nb_in_channel):
        super().__init__()
        self._nb_output_channel = nb_in_channel

    @classmethod
    def from_args(cls, nb_in_channel, args):
        return cls(nb_in_channel)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, x):
        return x
