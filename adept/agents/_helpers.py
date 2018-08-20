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
import torch


def obs_to_device(obs, device):
    """
    :param obs: An OrderedDict observation batch with keys mapping to different components of the observation.
    :param device: torch.device to move the tensors to
    :return:
    """
    batch = []
    for channels in zip(*obs.values()):
        tensor = torch.cat([channel.to(device).float() for channel in channels if isinstance(channel, torch.Tensor)])
        batch.append(tensor)
    return torch.stack(batch)
