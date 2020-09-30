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
from copy import copy, deepcopy
from adept.preprocess.base import MultiOperation, SimpleOperation


class _Preprocessor:
    def __init__(self, ops, observation_space, observation_dtypes=None):
        """
        Parameters
        ----------
        ops : list[gamebreaker.preprocess.Operation]
        observation_space : dict[str, Shape]
        observation_dtypes : dict[str, dtype]
        """
        cur_space = deepcopy(observation_space)
        cur_dtypes = deepcopy(observation_dtypes)

        self.ops = ops
        self.observation_space, self.observation_dtypes = self._update(
            cur_space, cur_dtypes
        )

    def _update(self, cur_space, cur_dtypes):
        cur_space = copy(cur_space)
        cur_dtypes = copy(cur_dtypes)
        for op in self.ops:
            if isinstance(op, SimpleOperation):
                output_shape = op.update_shape(
                    cur_space[op.input_field]
                )
                if output_shape:
                    cur_space[op.output_field] = output_shape
                else:
                    del cur_space[op.output_field]
                if cur_dtypes:
                    output_dtype = op.update_dtype(
                        cur_dtypes[op.input_field]
                    )
                    if output_dtype:
                        cur_dtypes[op.output_field] = output_dtype
                    else:
                        del cur_dtypes[op.output_field]
            elif isinstance(op, MultiOperation):
                input_shapes = [cur_space[k] for k in op.input_fields]
                input_dtypes = [cur_dtypes[k] for k in op.input_fields]
                output_shapes = op.update_shape(input_shapes)
                output_dtypes = op.update_dtype(input_dtypes)
                for k, shape, dtype in zip(
                    op.output_fields, output_shapes, output_dtypes
                ):
                    if shape:
                        cur_space[k] = shape
                    else:
                        del cur_space[k]
                    if cur_dtypes:
                        if dtype:
                            cur_dtypes[k] = dtype
                        else:
                            del cur_dtypes[k]
        return cur_space, cur_dtypes


class CPUPreprocessor(_Preprocessor):
    def __call__(self, obs):
        obs = copy(obs)
        for op in self.ops:
            if isinstance(op, SimpleOperation):
                output_tensor = op.preprocess_cpu(obs[op.input_field])
                if output_tensor is not None:
                    obs[op.output_field] = output_tensor
                else:
                    del obs[op.output_field]
            elif isinstance(op, MultiOperation):
                input_tensors = [obs[k] for k in op.input_fields]
                output_tensors = op.preprocess_cpu(input_tensors)
                for k, tensor in zip(op.output_fields, output_tensors):
                    if tensor is not None:
                        obs[k] = tensor
                    else:
                        del obs[k]
        return obs

    def reset(self):
        for o in self.ops:
            o.reset()


class GPUPreprocessor(_Preprocessor):
    def __call__(self, obs):
        obs = copy(obs)
        for op in self.ops:
            if isinstance(op, SimpleOperation):
                output_tensor = op.preprocess_gpu(obs[op.input_field])
                if output_tensor is not None:
                    obs[op.output_field] = output_tensor
                else:
                    del obs[op.output_field]
            elif isinstance(op, MultiOperation):
                input_tensors = [obs[k] for k in op.input_fields]
                output_tensors = op.preprocess_gpu(input_tensors)
                for k, tensor in zip(op.output_fields, output_tensors):
                    if tensor is not None:
                        obs[k] = tensor
                    else:
                        del obs[k]
        return obs

    def to(self, device):
        self.ops = [op.to(device) for op in self.ops]
        return self
