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
import torch

from adept.expcaches.rollout import RolloutCache
from adept.agents.dqn import OnlineDQN, DQN


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class OnlineQRDQN(OnlineDQN):
    args = {**OnlineDQN.args, **{'num_atoms': 51}}

    def __init__(
        self,
        network,
        device,
        reward_normalizer,
        gpu_preprocessor,
        engine,
        action_space,
        nb_env,
        nb_rollout,
        discount,
        target_copy_steps,
        double_dqn,
        num_atoms
    ):
        super(OnlineQRDQN, self).__init__(
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            engine,
            action_space,
            nb_env,
            nb_rollout,
            discount,
            target_copy_steps,
            double_dqn
        )
        self.num_atoms = num_atoms
        self._qr_density = (((2 * torch.arange(self.num_atoms, dtype=torch.float, requires_grad=False)) + 1) / (2.0 * self.num_atoms)).to(self.device)

    @classmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor, engine,
        action_space, nb_env=None
    ):
        if nb_env is None:
            nb_env = args.nb_env

        # if running in distrib mode, divide by number of processes
        denom = 1
        if hasattr(args, 'nb_proc') and args.nb_proc is not None:
            denom = args.nb_proc

        return cls(
            network, device, reward_normalizer, gpu_preprocessor, engine,
            action_space,
            nb_env=nb_env,
            nb_rollout=args.nb_rollout,
            discount=args.discount,
            target_copy_steps=args.target_copy_steps / denom,
            double_dqn=args.double_dqn,
            num_atoms=args.num_atoms
        )

    def _get_rollout_values(self, q_vals, action, batch_size):
        action_select = action.unsqueeze(1).expand(batch_size, 1, self.num_atoms)
        return q_vals.gather(1, action_select).squeeze(1)

    ### Start shared functions
    @staticmethod
    def output_space(action_space, args):
        head_dict = {}
        for k, shape in action_space.items():
            head_dict[k] = (shape[0] * args.num_atoms, )
        return head_dict

    def _action_from_q_vals(self, q_vals):
        return q_vals.mean(2).argmax(dim=-1, keepdim=True)

    def _compute_estimated_values(self, next_obs):
        # TODO make this general in basedqn
        # estimate value of next state
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            results, _ = self._target_net(next_obs_on_device, self.internals)
            target_q = self._get_qvals_from_pred(results)

            last_values = []
            for k in self._action_keys:
                # if double dqn estimate get target val for current estimated action
                if self.double_dqn:
                    current_results, _ = self.network(next_obs_on_device, self.internals)
                    current_q = self._get_qvals_from_pred(current_results)
                    action_select = current_q[k].mean(2).argmax(dim=-1, keepdim=True)
                    action_select = action_select.unsqueeze(-1).expand(-1, 1, self.num_atoms)
                else:
                    action_select = target_q[k].mean(2).argmax(dim=-1, keepdim=True)
                    action_select = action_select.unsqueeze(-1).expand(-1, 1, self.num_atoms)
                last_values.append(target_q[k].gather(1, action_select).squeeze(1))

            last_values = torch.cat(last_values, dim=1)
        return last_values

    def _loss_fn(self, batch_values, value_targets):
        # batched quantile huber loss
        diff = value_targets - batch_values
        dist_mask = torch.abs(self._qr_density - (diff.detach() < 0).float())
        # sum over atoms, mean over everything else
        return (huber(diff) * dist_mask).sum(-1).mean()

    def _get_qvals_from_pred(self, predictions):
        pred = {}
        for k, v in predictions.items():
            pred[k] = v.view(v.shape[0], -1, self.num_atoms)
        return pred


### Same as online QRDQN
class QRDQN(DQN):
    args = {**DQN.args, **{'num_atoms': 51}}

    def __init__(
        self,
        network,
        device,
        reward_normalizer,
        gpu_preprocessor,
        engine,
        action_space,
        nb_env,
        nb_rollout,
        discount,
        target_copy_steps,
        double_dqn,
        exp_size,
        exp_min_size,
        num_atoms
    ):
        super(QRDQN, self).__init__(
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            engine,
            action_space,
            nb_env,
            nb_rollout,
            discount,
            target_copy_steps,
            double_dqn,
            exp_size,
            exp_min_size
        )
        self.num_atoms = num_atoms
        self._qr_density = (((2 * torch.arange(self.num_atoms, dtype=torch.float, requires_grad=False)) + 1) / (2.0 * self.num_atoms)).to(self.device)

    @classmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor, engine,
        action_space, nb_env=None
    ):
        if nb_env is None:
            nb_env = args.nb_env

        # if running in distrib mode, divide by number of processes
        denom = 1
        if hasattr(args, 'nb_proc') and args.nb_proc is not None:
            denom = args.nb_proc

        return cls(
            network, device, reward_normalizer, gpu_preprocessor, engine,
            action_space,
            nb_env=nb_env,
            nb_rollout=args.nb_rollout,
            discount=args.discount,
            target_copy_steps=args.target_copy_steps / denom,
            double_dqn=args.double_dqn,
            exp_size=args.exp_size / denom,
            exp_min_size=args.exp_min_size / denom,
            num_atoms=args.num_atoms
        )

    ### Start shared functions
    @staticmethod
    def output_space(action_space, args):
        head_dict = {}
        for k, shape in action_space.items():
            head_dict[k] = (shape[0] * args.num_atoms, )
        return head_dict

    def _action_from_q_vals(self, q_vals):
        return q_vals.mean(2).argmax(dim=-1, keepdim=True)

    def _compute_estimated_values(self, next_obs):
        # TODO make this general in basedqn
        # estimate value of next state
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            results, _ = self._target_net(next_obs_on_device, self.internals)
            target_q = self._get_qvals_from_pred(results)

            last_values = []
            for k in self._action_keys:
                # if double dqn estimate get target val for current estimated action
                if self.double_dqn:
                    current_results, _ = self.network(next_obs_on_device, self.internals)
                    current_q = self._get_qvals_from_pred(current_results)
                    action_select = current_q[k].mean(2).argmax(dim=-1, keepdim=True)
                    action_select = action_select.unsqueeze(-1).expand(-1, 1, self.num_atoms)
                else:
                    action_select = target_q[k].mean(2).argmax(dim=-1, keepdim=True)
                    action_select = action_select.unsqueeze(-1).expand(-1, 1, self.num_atoms)
                last_values.append(target_q[k].gather(1, action_select).squeeze(1))

            last_values = torch.cat(last_values, dim=1)
        return last_values

    def _loss_fn(self, batch_values, value_targets):
        # batched quantile huber loss
        diff = value_targets - batch_values
        dist_mask = torch.abs(self._qr_density - (diff.detach() < 0).float())
        # sum over atoms, mean over everything else
        return (huber(diff) * dist_mask).sum(-1).mean()

    def _get_qvals_from_pred(self, predictions):
        pred = {}
        for k, v in predictions.items():
            pred[k] = v.view(v.shape[0], -1, self.num_atoms)
        return pred

