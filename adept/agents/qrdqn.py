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
from collections import OrderedDict
from copy import deepcopy

import torch
from torch.nn import functional as F

from adept.expcaches.rollout import RolloutCache
from adept.agents.agent_module import AgentModule


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class QRDQN(AgentModule):
    args = {
        'nb_rollout': 20,
        'discount': 0.99,
        'egreedy_final': 0.1,
        'egreedy_steps': 1000000,
        'target_copy_steps': 10000,
        'double_dqn': True,
        'num_atoms': 51,
    }

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
        egreedy_final,
        egreedy_steps,
        target_copy_steps,
        double_dqn,
        num_atoms
    ):
        super(QRDQN, self).__init__(
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            engine,
            action_space,
            nb_env
        )
        self.discount, self.egreedy_steps, self.egreedy_final = discount, egreedy_steps / nb_env, self.egreedy_final
        self.double_dqn = double_dqn
        self.num_atoms = num_atoms
        self.target_copy_steps = target_copy_steps / nb_env
        self._next_target_copy = self.target_copy_steps
        self._target_net = deepcopy(network)
        self._target_net.eval()
        self._act_count = 0
        self._qr_density = (((2 * torch.arange(self.num_atoms, dtype=torch.float)) + 1) / (2.0 * self.num_atoms)).to(self.device)

        self._exp_cache = RolloutCache(
            nb_rollout, device, reward_normalizer,
            ['values']
        )
        self._action_keys = list(sorted(action_space.keys()))

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
            egreedy_final=args.egreedy_final,
            egreedy_steps=args.egreedy_steps / denom,
            target_copy_steps=args.target_copy_steps / denom,
            double_dqn=args.double_dqn,
            num_atoms=args.num_atoms
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @staticmethod
    def output_space(action_space, args):
        head_dict = {}
        for k, shape in action_space.items():
            head_dict[k] = (shape[0] * args.num_atoms, )
        return head_dict

    def act(self, obs):
        self.network.train()
        self._act_count += 1
        return self._act_gym(obs)

    def _act_gym(self, obs):
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals
        )
        q_vals = self._get_qvals_from_pred(predictions)
        batch_size = predictions[self._action_keys[0]].shape[0]

        # reduce feature dim, build action_key dim
        actions = OrderedDict()
        values = []
        # TODO support multi-dimensional action spaces?
        for key in self._action_keys:
            # possible sample
            if self._act_count < self.egreedy_steps:
                epsilon = 1 - ((1-self.egreedy_final) / self.egreedy_steps) * self._act_count
            else:
                epsilon = self.egreedy_final

            # random action across some environments
            rand_mask = (epsilon > torch.rand(batch_size)).nonzero().squeeze(-1)
            action = q_vals[key].argmax(dim=-1, keepdim=True)
            rand_act = torch.randint(self.action_space[key][0], (rand_mask.shape[0], 1), dtype=torch.long).to(self.device)
            action[rand_mask] = rand_act

            actions[key] = action.squeeze(1).cpu().numpy()
            action_select = action.unsqueeze(-1).expand(batch_size, 1, self.num_atoms)
            values.append(q_vals[key].gather(1, action_select).squeeze(1))

        values = torch.cat(values, dim=1)

        self.exp_cache.write_forward(values=values)
        self.internals = internals
        return actions

    def act_eval(self, obs):
        self.network.eval()
        return self._act_eval_gym(obs)

    def _act_eval_gym(self, obs):
        raise NotImplementedError()
        with torch.no_grad():
            predictions, internals = self.network(
                self.gpu_preprocessor(obs, self.device), self.internals
            )

            # reduce feature dim, build action_key dim
            actions = OrderedDict()
            for key in self._action_keys:
                logit = predictions[key]
                prob = F.softmax(logit, dim=1)
                action = torch.argmax(prob, 1)
                actions[key] = action.cpu().numpy()

        self.internals = internals
        return actions

    def compute_loss(self, rollouts, next_obs):
        # copy target network
        if self._act_count > self._next_target_copy:
            self._target_net = deepcopy(self.network)
            self._target_net.eval()
            self._next_target_copy += self.target_copy_steps

        # estimate value of next state
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            results, _ = self._target_net(next_obs_on_device, self.internals)
            target_q = self._get_qvals_from_pred(results)

        last_values = []
        # if double dqn estimate get target val for current estimated action
        for k in self._action_keys:
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

        # compute nstep return and advantage over batch
        batch_values = torch.stack(rollouts.values)
        value_targets = self._compute_returns_advantages(last_values, rollouts.rewards, rollouts.terminals)

        diff = value_targets - batch_values
        dist_mask = torch.abs(self._qr_density - (diff.detach() < 0).float())

        # batched quantile huber loss
        # value_loss = torch.nn.functional.smooth_l1_loss(batch_values, value_targets)
        value_loss = (huber(diff) * dist_mask).mean()

        losses = {'value_loss': value_loss}
        metrics = {}
        return losses, metrics

    def _compute_returns_advantages(self, estimated_value, rewards, terminals):
        next_value = estimated_value
        # First step of nstep reward target is estimated value of t+1
        target_return = estimated_value
        nstep_target_returns = []
        for i in reversed(range(len(rewards))):
            # unsqueeze over action dim so it isn't broadcasted
            reward = rewards[i].unsqueeze(-1)
            terminal = terminals[i].unsqueeze(-1)

             # Nstep return is always calculated for the critic's target
            target_return = reward + self.discount * target_return * terminal
            nstep_target_returns.append(target_return)

        # reverse lists
        nstep_target_returns = torch.stack(list(reversed(nstep_target_returns))).data
        return nstep_target_returns

    def _get_qvals_from_pred(self, predictions):
        pred = {}
        for k, v in predictions.items():
            pred[k] = v.view(v.shape[0], -1, self.num_atoms)
        return pred

