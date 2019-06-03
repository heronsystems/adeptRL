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

from adept.expcaches.replay import PrioritizedExperienceReplay
from adept.agents.dqn import BaseDQN
from adept.utils import listd_to_dlist

class DQN(BaseDQN):
    exp_args = {
        'exp_size': 1000000,
        'exp_min_size': 1000,
    }
    args = {**BaseDQN.args, **exp_args}
    args['exp_update_rate'] = args['nb_rollout']

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
        exp_update_rate
    ):
        super(DQN, self).__init__(
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

        self.exp_size = int(exp_size / nb_env)
        self.exp_min_size = int(exp_min_size / nb_env)
        self.exp_update_rate = exp_update_rate
        self._exp_cache = PrioritizedExperienceReplay(
            0.6, self.exp_size, self.exp_min_size, nb_rollout, self.exp_update_rate, reward_normalizer, ['actions', 'internals']
        )

    @classmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor, engine,
        action_space, nb_env=None
    ):
        # Adding experience replay size and min size
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
            exp_update_rate=args.exp_update_rate
        )

    def act(self, obs):
        self._prepare_act()
        # exp replay doesn't save grads during forward
        with torch.no_grad():
            return self._act_gym(obs)

    def _get_rollout_values(self, *args, **kwargs):
        return None

    def _write_exp_cache(self, values, actions):
        detached_internals = {k: [i.detach() for i in v] for k, v in self.internals.items()}
        self.exp_cache.write_forward(internals=detached_internals, actions=actions)

    def compute_loss(self, rollouts, _):
        next_obs = rollouts.next_obs
        # rollout actions, terminals, rewards are lists/arrays convert to torch tensors
        rollout_actions = [{k: torch.from_numpy(v).to(self.device) for k, v in x.items()} for x in rollouts.actions]
        rewards = torch.stack(rollouts.rewards).to(self.device)
        terminals_mask = torch.stack(rollouts.terminals)  # keep on cpu
        terminals_mask_gpu = terminals_mask.to(self.device)
        importance_sample_weights = torch.from_numpy(rollouts.importance_sample_weights).float().to(self.device)
        # only need first internals
        rollout_internals = rollouts.internals[0]
        # obs are a list of dict of lists convert to dict of torch tensors
        torch_obs_dict = {k: torch.stack(v) for k, v in listd_to_dlist(rollouts.states).items()}

        self._possible_update_target()

        # recompute forward pass to get value estimates for states
        batch_values, internals = self._batch_forward(torch_obs_dict, rollout_actions, rollout_internals, terminals_mask)

        # estimate value of next state
        last_values = self._compute_estimated_values(next_obs, internals)

        # compute nstep return and advantage over batch
        value_targets = self._compute_returns_advantages(last_values, rewards, terminals_mask_gpu)

        # batched loss
        value_loss = self._loss_fn(batch_values, value_targets).squeeze(-1)

        # update experience cache td error, mean over envs
        # TODO: separate TD prioritiziation vs huber loss
        self.exp_cache.update_priorities(value_loss.mean(-1).detach().cpu())

        # weighted by sample broadcast over envs
        value_loss *= importance_sample_weights.unsqueeze(-1).expand_as(value_loss)

        losses = {'value_loss': value_loss.mean()}
        metrics = {}
        return losses, metrics

