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
from adept.actor import ACPPOActorTrain
from adept.exp import Rollout
from adept.learner import ACRolloutLearner
from .base.agent_module import AgentModule


class ActorCritic(AgentModule):
    args = {
        **Rollout.args,
        **ACPPOActorTrain.args,
        **ACRolloutLearner.args
    }

    def __init__(
            self,
            reward_normalizer,
            action_space,
            spec_builder,
            rollout_len,
            discount,
            normalize_advantage,
            entropy_weight,
            return_scale
    ):
        super().__init__(
            reward_normalizer,
            action_space
        )
        self.discount = discount
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight

        self._exp_cache = Rollout(spec_builder, rollout_len)
        self._actor = ACPPOActorTrain(action_space)
        self.reward_normalizer = reward_normalizer
        self.return_scale = return_scale

    @classmethod
    def from_args(
        cls, args, reward_normalizer,
        action_space, spec_builder, **kwargs
    ):
        return cls(
            reward_normalizer, action_space, spec_builder,
            rollout_len=args.rollout_len,
            discount=args.discount,
            normalize_advantage=args.normalize_advantage,
            entropy_weight=args.entropy_weight,
            return_scale=args.return_scale
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space):
        return ACPPOActorTrain._exp_spec(
            exp_len, batch_sz, obs_space, act_space, internal_space
        )

    @staticmethod
    def output_space(action_space):
        return ACPPOActorTrain.output_space(action_space)

    def compute_action_exp(self, predictions, internals, obs, available_actions):
        return self._actor.compute_action_exp(predictions, internals, obs, available_actions)

    def compute_loss_and_step(self, network, optimizer, next_obs, next_internals):
        r = rollouts
        rollout_len = len(r.rewards)

        # estimate value of next state
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            pred, _ = self.network(next_obs_on_device, self.internals)
            last_values = pred['critic'].squeeze(1).data

        # calc nsteps
        gae = 0.
        next_values = last_values
        gae_returns = []
        for i in reversed(range(rollout_len)):
            rewards = r.rewards[i]
            terminals = r.terminals[i]
            current_values = r.values[i].squeeze(1)
            # generalized advantage estimation
            delta_t = rewards + self.discount * next_values.data * terminals - current_values
            gae = gae * self.discount * self.tau * terminals + delta_t
            gae_returns.append(gae + current_values)
            next_values = current_values.data
        gae_returns = torch.stack(list(reversed(gae_returns))).data

        # Convert to torch tensors of [seq, num_env]
        old_values = torch.stack(r.values).squeeze(-1)
        adv_targets_batch = (gae_returns - old_values).data
        actions_device = torch.from_numpy(np.asarray(r.actions)).to(self.device)
        old_log_probs_batch = torch.stack(r.log_probs).data
        terminals_batch = torch.stack(r.terminals)

        # Normalize advantage
        if self.normalize_advantage:
            adv_targets_batch = (adv_targets_batch - adv_targets_batch.mean()) / \
                                (adv_targets_batch.std() + 1e-5)

        for e in range(self.nb_epoch):
            # setup minibatch iterator
            minibatch_inds = BatchSampler(SequentialSampler(range(rollout_len)), self.batch_size, drop_last=False)
            for i in minibatch_inds:
                starting_internals = self._detach_internals(r.internals[i[0]])
                gae_return = gae_returns[i]
                old_log_probs = old_log_probs_batch[i]
                sampled_actions = actions_device[i]
                adv_targets = adv_targets_batch[i]
                terminal_masks = terminals_batch[i]

                # States are list(dict) select batch and convert to dict(list)
                obs = listd_to_dlist([r.obs[batch_ind] for batch_ind in i])
                # convert to dict(tensors)
                obs = {k: torch.stack(v) for k, v in obs.items()}

                # forward pass
                cur_log_probs, cur_values, entropies = self.act_batch(obs, terminal_masks, sampled_actions,
                                                                      starting_internals)
                value_loss = 0.5 * torch.mean((cur_values - gae_return).pow(2))

                # calculate surrogate loss
                surrogate_ratio = torch.exp(cur_log_probs - old_log_probs)
                surrogate_loss = surrogate_ratio * adv_targets
                surrogate_loss_clipped = torch.clamp(surrogate_ratio, 1 - self.loss_clip,
                                                     1 + self.loss_clip) * adv_targets
                policy_loss = torch.mean(-torch.min(surrogate_loss, surrogate_loss_clipped)) 
                entropy_loss = torch.mean(-0.01 * entropies)

                losses = {'value_loss': value_loss, 'policy_loss': policy_loss, 'entropy_loss':
                          entropy_loss}
                # print('losses', losses)
                total_loss = torch.sum(torch.stack(tuple(loss for loss in losses.values())))
                update_handler.update(total_loss)

        metrics = {'advantage': torch.mean(adv_targets_batch)}
        return losses, total_loss, metrics

