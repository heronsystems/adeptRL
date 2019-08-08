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
import torch
from torch.nn import functional as F

from adept.expcaches.rollout import RolloutCache
from adept.utils.util import listd_to_dlist, dlist_to_listd
from adept.agents.agent_module import AgentModule


class ImpalaActor(AgentModule):
    """
    Reference implementation:
    Use https://github.com/deepmind/scalable_agent/blob/master/vtrace.py
    """
    args = {
        'nb_rollout': 20,
        'discount': 0.99,
        'minimum_importance_value': 1.0,
        'minimum_importance_policy': 1.0,
        'entropy_weight': 0.01
    }

    def __init__(
        self,
        network,
        device,
        reward_normalizer,
        gpu_preprocessor,
        policy,
        nb_env,
        nb_rollout,
        discount,
        minimum_importance_value,
        minimum_importance_policy,
        entropy_weight
    ):
        super(ImpalaActor, self).__init__(
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            policy,
            nb_env
        )
        self.discount = discount
        self.minimum_importance_value = minimum_importance_value
        self.minimum_importance_policy = minimum_importance_policy
        self.entropy_weight = entropy_weight

        self._exp_cache = RolloutCache(
            nb_rollout, device, reward_normalizer,
            ['log_prob_of_action', 'sampled_action']
        )

    @classmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor, policy,
        nb_env=None
    ):
        if nb_env is None:
            nb_env = args.nb_env

        return cls(
            network, device, reward_normalizer, gpu_preprocessor, policy,
            nb_env=nb_env,
            nb_rollout=args.nb_rollout,
            discount=args.discount,
            minimum_importance_value=args.minimum_importance_value,
            minimum_importance_policy=args.minimum_importance_policy,
            entropy_weight=args.entropy_weight
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @staticmethod
    def output_space(action_space):
        return {'critic': (1, ), **action_space}

    def seq_obs_to_pathways(self, obs, device):
        """
        Converts a dict of sequential observations to a list(of seq len) of
        dicts
        """
        pathway_dict = self.gpu_preprocessor(obs, device)
        return dlist_to_listd(pathway_dict)

    def act(self, obs):
        raise NotImplementedError
        self.network.train()
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals
        )
        available_actions = \
            obs['available_actions'] if 'available_actions' in obs \
            else None

        actions, extras = self.policy.act(predictions, available_actions)
        log_probs, entropies = extras['log_probs_host'], extras['entropies']

        return actions

    def act_eval(self, obs):
        self.network.eval()
        with torch.no_grad():
            predictions, internals = self.network(
                self.gpu_preprocessor(obs, self.device), self.internals
            )
        if 'available_actions' in obs:
            actions = self.policy.act_eval(
                predictions, obs['available_actions']
            )
        else:
            actions = self.policy.act_eval(predictions)

        self.internals = internals
        return actions

    def act_on_host(
        self, obs, next_obs, terminal_masks, sampled_actions, internals
    ):
        """
        This is the method to recompute the forward pass on the host, it
        must return log_probs, values and entropies Obs, sampled_actions,
        terminal_masks here are [seq, batch], internals must be reset if
        terminal
        """
        self.network.train()
        next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)

        values = []
        log_probs_of_action = []
        entropies = []

        # if network is modular,
        # trunk can be sped up by combining batch & seq dim
        def get_results_generator():
            obs_on_device = self.seq_obs_to_pathways(obs, self.device)

            def get_results(seq_ind, internals):
                obs_of_seq_ind = obs_on_device[seq_ind]
                return self.network(obs_of_seq_ind, internals)

            return get_results

        result_fn = get_results_generator()
        for seq_ind in range(terminal_masks.shape[0]):
            results, internals = result_fn(seq_ind, internals)
            logits_seq = {k: v for k, v in results.items() if k != 'critic'}
            log_probs_action_seq, entropies_seq = self._predictions_to_logprobs_ents_host(
                seq_ind, obs, logits_seq, sampled_actions[seq_ind]
            )
            # seq lists
            values.append(results['critic'].squeeze(1))
            log_probs_of_action.append(log_probs_action_seq)
            entropies.append(entropies_seq)

            # if this state was terminal reset internals
            for batch_ind, t_mask in enumerate(terminal_masks[seq_ind]):
                if t_mask == 0:
                    reset_internals = self.network.new_internals(self.device)
                    for k, v in reset_internals.items():
                        internals[k][batch_ind] = v

        # forward on state t+1
        with torch.no_grad():
            results, _ = self.network(next_obs_on_device, internals)
            last_values = results['critic'].squeeze(1)

        return torch.stack(log_probs_of_action), torch.stack(
            values
        ), last_values, torch.stack(entropies)

    def compute_loss(self, rollouts, available_actions):
        # rollouts here are a list of [seq, nb_env]
        # cat along the 1 dim gives [seq, batch = nb_env*nb_batches]
        # pull from rollout and convert to tensors of [seq, batch, ...]
        rewards = torch.cat(rollouts['rewards'], 1).to(self.device)
        terminals_mask = torch.cat(rollouts['terminals'], 1).to(self.device)
        discount_terminal_mask = self.discount * terminals_mask
        states = {
            k.split('-')[-1]: torch.cat(rollouts[k], 1)
            for k, v in rollouts.items() if 'rollout_obs-' in k
        }
        next_states = {
            k.split('-')[-1]:
            torch.cat(rollouts[k],
                      0)  # 0 dim here is batch since next obs has no seq
            for k, v in rollouts.items() if 'next_obs-' in k
        }
        behavior_log_prob_of_action = torch.cat(
            rollouts['log_prob_of_action'], 1
        ).to(self.device)
        behavior_sampled_action = torch.cat(rollouts['sampled_action'],
                                            1).long().to(self.device)
        # internals are prefixed like internals-
        # they are a list[]
        behavior_starting_internals = {
            # list flattening
            # https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
            k.split('-')[-1]:
            [item.to(self.device) for sublist in v for item in sublist]
            for k, v in rollouts.items() if 'internals' in k
        }

        # compute current policy/critic forward
        current_log_prob_of_action, current_values, estimated_value, current_entropies = self.act_on_host(
            states, next_states, terminals_mask, behavior_sampled_action,
            behavior_starting_internals
        )

        # compute target for current value and advantage
        with torch.no_grad():
            # create importance sampling
            log_diff_behavior_vs_current = current_log_prob_of_action - behavior_log_prob_of_action
            value_trace_target, pg_advantage, importance = self._vtrace_returns(
                log_diff_behavior_vs_current, discount_terminal_mask, rewards,
                current_values, estimated_value, self.minimum_importance_value,
                self.minimum_importance_policy
            )

        # using torch.no_grad so detach is unnecessary
        value_loss = 0.5 * torch.mean(
            (value_trace_target - current_values).pow(2)
        )
        policy_loss = torch.mean(-current_log_prob_of_action * pg_advantage)
        entropy_loss = torch.mean(-current_entropies) * self.entropy_weight

        losses = {
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss
        }
        metrics = {'importance': importance.mean()}
        return losses, metrics

    @staticmethod
    def _vtrace_returns(
        log_diff_behavior_vs_current, discount_terminal_mask, rewards, values,
        estimated_value, minimum_importance_value, minimum_importance_policy
    ):
        """
        :param log_diff_behavior_vs_current:
        :param discount_terminal_mask: should be shape [seq, batch] of
        discount * (1 - terminal)
        :param rewards:
        :param values:
        :param estimated_value:
        :param minimum_importance_value:
        :param minimum_importance_policy:
        :return:
        """
        importance = torch.exp(log_diff_behavior_vs_current)
        clamped_importance_value = importance.clamp(
            max=minimum_importance_value
        )
        # if multiple actions take the average, (dim 3 is seq, batch, # actions)
        if clamped_importance_value.dim() == 3:
            clamped_importance_value = clamped_importance_value.mean(-1)

        # create nstep vtrace return
        # first create d_tV of function 1 in the paper
        values_t_plus_1 = torch.cat(
            (values[1:], estimated_value.unsqueeze(0)), 0
        )
        diff_value_per_step = clamped_importance_value * (
            rewards + discount_terminal_mask * values_t_plus_1 - values
        )

        # reverse over the values to create the summed importance weighted
        # return everything on the right side of the plus in function 1 of
        # the paper
        vs_minus_v_xs = []
        nstep_v = 0.0
        # TODO: this uses a different clamping if != 1
        for i in reversed(range(diff_value_per_step.shape[0])):
            nstep_v = diff_value_per_step[i] + discount_terminal_mask[
                i] * clamped_importance_value[i] * nstep_v
            vs_minus_v_xs.append(nstep_v)
        # reverse to a forward in time list
        vs_minus_v_xs = torch.stack(list(reversed(vs_minus_v_xs)))

        # Add V(s) to finish computation of v_s
        v_s = values + vs_minus_v_xs

        # advantage is pg_importance * (v_s of t+1 - values)
        clamped_importance_pg = importance.clamp(max=minimum_importance_policy)

        v_s_tp1 = torch.cat((v_s[1:], estimated_value.unsqueeze(0)), 0)
        advantage = rewards + discount_terminal_mask * v_s_tp1 - values

        # if multiple actions broadcast the advantage to be weighted by the
        # different actions importance
        # (dim 3 is seq, batch, # actions)
        if clamped_importance_pg.dim() == 3:
            advantage = advantage.unsqueeze(-1)

        weighted_advantage = clamped_importance_pg * advantage
        return v_s, weighted_advantage, importance
