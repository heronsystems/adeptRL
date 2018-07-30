# Use https://github.com/deepmind/scalable_agent/blob/master/vtrace.py for reference
import torch
from gym import spaces
from torch.nn import functional as F

from adept.expcaches.rollout import RolloutCache
from adept.utils.util import listd_to_dlist
from .._base import Agent, EnvBase


class ActorCriticVtrace(Agent, EnvBase):
    def __init__(self, network, device, reward_normalizer, nb_env, nb_rollout, discount,
                 minimum_importance_value=1.0, minimum_importance_policy=1.0, entropy_weight=0.01):
        self.discount = discount
        self.minimum_importance_value = minimum_importance_value
        self.minimum_importance_policy = minimum_importance_policy
        self.entropy_weight = entropy_weight

        self._network = network.to(device)
        self._exp_cache = RolloutCache(nb_rollout, device, reward_normalizer, ['log_prob_of_action', 'sampled_action'])
        self._internals = listd_to_dlist([self.network.new_internals(device) for _ in range(nb_env)])
        self._device = device
        self.network.train()

    @property
    def exp_cache(self):
        return self._exp_cache

    @property
    def network(self):
        return self._network

    @property
    def device(self):
        return self._device

    @property
    def internals(self):
        return self._internals

    @internals.setter
    def internals(self, new_internals):
        self._internals = new_internals

    @staticmethod
    def output_shape(action_space):
        if isinstance(action_space, spaces.Discrete):
            head_dict = {'critic': 1, 'actor': action_space.n}
        elif isinstance(action_space, spaces.Dict):
            # TODO support nested dicts
            # currently only works for dicts of Discrete action_spaces's
            head_dict = {**{'critic': 1}, **{k: a_space.n for k, a_space in action_space.spaces.items()}}
        else:
            raise ValueError('Unrecognized action space {}'.format(action_space))
        return head_dict

    def seq_obs_to_pathways(self, obs, device):
        """
            Converts a dict of sequential observations to a list(of seq len) of dicts
        """
        pathway_dict = self.obs_to_pathways(obs, device, visual_dim=4, discrete_dim=2)
        visual = pathway_dict['visual']
        discrete = pathway_dict['discrete']
        # visual or discrete can be empty tensor
        if visual.shape[0] > 0:
            # visual and discrete
            if discrete.shape[0] > 0:
                seq = [{
                    'visual': visual[i],
                    'discrete': discrete[i]
                } for i in range(visual.shape[0])]
            # visual only
            else:
                seq = [{
                    'visual': visual[i],
                    'discrete': discrete
                } for i in range(visual.shape[0])]
        # only discrete
        else:
            seq = [{
                'visual': visual,
                'discrete': discrete[i]
            } for i in range(discrete.shape[0])]

        return seq

    def act(self, obs):
        """
            This is the method called on each worker so it does not require grads and must
            keep track of it's internals. IMPALA only needs log_probs(a) and the sampled action from the worker
            Additionally, no gradients are needed here
        """
        with torch.no_grad():
            results, internals = self.network(self.obs_to_pathways(obs, self.device), self.internals)
            logits = {k: v for k, v in results.items() if k != 'critic'}

            logits = self.preprocess_logits(logits)
            actions, log_prob = self.process_logits(logits, obs)
            compressed_actions, compressed_log_probs = self.compress_actions_log_probs(actions, log_prob)

            self.exp_cache.write_forward(
                log_prob_of_action=compressed_log_probs,
                sampled_action=compressed_actions
            )
            self.internals = internals
            return actions

    def compress_actions_log_probs(self, actions, log_probs):
        """
            Used to convert actions & log_probs into something that can be sent across MPI
            Must be overriden for envs with complex action spaces
            Must return an int, array, or tensor
        """
        return actions, log_probs

    def act_eval(self, obs):
        self.network.eval()
        with torch.no_grad():
            results, internals = self.network(self.obs_to_pathways(obs, self.device), self.internals)
            logits = {k: v for k, v in results.items() if k != 'critic'}

            logits = self.preprocess_logits(logits)
            actions, _ = self.process_logits(logits, obs, deterministic=True)
        self.internals = internals
        return actions

    def act_on_host(self, obs, next_obs, terminal_masks, sampled_actions, internals):
        """
            This is the method to recompute the forward pass on the host, it must return log_probs, values and entropies
            Obs, sampled_actions, terminal_masks here are [seq, batch], internals must be reset if terminal
        """
        obs_on_device = self.seq_obs_to_pathways(obs, self.device)
        next_obs_on_device = self.obs_to_pathways(next_obs, self.device)

        values = []
        log_probs_of_action = []
        entropies = []
        # TODO: log_probs needs to support multiple actions as a dict
        for seq_ind in range(terminal_masks.shape[0]):
            obs_of_seq_ind = obs_on_device[seq_ind]
            results, internals = self.network(obs_of_seq_ind, internals)
            # TODO: when states are dicts use the below
            # results, internals = self.network({k: v[seq_ind] for k, v in obs_on_device.items()}, self.internals)

            logits_seq = {k: v for k, v in results.items() if k != 'critic'}
            logits_seq = self.preprocess_logits(logits_seq)
            log_probs_action_seq, entropies_seq = self.process_logits_host(logits_seq, sampled_actions[seq_ind], obs_of_seq_ind)
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

        # TODO: can't stack if dict
        return torch.stack(log_probs_of_action), torch.stack(values), last_values, torch.stack(entropies)

    def preprocess_logits(self, logits):
        return logits['actor']

    def process_logits(self, logits, obs, deterministic=False):
        prob = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        if not deterministic:
            actions = prob.multinomial(1)
        else:
            actions = torch.argmax(prob, 1, keepdim=True)
        log_probs = log_probs.gather(1, actions)

        return actions.squeeze(1).cpu().numpy(), log_probs.squeeze(1)

    def process_logits_host(self, logits, actions, obs):
        prob = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropies = -(log_probs * prob).sum(1)
        log_probs = log_probs.gather(1, actions.unsqueeze(1))

        return log_probs.squeeze(1), entropies

    def compute_loss(self, rollouts):
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
            k.split('-')[-1]: torch.cat(rollouts[k], 0)  # 0 dim here is batch since next obs has no seq
            for k, v in rollouts.items() if 'next_obs-' in k
        }
        behavior_log_prob_of_action = torch.cat(rollouts['log_prob_of_action'], 1).to(self.device)
        behavior_sampled_action = torch.cat(rollouts['sampled_action'], 1).long().to(self.device)
        # internals are prefixed like internals-
        # they are a list[]
        behavior_starting_internals = {
            # list flattening https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
            k.split('-')[-1]: [item.to(self.device) for sublist in v for item in sublist]
            for k, v in rollouts.items() if 'internals' in k
        }

        # compute current policy/critic forward
        current_log_prob_of_action, current_values, estimated_value, current_entropies = self.act_on_host(states, next_states, terminals_mask,
                                                                                                          behavior_sampled_action,
                                                                                                          behavior_starting_internals)

        # compute target for current value and advantage
        with torch.no_grad():
            # create importance sampling
            log_diff_behavior_vs_current = current_log_prob_of_action - behavior_log_prob_of_action
            value_trace_target, pg_advantage, importance = self._vtrace_returns(log_diff_behavior_vs_current, discount_terminal_mask, rewards,
                                                                                current_values, estimated_value, self.minimum_importance_value,
                                                                                self.minimum_importance_policy)

        # using torch.no_grad so detach is unnecessary
        value_loss = 0.5 * torch.mean((value_trace_target - current_values).pow(2))
        policy_loss = torch.mean(-current_log_prob_of_action * pg_advantage)
        entropy_loss = torch.mean(-current_entropies) * self.entropy_weight

        losses = {'value_loss': value_loss, 'policy_loss': policy_loss, 'entropy_loss': entropy_loss}
        metrics = {'importance': importance.mean()}
        return losses, metrics

    @staticmethod
    def _vtrace_returns(log_diff_behavior_vs_current, discount_terminal_mask, rewards, values, estimated_value,
                        minimum_importance_value, minimum_importance_policy):
        """
            Args:
                discount_terminal_mask: should be shape [seq, batch] of discount * (1 - terminal)
            Returns target for current critic and advantage for policy
        """
        importance = torch.exp(log_diff_behavior_vs_current)
        clamped_importance_value = importance.clamp(max=minimum_importance_value)
        # if multiple actions take the average, (dim 3 is seq, batch, # actions)
        if clamped_importance_value.dim() == 3:
            clamped_importance_value = clamped_importance_value.mean(-1)

        # create nstep vtrace return
        # first create d_tV of function 1 in the paper
        values_t_plus_1 = torch.cat((values[1:], estimated_value.unsqueeze(0)), 0)
        diff_value_per_step = clamped_importance_value * (rewards + discount_terminal_mask * values_t_plus_1 - values)

        # reverse over the values to create the summed importance weighted return
        # everything on the right side of the plus in function 1 of the paper
        vs_minus_v_xs = []
        nstep_v = 0.0
        # TODO: this uses a different clamping if != 1
        for i in reversed(range(diff_value_per_step.shape[0])):
            nstep_v = diff_value_per_step[i] + discount_terminal_mask[i] * clamped_importance_value[i] * nstep_v
            vs_minus_v_xs.append(nstep_v)
        # reverse to a forward in time list
        vs_minus_v_xs = torch.stack(list(reversed(vs_minus_v_xs)))

        # Add V(s) to finish computation of v_s
        v_s = values + vs_minus_v_xs

        # advantage is pg_importance * (v_s of t+1 - values)
        clamped_importance_pg = importance.clamp(max=minimum_importance_policy)

        v_s_tp1 = torch.cat((v_s[1:], estimated_value.unsqueeze(0)), 0)
        advantage = rewards + discount_terminal_mask * v_s_tp1 - values

        # if multiple actions broadcast the advantage to be weighted by the differen actions importance
        # (dim 3 is seq, batch, # actions)
        if clamped_importance_pg.dim() == 3:
            advantage = advantage.unsqueeze(-1)

        weighted_advantage = clamped_importance_pg * advantage
        return v_s, weighted_advantage, importance
