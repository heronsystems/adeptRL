import torch

from .base.learner_module import LearnerModule
from .base.dm_return_scale import DeepMindReturnScaler


class DQNRolloutLearner(LearnerModule):
    """
    DQN Rollout Learner
    """
    args = {
        'discount': 0.99,
        'return_scale': False,
        'double_dqn': True
    }

    def __init__(
            self,
            reward_normalizer,
            discount,
            return_scale,
            double_dqn
    ):
        self.reward_normalizer = reward_normalizer
        self.discount = discount
        self.return_scale = return_scale
        self.double_dqn = double_dqn
        if return_scale:
            self.dm_scaler = DeepMindReturnScaler(10. ** -3)

    @classmethod
    def from_args(cls, args, reward_normalizer):
        return cls(
            reward_normalizer,
            args.discount,
            args.return_scale,
            args.double_dqn
        )

    def compute_loss(self, network, experiences, next_obs, internals):
        # TODO: target network
        # estimate value of next state
        last_values = self.compute_estimated_values(network, network, next_obs, internals)

        # compute nstep return and advantage over batch
        batch_values = torch.stack(experiences.values)
        value_targets = self.compute_returns(last_values, experiences.rewards, experiences.terminals)

        # batched loss
        value_loss = self.loss_fn(batch_values, value_targets)

        losses = {'value_loss': value_loss.mean()}
        metrics = {}
        return losses, metrics

    def loss_fn(self, batch_values, value_targets):
        return 0.5 * (value_targets - batch_values).pow(2)

    def compute_estimated_values(self, network, target_network, next_obs, internals):
        # estimate value of next state
        with torch.no_grad():
            results, _ = target_network(next_obs, internals)
            # TODO: hack, where can this class get action keys?
            self.action_keys = list(results.keys())
            target_q = self._get_qvals_from_pred(results)

            # if double dqn estimate get target val for current estimated action
            if self.double_dqn:
                current_results, _ = network(next_obs, internals)
                current_q = self._get_qvals_from_pred(current_results)
                last_actions = [self._action_from_q_vals(current_q[k]) for k in self.action_keys]
                last_values = torch.stack([target_q[k].gather(1, a)[:, 0].data for k, a in zip(self.action_keys, last_actions)], dim=1)
            else:
                last_values = torch.stack([torch.max(target_q[k], 1)[0].data for k in self.action_keys], dim=1)

        return last_values

    def compute_returns(self, estimated_value, rewards, terminals):
        next_value = estimated_value
        # First step of nstep reward target is estimated value of t+1
        target_return = estimated_value
        nstep_target_returns = []
        for i in reversed(range(len(rewards))):
            # unsqueeze over action dim so it isn't broadcasted
            reward = rewards[i].unsqueeze(-1)
            terminal_mask = 1. - terminals[i].unsqueeze(-1).float()

            # Nstep return is always calculated for the critic's target
            if self.return_scale:
                target_return = self.dm_scaler.calc_scale(
                    reward + self.discount * self.dm_scaler.calc_inverse_scale(target_return) * terminal_mask
                )
            else:
                target_return = reward + self.discount * target_return * terminal_mask
            nstep_target_returns.append(target_return)

        # reverse lists
        nstep_target_returns = torch.stack(list(reversed(nstep_target_returns))).data
        return nstep_target_returns

    # TODO: this is duplicated from rollout actor
    def _get_qvals_from_pred(self, preds):
        return preds

    def _action_from_q_vals(self, q_vals):
        return q_vals.argmax(dim=-1, keepdim=True)

