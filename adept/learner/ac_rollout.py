import torch

from .base.learner_module import LearnerModule
from .base.dm_return_scale import DeepMindReturnScaler


class ACRolloutLearner(LearnerModule):
    """
    Actor Critic Rollout Learner
    """
    args = {
        'discount': 0.99,
        'normalize_advantage': False,
        'entropy_weight': 0.01,
        'return_scale': False
    }

    def __init__(
            self,
            discount,
            normalize_advantage,
            entropy_weight,
            return_scale
    ):
        self.discount = discount
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight
        self.return_scale = return_scale
        if return_scale:
            self.dm_scaler = DeepMindReturnScaler(10. ** -3)

    @classmethod
    def from_args(cls, args):
        return cls(
            args.discount,
            args.normalize_advantage,
            args.entropy_weight,
            args.return_scale
        )

    def compute_loss(self, network, experiences, next_obs, internals):
        # estimate value of next state
        with torch.no_grad():
            results, _ = network(next_obs, internals)
            last_values = results['critic'].squeeze(1).data

        # compute nstep return and advantage over batch
        batch_values = torch.stack(experiences.values)
        batch_tgt_returns = self.compute_returns(
            last_values, experiences.rewards, experiences.terminals
        )
        batch_advantages = batch_tgt_returns - batch_values.data

        # batched value loss
        value_loss = 0.5 * torch.mean((batch_tgt_returns - batch_values).pow(2))

        # normalize advantage so that an even number
        # of actions are reinforced and penalized
        if self.normalize_advantage:
            batch_advantages = (batch_advantages - batch_advantages.mean()) \
                               / (batch_advantages.std() + 1e-5)
        policy_loss = 0.
        entropy_loss = 0.

        rollout_len = len(experiences.rewards)
        for i in range(rollout_len):
            log_probs = experiences.log_probs[i]
            entropies = experiences.entropies[i]

            policy_loss = policy_loss - (
                    log_probs * batch_advantages[i].unsqueeze(1).data
            ).sum(1)
            entropy_loss = entropy_loss - (
                    self.entropy_weight * entropies
            ).sum(1)

        batch_size = policy_loss.shape[0]
        nb_action = log_probs.shape[1]

        denom = batch_size * rollout_len * nb_action
        policy_loss = policy_loss.sum(0) / denom
        entropy_loss = entropy_loss.sum(0) / denom

        losses = {
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss
        }
        metrics = {}
        return losses, metrics

    def compute_returns(self, estimated_value, rewards, terminals):
        # First step of nstep reward target is estimated value of t+1
        target_return = estimated_value
        nstep_target_returns = []
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            terminal_mask = 1. - terminals[i].float()

            if self.return_scale:
                target_return = self.dm_scaler.calc_scale(
                    reward +
                    self.discount *
                    self.dm_scaler.calc_inverse_scale(target_return) *
                    terminal_mask
                )
            else:
                target_return = reward + (
                        self.discount * target_return * terminal_mask
                )
            nstep_target_returns.append(target_return)

        # reverse lists
        nstep_target_returns = torch.stack(
            list(reversed(nstep_target_returns))
        ).data

        return nstep_target_returns
