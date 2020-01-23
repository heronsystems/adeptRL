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
            reward_normalizer,
            discount,
            normalize_advantage,
            entropy_weight,
            return_scale,
            optimizer
    ):
        super(ACRolloutLearner, self).__init__(optimizer)
        self.reward_normalizer = reward_normalizer
        self.discount = discount
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight
        self.return_scale = return_scale
        if return_scale:
            self.dm_scaler = DeepMindReturnScaler(10. ** -3)

    @classmethod
    def from_args(cls, args, reward_normalizer, optimizer):
        return cls(
            reward_normalizer,
            args.discount,
            args.normalize_advantage,
            args.entropy_weight,
            args.return_scale,
            args.optimizer
        )

    def compute_loss(self, network, experiences, next_obs, internals):
        # normalize rewards
        rewards = self.reward_normalizer(torch.stack(experiences.rewards))

        # estimate value of next state
        with torch.no_grad():
            results, _, _ = network(next_obs, internals)
            last_values = results['critic'].squeeze(1).data

        # compute nstep return and advantage over batch
        r_values = torch.stack(experiences.values)
        r_tgt_returns = self.compute_returns(
            last_values, rewards, experiences.terminals
        )
        r_advantages = r_tgt_returns - r_values.data

        # normalize advantage so that an even number
        # of actions are reinforced and penalized
        if self.normalize_advantage:
            r_advantages = (r_advantages - r_advantages.mean()) \
                               / (r_advantages.std() + 1e-5)
        policy_loss = 0.
        entropy_loss = 0.

        rollout_len = len(rewards)
        for i in range(rollout_len):
            log_probs = experiences.log_probs[i]
            entropies = experiences.entropies[i]

            policy_loss = policy_loss - (
                    log_probs * r_advantages[i].unsqueeze(1).data
            ).sum(1)
            entropy_loss = entropy_loss - (
                    self.entropy_weight * entropies
            ).sum(1)

        batch_size = policy_loss.shape[0]
        nb_action = log_probs.shape[1]

        value_loss = 0.5 * (r_tgt_returns - r_values).pow(2).mean()
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

    def compute_returns(self, bootstrap_value, rewards, terminals):
        """
        R = Rollout Length
        B = Batch Size

        :param bootstrap_value:
        :param rewards: Tensor[R, B]
        :param terminals: Tensor[R, B]
        :return:
        """
        # First step of nstep reward target is estimated value of t+1
        target_return = bootstrap_value
        rollout_len = len(rewards)
        nstep_target_returns = []
        for i in reversed(range(rollout_len)):
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
