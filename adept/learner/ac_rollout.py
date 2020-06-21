import torch

from .base.learner_module import LearnerModule
from .base.dm_return_scale import DeepMindReturnScaler


class ACRolloutLearner(LearnerModule):
    """
    Actor Critic Rollout Learner
    """

    args = {
        "discount": 0.99,
        "normalize_advantage": False,
        "entropy_weight": 0.01,
        "return_scale": False,
    }

    def __init__(
        self,
        reward_normalizer,
        discount,
        normalize_advantage,
        entropy_weight,
        return_scale,
    ):
        self.reward_normalizer = reward_normalizer
        self.discount = discount
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight
        self.return_scale = return_scale
        if return_scale:
            self.dm_scaler = DeepMindReturnScaler(10.0 ** -3)

    @classmethod
    def from_args(cls, args, reward_normalizer):
        return cls(
            reward_normalizer,
            args.discount,
            args.normalize_advantage,
            args.entropy_weight,
            args.return_scale,
        )

    def learn_step(self, updater, network, experiences, next_obs, internals):
        # normalize rewards
        rewards = self.reward_normalizer(torch.stack(experiences.rewards))

        # torch stack rollouts
        r_log_probs_action = torch.stack(experiences.log_probs)
        r_values = torch.stack(experiences.values)
        r_entropies = torch.stack(experiences.entropies)

        # estimate value of next state
        with torch.no_grad():
            results, _, _ = network(next_obs, internals)
            last_values = results["critic"].squeeze(1).data

        # compute nstep return and advantage over batch
        r_tgt_returns = self.compute_returns(
            last_values, rewards, experiences.terminals
        )
        r_advantages = r_tgt_returns - r_values.data

        # normalize advantage so that an even number
        # of actions are reinforced and penalized
        if self.normalize_advantage:
            r_advantages = (r_advantages - r_advantages.mean()) / (
                r_advantages.std() + 1e-5
            )

        # batched losses
        policy_loss = -(r_log_probs_action) * r_advantages.unsqueeze(-1)
        # mean over actions, seq, batch
        policy_loss = policy_loss.mean()
        entropy_loss = -r_entropies.mean() * self.entropy_weight
        value_loss = 0.5 * (r_tgt_returns - r_values).pow(2).mean()

        updater.step(value_loss + policy_loss + entropy_loss)

        losses = {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
        }
        metrics = {}
        return losses, metrics

    def compute_returns(self, bootstrap_value, rewards, terminals):
        # First step of nstep reward target is estimated value of t+1
        target_return = bootstrap_value
        rollout_len = len(rewards)
        nstep_target_returns = []
        for i in reversed(range(rollout_len)):
            reward = rewards[i]
            terminal_mask = 1.0 - terminals[i].float()

            if self.return_scale:
                target_return = self.dm_scaler.calc_scale(
                    reward
                    + self.discount
                    * self.dm_scaler.calc_inverse_scale(target_return)
                    * terminal_mask
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
