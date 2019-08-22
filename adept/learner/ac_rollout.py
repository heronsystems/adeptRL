import torch

from adept.learner.base.learner_module import LearnerModule


class ACRolloutLearner(LearnerModule):
    """
    Actor Critic Rollout Learner
    """
    args = {
        'discount': 0.99,
        'gae': True,
        'tau': 1.,
        'normalize_advantage': False,
        'entropy_weight': 0.01
    }

    def __init__(
            self,
            discount,
            gae,
            tau,
            normalize_advantage,
            entropy_weight
    ):
        self.discount = discount
        self.gae = gae
        self.tau = tau
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight

    @classmethod
    def from_args(cls, args):
        return cls(
            args.discount,
            args.gae,
            args.tau,
            args.normalize_advantage,
            args.entropy_weight
        )

    def compute_loss(self, network, experiences, next_obs, internals):
        # estimate value of next state
        with torch.no_grad():
            results, _ = network(next_obs, internals)
            last_values = results['critic'].squeeze(1).data

        # compute nstep return and advantage over batch
        batch_values = torch.stack(experiences.values)
        value_targets, batch_advantages = self._compute_returns_advantages(
            batch_values, last_values, experiences.rewards, experiences.terminals
        )

        # batched value loss
        value_loss = 0.5 * torch.mean((value_targets - batch_values).pow(2))

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

    def _compute_returns_advantages(
            self, values, estimated_value, rewards, terminals
    ):
        if self.gae:
            gae = 0.
            gae_advantages = []

        next_value = estimated_value
        # First step of nstep reward target is estimated value of t+1
        target_return = estimated_value
        nstep_target_returns = []
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            terminal_mask = 1. - terminals[i].float()

            # Nstep return is always calculated for the critic's target
            # using the GAE target for the critic results in the
            # same or worse performance
            target_return = reward + self.discount * target_return * terminal_mask
            nstep_target_returns.append(target_return)

            # Generalized Advantage Estimation
            if self.gae:
                delta_t = reward \
                          + self.discount * next_value * terminal_mask \
                          - values[i].data
                gae = gae * self.discount * self.tau * terminal_mask + delta_t
                gae_advantages.append(gae)
                next_value = values[i].data

        # reverse lists
        nstep_target_returns = torch.stack(
            list(reversed(nstep_target_returns))
        ).data

        if self.gae:
            advantages = torch.stack(list(reversed(gae_advantages))).data
        else:
            advantages = nstep_target_returns - values.data

        return nstep_target_returns, advantages
