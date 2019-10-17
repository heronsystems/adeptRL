import torch

from .dqn_rollout import DQNRolloutLearner


class DDQNRolloutLearner(DQNRolloutLearner):
    """
    DDQN Rollout Learner
    Dueling DQN
    """
    # TODO: this is duplicated from rollout actor
    def _get_qvals_from_pred(self, predictions):
        q = {}
        for k in self.action_keys:
            norm_adv = predictions[k] - predictions[k].mean(-1, keepdim=True)
            q[k] = norm_adv + predictions['value']
        return q


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class QRDDQNRolloutLearner(DQNRolloutLearner):
    """
    QRDDQN Rollout Learner
    Quantile Regression Dueling DQN
    """
    args = {**DQNRolloutLearner.args, 'num_atoms': 51}
    def __init__(
            self,
            reward_normalizer,
            discount,
            return_scale,
            double_dqn,
            num_atoms
    ):
        if not double_dqn:
            raise NotImplementedError()
        super().__init__(reward_normalizer, discount, return_scale, double_dqn)
        self.num_atoms = num_atoms
        self._qr_density = (((2 * torch.arange(self.num_atoms, dtype=torch.float, requires_grad=False)) + 1) / (2.0 * self.num_atoms))

    @classmethod
    def from_args(cls, args, reward_normalizer):
        return cls(
            reward_normalizer,
            args.discount,
            args.return_scale,
            args.double_dqn,
            args.num_atoms
        )

    def compute_loss(self, network, experiences, next_obs, internals):
        # move qr density to network device
        network_device = next(network.parameters()).device
        if self._qr_density.device != network_device:
            self._qr_density = self._qr_density.to(network_device)
        return super().compute_loss(network, experiences, next_obs, internals)

    def loss_fn(self, batch_values, value_targets):
        # Broadcast temporal difference to compare every combination of quantiles
        # This is the formula for loss from the Implicit Quantile Networks paper
        diff = value_targets.unsqueeze(3) - batch_values.unsqueeze(2)
        dist_mask = torch.abs(self._qr_density - (diff.detach() < 0).float())
        return (huber(diff) * dist_mask).sum(-1).mean(-1, keepdim=True)

    # TODO: this is duplicated from rollout actor
    def _get_qvals_from_pred(self, predictions):
        pred = {}
        for k in self.action_keys:
            v = predictions[k]
            adv = v.view(v.shape[0], -1, self.num_atoms)
            norm_adv = adv - adv.mean(1, keepdim=True)
            pred[k] = norm_adv + predictions['value'].unsqueeze(1)
        return pred

    # TODO: this is duplicated from rollout actor
    def _action_from_q_vals(self, q_vals):
        # mean atoms, argmax over mean
        return q_vals.mean(-1).argmax(dim=-1, keepdim=True)

    # TODO: this is duplicated from rollout actor
    def _get_action_values(self, q_vals, action, batch_size):
        # TODO: need to store num atoms in self
        action_select = action.unsqueeze(1).expand(batch_size, 1, 51)
        return q_vals.gather(1, action_select).squeeze(1)

