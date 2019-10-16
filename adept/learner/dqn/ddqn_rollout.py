import torch

from .dqn_rollout import DQNRolloutLearner


class DDQNRolloutLearner(DQNRolloutLearner):
    """
    DDQN Rollout Learner
    """
    # TODO: this is duplicated from rollout actor
    def _get_qvals_from_pred(self, predictions):
        q = {}
        for k in self._action_keys:
            norm_adv = predictions[k] - predictions[k].mean(-1, keepdim=True)
            q[k] = norm_adv + predictions['value']
        return q

