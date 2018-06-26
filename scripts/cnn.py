from __future__ import division

import os

import numpy as np
import torch
from torch.nn import functional as F

from scripts._base import BaseTrainingLoop
from src.models.cnn import SimpleCNN
from src.utils import RolloutCache, base_parser
from src.utils import from_numpy


def unroll(cache, gamma, tau):
    returns = cache['valuess'][-1].data
    policy_loss = 0.
    value_loss = 0.
    gae = torch.zeros_like(returns)

    rollout_len = len(cache['rewardss']) - 1
    for i in reversed(range(rollout_len)):
        rewards = cache['rewardss'][i]
        masks = cache['maskss'][i]
        values = cache['valuess'][i]
        values_t1 = cache['valuess'][i + 1]
        log_probs = cache['log_probss'][i]
        entropies = cache['entropiess'][i]

        returns = rewards + gamma * returns * masks
        advantages = returns - values
        value_loss = value_loss + 0.5 * advantages.pow(2)

        # Generalized Advantage Estimation
        delta_t = rewards + gamma * values_t1.data * masks - values.data
        gae = gae * gamma * tau * masks + delta_t

        policy_loss = policy_loss - log_probs * gae - 0.01 * entropies
    policy_loss = policy_loss / rollout_len
    value_loss = value_loss / rollout_len
    losses = {
        'value_loss': torch.mean(value_loss),
        'policy_loss': torch.mean(policy_loss)
    }
    metrics = {}
    return losses, metrics


def action_train(model, states):
    model.train()
    values, logits = model(states)
    prob = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropies = -(log_probs * prob).sum(1)
    actions = prob.multinomial(1)
    log_probs = log_probs.gather(1, actions)
    return actions.squeeze(1), values.squeeze(1), log_probs.squeeze(1), entropies


def action_test(model, states, hxs, cxs):
    model.eval()
    with torch.no_grad():
        value, logit, hxs, cxs = model(states, hxs, cxs)
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
    return action


class TrainingLoop(BaseTrainingLoop):
    def __init__(self, args):
        frame_stack = True
        rollout_cache = RolloutCache('rewardss', 'valuess', 'entropiess', 'log_probss', 'maskss')
        super(TrainingLoop, self).__init__(args, frame_stack, rollout_cache)

    def _setup_model(self):
        return SimpleCNN(
            self.envs.observation_space.shape[0], self.envs.action_space.n, self.args.batch_norm
        ).to(self.device)

    def _forward_step(self):
        actions, values, log_probs, entropies = action_train(self.model, from_numpy(self.states, self.device))
        states, rewards_unclipped, dones, infos = self.envs.step(actions.cpu().numpy())

        self.states = states
        rewards = torch.tensor([max(min(reward, 1), -1) for reward in rewards_unclipped]).to(self.device)
        masks = (1. - torch.from_numpy(np.array(dones, dtype=np.float32))).unsqueeze(1).to(self.device)
        self.rollout_cache.append({
            'rewardss': rewards,
            'valuess': values,
            'entropiess': entropies,
            'log_probss': log_probs,
            'maskss': masks
        })
        return rewards_unclipped, dones, infos

    def _unroll(self):
        return unroll(self.rollout_cache, self.args.gamma, self.args.tau)

    def _weight_losses(self, loss_dict):
        return loss_dict['policy_loss'] + 0.5 * loss_dict['value_loss']


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = base_parser()
    parser.add_argument('--name', default='cnn', help='logdir/tensorboard name')
    args = parser.parse_args()
    TrainingLoop(args).run()
