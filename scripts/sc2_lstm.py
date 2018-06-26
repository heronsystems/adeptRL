from __future__ import division

import os

import numpy as np
import torch
from absl import flags
from pysc2.lib.actions import FUNCTION_TYPES, FUNCTIONS
from torch.nn import functional as F

from scripts._base import BaseTrainingLoop
from src.environments.sc2 import make_sc2_env, SC2SubprocEnv, SC2SingleEnv
from src.models.lstm import LSTMModelSC2
from src.utils import RolloutCache, base_parser

FLAGS = flags.FLAGS
FLAGS(['sc2_lstm.py'])


def unroll(cache, gamma, tau):
    returns = cache['valuess'][-1].data
    policy_loss = 0.
    value_loss = 0.
    gae = torch.zeros_like(returns)

    policy_loss_count = 0
    rollout_len = len(cache['rewardss']) - 1
    for i in reversed(range(rollout_len)):
        rewards = cache['rewardss'][i]       # Tensor{B}
        masks = cache['maskss'][i]           # Tensor{B}
        values = cache['valuess'][i]         # Tensor{B}
        values_t1 = cache['valuess'][i + 1]  # Tensor{B}
        log_probs_batch = cache['log_probss'][i]   # List{B}[Dict{.}[headname: Tensor{1}]]
        entropies_batch = cache['entropiess'][i]   # List{B}[Dict{.}[headname: Tensor{1}]]

        returns = rewards + gamma * returns * masks
        advantages = returns - values
        value_loss = value_loss + 0.5 * advantages.pow(2)

        # Generalized Advantage Estimation
        delta_t = rewards + gamma * values_t1.data * masks - values.data
        gae = gae * gamma * tau * masks + delta_t

        batch_idx = 0
        for log_probs_by_hn, entropies_by_hn in zip(log_probs_batch, entropies_batch):
            for log_prob, entropy in zip(log_probs_by_hn.values(), entropies_by_hn.values()):
                policy_loss_count += 1
                policy_loss = policy_loss - log_prob * gae[batch_idx] - 0.01 * entropy
            batch_idx += 1

    policy_loss = policy_loss / policy_loss_count
    value_loss = value_loss / rollout_len
    losses = {
        'value_loss': torch.mean(value_loss),
        'policy_loss': policy_loss
    }
    metrics = {}
    return losses, metrics


def action_train(model, states, hxs, cxs, available_actions):
    model.train()
    values, logits_batch_by_hn, hxs, cxs = model(states, hxs, cxs)
    acts_batch_by_hn, lprobs_batch_by_hn, ents_batch_by_hn =\
        process_logits_by_head(logits_batch_by_hn, available_actions)

    arg_names_batch = lookup_arg_names_batchwise(acts_batch_by_hn['func_id'])
    acts_result_batch, lprobs_result_batch, ents_result_batch = select_heads_batchwise(
        arg_names_batch,
        acts_batch_by_hn,
        lprobs_batch_by_hn,
        ents_batch_by_hn
    )
    return acts_result_batch, values.squeeze(1), lprobs_result_batch, ents_result_batch, hxs, cxs


def mask_action_probs(probs, available_actions_batch):
    """
    :param probs: Tensor{B, C}
    :param available_actions_batch: List{B}[List{.}[func_ids]]
    :return:
    """
    mask = torch.zeros_like(probs)
    # TODO vectorize
    for batch_idx, available_actions in enumerate(available_actions_batch):
        for available_action in available_actions:
            mask[batch_idx, available_action] = 1
    return probs * mask


def select_heads_batchwise(arg_names_batch, acts_batch_by_hn, lprobs_batch_by_hn, ents_batch_by_hn):
    acts_result_batch, lprobs_result_batch, ents_result_batch = [], [], []
    for batch_idx in range(len(arg_names_batch)):
        action_by_hn = {'func_id': acts_batch_by_hn['func_id'][batch_idx].item()}
        lprob_by_hn = {'func_id': lprobs_batch_by_hn['func_id'][batch_idx]}
        ent_by_hn = {'func_id': ents_batch_by_hn['func_id'][batch_idx]}
        for arg_name in arg_names_batch[batch_idx]:
            # arg_names == headnames except for screen, minimap, and screen2
            if arg_name == 'screen':
                screen_y = acts_batch_by_hn['screen_y'][batch_idx]
                screen_x = acts_batch_by_hn['screen_x'][batch_idx]
                action_by_hn['screen'] = [screen_y.item(), screen_x.item()]
                lprob_by_hn['screen_y'] = lprobs_batch_by_hn['screen_y'][batch_idx]
                lprob_by_hn['screen_x'] = lprobs_batch_by_hn['screen_x'][batch_idx]
                ent_by_hn['screen_y'] = ents_batch_by_hn['screen_y'][batch_idx]
                ent_by_hn['screen_x'] = ents_batch_by_hn['screen_x'][batch_idx]
            elif arg_name == 'minimap':
                minimap_y = acts_batch_by_hn['minimap_y'][batch_idx]
                minimap_x = acts_batch_by_hn['minimap_x'][batch_idx]
                action_by_hn['minimap'] = [minimap_y.item(), minimap_x.item()]
                lprob_by_hn['minimap_y'] = lprobs_batch_by_hn['minimap_y'][batch_idx]
                lprob_by_hn['minimap_x'] = lprobs_batch_by_hn['minimap_x'][batch_idx]
                ent_by_hn['minimap_y'] = ents_batch_by_hn['minimap_y'][batch_idx]
                ent_by_hn['minimap_x'] = ents_batch_by_hn['minimap_x'][batch_idx]
            elif arg_name == 'screen2':
                screen2_y = acts_batch_by_hn['screen2_y'][batch_idx]
                screen2_x = acts_batch_by_hn['screen2_x'][batch_idx]
                action_by_hn['screen2'] = [screen2_y.item(), screen2_x.item()]
                lprob_by_hn['screen2_y'] = lprobs_batch_by_hn['screen2_y'][batch_idx]
                lprob_by_hn['screen2_x'] = lprobs_batch_by_hn['screen2_x'][batch_idx]
                ent_by_hn['screen2_y'] = ents_batch_by_hn['screen2_y'][batch_idx]
                ent_by_hn['screen2_x'] = ents_batch_by_hn['screen2_x'][batch_idx]
            else:
                action_by_hn[arg_name] = [acts_batch_by_hn[arg_name][batch_idx].item()]
                lprob_by_hn[arg_name] = lprobs_batch_by_hn[arg_name][batch_idx]
                ent_by_hn[arg_name] = ents_batch_by_hn[arg_name][batch_idx]
        acts_result_batch.append(action_by_hn)
        lprobs_result_batch.append(lprob_by_hn)
        ents_result_batch.append(ent_by_hn)
    return acts_result_batch, lprobs_result_batch, ents_result_batch


def process_logits_by_head(logits_batch_by_hn, available_actions_batch):
    """
    :param logits_batch_by_hn: Dict[head_name: logits_batch{B, C}]
    :param available_actions_batch: List{B}[List{.}[int]]
    :return:
        acts_batch_by_hn : Dict[headname: action_tensor{B}],
        lprobs_batch_by_hn : Dict[headname: lprob_tensor{B}],
        ents_batch_by_hn : Dict[headname: ent_tensor{B}]
    """
    acts_batch_by_hn, lprobs_batch_by_hn, ents_batch_by_hn = {}, {}, {}
    for headname, logits_exmpl in logits_batch_by_hn.items():
        if headname == 'func_id':
            acts_batch_by_hn[headname], lprobs_batch_by_hn[headname], ents_batch_by_hn[headname] = \
                process_logits_by_batch(logits_batch_by_hn[headname], available_actions_batch)
        else:
            acts_batch_by_hn[headname], lprobs_batch_by_hn[headname], ents_batch_by_hn[headname] = \
                process_logits_by_batch(logits_batch_by_hn[headname])
    return acts_batch_by_hn, lprobs_batch_by_hn, ents_batch_by_hn


def process_logits_by_batch(logits_batch, available_actions_batch=None):
    """
    :param logits_batch: Tensor{B}
    :param available_actions_batch: List{B}[List{.}[func_ids]]
    :return:
        actions_batch: Tensor{B}
        lprobs_batch: Tensor{B}
        ents_batch: Tensor{B}
    """
    probs_batch = F.softmax(logits_batch, dim=1)
    probs_batch = probs_batch if available_actions_batch is None \
        else mask_action_probs(probs_batch, available_actions_batch)
    actions_batch = probs_batch.multinomial(1)
    lprobs_batch = F.log_softmax(logits_batch, dim=1).gather(1, actions_batch)
    ents_batch = -(lprobs_batch * probs_batch).sum(1)
    actions_batch = actions_batch.squeeze(1)
    lprobs_batch = lprobs_batch.squeeze(1)
    return actions_batch, lprobs_batch, ents_batch


def lookup_arg_names_batchwise(actions_batch):
    """
    :param actions_batch: Tensor{B}
    :return: List{B}[List{.}[arg_name]]
    """
    arg_names_batch = []
    for action in actions_batch:
        action = action.item()
        arg_names_batch.append(lookup_arg_names_by_id(action))
    return arg_names_batch


def lookup_arg_names_by_id(func_id):
    """
    :param func_id: int
    :return: argument_names: List{.}[arg_name]
    """
    func = FUNCTIONS[func_id]
    required_args = FUNCTION_TYPES[func.function_type]
    argument_names = []
    for arg in required_args:
        argument_names.append(arg.name)
    return argument_names


class TrainingLoop(BaseTrainingLoop):
    def __init__(self, args):
        frame_stack = False
        rollout_cache = RolloutCache('rewardss', 'valuess', 'entropiess', 'log_probss', 'maskss')
        super(TrainingLoop, self).__init__(args, frame_stack, rollout_cache)
        self.hxs = [torch.zeros(512).to(self.device) for _ in range(args.workers)]
        self.cxs = [torch.zeros(512).to(self.device) for _ in range(args.workers)]

    def _setup_model(self):
        return LSTMModelSC2(
            266, self.args.batch_norm  # TODO, don't hardcode number of features
        ).to(self.device)

    def _forward_step(self):
        available_actions_batch = [state['available_actions'] for state in self.states_batch]
        actions_batch, values_batch, log_probs_batch, entropies_batch, self.hxs, self.cxs = action_train(
            self.model,
            self._states_to_device(self.states_batch),
            self.hxs,
            self.cxs,
            available_actions_batch
        )

        # Make sure we didn't sample an invalid action (prob supposed to be 0)
        for i in range(len(available_actions_batch)):
            func_id = actions_batch[i]['func_id']
            available_actions = set(available_actions_batch[i])
            if func_id not in available_actions:
                actions_batch[i] = {'func_id': 0}  # no op
                print('warning: sampled an invalid action', func_id, available_actions)

        states_batch, rewards_unclipped, dones, infos = self.envs.step(actions_batch)

        for i in range(self.args.workers):
            if dones[i]:
                self.cxs[i] = torch.zeros_like(self.cxs[i])
                self.hxs[i] = torch.zeros_like(self.hxs[i])

        self.states_batch = states_batch
        rewards = torch.tensor([max(min(reward, 1), -1) for reward in rewards_unclipped]).to(self.device)
        masks = (1. - torch.from_numpy(np.array(dones, dtype=np.float32))).to(self.device)
        self.rollout_cache.append({
            'rewardss': rewards,
            'valuess': values_batch,
            'entropiess': entropies_batch,
            'log_probss': log_probs_batch,
            'maskss': masks
        })
        return rewards_unclipped, dones, infos

    def _unroll(self):
        return unroll(self.rollout_cache, self.args.gamma, self.args.tau)

    def _weight_losses(self, loss_dict):
        return loss_dict['policy_loss'] + 0.5 * loss_dict['value_loss']

    def _after_backwards(self):
        super(TrainingLoop, self)._after_backwards()
        self.cxs = [cx.detach() for cx in self.cxs]
        self.hxs = [hx.detach() for hx in self.hxs]

    def _setup_envs(self):
        if args.workers > 1:
            return SC2SubprocEnv(
                [make_sc2_env(args.env, args.seed + i) for i in range(args.workers)]
            )
        else:
            return SC2SingleEnv(
                [make_sc2_env(args.env, args.seed + i) for i in range(args.workers)]
            )

    def _states_to_device(self, states_batch):
        float_states = []
        binary_states = []
        for state_dict in states_batch:
            float_states.append(state_dict['float_state'])
            binary_states.append(state_dict['binary_state'])
        # float_states = np.array(float_states, dtype=np.float32)
        # binary_states = np.array(binary_states, dtype=np.uint8)
        # float_states = torch.from_numpy(float_states).to(self.device)
        # binary_states = torch.from_numpy(binary_states).to(self.device)
        return torch.cat(
            [torch.stack(float_states).to(self.device), torch.stack(binary_states).to(self.device).float()],
            dim=1
        )


if __name__ == '__main__':
    # from absl import flags
    # from absl.flags import Flag
    # FLAGS = flags.FLAGS
    # FLAGS(['sc2_lstm.py'])
    # FLAGS['sc2_run_config'].parse('Linux')
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = base_parser()
    parser.add_argument('--name', default='sc2_lstm', help='logdir/tensorboard name')
    args = parser.parse_args()
    TrainingLoop(args).run()
