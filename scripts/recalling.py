from __future__ import division

import argparse
import os
from time import time

import numpy as np
import torch
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from src.environments.atari import make_atari_env
from src.models.a2c_recalling import RecallingModel
from src.utils import from_numpy, RolloutCache, read_config
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from copy import copy

ROOT_DIR = os.path.abspath(os.pardir)
os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(description='A2C')
parser.add_argument(
    '--lr',
    type=float,
    default=7e-4,
    metavar='LR',
    help='learning rate (default: 7e-4)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=32,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='PongNoFrameskip-v4',
    metavar='ENV',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--load-model-dir',
    default=os.path.join(ROOT_DIR, 'trained_models/'),
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--log-dir', default=os.path.join(ROOT_DIR, 'logs/'), metavar='LG', help='folder to save logs')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    help='number of game steps for rollout'
)
parser.add_argument(
    '--query-breadth',
    type=int,
    default=50,
    help='number of memories to consider'
)
parser.add_argument(
    '--ltm-max-len',
    type=int,
    default=2048,
    help='how many unique memories can be stored'
)
parser.add_argument(
    '--name',
    default='a2c_recalling',
    help='name of the experiment'
)


def unroll(cache, gamma, tau):
    returns = cache['valuess'][-1].data.squeeze()
    policy_loss = 0.
    value_loss = 0.
    gae = torch.zeros_like(returns)

    rollout_len = len(cache['rewardss']) - 1
    for i in reversed(range(rollout_len)):
        rewards = cache['rewardss'][i]
        masks = cache['maskss'][i]
        values = cache['valuess'][i].squeeze()
        values_t1 = cache['valuess'][i + 1].squeeze()
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
    return {
        'value_loss': torch.mean(value_loss),
        'policy_loss': torch.mean(policy_loss)
    }


def action_train(model, states, hxs, cxs, values, actions):
    model.train()
    actions_1hot = torch.zeros(values.size(0), nb_action).float().to(device)
    actions_1hot.scatter_(1, actions, 1.)
    env_embs = model('embed', states, actions_1hot, values)

    values, logit, hxs, cxs, inds, weights = model('recall', env_embs, hxs, cxs)
    prob = F.softmax(logit, dim=1)
    log_probs = F.log_softmax(logit, dim=1)
    entropies = -(log_probs * prob).sum(1)
    actions = prob.multinomial(1)
    log_probs = log_probs.gather(1, actions)
    return actions, values, log_probs.squeeze(1), entropies, hxs, cxs, env_embs, inds, weights


def action_test(model, states, hxs, cxs):
    model.eval()
    with torch.no_grad():
        value, logit, hxs, cxs = model(states, hxs, cxs)
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
    return action


if __name__ == '__main__':
    args = parser.parse_args()
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    envs = SubprocVecEnv([make_atari_env(args.env, env_conf, args, args.seed + i) for i in range(args.workers)])
    dones = [True for i in range(args.workers)]
    values = torch.zeros(args.workers, 1).to(device)
    episode_reward_buff = torch.zeros(args.workers)

    nb_input_chan = envs.observation_space.shape[0]
    nb_action = envs.action_space.n
    model = RecallingModel(nb_input_chan, nb_action, args.query_breadth, args.ltm_max_len).to(device)
    cxs = [torch.zeros(1, 512).to(device) for i in range(args.workers)]
    hxs = [torch.zeros(1, 512).to(device) for i in range(args.workers)]
    actions = torch.zeros(args.workers, 1).to(device).long()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-5, alpha=0.99)

    rollout_cache = RolloutCache(
        'rewardss',
        'valuess',
        'entropiess',
        'log_probss',
        'maskss',
        'env_embss',
        'indss',
        'weightss',
        'cxss'
    )
    states = from_numpy(envs.reset(), device)
    step_count = 0
    writer = SummaryWriter(os.path.join(args.log_dir, args.name, args.env))
    t = time()
    while True:
        [cx.detach_() for cx in cxs]
        [hx.detach_() for hx in hxs]
        values.detach_()

        for step in range(args.num_steps + 1):
            step_count += args.workers
            rollout_cache['cxss'].append(copy(cxs))  # Shallow copy of cell states before they're modified
            actions, values, log_probs, entropies, hxs, cxs, env_embs, inds, weights =\
                action_train(model, states, hxs, cxs, values, actions)
            states, rewards_unclipped, dones, _ = envs.step(actions.squeeze(1).cpu().numpy())

            for i in range(args.workers):
                if dones[i]:
                    cxs[i] = torch.zeros(1, 512).to(device)
                    hxs[i] = torch.zeros(1, 512).to(device)

            states = from_numpy(states, device)
            rewards = torch.tensor([max(min(reward, 1), -1) for reward in rewards_unclipped]).to(device)
            masks = (1. - torch.from_numpy(np.array(dones, dtype=np.float32))).unsqueeze(1).to(device)
            rollout_cache['rewardss'].append(rewards)
            rollout_cache['valuess'].append(values)
            rollout_cache['entropiess'].append(entropies)
            rollout_cache['log_probss'].append(log_probs)
            rollout_cache['maskss'].append(masks)
            rollout_cache['env_embss'].append(env_embs)
            rollout_cache['indss'].append(inds)
            rollout_cache['weightss'].append(weights)

            episode_reward_buff += torch.tensor(rewards_unclipped).float()
            ep_rewards = []
            for ep_reward, done in zip(episode_reward_buff, dones):
                if done:
                    ep_rewards.append(ep_reward.item())
                    ep_reward.zero_()
            if ep_rewards:
                reward_result = np.mean(ep_rewards)
                writer.add_scalar('reward', reward_result, step_count)
                print('step', step_count, 'reward', reward_result, 'sps', step_count / (time() - t))

        losses = unroll(rollout_cache, args.gamma, args.tau)
        total_loss = losses['policy_loss'] + 0.5 * losses['value_loss']

        total_loss.backward()
        if step_count % 100 == 0:
            writer.add_scalar('loss/policy_loss', losses['policy_loss'].item() / args.workers, step_count)
            writer.add_scalar('loss/value_loss', losses['value_loss'].item() / args.workers, step_count)
            writer.add_scalar('macro_loss/total_loss', total_loss.item() / args.workers, step_count)
            for p_name, param in model.named_parameters():
                p_name = p_name.replace('.', '/')
                writer.add_scalar(p_name, torch.norm(param).item(), step_count)
                if param.grad is not None:
                    writer.add_scalar(p_name + '.grad', torch.norm(param.grad).item(), step_count)

        optimizer.step()
        optimizer.zero_grad()

        # Long term memory housekeeping
        # model.ltm.detach()
        model.eval()
        for i in reversed(range(len(rollout_cache['rewardss']) - 1)):
            # Key
            old_cxs = rollout_cache['cxss'][i]
            # Value
            en_embs = rollout_cache['env_embss'][i]
            # Buffer
            wgts = rollout_cache['weightss'][i]
            ixs = rollout_cache['indss'][i]

            for j in range(args.workers):
                with torch.no_grad():
                    normed_cxs = model.cx_layer_norm(old_cxs[j].data)
                model.ltm.append(normed_cxs, en_embs[j].data)
                model.ltm.update_buff(ixs[j], wgts[j].data)
        model.train()
        rollout_cache.clear()
        t = time()
