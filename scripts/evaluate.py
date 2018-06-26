from __future__ import division

import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from src.environments.atari import atari_env
from src.utils import read_config, from_numpy
from src.models.cnn import SimpleCNN
from src.models.lstm import LSTMModelAtari
from src.models.attention import AttentionCNN
import time
from torch.nn import functional as F
import numpy as np
from src.utils import base_parser

#undo_logger_setup()
ROOT_DIR = os.path.abspath(os.pardir)



class Agent:
    def __init__(self, device):
        self.device = device

    def to(self, device):
        return self

    def __call__(self, model, state):
        pass

    def reset(self):
        return


class StatelessAgent(Agent):
    def __call__(self, model, state):
        state = state.unsqueeze(0)
        model.eval()
        with torch.no_grad():
            value, logit = model(state)
            prob = F.softmax(logit, dim=1)
            action = torch.argmax(prob, 1).item()
        return action


class LSTMAgent(Agent):
    def __init__(self, device):
        super(LSTMAgent, self).__init__(device)
        self.cxs = [torch.zeros(1, 512)]
        self.hxs = [torch.zeros(1, 512)]
        self.to(device)

    def to(self, device):
        self.cxs = [self.cxs[0].to(device)]
        self.hxs = [self.hxs[0].to(device)]
        return self

    def __call__(self, model, state):
        state = state.unsqueeze(0)
        model.eval()
        with torch.no_grad():
            _, logit, self.hxs, self.cxs = model(state, self.hxs, self.cxs)
            prob = F.softmax(logit, dim=1)
            action = torch.argmax(prob, 1).item()
        return action

    def reset(self):
        self.cxs[0] = torch.zeros_like(self.cxs[0])
        self.hxs[0] = torch.zeros_like(self.hxs[0])


models_by_name = {
    'SimpleCNN': SimpleCNN,
    'LSTMModelAtari': LSTMModelAtari,
    'AttentionCNN': AttentionCNN
}

framestack_by_name = {
    'SimpleCNN': True,
    'LSTMModelAtari': False,
    'AttentionCNN': True
}

get_agent_by_name = {
    'SimpleCNN': StatelessAgent,
    'LSTMModelAtari': LSTMAgent,
    'AttentionCNN': StatelessAgent
}

def main(args):
    args = parser.parse_args()
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]

    env = atari_env(args.env, env_conf, args, framestack_by_name[args.model_name])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_mean = 0
    best_std = 0
    mean = 0
    std = 0
    for i in range(3):
        model_file = os.path.join(args.log_dir, args.experiment_name, args.env, 'model{}.dat'.format(i + 1))
        model = models_by_name[args.model_name](
            env.observation_space.shape[0],
            env.action_space.n,
            args.batch_norm
        ).to(device)
        model.load_state_dict(torch.load(model_file))
        agent = get_agent_by_name[args.model_name](device)

        test_count = 0
        eps_len = 0
        episode_rewards = []
        for i_episode in range(args.num_episodes):
            env.seed(args.seed + i_episode)
            agent.reset()
            state = from_numpy(env.reset(), device)
            eps_len += 2
            episode_reward = 0
            while True:
                eps_len += 1
                if args.render:
                    if i_episode % args.render_freq == 0:
                        env.render()

                action = agent(model, state)
                state, reward, done, info = env.step(action)
                state = from_numpy(state, device)
                episode_reward += reward

                if done and not info:
                    agent.reset()
                    state = from_numpy(env.reset(), device)
                    eps_len += 2
                elif info:
                    test_count += 1
                    episode_rewards.append(episode_reward)
                    print('model', str(i + 1), '/', '3', 'test', test_count, '/', args.num_episodes)
                    eps_len = 0
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        print('model' + str(i + 1), 'mean', mean, 'std', std)
        if mean > best_mean:
            best_mean = mean
            best_std = std
    print('final mean', best_mean, 'final std', best_std)


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = base_parser()
    parser.add_argument('--model-name', default='SimpleCNN', help='SimpleCNN / SimpleLSTM / AttentionCNN')
    parser.add_argument('--experiment-name', default='cnn')
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=30,
        metavar='NE',
        help='how many episodes in evaluation (default: 30)')
    parser.add_argument(
        '--render',
        dest='render',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--render-freq',
        type=int,
        default=1,
        metavar='RF',
        help='Frequency to watch rendered game play'
    )

    args = parser.parse_args()
    main(args)
