import torch
from src.models._base import StateEmbedder, ForwardModel, InverseModel
from torch import nn as nn

from src.utils import norm_col_init


class CuriosityModel(torch.nn.Module):
    def __init__(self, nb_input, nb_action):
        super(CuriosityModel, self).__init__()
        self.nb_input = nb_input
        self.nb_action = nb_action
        self.state_embedder = StateEmbedder(nb_input)
        self.actor_critic = ActorCritic(nb_action)
        self.forward_model = ForwardModel(nb_action)
        self.inverse_model = InverseModel(nb_action)
        self.train()

    def forward(self, mode, *args):
        if mode == 'embed':
            x, = args
            return self.state_embedder(x)
        elif mode == 'ac':
            inputs, hx, cx = args
            return self.actor_critic(inputs, hx, cx)
        elif mode == 'fm':
            state_emb, action = args
            return self.forward_model(state_emb, action)
        elif mode == 'im':
            state_emb, next_state_emb = args
            return self.inverse_model(state_emb, next_state_emb)
        else:
            raise Exception('Invalid mode:', mode)


class ActorCritic(torch.nn.Module):
    def __init__(self, nb_action):
        super(ActorCritic, self).__init__()
        self.lstm = nn.LSTMCell(800, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, nb_action)
        self.init()

    def init(self):
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, xs, hxs, cxs):
        hx, cx = None, None
        hs, cs = [], []
        for x, hx, cx in zip(xs, hxs, cxs):
            h, c = self.lstm(x.unsqueeze(0), (hx, cx))
            hs.append(h)
            cs.append(c)

        if hx is None or cx is None:
            raise ValueError('Length must be >= 1: len(hxs) = {}, len(cxs): {}'.format(len(hxs), len(cxs)))

        # Rebatch the results
        result = torch.cat(hs, dim=0)

        return self.critic_linear(result), self.actor_linear(result), hs, cs
