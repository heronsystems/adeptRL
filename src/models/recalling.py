import torch
from torch import nn as nn
from torch.nn import functional as F

from src.models._base import StateEmbedder, ForwardModel, InverseModel
from src.utils import norm_col_init


class RecallingModel(torch.nn.Module):
    def __init__(self, nb_input, nb_action, query_width):
        super(RecallingModel, self).__init__()
        self.nb_input = nb_input
        self.nb_action = nb_action
        self.query_width = query_width
        self.state_embedder = StateEmbedder(nb_input)
        self.actor_critic = ActorCritic(nb_action)
        self.forward_model = ForwardModel(nb_action)
        self.inverse_model = InverseModel(nb_action)
        self.experience_model = ExperienceModel(nb_action)
        self.train()

    def forward(self, mode, *args):
        if mode == 'embed':
            x, = args
            return self.state_embedder(x)
        elif mode == 'ac':
            inputs, hx, cx, dnd = args
            m = dnd.forward(cx)
            experience = self.experience_model(m)
            critic, actor, hx, cx, = self.actor_critic(inputs, hx, cx)
            return critic, actor, hx + experience, cx
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

    def forward(self, inputs, hx, cx):
        hx, cx = self.lstm(inputs, (hx, cx))
        return self.critic_linear(hx), self.actor_linear(hx), hx, cx


class ExperienceModel(torch.nn.Module):
    def __init__(self, nb_action):
        super(ExperienceModel, self).__init__()
        self.hidden = nn.Linear(512 + nb_action + 1, 512)
        self.out_linear = nn.Linear(512, 512)
        self.init()

    def init(self):
        self.hidden.weight.data = norm_col_init(self.hidden.weight.data, 0.01)
        self.hidden.bias.data.fill_(0)
        self.out_linear.weight.data = norm_col_init(self.out_linear.weight.data, 0.01)
        self.out_linear.bias.data.fill_(0)

    def forward(self, action_value_next_s):
        x = F.leaky_relu(self.hidden(action_value_next_s))
        x = self.out_linear(x)
        # x = x / torch.norm(x, 2).detach()
        return x
