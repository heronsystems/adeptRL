from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import norm_col_init, weights_init
from torch.nn import BatchNorm2d
from src.models._base import Identity, LSTMCellLayerNorm, Residual2D


class LSTMModelBase(torch.nn.Module):
    def __init__(self, nb_input, normalize):
        super(LSTMModelBase, self).__init__()
        self.conv1 = nn.Conv2d(nb_input, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        if normalize:
            self.lstm = LSTMCellLayerNorm(1024, 512)
        else:
            self.lstm = nn.LSTMCell(1024, 512)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(64)
            self.bn4 = BatchNorm2d(64)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn4 = Identity()

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        if not normalize:
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    def forward(self, ins, hxs, cxs):
        xs = F.relu(self.bn1(self.maxp1(self.conv1(ins))))
        xs = F.relu(self.bn2(self.maxp2(self.conv2(xs))))
        xs = F.relu(self.bn3(self.maxp3(self.conv3(xs))))
        xs = F.relu(self.bn4(self.maxp4(self.conv4(xs))))

        xs = xs.view(xs.size(0), -1)

        hxs = torch.stack(hxs)
        cxs = torch.stack(cxs)
        hxs, cxs = self.lstm(xs, (hxs, cxs))

        return hxs, list(torch.unbind(hxs, dim=0)), list(torch.unbind(cxs, dim=0))


class LSTMModelAtari(torch.nn.Module):
    def __init__(self, nb_input, nb_output, normalize):
        super(LSTMModelAtari, self).__init__()
        self.lstm_base = LSTMModelBase(nb_input, normalize)
        self.actor_linear = nn.Linear(512, nb_output)
        self.critic_linear = nn.Linear(512, 1)

        self.apply(weights_init)
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, ins, hxs, cxs):
        result, hxs, cxs = self.lstm_base(ins, hxs, cxs)
        return self.critic_linear(result), self.actor_linear(result), hxs, cxs


class LSTMModelSC2(torch.nn.Module):
    def __init__(self, nb_input, normalize):
        super(LSTMModelSC2, self).__init__()
        self.lstm_base = LSTMModelBase(nb_input, normalize)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = SC2Actor()
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, ins, hxs, cxs):
        result, hxs, cxs = self.lstm_base(ins, hxs, cxs)
        return self.critic_linear(result), self.actor_linear(result), hxs, cxs


class SC2Actor(torch.nn.Module):
    def __init__(self):
        super(SC2Actor, self).__init__()
        self.func_id = nn.Linear(512, 524)
        self.screen_x = nn.Linear(512, 80)
        self.screen_y = nn.Linear(512, 80)
        self.minimap_x = nn.Linear(512, 80)
        self.minimap_y = nn.Linear(512, 80)
        self.screen2_x = nn.Linear(512, 80)
        self.screen2_y = nn.Linear(512, 80)
        self.queued = nn.Linear(512, 2)
        self.control_group_act = nn.Linear(512, 4)
        self.control_group_id = nn.Linear(512, 10)
        self.select_point_act = nn.Linear(512, 4)
        self.select_add = nn.Linear(512, 2)
        self.select_unit_act = nn.Linear(512, 4)
        self.select_unit_id = nn.Linear(512, 500)
        self.select_worker = nn.Linear(512, 4)
        self.unload_id = nn.Linear(512, 500)
        self.build_queue_id = nn.Linear(512, 10)

        self.func_id.weight.data = norm_col_init(self.func_id.weight.data, 0.01)
        self.func_id.bias.data.fill_(0)
        self.screen_x.weight.data = norm_col_init(self.screen_x.weight.data, 0.01)
        self.screen_x.bias.data.fill_(0)
        self.screen_y.weight.data = norm_col_init(self.screen_y.weight.data, 0.01)
        self.screen_y.bias.data.fill_(0)
        self.minimap_x.weight.data = norm_col_init(self.minimap_x.weight.data, 0.01)
        self.minimap_x.bias.data.fill_(0)
        self.minimap_y.weight.data = norm_col_init(self.minimap_y.weight.data, 0.01)
        self.minimap_y.bias.data.fill_(0)
        self.screen2_x.weight.data = norm_col_init(self.screen2_x.weight.data, 0.01)
        self.screen2_x.bias.data.fill_(0)
        self.screen2_y.weight.data = norm_col_init(self.screen2_y.weight.data, 0.01)
        self.screen2_y.bias.data.fill_(0)
        self.queued.weight.data = norm_col_init(self.queued.weight.data, 0.01)
        self.queued.bias.data.fill_(0)
        self.control_group_act.weight.data = norm_col_init(self.control_group_act.weight.data, 0.01)
        self.control_group_act.bias.data.fill_(0)
        self.control_group_id.weight.data = norm_col_init(self.control_group_id.weight.data, 0.01)
        self.control_group_id.bias.data.fill_(0)
        self.select_point_act.weight.data = norm_col_init(self.select_point_act.weight.data, 0.01)
        self.select_point_act.bias.data.fill_(0)
        self.select_add.weight.data = norm_col_init(self.select_add.weight.data, 0.01)
        self.select_add.bias.data.fill_(0)
        self.select_unit_act.weight.data = norm_col_init(self.select_unit_act.weight.data, 0.01)
        self.select_unit_act.bias.data.fill_(0)
        self.select_unit_id.weight.data = norm_col_init(self.select_unit_id.weight.data, 0.01)
        self.select_unit_id.bias.data.fill_(0)
        self.select_worker.weight.data = norm_col_init(self.select_worker.weight.data, 0.01)
        self.select_worker.bias.data.fill_(0)
        self.unload_id.weight.data = norm_col_init(self.unload_id.weight.data, 0.01)
        self.unload_id.bias.data.fill_(0)
        self.build_queue_id.weight.data = norm_col_init(self.build_queue_id.weight.data, 0.01)
        self.build_queue_id.bias.data.fill_(0)

    def forward(self, x):
        return {
            'func_id': self.func_id(x),
            'screen_x': self.screen_x(x),
            'screen_y': self.screen_y(x),
            'minimap_x': self.minimap_x(x),
            'minimap_y': self.minimap_y(x),
            'screen2_x': self.screen2_x(x),
            'screen2_y': self.screen2_y(x),
            'queued': self.queued(x),
            'control_group_act': self.control_group_act(x),
            'control_group_id': self.control_group_id(x),
            'select_point_act': self.select_point_act(x),
            'select_add': self.select_add(x),
            'select_unit_act': self.select_unit_act(x),
            'select_unit_id': self.select_unit_id(x),
            'select_worker': self.select_worker(x),
            'unload_id': self.unload_id(x),
            'build_queue_id': self.build_queue_id(x)
        }


class ResNetLSTM(torch.nn.Module):
    def __init__(self, nb_input, nb_output, normalize):
        super(ResNetLSTM, self).__init__()
        self.res1 = Residual2D(nb_input, 32, stride=2)  # 40x40
        self.res2 = Residual2D(32, 32, stride=1)
        self.res3 = Residual2D(32, 32, stride=2)  # 20x20
        self.res4 = Residual2D(32, 32, stride=1)
        self.res5 = Residual2D(32, 8, stride=2)  # 10x10
        self.res6 = Residual2D(8, 8, stride=1)
        if normalize:
            self.lstm = LSTMCellLayerNorm(800, 512)
        else:
            self.lstm = nn.LSTMCell(800, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, nb_output)

        if not normalize:
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    def forward(self, ins, hxs, cxs):
        xs = F.relu(self.res1(ins))
        xs = F.relu(self.res2(xs))
        xs = F.relu(self.res3(xs))
        xs = F.relu(self.res4(xs))
        xs = F.relu(self.res5(xs))
        xs = F.relu(self.res6(xs))

        xs = xs.view(xs.size(0), -1)

        hxs = torch.stack(hxs)
        cxs = torch.stack(cxs)
        hxs, cxs = self.lstm(xs, (hxs, cxs))

        return hxs, list(torch.unbind(hxs, dim=0)), list(torch.unbind(cxs, dim=0))
