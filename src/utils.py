from __future__ import division

import argparse
import gc
import json
import logging
import os
import tracemalloc

import numpy as np
import torch
import torch.multiprocessing as mp
from src.environments.atari import LazyFrames
from torch.autograd import Variable


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def update_shared_model(model, shared_model, is_gpu=False):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not is_gpu:
            return
        elif not is_gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def is_gpu(gpu_id):
    return gpu_id >= 0


def get_device(gpu_id):
    return torch.device("cuda:0") if gpu_id >= 0 else torch.device("cpu")


def from_numpy(arr, device):
    if isinstance(arr, LazyFrames):
        arr = arr._force()
    return Variable(torch.from_numpy(arr).float().to(device))


def isnan(tensor):
    # Gross: https://github.com/pytorch/pytorch/issues/4767
    return (tensor != tensor)


def hasnan(tensor):
    return isnan(tensor).any()


class RolloutCache(dict):
    def __init__(self, *keys):
        super(RolloutCache, self).__init__()
        for k in keys:
            self[k] = []

    def clear(self):
        for k in self.keys():
            self[k] = []
        return self

    def __len__(self):
        for k in self.keys():
            return len(self[k])

    def apply(self, fn):
        for k in self.keys():
            fn(self[k])

    def append(self, d):
        for k, v in d.items():
            self[k].append(v)


class Optional:
    def __init__(self, value=None):
        if value is None:
            self._value = []
        else:
            self._value = [value]

    def get_or_else(self, fn, *fn_args):
        if self.not_empty():
            return self._value[0]
        else:
            return fn(*fn_args)

    def map(self, fn):
        map(fn, self._value)

    def apply(self, fn):
        for v in self._value:
            fn(v)

    def is_empty(self):
        return self._value

    def not_empty(self):
        return not self._value

    def get(self):
        if self.is_empty():
            raise LookupError("Can't get an empty optional.")
        else:
            return self._value[0]


class LockingCounter(object):
    def __init__(self, initval=0):
        self.i = mp.Value('i', initval)
        self.lock = mp.Lock()

    def get_value_and_increment(self):
        with self.lock:
            v = self.i.value
            self.i.value += 1
            return v

    def increment(self, size=1):
        with self.lock:
            self.i.value += size

    def value(self):
        with self.lock:
            return self.i.value


class TMLeakFinder:
    def __init__(self):
        self.snapshot = None
        tracemalloc.start(10)

    def trace_print(self):
        snapshot2 = tracemalloc.take_snapshot()
        snapshot2 = snapshot2.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
            tracemalloc.Filter(False, tracemalloc.__file__)
        ))

        if self.snapshot is not None:
            print("================================== Begin Trace:")
            # stats = snapshot2.compare_to(self.snapshot, 'lineno', cumulative=True)
            stats = snapshot2.compare_to(self.snapshot, 'traceback')[:10]
            for stat in stats:
                print(stat)
        self.snapshot = snapshot2


class GCLeakFinder:
    def __init__(self):
        self.prev_len = None
        self.count_by_tsize = {}

    def compare(self):
        objs = {id(obj): obj for obj in gc.get_objects()}

        if self.prev_len is not None:
            print(len(objs) - self.prev_len)
        # for obj in objs:
        #     if isinstance(obj, torch.Tensor):
        #         if obj.size()

        self.prev_len = len(objs)


class CircularBuffer(object):
    def __init__(self, size):
        self.index = 0
        self.size = size
        self._data = []

    def append(self, value):
        if len(self._data) == self.size:
            self._data[self.index] = value
        else:
            self._data.append(value)
        self.index = (self.index + 1) % self.size

    def __getitem__(self, key):
        """get element by index like a regular array"""
        return self._data[key]

    def __repr__(self):
        """return string representation"""
        return self._data.__repr__() + ' (' + str(len(self._data))+' items)'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def base_parser():
    root_dir = os.path.abspath(os.pardir)

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
        default=os.path.join(root_dir, 'config.json'),
        metavar='EC',
        help='environment to crop and resize info (default: config.json)')
    parser.add_argument(
        '--save-max',
        default=True,
        metavar='SM',
        help='Save model on every test run high score matched or bested'
    )
    parser.add_argument(
        '--log-dir', default=os.path.join(root_dir, 'logs/'), metavar='LG', help='folder to save logs')
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
        '--max-steps',
        type=int,
        default=10e6
    )
    parser.add_argument('--batch-norm', default=True)
    parser.add_argument('--profile', type=str2bool, nargs='?', const=True, default=False)
    return parser
