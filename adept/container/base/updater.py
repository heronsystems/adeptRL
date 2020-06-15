import abc


class Updater(metaclass=abc.ABCMeta):
    def __init__(self, optimizer, network, grad_norm_clip):
        self.optimizer = optimizer
        self.network = network
        self.grad_norm_clip = grad_norm_clip

    def step(self, loss):
        raise NotImplementedError
