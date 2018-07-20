import abc


class NormalizerBase(abc.ABC):
    @abc.abstractmethod
    def __call__(self, item):
        raise NotImplementedError


class Clip(NormalizerBase):
    def __init__(self, floor=-1, ceil=1):
        self.floor = floor
        self.ceil = ceil

    def __call__(self, item):
        return float(max(min(item, self.ceil), self.floor))


class Scale(NormalizerBase):
    def __init__(self, coefficient):
        self.coefficient = coefficient

    def __call__(self, item):
        return self.coefficient * item
