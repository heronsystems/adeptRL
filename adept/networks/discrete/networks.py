from .._base import InputNetwork


class DiscreteIdentity(InputNetwork):
    def __init__(self, nb_in_channel):
        super().__init__()
        self._nb_output_channel = nb_in_channel

    @classmethod
    def from_args(cls, nb_in_channel, args):
        return cls(nb_in_channel)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, x):
        return x
