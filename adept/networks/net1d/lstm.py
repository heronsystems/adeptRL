import torch
from torch.nn import LSTMCell

from adept.modules import LSTMCellLayerNorm
from adept.networks.net1d.submodule_1d import SubModule1D


class LSTM(SubModule1D):
    args = {
        'lstm_normalize': True,
        'lstm_nb_hidden': 512
    }

    def __init__(self, input_shape, id, normalize, nb_hidden):
        super().__init__(input_shape, id)
        self._nb_hidden = nb_hidden

        if normalize:
            self.lstm = LSTMCellLayerNorm(
                input_shape[0], nb_hidden
            )
        else:
            self.lstm = LSTMCell(input_shape[0], nb_hidden)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    @classmethod
    def from_args(cls, args, input_shape, id):
        return cls(input_shape, id, args.lstm_normalize, args.lstm_nb_hidden)

    @property
    def _output_shape(self):
        return (self._nb_hidden, )

    def _forward(self, xs, internals, **kwargs):
        hxs = torch.stack(internals['hx'])
        cxs = torch.stack(internals['cx'])
        hxs, cxs = self.lstm(xs, (hxs, cxs))

        return (
            hxs, {
                'hx': list(torch.unbind(hxs, dim=0)),
                'cx': list(torch.unbind(cxs, dim=0))
            }
        )

    def _new_internals(self):
        return {
            'hx': torch.zeros(self._nb_hidden),
            'cx': torch.zeros(self._nb_hidden)
        }
