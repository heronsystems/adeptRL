import torch


class DeepMindReturnScaler:
    """
    Scale returns as in R2D2.
    https://openreview.net/pdf?id=r1lyTjAqYX
    """

    def __init__(self, scale):
        self.scale = scale

    def calc_scale(self, x):
        return (
            torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + self.scale * x
        )

    def calc_inverse_scale(self, x):
        sign = torch.sign(x)
        sqrt = torch.sqrt(1 + 4 * self.scale * (torch.abs(x) + 1 + self.scale))
        return sign * ((((sqrt - 1) / (2 * self.scale)) ** 2) - 1)
