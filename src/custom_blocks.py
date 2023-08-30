from torch import Tensor
from torch.nn import Conv2d, Sequential, Module, Parameter, BatchNorm2d, LeakyReLU
import torch


class BottleNeck(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            rezero: bool,
    ) -> None:
        super(BottleNeck, self).__init__()
        self.rezero = rezero
        mid_channels = int(in_channels / 4.0)
        if self.rezero:
            self.left = Sequential(
                LeakyReLU(0.2, inplace=True),
                Conv2d(in_channels, mid_channels, 1, 1),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, out_channels, 1, 1),
            )
            self.alpha = Parameter(torch.tensor(0.0))
        else:
            self.left = Sequential(
                BatchNorm2d(in_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(in_channels, mid_channels, 1, 1),
                BatchNorm2d(mid_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1),
                BatchNorm2d(mid_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, out_channels, 1, 1),
            )

        if in_channels == out_channels:
            self.right = None
        else:
            self.right = Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        factor = 1.0
        if self.rezero:
            factor = self.alpha
        x1 = factor * self.left(x) + (x if self.right is None else self.right(x))
        return x1
