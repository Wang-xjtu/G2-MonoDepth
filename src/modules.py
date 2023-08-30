from torch import Tensor
from torch.nn import Module, Conv2d, Sequential, UpsamplingNearest2d
from .custom_blocks import BottleNeck
import torch


class StackedBottleNeck(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            rezero: bool,
    ) -> None:
        super(StackedBottleNeck, self).__init__()

        self.block = Sequential(
            BottleNeck(in_channels, out_channels, rezero),
            BottleNeck(out_channels, out_channels, rezero),
            BottleNeck(out_channels, out_channels, rezero),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.block(x)
        return x1


class FirstModule(Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            rezero: bool,
    ):
        super(FirstModule, self).__init__()
        self.block = Sequential(
            Conv2d(in_channel, out_channel, 3, 2, 1),
            StackedBottleNeck(out_channel, out_channel, rezero),
            UpsamplingNearest2d(scale_factor=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = torch.cat((x, self.block(x)), dim=1)
        return x1


class UNetModule(Module):
    def __init__(
            self,
            mid_module: Module,
            in_channel: int,
            mid_channel: int,
            rezero: bool,
    ):
        super(UNetModule, self).__init__()
        self.block = Sequential(
            Conv2d(in_channel, mid_channel, 3, 2, 1),
            StackedBottleNeck(mid_channel, mid_channel, rezero),
            mid_module,
            StackedBottleNeck(2 * mid_channel, in_channel, rezero),
            UpsamplingNearest2d(scale_factor=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = torch.cat((x, self.block(x)), dim=1)
        return x1
