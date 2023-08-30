import torch
from torch import Tensor
from torch.nn import Module, Conv2d, init, Sequential
from .modules import FirstModule, UNetModule, StackedBottleNeck

chan = [512, 512, 512, 512, 256, 128, 64]


class UNet(Module):
    def __init__(self, layer_num: int = 7, rezero: bool = True):
        super(UNet, self).__init__()

        layer = FirstModule(chan[1], chan[0], rezero)
        for i in range(2, layer_num):
            layer = UNetModule(layer, chan[i], chan[i - 1], rezero)

        self.block = Sequential(
            Conv2d(5, chan[i], 3, 1, 1),
            StackedBottleNeck(chan[i], chan[i], rezero),
            layer,
            StackedBottleNeck(2 * chan[i], chan[i], rezero),
            Conv2d(chan[i], 1, 3, 1, 1),
        )

        # initializing
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, a=0.2)
                init.zeros_(m.bias)

    def forward(
            self,
            rgb: Tensor,
            point: Tensor,
            hole_point: Tensor,
    ) -> Tensor:
        x1 = torch.cat((rgb, point, hole_point), dim=1)
        depth = self.block(x1)
        return depth
