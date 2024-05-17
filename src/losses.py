import torch
from torch import Tensor
from torch.nn import Module, Parameter, L1Loss
from torch.nn.functional import interpolate, conv2d


class Gradient2D(Module):
    def __init__(self):
        super(Gradient2D, self).__init__()
        kernel_x = torch.FloatTensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        kernel_y = torch.FloatTensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        self.weight_x = Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x: Tensor):
        grad_x = conv2d(x, self.weight_x)
        grad_y = conv2d(x, self.weight_y)
        return grad_x, grad_y


class WeightedDataLoss(Module):
    def __init__(
            self,
    ) -> None:
        super(WeightedDataLoss, self).__init__()
        self.loss_fun = L1Loss(reduction='sum')
        self.eps = 1e-6

    def forward(self, output: Tensor, target: Tensor, hole: Tensor) -> Tensor:
        assert output.shape == target.shape, "output size != target size"
        sampled_output = hole * output + (1.0 - hole) * target
        number_valid = torch.sum(hole) + self.eps
        loss = self.loss_fun(sampled_output, target)
        return loss / number_valid


class WeightedMSGradLoss(Module):
    def __init__(
            self,
            k: int = 4,
            sobel: bool = True,
    ):
        super(WeightedMSGradLoss, self).__init__()
        if sobel:
            self.grad_fun = Gradient2D().cuda()
        else:
            self.grad_fun = L1Loss(reduction='sum')
        self.eps = 1e-6
        self.k = k
        self.sobel = sobel

    def __gradient_loss__(self, residual: Tensor) -> Tensor:
        if self.sobel:
            loss_x, loss_y = self.grad_fun(residual)
            loss_x = torch.sum(torch.abs(loss_x))
            loss_y = torch.sum(torch.abs(loss_y))
        else:
            loss_x = self.grad_fun(residual[:, :, 1:, :], residual[:, :, :-1, :])
            loss_y = self.grad_fun(residual[:, :, :, 1:], residual[:, :, :, :-1])

        loss = loss_x + loss_y
        return loss

    def forward(self, output: Tensor, target: Tensor, hole_target: Tensor) -> Tensor:
        assert output.shape == target.shape, "sampled_output size != target size"
        sampled_output = hole_target * output + (1.0 - hole_target) * target
        residual = sampled_output - target
        number_valid = torch.sum(hole_target) + self.eps
        loss = 0.
        for i in range(self.k):
            scale_factor = 1.0 / (2 ** i)
            if i == 0:
                k_residual = residual
            else:
                k_residual = interpolate(residual, scale_factor=scale_factor, recompute_scale_factor=True)
            loss += self.__gradient_loss__(k_residual)
        return loss / number_valid
