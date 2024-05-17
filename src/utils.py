import os
import cv2
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
import numpy as np


class StandardizeData(torch.nn.Module):
    def __init__(self):
        super(StandardizeData, self).__init__()
        self.fn = self.__masked_mean_robust_standardization__

    @staticmethod
    def __masked_mean_robust_standardization__(depth, mask, eps=1e-6):
        mask_num = torch.sum(mask, dim=(1, 2, 3))
        mask_num[mask_num == 0] = eps
        depth_mean = (torch.sum(depth * mask, dim=(1, 2, 3)) / mask_num).view(
            depth.shape[0], 1, 1, 1
        )
        depth_std = (
            torch.sum(torch.abs((depth - depth_mean) * mask), dim=(1, 2, 3)) / mask_num
        )
        return depth_mean, depth_std.view(depth.shape[0], 1, 1, 1) + eps

    def forward(self, depth, gt, mask_hole):
        t_d, s_d = self.fn(depth, mask_hole)
        t_g, s_g = self.fn(gt, mask_hole)
        sta_depth = (depth - t_d) / s_d
        sta_gt = (gt - t_g) / s_g
        return sta_depth, sta_gt


def min_max_norm(depth):  # (1,c,h,w)
    max_value = torch.max(depth)
    min_value = torch.min(depth)
    norm_depth = (depth - min_value) / (max_value - min_value)
    return norm_depth


def save_img(
    data: Tensor,
    file_name: Path,
) -> None:
    data = data.squeeze()

    if data.ndim == 3:
        ndarr = (
            data.mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        )
        ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    else:
        ndarr = data.squeeze(0).mul(255).clamp_(0, 255).to("cpu", torch.uint8).numpy()
        ndarr = cv2.applyColorMap(ndarr, cv2.COLORMAP_JET)
    cv2.imwrite(str(file_name), ndarr)


class DDPutils(object):
    @staticmethod
    def setup(rank, world_size, port):
        # rank: the serial number  of GPU
        # world_size: the number of GPUs

        # environment setting: localhost is the ip of local，6005 is interface number
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        # initializing
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    @staticmethod
    def run_demo(demo_fn, world_size):
        # demo_fn: name of your main function, like "train"
        # world_size: the number of GPUs
        mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of params: %.2fM" % (total / 1e6))
