from src.src_main import G2_MonoDepth
from src.utils import DDPutils
from config import Configs
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # gpus

# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def DDP_main(rank, world_size):
    cf = Configs(world_size)

    # DDP components
    DDPutils.setup(rank, world_size, 6003)
    if rank == 0:
        print(f"Selected arguments: {cf.__dict__}")

    trainer = G2_MonoDepth(cf, rank=rank)
    trainer.train(cf)
    DDPutils.cleanup()


if __name__ == "__main__":
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        DDPutils.run_demo(DDP_main, n_gpus)
