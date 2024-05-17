import datetime
import time
import timeit

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import save_img, min_max_norm, print_model_parm_nums, StandardizeData
from .data_tools import get_dataloader
from .losses import (
    WeightedDataLoss,
    WeightedMSGradLoss,
)
from .networks import UNet


class G2_MonoDepth:
    def __init__(self, cf, rank):
        self.configs = cf
        self.rank = rank
        self.network = UNet(rezero=True).cuda().train()
        if rank == 0:
            print_model_parm_nums(self.network)  # the number of parameters
        self.network = DDP(
            self.network, device_ids=[self.rank], static_graph=True
        )  # Use DistributedDataParallel:
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.network.parameters(),
                    "initial_lr": cf.lr,
                }
            ],
            lr=cf.lr,
            weight_decay=cf.wd,
        )  # create optimizers
        self.network = torch.compile(self.network)  # pytorch 2.0
        # dataloader and datasampler
        self.loader, self.sampler = get_dataloader(
            cf.rgbd_dirs,
            cf.hole_dirs,
            cf.batch_size,
            cf.sizes,
            self.rank,
            cf.num_workers,
        )
        self.iteration_num = len(self.loader)
        # learning rate scheduler: cosine decay
        self.scheduler = CosineAnnealingLR(
            self.optimizer, self.iteration_num * cf.epochs
        )
        # use amp
        self.scaler = GradScaler() if cf.amp else None
        self.start_epoch = 0
        # resume train
        if cf.checkpoint is not None:
            if self.rank == 0:
                print("resume training...")
            # load everything
            map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
            checkpoint = torch.load(cf.checkpoint, map_location=map_location)
            self.start_epoch = checkpoint["epoch"]
            self.network.module.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.reg_function = WeightedDataLoss()
        self.sta_tool = StandardizeData()
        self.grad_function = WeightedMSGradLoss()

    def optimize_one_iteration(self, rgb, gt, raw):
        hole_gt = torch.where(gt == 0, torch.zeros_like(gt), torch.ones_like(gt))
        hole_raw = torch.where(raw == 0, torch.zeros_like(raw), torch.ones_like(raw))

        with autocast(enabled=(self.scaler is not None)):
            # loss in absolute domain
            depth = self.network(rgb, raw, hole_raw)
            loss_adepth = self.reg_function(depth, gt, hole_raw)
            # loss in relative domain
            sta_depth, sta_gt = self.sta_tool(depth, gt, hole_gt)
            loss_rdepth = self.reg_function(sta_depth, sta_gt, hole_gt)
            # sobel grad
            loss_rgrad = self.grad_function(sta_depth, sta_gt, hole_gt)

            loss = loss_adepth + loss_rdepth + 0.5 * loss_rgrad

        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return loss, depth

    def feedback_module(self, elapsed, iter_step, loss, summary, global_step):
        print(
            "Elapsed:[%s]|batch:[%d/%d]|loss:%.4f"
            % (elapsed, iter_step, self.iteration_num, float(loss))
        )
        # log loss
        summary.add_scalar("loss/loss", loss, global_step=global_step)

    @staticmethod
    def save_imgs(rgb, gt, raw, pred, log_dir, epoch, iter_step):
        # make dir
        epoch_dir = log_dir / ("epoch_" + str(epoch))
        epoch_dir.mkdir(exist_ok=True)
        file_last = f"_{epoch}_{iter_step}.png"
        # save the images:
        save_img(rgb[0], epoch_dir / ("rgb" + file_last))
        save_img(gt[0], epoch_dir / ("gt" + file_last))
        save_img(raw[0], epoch_dir / ("raw" + file_last))
        save_img(pred[0], epoch_dir / ("pred" + file_last))
        save_img(min_max_norm(gt[0]), epoch_dir / ("norm_gt" + file_last))
        save_img(min_max_norm(pred[0]), epoch_dir / ("norm_pred" + file_last))

    def train(self, cf):
        if self.rank == 0:
            # model/log rgbd_dirs
            model_dir, log_dir = cf.save_dir / "models", cf.save_dir / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            # tensorboard summarywriter
            summary = SummaryWriter(str(log_dir / "tensorboard"))
            # create a global time counter
            global_time = time.time()
            print("Starting the training process ... ")
        # epoch start
        global_step = self.start_epoch * self.iteration_num
        for epoch in range(self.start_epoch + 1, cf.epochs + 1):
            # set epoch in Distributed samplers
            self.sampler.set_epoch(epoch)
            # record time at the start of epoch
            if self.rank == 0:
                start = timeit.default_timer()
                print(f"\nEpoch: [{epoch}/{cf.epochs}]")
            for i, (rgb, gt, raw) in enumerate(self.loader, start=1):
                # get data
                rgb = rgb.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                raw = raw.cuda(non_blocking=True)
                # optimizing
                loss, pred = self.optimize_one_iteration(rgb, gt, raw)
                self.scheduler.step()  # learning rate decay
                # logging
                global_step += 1
                # log a loss feedback
                if self.rank == 0 and ((i % cf.feedback_iteration == 0) or (i == 1)):
                    with torch.no_grad():
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        # log and print
                        self.feedback_module(elapsed, i, loss, summary, global_step)
                        # # save intermediate results
                        self.save_imgs(rgb, gt, raw, pred, log_dir, epoch, i)
            # logging checkpoint
            if self.rank == 0:
                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))
                if epoch % cf.checkpoint_epoch == 0 or epoch == cf.epochs:
                    save_file = model_dir / f"epoch_{epoch}.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "network": self.network.module.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "scaler": self.scaler.state_dict(),
                        },
                        save_file,
                    )

        if self.rank == 0:
            print("Training completed ...")
