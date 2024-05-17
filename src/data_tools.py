from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as trans
import torchvision.transforms.functional as TF
import os
import pickle


def rgb_read(filename: Path) -> Tensor:
    data = Image.open(filename)
    rgb = (np.array(data) / 255.0).astype(np.float32)
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
    data.close()
    return rgb


def depth_read(filename: Path) -> Tensor:
    data = Image.open(filename)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    depth = (np.array(data) / 65535.0).astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(0)
    data.close()
    return depth


def hole_read(filename: Path) -> Tensor:
    data = Image.open(filename)
    hole = (np.array(data) / 255.0).astype(np.float32)
    hole = torch.from_numpy(hole).unsqueeze(0)
    data.close()
    return hole


class RandomResizedCropRGBD(trans.RandomResizedCrop):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        hole = torch.where(
            img[3, :, :] == 0,
            torch.zeros_like(img[3, :, :]),
            torch.ones_like(img[3, :, :]),
        )
        img = torch.cat([img, hole.unsqueeze(0)], dim=0)
        img = TF.resized_crop(
            img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias
        )
        hole = torch.where(
            img[4, :, :] < 1,
            torch.zeros_like(img[4, :, :]),
            torch.ones_like(img[4, :, :]),
        )
        img = torch.cat([img[:3, :, :], (img[3, :, :] * hole).unsqueeze(0)], dim=0)
        return img

        # rgb = TF.resized_crop(img[:3, :, :], i, j, h, w, self.size, TF.InterpolationMode.BILINEAR, self.antialias)
        # gt = TF.resized_crop(img[3, :, :].unsqueeze(0), i, j, h, w, self.size, TF.InterpolationMode.NEAREST,
        #                      self.antialias)
        # return torch.cat([rgb, gt], dim=0)


class RandomDepth(torch.nn.Module):
    def __init__(self, factor):
        super(RandomDepth, self).__init__()
        self.factor = factor

    def forward(self, depth):
        """
        Args:
            depth (Tensor): depth map to be scaled and shifted.

        Returns:
            Tensor: scaled and shifted depth.
        """
        max_depth = torch.max(depth)
        scale_factor = np.random.uniform(
            1 - self.factor, np.clip(1 + self.factor, a_min=1, a_max=1 / max_depth)
        )
        depth = depth * scale_factor
        return depth


class TransformUtils(object):
    def __init__(self, size):
        self._rgbgt_transform = trans.Compose(
            [
                trans.RandomCrop(size),
                RandomResizedCropRGBD(
                    size, (0.64, 1.0), antialias=True
                ),  # augmentation introduced in next paper
                trans.RandomHorizontalFlip(0.5),
            ]
        )
        self._rgb_transform = trans.ColorJitter(0.2, 0.2, 0.2)
        self._gt_transform = RandomDepth(0.2)  # augmenation introduced in next paper
        self._hole_transform = trans.Compose(
            [
                trans.RandomCrop(size),
                trans.RandomAffine(
                    degrees=180,
                    translate=(0.5, 0.5),
                    scale=(0.5, 4.0),
                    shear=60,
                    fill=1.0,
                ),
                trans.RandomHorizontalFlip(0.5),
                trans.RandomVerticalFlip(0.5),
            ]
        )
        self.pro_list = [0.2, 0.4, 0.6]

    def trans_rgbgt(self, rgb: Tensor, gt: Tensor):
        # together transforming
        rgbgt = self._rgbgt_transform(torch.cat([rgb, gt], dim=0))
        rgb = rgbgt[:3, :, :]
        gt = rgbgt[3, :, :].unsqueeze(0)
        # individual transforming
        rgb = self._rgb_transform(rgb)
        gt = self._gt_transform(gt)
        return rgb, gt

    def _holes(self, raw, hole_ls, p=0.8):
        if np.random.uniform(0.0, 1.0) < p:
            hole = hole_read(hole_ls[np.random.randint(0, len(hole_ls))])
            hole = self._hole_transform(hole)
            raw = raw * hole
        return raw

    def trans_raw(self, gt, hole_gt, hole_ls):
        raw = gt.clone()
        # Random sampling
        p_blur = 0.5
        random_factor = np.random.uniform(0.0, 1.0)
        if random_factor < self.pro_list[0]:
            zero_rate = 0.0  # depth recovery
            p_blur = 1.0
        elif random_factor < self.pro_list[1]:
            zero_rate = np.random.uniform(0.0, 0.9)  # not very sparse depth completion
        elif random_factor < self.pro_list[2]:
            zero_rate = np.random.uniform(0.9, 1.0)  # very sparse depth completion
        else:
            zero_rate = 1.0  # depth estimation
        # add noise blur
        if zero_rate < 1:
            raw = self._noiseblur(raw, hole_gt, p_blur=p_blur)
            raw = self._holes(raw, hole_ls)
        raw = self._sample(raw, zero_rate)
        return raw

    @staticmethod
    def _sample(raw, zero_rate) -> Tensor:
        data_shape = raw.shape
        if zero_rate == 0.0:
            return raw
        elif zero_rate == 1.0:
            return torch.zeros(data_shape)
        else:
            random_point = torch.ones(data_shape).uniform_(0.0, 1.0)
            random_point[random_point <= zero_rate] = 0.0
            random_point[random_point > zero_rate] = 1.0
            return raw * random_point

    def _noiseblur(self, raw, hole_gt, p_noise=0.5, p_blur=0.5):
        raw_shape = raw.shape
        # add noise
        if np.random.uniform(0.0, 1.0) < p_noise:
            gaussian_noise = torch.ones(raw_shape).normal_(
                0, np.random.uniform(0.01, 0.1)
            )
            gaussian_noise = self._sample(gaussian_noise, np.random.uniform(0.0, 1.0))
            raw = raw + gaussian_noise * hole_gt
        # add blur
        if np.random.uniform(0.0, 1.0) < p_blur:
            sample_factor = 2 ** (np.random.randint(1, 5))
            blur_trans = trans.Compose(
                [
                    trans.Resize(
                        (raw_shape[1] // sample_factor, raw_shape[2] // sample_factor),
                        interpolation=TF.InterpolationMode.NEAREST,
                        antialias=True,
                    ),
                    trans.Resize(
                        (raw_shape[1], raw_shape[2]),
                        interpolation=TF.InterpolationMode.NEAREST,
                        antialias=True,
                    ),
                ]
            )
            raw = blur_trans(raw)
        return torch.clamp(raw, 0.0, 1.0)


class RGBDHDataset(Dataset):
    def __init__(self, rgbd_dir, hole_dir, size):
        super(RGBDHDataset, self).__init__()
        self.transforms = TransformUtils(size)
        rgbd_list_dir = "data_list/" + str(rgbd_dir.name)
        os.makedirs(rgbd_list_dir, exist_ok=True)

        # rgbd list
        if os.path.exists(rgbd_list_dir + "/rgb_ls.pkl"):
            rgb_list = open(rgbd_list_dir + "/rgb_ls.pkl", "rb")
            self.rgb_ls = pickle.load(rgb_list)
            rgb_list.close()
            depth_list = open(rgbd_list_dir + "/depth_ls.pkl", "rb")
            self.depth_ls = pickle.load(depth_list)
            depth_list.close()
        else:
            self.rgb_ls, self.depth_ls = self.__getrgbd__(rgbd_dir)
            rgb_list = open(rgbd_list_dir + "/rgb_ls.pkl", "wb")
            pickle.dump(self.rgb_ls, rgb_list)
            rgb_list.close()
            depth_list = open(rgbd_list_dir + "/depth_ls.pkl", "wb")
            pickle.dump(self.depth_ls, depth_list)
            depth_list.close()

        # hole list
        if os.path.exists("data_list/hole_ls.pkl"):
            hole_list = open("data_list/hole_ls.pkl", "rb")
            self.hole_ls = pickle.load(hole_list)
            hole_list.close()
        else:
            self.hole_ls = self.__gethole__(hole_dir)
            hole_list = open("data_list/hole_ls.pkl", "wb")
            pickle.dump(self.hole_ls, hole_list)
            hole_list.close()

    @staticmethod
    def __getrgbd__(rgbd_dir):
        rgb_ls = []
        depth_ls = []
        for file in rgbd_dir.rglob("*.png"):
            str_file = str(file)
            if "/rgb/" in str_file:
                rgb_ls.append(file)
                depth_file = str_file.replace("/rgb/", "/depth/", 1)
                depth_ls.append(Path(depth_file))

        return rgb_ls, depth_ls

    @staticmethod
    def __gethole__(hole_dir):
        hole = []
        for file in hole_dir.rglob("*.png"):
            hole.append(file)
        return hole

    def __len__(self):
        assert len(self.rgb_ls) == len(
            self.depth_ls
        ), f"The number of RGB and gen_depth is unpaired"
        return len(self.rgb_ls)

    def __getitem__(self, item):
        # names of RGB and depth should be paired
        rgb_path = self.rgb_ls[item]
        depth_path = self.depth_ls[item]
        assert (
            rgb_path.name[:-4] == depth_path.name[:-4]
        ), f"The RGB {str(self.rgb_ls[item])} and gen_depth {str(self.depth_ls[item])} is unpaired"
        rgb = rgb_read(rgb_path)
        gt = depth_read(depth_path)
        rgb, gt = self.transforms.trans_rgbgt(rgb, gt)
        hole_gt = torch.where(gt == 0, torch.zeros_like(gt), torch.ones_like(gt))
        raw = self.transforms.trans_raw(gt, hole_gt, self.hole_ls)
        return rgb, gt, raw


def get_dataloader(rgbd_dirs, hole_dirs, batch_size, sizes, rank, num_workers):
    rgbd_dataset = RGBDHDataset(rgbd_dirs, hole_dirs, sizes)
    if rank == 0:
        print(f"Loaded the rgbd dataset with: {len(rgbd_dataset)} images...\n")

    # initialize dataloaders
    rgbd_sampler = DistributedSampler(rgbd_dataset)
    rgbd_data = DataLoader(
        rgbd_dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=rgbd_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return rgbd_data, rgbd_sampler
