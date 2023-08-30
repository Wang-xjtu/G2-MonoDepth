import torch
import numpy as np
import random
from PIL import Image


def rgb_read(filename):
    data = Image.open(filename)
    rgb = (np.array(data) / 255.).astype(np.float32)
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
    data.close()
    return rgb


def depth_read(filename):
    data = Image.open(filename)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    depth = (np.array(data) / 65535.).astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(0)
    data.close()
    return depth


class RGBPReader(object):
    def __init__(self):
        self.rel = False

    @staticmethod
    def read_rgbraw(rgb_path, raw_path):
        rgb = rgb_read(rgb_path)
        if '_0%' in raw_path:
            raw = torch.zeros((1, 1, rgb.shape[1], rgb.shape[2]))
        else:
            raw = depth_read(raw_path).unsqueeze(0)
            if raw.shape[2] != rgb.shape[1]:
                raw = torch.nn.functional.interpolate(raw, (rgb.shape[1], rgb.shape[2]), mode='nearest')

        return rgb.unsqueeze(0).to(torch.float32), raw.to(torch.float32)

    def read_data(self, rgb_path, raw_path):
        rgb, raw = self.read_rgbraw(rgb_path, raw_path)
        hole_raw = torch.ones_like(raw)
        hole_raw[raw == 0] = 0
        if torch.sum(hole_raw) == 0:
            self.rel = True
        else:
            self.rel = False
        return rgb, raw, hole_raw

    @staticmethod
    def __min_max_norm__(depth):
        max_value = np.max(depth)
        min_value = np.min(depth)
        norm_depth = (depth - min_value) / (max_value - min_value + 1e-6)
        return norm_depth

    def adjust_domain(self, pred):
        pred = pred.squeeze().to('cpu').numpy()
        if self.rel:
            pred = self.__min_max_norm__(pred)
        pred = np.clip(pred * 65535., 0, 65535).astype(np.int32)
        return pred


class DepthEvaluation(object):
    @staticmethod
    def rmse(depth, ground_truth):
        residual = ((depth - ground_truth) / 256.) ** 2
        residual[ground_truth == 0.] = 0.
        value = np.sqrt(np.sum(residual) / np.count_nonzero(ground_truth))
        return value

    @staticmethod
    def absRel(depth, ground_truth):
        diff = depth - ground_truth
        diff[ground_truth == 0] = 0.
        rel = np.sum(abs(diff) / (ground_truth + 1e-6)) / np.count_nonzero(ground_truth)
        return rel

    @staticmethod
    def srmse(depth, ground_truth):
        mask = np.ones_like(depth)
        mask[ground_truth == 0] = 0
        number_valid = np.sum(mask) + 1e-6
        # sta depth
        mean_d = np.sum(depth * mask) / number_valid
        std_d = np.sum(np.abs(depth - mean_d) * mask) / number_valid
        sta_dep = (depth - mean_d) / (std_d + 1e-6)
        # sta gt
        mean_gt = np.sum(ground_truth * mask) / number_valid
        std_gt = np.sum(np.abs(ground_truth - mean_gt) * mask) / number_valid
        sta_gt = (ground_truth - mean_gt) / (std_gt + 1e-6)

        sta_rmse = np.sqrt(np.sum(mask * (sta_dep - sta_gt) ** 2) / number_valid)
        return sta_rmse

    @staticmethod
    def oe(depth, ground_truth):
        ordinal_error = OrdinalError()
        value = ordinal_error.error(depth, ground_truth)
        return value


class OrdinalError(object):
    def __init__(self, t=0.01):
        super(OrdinalError, self).__init__()
        self.t = t
        self.eps = 1e-8

    def ordinal_label(self, sampled_a, sampled_b):
        label = np.zeros_like(sampled_a)
        ratio = sampled_a / (sampled_b + self.eps)
        label[ratio >= (1 + self.t)] = 1
        label[ratio <= (1 / (1 + self.t))] = -1
        return label

    def sampling(self, pred, gt):
        nonzero_pixels_index = np.nonzero(gt)
        pixel_num = nonzero_pixels_index[0].shape[0]
        # select_num = int(0.5 * pixel_num)
        select_num = int(0.3 * pixel_num)

        select_pixels_index_a = [i for i in range(pixel_num)]
        select_pixels_index_b = [i for i in range(pixel_num)]
        random.seed(1)
        random.shuffle(select_pixels_index_a)
        point_a_index_list = select_pixels_index_a[:select_num]
        random.seed(2)
        random.shuffle(select_pixels_index_b)
        point_b_index_list = select_pixels_index_b[:select_num]

        point_a_x = nonzero_pixels_index[0][point_a_index_list]
        point_a_y = nonzero_pixels_index[1][point_a_index_list]
        point_b_x = nonzero_pixels_index[0][point_b_index_list]
        point_b_y = nonzero_pixels_index[1][point_b_index_list]

        pred_sampled_a = pred[point_a_x, point_a_y]
        gt_sampled_a = gt[point_a_x, point_a_y]
        pred_sampled_b = pred[point_b_x, point_b_y]
        gt_sampled_b = gt[point_b_x, point_b_y]

        return pred_sampled_a, pred_sampled_b, gt_sampled_a, gt_sampled_b

    def error(self, pred, gt):
        pred_sampled_a, pred_sampled_b, gt_sampled_a, gt_sampled_b = self.sampling(pred, gt)
        label_pred = self.ordinal_label(pred_sampled_a, pred_sampled_b)
        label_gt = self.ordinal_label(gt_sampled_a, gt_sampled_b)
        loss = np.where(label_pred == label_gt, np.zeros_like(label_pred), np.ones_like(label_pred))
        loss = np.mean(loss)
        return loss
