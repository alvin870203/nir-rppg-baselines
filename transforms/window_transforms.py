import random
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
from torchvision.transforms import v2

# NOTE: torchvision.transforms.v2 apply same transform and random seed to the input in a single call, no matter it is C*H*W or N*C*H*W.

@dataclass
class WindowTransformConfig:
    img_h: int = 128  # input image height of the model
    img_w: int = 128  # input image width of the model
    bbox_scale: float = 1.0
    window_shift: float = 0.0  # augmented bbox center_{x or y} = center_{x or y} + bbox_{w or h} * random.uniform(-max, max))
    window_shift_p: float = 0.0  # probability of applying random bbox shift
    window_scale_range: tuple[float, float] = (1.0, 1.0)  # augmented bbox_scale = bbox_scale * random.uniform(min, max)
    window_scale_p: float = 0.0  # probability of applying random bbox scale
    window_hflip_p: float = 0.0


class WindowTransform(nn.Module):
    def __init__(self, config: WindowTransformConfig):
        super().__init__()
        self.config = config
        self.transform = []
        if config.window_shift != 0.0:
            self.transform.append("random_bbox_shift")
        if config.window_scale_range != (1.0, 1.0):
            self.transform.append("random_bbox_scale")


    def forward(self, nir_imgs: torch.Tensor, ppg_labels: torch.Tensor, face_locations: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        # Input:
        #   nir_imgs: (window_size, 1, H, W)
        #   ppg_labels: (window_size, 1)
        #   face_locations: (window_size, 4) or (window_size, 0=None)
        # Output:
        #   nir_imgs_transformed: (window_size, 1, img_h, img_w)
        #   ppg_labels_transformed: (window_size, 1)

        # Same random transform for all frames in a window
        bbox_shift_x = np.random.choice([0.0, np.random.uniform(-self.config.window_shift, self.config.window_shift)],
                                        p=[1 - self.config.window_shift_p, self.config.window_shift_p])
        bbox_shift_y = np.random.choice([0.0, np.random.uniform(-self.config.window_shift, self.config.window_shift)],
                                        p=[1 - self.config.window_shift_p, self.config.window_shift_p])
        bbox_scale_factor = np.random.choice([1.0, np.random.uniform(*self.config.window_scale_range)],
                                             p=[1 - self.config.window_scale_p, self.config.window_scale_p])
        nir_imgs_transformed = []
        ppg_labels_transformed = []
        for nir_img, ppg_label, face_location in zip(nir_imgs, ppg_labels, face_locations):
            orig_nir_img_h, orig_nir_img_w = nir_img.shape[1:]
            left, top, right, bottom = (0, 0, orig_nir_img_w, orig_nir_img_h) if face_location is None else face_location
            center_x, center_y = (left + right) // 2, (top + bottom) // 2
            face_aspect_ratio = (right - left) / (bottom - top)
            bbox_aspect_ratio = self.config.img_w / self.config.img_h
            if face_aspect_ratio > bbox_aspect_ratio:
                bbox_w = right - left
                bbox_h = bbox_w / bbox_aspect_ratio
            else:
                bbox_h = bottom - top
                bbox_w = bbox_h * bbox_aspect_ratio

            if "random_bbox_shift" in self.transform:
                center_x = center_x + bbox_w * bbox_shift_x
                center_y = center_y + bbox_h * bbox_shift_y

            base_bbox_scale = 1.0 if face_location is None else self.config.bbox_scale
            if "random_bbox_scale" in self.transform:
                bbox_scale = base_bbox_scale * bbox_scale_factor
            else:
                bbox_scale = base_bbox_scale
            bbox_w *= bbox_scale
            bbox_h *= bbox_scale

            left = int(center_x - bbox_w // 2)
            top = int(center_y - bbox_h // 2)
            right = int(center_x + bbox_w // 2)
            bottom = int(center_y + bbox_h // 2)
            left_pad = max(0, -left)
            top_pad = max(0, -top)
            right_pad = max(0, right - orig_nir_img_w)
            bottom_pad = max(0, bottom - orig_nir_img_h)
            nir_img = v2.functional.pad(nir_img, (left_pad, top_pad, right_pad, bottom_pad), fill=0, padding_mode='constant')
            left += left_pad
            top += top_pad
            right += left_pad
            bottom += top_pad
            height = bottom - top
            width = right - left
            nir_img = v2.functional.crop(nir_img, top, left, height, width)
            if (orig_nir_img_h, orig_nir_img_w) != (self.config.img_h, self.config.img_w):
                nir_img = v2.functional.resize(nir_img, (self.config.img_h, self.config.img_w), antialias=True)

            nir_imgs_transformed.append(nir_img)
            ppg_labels_transformed.append(ppg_label)

        nir_imgs_transformed = torch.stack(nir_imgs_transformed)
        ppg_labels_transformed = torch.stack(ppg_labels_transformed)

        return nir_imgs_transformed, ppg_labels_transformed
