from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

# NOTE: torchvision.transforms.v2 apply same transform and random seed to the input in a single call, no matter it is C*H*W or N*C*H*W.

@dataclass
class FrameTransformConfig:
    img_size_h: int = 128
    img_size_w: int = 128
    bbox_scale: float = 1.0
    frame_shift: float = 0.0  # augmented bbox center_{x or y} = center_{x or y} + bbox_{w or h} * random.uniform(-max, max))
    frame_shift_p: float = 0.2  # probability of applying random bbox shift
    frame_scale_range: tuple[float, float] = (1.0, 1.0)  # augmented bbox_scale = bbox_scale * random.uniform(min, max)
    frame_scale_p: float = 0.2  # probability of applying random bbox scale


class FrameTransform(nn.Module):
    def __init__(self, config: FrameTransformConfig):
        super().__init__()
        self.config = config
        self.transform = []
        if config.frame_shift != 0.0:
            self.transform.append("random_bbox_shift")
        if config.frame_scale_range != (1.0, 1.0):
            self.transform.append("random_bbox_scale")


    def forward(self, nir_img: torch.Tensor, ppg_label: torch.Tensor,
                face_location: None | np.ndarray = None) -> tuple[torch.Tensor, torch.Tensor]:
        orig_nir_img_h, orig_nir_img_w = nir_img.shape[1:]
        left, top, right, bottom = (0, 0, orig_nir_img_w, orig_nir_img_h) if face_location is None else face_location
        center_x, center_y = (left + right) // 2, (top + bottom) // 2
        face_aspect_ratio = (right - left) / (bottom - top)
        bbox_aspect_ratio = self.config.img_size_w / self.config.img_size_h
        if face_aspect_ratio > bbox_aspect_ratio:
            bbox_w = right - left
            bbox_h = bbox_w / bbox_aspect_ratio
        else:
            bbox_h = bottom - top
            bbox_w = bbox_h * bbox_aspect_ratio

        if "random_bbox_shift" in self.transform:
            # TODO: Same or similar random shift for all frames in a window or not?
            frame_shift_x = np.random.choice([0.0, np.random.uniform(-self.config.frame_shift, self.config.frame_shift)],
                                             p=[1 - self.config.frame_shift_p, self.config.frame_shift_p])
            frame_shift_y = np.random.choice([0.0, np.random.uniform(-self.config.frame_shift, self.config.frame_shift)],
                                             p=[1 - self.config.frame_shift_p, self.config.frame_shift_p])
            center_x = center_x + bbox_w * frame_shift_x
            center_y = center_y + bbox_h * frame_shift_y

        base_bbox_scale = 1.0 if face_location is None else self.config.bbox_scale
        if "random_bbox_scale" in self.transform:
            # TODO: Same or similar random scale for all frames in a window or not?
            frame_scale = np.random.choice([1.0, np.random.uniform(*self.config.frame_scale_range)],
                                           p=[1 - self.config.frame_scale_p, self.config.frame_scale_p])
            bbox_scale = base_bbox_scale * frame_scale
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
        if (orig_nir_img_h, orig_nir_img_w) != (self.config.img_size_h, self.config.img_size_w):
            nir_img = v2.functional.resize(nir_img, (self.config.img_size_h, self.config.img_size_w), antialias=True)
        return nir_img, ppg_label
