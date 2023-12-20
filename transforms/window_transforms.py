import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.transforms import v2

# NOTE: torchvision.transforms.v2 apply same transform and random seed to the input in a single call, no matter it is C*H*W or N*C*H*W.

@dataclass
class WindowTransformConfig:
    img_h: int = 128  # input image height of the model
    img_w: int = 128  # input image width of the model
    window_hflip_p: float = 0.0
    window_affine_rotate: float = 0.0  # unit: degrees
    window_affine_shift: tuple[float, float] = (0.0, 0.0)  # horizontal and vertical shift is btw +-img_w*a and +-img_h*b
    window_scale_range: tuple[float, float] = (1.0, 1.0)  # (min, max) scale factor


class WindowTransform(nn.Module):
    def __init__(self, config: WindowTransformConfig):
        super().__init__()
        self.config = config
        transform_list = []
        if config.window_hflip_p > 0:
            transform_list.append(v2.RandomHorizontalFlip(config.window_hflip_p))
        if config.window_affine_rotate != 0 or config.window_affine_shift != (0.0, 0.0) or config.window_scale_range != (1.0, 1.0):
            transform_list.append(v2.RandomAffine(degrees=config.window_affine_rotate,
                                                  translate=config.window_affine_shift,
                                                  scale=config.window_scale_range))
        # FUTURE: transform related to crop and face_location to simulate the unstable face detection
        self.transform = v2.Compose(transform_list) if len(transform_list) > 0 else None


    def forward(self, nir_imgs: torch.Tensor, ppg_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transform is not None:
            nir_imgs, ppg_labels = self.transform(nir_imgs, ppg_labels)
        return nir_imgs, ppg_labels
