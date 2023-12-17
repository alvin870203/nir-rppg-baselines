import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.transforms import v2

# NOTE: torchvision.transforms.v2 apply same transform and random seed to the input in a single call, no matter it is C*H*W or N*C*H*W.

@dataclass
class WindowTransformConfig:
    img_size_h: int = 128
    img_size_w: int = 128
    window_hflip_p: float = 0.


class WindowTransform(nn.Module):
    def __init__(self, config: WindowTransformConfig):
        super().__init__()
        self.config = config
        transform_list = []
        if config.window_hflip_p > 0:
            transform_list.append(v2.RandomHorizontalFlip(config.window_hflip_p))
        # FUTURE: transform related to crop and face_location to simulate the unstable face detection
        self.transform = v2.Compose(transform_list) if len(transform_list) > 0 else None


    def forward(self, nir_imgs: torch.Tensor, ppg_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transform is not None:
            nir_imgs, ppg_labels = self.transform(nir_imgs, ppg_labels)
        return nir_imgs, ppg_labels
