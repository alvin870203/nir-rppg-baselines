from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

# NOTE: torchvision.transforms.v2 apply same transform and random seed to the input in a single call, no matter it is C*H*W or N*C*H*W.

@dataclass
class FrameTransformConfig:
    img_h: int = 128  # input image height of the model
    img_w: int = 128  # input width width of the model


class FrameTransform(nn.Module):
    def __init__(self, config: FrameTransformConfig):
        super().__init__()
        self.config = config
        self.transform = []


    def forward(self, nir_img: torch.Tensor, ppg_label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Is there anything to do here?
        return nir_img, ppg_label
