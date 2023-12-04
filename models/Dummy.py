import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class DummyConfig:
    img_size_h: int = 640
    img_size_w: int = 640
    out_dim: int = 1
    bias: bool = True


class Dummy(nn.Module):
    def __init__(self, config: DummyConfig):
        super().__init__()
        self.config = config

        # A simple CNN
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=config.bias),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, bias=config.bias),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (config.img_size_h // 4) * (config.img_size_w // 4), config.out_dim, bias=config.bias)
        )


    def forward(self, nir_imgs: torch.Tensor, ppg_signals: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # nir_imgs: (batch_size, window_size, 1, img_size_h, img_size_w)
        # ppg_signals: (batch_size, window_size, 1)

        device = nir_imgs.device
        logits = self.model(nir_imgs[:, 1] - nir_imgs[:, 0])
        if ppg_signals is not None:
            # if we are given some desired targets also calculate the loss
            if self.config.out_dim == 1:
                # for differentiated ppg regression
                labels = ppg_signals[:, 1] - ppg_signals[:, 0]
                loss = F.mse_loss(logits, labels)
            else:
                pass  # TODO: for seq2seq

        return logits, loss