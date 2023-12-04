import math
import inspect
from dataclasses import dataclass
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass
class DeepPhysConfig:
    img_size_h: int = 640
    img_size_w: int = 640
    out_dim: int = 1
    bias: bool = True


class DeepPhys(nn.Module):
    def __init__(self, config: DeepPhysConfig, train_dataset: Dataset):
        super().__init__()
        self.config = config

        self.rppg_signals_diff_std = self.get_rppg_signals_diff_std(train_dataset)

        self.mask1 = None
        self.mask2 = None

        # Implementation by terbed/Deep-rPPG
        # Appearance stream
        self.a_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn1 = nn.BatchNorm2d(32)

        self.a_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn2 = nn.BatchNorm2d(32)
        self.a_d1 = nn.Dropout2d(p=0.50)

        self.a_softconv1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=config.bias)
        self.a_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.a_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.a_bn3 = nn.BatchNorm2d(64)

        self.a_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.a_bn4 = nn.BatchNorm2d(64)
        self.a_d2 = nn.Dropout2d(p=0.50)
        self.a_softconv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=config.bias)

        # Motion stream
        self.m_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.m_bn1 = nn.BatchNorm2d(32)
        self.m_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.m_bn2 = nn.BatchNorm2d(32)
        self.d1 = nn.Dropout2d(p=0.50)

        self.m_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.m_bn3 = nn.BatchNorm2d(64)
        self.m_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.m_bn4 = nn.BatchNorm2d(64)
        self.d2 = nn.Dropout2d(p=0.50)
        self.m_avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected blocks
        self.d3 = nn.Dropout(p=0.25)
        self.fully1 = nn.Linear(in_features=64 * (self.config.img_size_h // 4) * (self.config.img_size_w // 4),
                                out_features=128, bias=config.bias)
        self.fully2 = nn.Linear(in_features=128, out_features=config.out_dim, bias=config.bias)


    def get_rppg_signals_diff_std(self, train_dataset: Dataset) -> float:
        rppg_signals_diff = []
        for _, ppg_signals in train_dataset.data:
            rppg_signals_diff.append(ppg_signals[1:] - ppg_signals[:-1])
        rppg_signals_diff = np.concatenate(rppg_signals_diff)
        return np.std(rppg_signals_diff)


    def forward(self, nir_imgs: torch.Tensor, ppg_signals: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # nir_imgs: (batch_size, window_size, 1, img_size_h, img_size_w)
        # ppg_signals: (batch_size, window_size, 1)

        device = nir_imgs.device

        # Implementation by terbed/Deep-rPPG
        A = nir_imgs[:, 0]
        M = nir_imgs[:, 1] - nir_imgs[:, 0]

        # (A) - Appearance stream -------------------------------------------------------------
        # First two convolution layer
        A = torch.tanh(self.a_bn1(self.a_conv1(A)))
        A = torch.tanh(self.a_bn2(self.a_conv2(A)))
        A = self.a_d1(A)

        # Calculating attention mask1 with softconv1
        mask1 = torch.sigmoid(self.a_softconv1(A))
        B, _, H, W = A.shape
        norm = 2 * torch.norm(mask1, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask1 = torch.div(mask1 * H * W, norm)
        self.mask1 = mask1

        # Pooling
        A = self.a_avgpool(A)
        # Last two convolution
        A = torch.tanh(self.a_bn3(self.a_conv3(A)))
        A = torch.tanh(self.a_bn4(self.a_conv4(A)))
        A = self.a_d2(A)

        # Calculating attention mask2 with softconv2
        mask2 = torch.sigmoid(self.a_softconv2(A))
        B, _, H, W = A.shape
        norm = 2 * torch.norm(mask2, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask2 = torch.div(mask2 * H * W, norm)
        self.mask2 = mask2

        # (M) - Motion stream --------------------------------------------------------------------
        M = torch.tanh(self.m_bn1(self.m_conv1(M)))
        M = self.m_bn2(self.m_conv2(M))
        M = torch.tanh(torch.mul(M, mask1))  # multiplying with attention mask1
        M = self.d1(M)  # Dropout layer 1
        # Pooling
        M = self.m_avgpool1(M)
        # Last convs
        M = torch.tanh(self.m_bn3(self.m_conv3(M)))
        M = self.m_bn4(self.m_conv4(M))
        M = torch.tanh(torch.mul(M, mask2))  # multiplying with attention mask2
        M = self.d2(M)  # Dropout layer 2
        M = self.m_avgpool2(M)

        # (F) - Fully connected part -------------------------------------------------------------
        # Flatten layer out
        out = torch.flatten(M, start_dim=1)  # start_dim=1 to handle batches
        out = self.d3(out)  # dropout layer 3
        out = torch.tanh(self.fully1(out))
        logits = self.fully2(out)


        if ppg_signals is not None:
            # if we are given some desired targets also calculate the loss
            if self.config.out_dim == 1:
                # for differentiated ppg regression
                labels = (ppg_signals[:, 1] - ppg_signals[:, 0]) / self.rppg_signals_diff_std
                loss = F.mse_loss(logits, labels)
            else:
                pass  # TODO: for seq2seq

        return logits, loss