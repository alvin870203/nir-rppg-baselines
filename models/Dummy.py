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


    def forward(self, nir_imgs: torch.Tensor, ppg_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # nir_imgs: (batch_size, window_size, 1, img_size_h, img_size_w)
        # ppg_labels: (batch_size, window_size, 1)

        device = nir_imgs.device
        logits = self.model(nir_imgs[:, 1] - nir_imgs[:, 0])
        if ppg_labels is not None:
            # if we are given some desired targets also calculate the loss
            if self.config.out_dim == 1:
                # for differentiated ppg regression
                labels = ppg_labels[:, 1] - ppg_labels[:, 0]
                loss = F.mse_loss(logits, labels)
            else:
                # for seq2seq
                labels = ppg_labels.squeeze()
                loss = F.mse_loss(logits, labels)
        else:
            loss = None

        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups.
        # Example:
        #   To make any parameters that is 2D will be weight decayed, otherwise no.
        #   i.e. all weight tensors in matmuls + embeddings decay, all biases and batchnorms don't.
        #   decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        #   nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        decay_params = [p for n, p in param_dict.items()]
        nodecay_params = []
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
