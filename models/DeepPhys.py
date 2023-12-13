import math
import inspect
from dataclasses import dataclass
import numpy as np
import scipy.signal as sig

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
    dropout: float = 0.50


class DeepPhys(nn.Module):
    def __init__(self, config: DeepPhysConfig, train_dataset: Dataset):
        super().__init__()
        self.config = config

        self.rppg_labels_diff_std = self.get_rppg_labels_diff_std(train_dataset)

        # Implementation by terbed/Deep-rPPG
        # Appearance stream
        self.a_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn1 = nn.BatchNorm2d(32)

        self.a_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn2 = nn.BatchNorm2d(32)
        self.a_d1 = nn.Dropout2d(p=config.dropout)

        self.a_softconv1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=config.bias)
        self.a_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.a_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.a_bn3 = nn.BatchNorm2d(64)

        self.a_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.a_bn4 = nn.BatchNorm2d(64)
        self.a_d2 = nn.Dropout2d(p=config.dropout)
        self.a_softconv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=config.bias)

        # Motion stream
        self.m_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.m_bn1 = nn.BatchNorm2d(32)
        self.m_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.m_bn2 = nn.BatchNorm2d(32)
        self.d1 = nn.Dropout2d(p=config.dropout)

        self.m_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.m_bn3 = nn.BatchNorm2d(64)
        self.m_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=config.bias)
        self.m_bn4 = nn.BatchNorm2d(64)
        self.d2 = nn.Dropout2d(p=config.dropout)
        self.m_avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected blocks
        self.d3 = nn.Dropout(p=config.dropout / 2)
        self.fully1 = nn.Linear(in_features=64 * (self.config.img_size_h // 4) * (self.config.img_size_w // 4),
                                out_features=128, bias=config.bias)
        self.fully2 = nn.Linear(in_features=128, out_features=config.out_dim, bias=config.bias)


    def get_rppg_labels_diff_std(self, train_dataset: Dataset) -> float:
        rppg_labels_diff = []
        for _, ppg_labels in train_dataset:
            rppg_labels_diff.append((ppg_labels[1] - ppg_labels[0]))
        rppg_labels_diff = np.concatenate(rppg_labels_diff)
        # np.save("rppg_signals_diff.npy", rppg_labels_diff)  # For inspecting the distribution by test_MR-NIRP_Indoor_statistics.ipynb
        return rppg_labels_diff.std()


    def forward(self, nir_imgs: torch.Tensor, ppg_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # nir_imgs: (batch_size, window_size, 1, img_size_h, img_size_w)
        # ppg_labels: (batch_size, window_size, 1)

        device = nir_imgs.device

        # Implementation by terbed/Deep-rPPG
        A = nir_imgs[:, 0]
        M = torch.div(nir_imgs[:, 1] - nir_imgs[:, 0], nir_imgs[:, 1] + nir_imgs[:, 0] + 1e-8)  # +1e-8 to avoid division by zero

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


        if ppg_labels is not None:
            # if we are given some desired targets also calculate the loss
            # for differentiated ppg regression
            labels = (ppg_labels[:, 1] - ppg_labels[:, 0]) / self.rppg_labels_diff_std
            loss = F.mse_loss(logits, labels)
        else:
            loss = None

        return logits, loss


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def _init_weights(self, module):
        # FUTURE: implement this & more investigations around better init etc
        # for example:
        #   if isinstance(module, nn.Linear):
        #       torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #       if module.bias is not None:
        #           torch.nn.init.zeros_(module.bias)
        #   elif ...
        raise NotImplementedError()


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # FUTURE: implement this & init from pretrained
        raise NotImplementedError()


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


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of ,e.g., A100 bfloat16 peak FLOPS """
        # FUTURE: implement this
        raise NotImplementedError()


    @torch.no_grad()
    def generate(self, test_dataset: Dataset, device: str) -> dict[str, dict[str, np.ndarray]]:
        """
        Predict on test set, usually with non-overlapping windows of 30s
        and calculate PPG figure, BPM error, and other frequency stuffs.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        result = {}  # {subject_name: {start_idxs, losses, ppg_predicts, ppg_labels, spectrum_predicts, spectrum_labels, bpm_predicts, bpm_labels}}
        for subject_name, (start_idxs, bpm_labels, spectrum_labels) in test_dataset.test_data.items():
            losses = []
            ppg_predicts, ppg_labels = [], []
            spectrum_predicts = []
            bpm_predicts = []
            nir_imgs, ppg_label = test_dataset.data[subject_name]
            ppg_predict_first = ppg_label[0]
            for start_idx, bpm_label in zip(start_idxs, bpm_labels):
                # There's only (test_dataset.config.test_window_size - 1) windows in a test_window
                start_idx_t0, end_idx_t0 = start_idx, start_idx + test_dataset.config.test_window_size - 1
                start_idx_t1, end_idx_t1 = start_idx_t0 + 1, end_idx_t0 + 1
                nir_imgs_torch = torch.stack((torch.from_numpy(nir_imgs[start_idx_t0 : end_idx_t0]),
                                              torch.from_numpy(nir_imgs[start_idx_t1 : end_idx_t1])), dim=1).unsqueeze(2).float().to(device)
                ppg_labels_torch = torch.stack((torch.from_numpy(ppg_label[start_idx_t0 : end_idx_t0]),
                                                 torch.from_numpy(ppg_label[start_idx_t1 : end_idx_t1])), dim=1).unsqueeze(2).float().to(device)

                logits, loss = self(nir_imgs_torch, ppg_labels_torch)

                ppg_predict = torch.cumsum(logits.squeeze(), dim=0).cpu().numpy() * self.rppg_labels_diff_std
                ppg_predict = np.concatenate((np.zeros(1), ppg_predict)) + ppg_predict_first  # Add last ppg_predict or ppg_labels[start_idx_t0] as bias
                ppg_predict_first = ppg_predict[-1]
                assert len(ppg_predict) == test_dataset.config.test_window_size, f"len of predicted ppg is not the same as ground truth ppg!"
                ppg_predict_detrend = sig.detrend(ppg_predict)
                spectrum_predict = np.abs(np.fft.rfft(ppg_predict_detrend))
                freq = np.fft.rfftfreq(len(ppg_predict_detrend), d=1./test_dataset.config.video_fps)
                freq_range = np.logical_and(freq <= test_dataset.config.max_heart_rate / 60, freq >= test_dataset.config.min_heart_rate / 60)
                max_idx = np.argmax(spectrum_predict[freq_range])
                max_freq = freq[freq_range][max_idx]
                bpm_predict = max_freq * 60

                losses.append(loss.item())
                ppg_predicts.append(ppg_predict)
                ppg_labels.append(ppg_label[start_idx_t0 : end_idx_t1])
                spectrum_predicts.append(spectrum_predict)
                bpm_predicts.append(bpm_predict)

            result[subject_name] = {'start_idxs': start_idxs, 'losses': np.array(losses),
                                    'ppg_predicts': np.array(ppg_predicts), 'ppg_labels': np.array(ppg_label),
                                    'spectrum_predicts': np.array(spectrum_predicts), 'spectrum_labels': spectrum_labels,
                                    'bpm_predicts': np.array(bpm_predicts), 'bpm_labels': bpm_labels}

        return result
