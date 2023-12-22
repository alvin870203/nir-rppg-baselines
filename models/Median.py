import math
import inspect
from dataclasses import dataclass
import numpy as np
import scipy.signal as sig

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

@dataclass
class MedianConfig:
    out_dim: int = 1
    rppg_labels_diff_std: float = 1.0


class Median(nn.Module):
    def __init__(self, config: MedianConfig, train_dataset: Dataset):
        super().__init__()
        self.config = config

        # Get the median training dataset labels
        ppg_labels_all = []
        ppg_labels_diff_all = []

        for _, ppg_labels in train_dataset:
            ppg_labels_all.append(ppg_labels)
            ppg_labels_diff_all.append(torch.diff(ppg_labels, axis=0))

        ppg_labels_all = torch.stack(ppg_labels_all, axis=0)
        ppg_labels_diff_all = torch.stack(ppg_labels_diff_all, axis=0)

        self.ppg_labels_all_median = ppg_labels_all.median(axis=0).values.squeeze()
        self.ppg_labels_diff_all_median = ppg_labels_diff_all.median(axis=0).values.squeeze()
        print(f"ppg_labels_all_median: {self.ppg_labels_all_median}")
        print(f"ppg_labels_diff_all_median: {self.ppg_labels_diff_all_median}")


    def forward(self, nir_imgs: torch.Tensor, ppg_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # nir_imgs: (batch_size, window_size, 1, img_h, img_w)
        # ppg_labels: (batch_size, window_size, 1)

        device = nir_imgs.device
        batch_size = nir_imgs.shape[0]
        if self.config.out_dim == 1:
            # for differentiated ppg regression
            logits = self.ppg_labels_diff_all_median.repeat(batch_size, 1).to(device)
        else:
            # for seq2seq
            logits = self.ppg_labels_all_median.repeat(batch_size, 1).to(device)


        if ppg_labels is not None:
            # if we are given some desired targets also calculate the loss
            if self.config.out_dim == 1:
                # for differentiated ppg regression
                labels = (ppg_labels[:, 1] - ppg_labels[:, 0]) / self.config.rppg_labels_diff_std
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


    @torch.no_grad()
    def generate(self, test_dataset: Dataset, device: str) -> dict[str, dict[str, np.ndarray]]:
        """
        Predict on test set, usually with non-overlapping windows of 30s
        and calculate PPG figure, BPM error, and other frequency stuffs.
        """
        # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        self.eval()

        result = {}  # {subject_name: {start_idxs, losses, ppg_predicts, ppg_labels, spectrum_predicts, spectrum_labels, bpm_predicts, bpm_labels}}
        for subject_name, (start_idxs, bpm_labels, spectrum_labels) in test_dataset.test_list.items():
            losses = []
            ppg_predicts, ppg_labels = [], []
            spectrum_predicts = []
            bpm_predicts = []
            ppg_predict_first = test_dataset.data[subject_name]['ppg_labels'][0]
            for start_idx, bpm_label in zip(start_idxs, bpm_labels):
                nir_imgs_window, ppg_labels_window = test_dataset.get_test_data(subject_name, start_idx)
                nir_imgs_window, ppg_labels_window = nir_imgs_window.to(device), ppg_labels_window.to(device)

                # Visualization for debug
                # for i, nir_img in enumerate(nir_imgs_window[0, :, 0]):
                #     cv2.imshow(f'nir_imgs{i}', nir_img.cpu().numpy())
                # cv2.waitKey(0)

                logits, loss = self(nir_imgs_window, ppg_labels_window)

                ppg_predicts_window = torch.cumsum(logits.squeeze(), dim=0).cpu().numpy() * self.config.rppg_labels_diff_std
                ppg_predicts_window = np.concatenate((np.zeros(1), ppg_predicts_window)) + ppg_predict_first  # Add last ppg_predict or ppg_labels[0] as bias
                ppg_predict_first = ppg_predicts_window[-1]
                assert len(ppg_predicts_window) == test_dataset.config.test_window_size, f"len of predicted ppg is not the same as ground truth ppg!"
                ppg_predicts_detrend = sig.detrend(ppg_predicts_window)
                spectrum_predict = np.abs(np.fft.rfft(ppg_predicts_detrend))
                freq = np.fft.rfftfreq(len(ppg_predicts_detrend), d=1./test_dataset.config.video_fps)
                freq_range = np.logical_and(freq <= test_dataset.config.max_heart_rate / 60, freq >= test_dataset.config.min_heart_rate / 60)
                max_idx = np.argmax(spectrum_predict[freq_range])
                max_freq = freq[freq_range][max_idx]
                bpm_predict = max_freq * 60

                losses.append(loss.item())
                ppg_predicts.append(ppg_predicts_window)
                ppg_labels.append(test_dataset.data[subject_name]['ppg_labels'][start_idx : start_idx + test_dataset.config.test_window_size])
                spectrum_predicts.append(spectrum_predict)
                bpm_predicts.append(bpm_predict)

            result[subject_name] = {'start_idxs': start_idxs, 'losses': np.array(losses),
                                    'ppg_predicts': np.array(ppg_predicts), 'ppg_labels': np.array(ppg_labels),
                                    'spectrum_predicts': np.array(spectrum_predicts), 'spectrum_labels': spectrum_labels,
                                    'bpm_predicts': np.array(bpm_predicts), 'bpm_labels': bpm_labels}
        self.train()
        return result
