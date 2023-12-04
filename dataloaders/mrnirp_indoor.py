import os
import glob
import numpy as np
import cv2
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


@dataclass
class MRNIRPIndoorDatasetConfig:
    dataset_root_path: str = '/mnt/Data/MR-NIRP_Indoor/'
    window_size: int = 2  # unit: frames
    window_stride: int = 1  # unit: frames
    img_size_h: int = 640
    img_size_w: int = 640
    video_fps: float = 30.
    ppg_fps: float = 60.


class MRNIRPIndoorDataset(Dataset):
    def __init__(self, config: MRNIRPIndoorDatasetConfig, split: str) -> None:
        assert split in ['train', 'val', 'test']
        self.config = config
        self.split = split
        self.data_raw = self.load_data()
        self.data = self.window_data()


    def load_data(self) -> dict[str, tuple[list[str], np.ndarray]]:
        data_raw = {}
        for subject_folder in glob.glob(os.path.join(self.config.dataset_root_path, '*')):
            # TODO: select subject by self.split
            if self.split == 'train' and os.path.basename(subject_folder) == 'Subject1_motion_940':
                continue
            elif self.split == 'val' and os.path.basename(subject_folder) != 'Subject1_motion_940':
                continue
            else:
                pass  # correct split
            if not os.path.isdir(subject_folder):
                continue

            if os.path.isdir(os.path.join(subject_folder, "NIR")):
                nir_path_list = sorted(glob.glob(os.path.join(subject_folder, "NIR", "*.pgm")))
            else:
                nir_path_list = sorted(glob.glob(os.path.join(subject_folder, "cam_flea3_1", "*.pgm")))

            ppg_mat = sio.loadmat(os.path.join(subject_folder, "PulseOX", "pulseOx.mat"))
            ppg_signal_corrupted = ppg_mat["pulseOxRecord"].squeeze()
            ppg_time_corrupted = ppg_mat["pulseOxTime"][0]
            ppg_signal, ppg_time = [], []
            for idx, (value, time) in enumerate(zip(ppg_signal_corrupted, ppg_time_corrupted)):
                num_values = len(value[0]) if isinstance(value, np.ndarray) else 1
                if num_values > 1:  # Multiple values at a time step due to queued delayed ppg signal
                    for sub_idx, sub_value in enumerate(value[0]):
                        ppg_time.append(ppg_time_corrupted[idx-1] + ((sub_idx+1) / num_values) * (time - ppg_time_corrupted[idx-1]))
                        ppg_signal.append(sub_value)
                else:
                    ppg_time.append(time)
                    ppg_signal.append(value.item())
            ppg_signal = np.array(ppg_signal)
            ppg_time = np.array(ppg_time)
            # Resample ppg_signal to the same size as nir_path_list
            ppg_signal_resampled = sig.resample(ppg_signal, len(nir_path_list))
            ppg_time_resampled = np.linspace(ppg_time[0], ppg_time[-1], len(nir_path_list))
            data_raw[os.path.basename(subject_folder)] = (nir_path_list, ppg_signal_resampled)

        return data_raw


    def window_data(self) -> list[tuple[list[str], np.ndarray]]:
        data = []
        for subject_name, (nir_path_list, ppg_signal) in self.data_raw.items():
            for idx in range(0, len(nir_path_list) - self.config.window_size + 1, self.config.window_stride):
                if np.any(ppg_signal[idx : idx + self.config.window_size] < 1):
                    continue
                data.append((nir_path_list[idx : idx + self.config.window_size],
                             ppg_signal[idx : idx + self.config.window_size]))

        return data


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # nir_imgs: (batch_size, window_size, 1, img_size_h, img_size_w)
        # ppg_signals: (batch_size, window_size, 1)
        # NOTE: Important to normalize the images to [0, 1] range, or else the training loss will not converge and only random accuracy will be achieved
        # FIXME: Cannot normalize each image separately, or else the rppg intensity will be lost
        nir_imgs = torch.stack([torch.from_numpy(
                                    cv2.normalize(cv2.resize(cv2.imread(nir_path, cv2.IMREAD_UNCHANGED), (self.config.img_size_w, self.config.img_size_h), interpolation=cv2.INTER_AREA),
                                                  None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.float32)[np.newaxis, ...]).float()
                                for nir_path in self.data[idx][0]])
        ppg_signals = torch.from_numpy(self.data[idx][1][..., np.newaxis]).float()
        return nir_imgs, ppg_signals
