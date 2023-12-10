import os
import glob
import numpy as np
import cv2
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

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
    train_list: tuple[str] = ()
    val_list: tuple[str] = ()
    test_list: tuple[str] = ()


class MRNIRPIndoorDataset(Dataset):
    def __init__(self, config: MRNIRPIndoorDatasetConfig, split: str) -> None:
        assert split in ['train', 'val', 'test']
        self.config = config
        self.split = split
        self.data = self.load_data()  # dict of (subject_name, (nir_img_array, ppg_signal))
        self.window_list = self.window_data()  # list of (subject_name, start_idx)


    def load_data(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        data = {}
        for subject_npz_path in glob.glob(os.path.join(self.config.dataset_root_path, '*.npz')):
            subject_name = os.path.basename(subject_npz_path).split('.')[0]
            if self.split == 'train' and subject_name not in self.config.train_list:
                continue
            elif self.split == 'val' and subject_name not in self.config.val_list:
                continue
            elif self.split == 'test' and subject_name not in self.config.test_list:
                continue
            else:
                pass  # correct split

            subject_npz = np.load(subject_npz_path)
            # NOTE: Important to normalize the images to [0, 1] range, or else the training loss will not converge and only random accuracy will be achieved.
            #       The nir_img_array is already normalized to [0, 1] range in the npz file. See preparation/ for details.
            nir_img_array = subject_npz["nir_img_array"]
            ppg_signal = subject_npz["ppg_signal"]
            ppg_time = subject_npz["ppg_time"]

            # Resize NIR images
            nir_img_array = np.array([cv2.resize(nir_img, (self.config.img_size_w, self.config.img_size_h), interpolation=cv2.INTER_AREA)
                                      for nir_img in nir_img_array])

            # Resample ppg_signal to the same size as nir_path_list
            ppg_signal_resampled = sig.resample(ppg_signal, len(nir_img_array))
            ppg_time_resampled = np.linspace(ppg_time[0], ppg_time[-1], len(nir_img_array))

            data[subject_name] = (nir_img_array, ppg_signal_resampled)

        return data


    def window_data(self) -> list[tuple[str, int]]:
        window_list = []
        for subject_name, (nir_imgs, ppg_signal) in self.data.items():
            for idx in range(0, len(nir_imgs) - self.config.window_size + 1, self.config.window_stride):
                if np.any(ppg_signal[idx : idx + self.config.window_size] < 1):
                    continue
                window_list.append((subject_name, idx))

        return window_list


    def __len__(self) -> int:
        return len(self.window_list)


    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # nir_imgs: (batch_size, window_size, 1, img_size_h, img_size_w)
        # ppg_signals: (batch_size, window_size, 1)
        subject_name, start_idx = self.window_list[idx]
        nir_imgs = torch.from_numpy(self.data[subject_name][0][start_idx : start_idx + self.config.window_size, np.newaxis, ...]).float()
        ppg_signals = torch.from_numpy(self.data[subject_name][1][start_idx : start_idx + self.config.window_size, np.newaxis]).float()
        return nir_imgs, ppg_signals
