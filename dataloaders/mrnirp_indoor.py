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

from utils.preprocess import pad_crop_resize


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
    test_window_size: int = 900  # unit: frames (30 seconds)
    test_window_stride: int = 900  # unit: frames (non-overlapping)
    max_heart_rate: int = 250  # unit: bpm
    min_heart_rate: int = 40  # unit: bpm
    crop_face_type: str = 'video_first'  # 'no', 'video_first', 'window_first', 'every'
    bbox_scale: float = 1.6


class MRNIRPIndoorDataset(Dataset):
    def __init__(self, config: MRNIRPIndoorDatasetConfig, split: str) -> None:
        assert split in ['train', 'val', 'test']
        self.config = config
        self.split = split
        self.data = self.load_data()  # dict of {subject_name: {nir_img_array, ppg_labels, face_locations}}
        self.window_list = self.get_window_list()  # list of (subject_name, start_idx)
        self.test_data = self.get_test_data()  # dict of {subject_name: (start_idx_array, bpm_array)}


    def load_data(self) -> dict[str, dict[str, np.ndarray]]:
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
            #       The nir_imgs is already normalized to [0, 1] range in the npz file. See preparation/ for details.
            nir_imgs = subject_npz["nir_imgs"]
            ppg_labels = subject_npz["ppg_labels"]
            ppg_time = subject_npz["ppg_time"]
            face_locations = subject_npz["face_locations"]  # left, top, right, bottom

            # Resample ppg_labels to the same size as nir_path_list
            ppg_labels_resampled = sig.resample(ppg_labels, len(nir_imgs))
            ppg_time_resampled = np.linspace(ppg_time[0], ppg_time[-1], len(nir_imgs))

            data[subject_name] = {"nir_imgs": nir_imgs, "ppg_labels": ppg_labels_resampled, "face_locations": face_locations}

        return data


    def get_window_list(self) -> list[tuple[str, int]]:
        window_list = []
        for subject_name, subject_data in self.data.items():
            nir_imgs = subject_data['nir_imgs']
            ppg_labels = subject_data['ppg_labels']
            face_locations = subject_data['face_locations']
            for idx in range(0, len(nir_imgs) - self.config.window_size + 1, self.config.window_stride):
                if np.any(ppg_labels[idx : idx + self.config.window_size] < 1):
                    continue
                window_list.append((subject_name, idx))  # FUTURE: Add bpm to the tuple

        return window_list


    def get_test_data(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        test_data = {}
        for subject_name, subject_data in self.data.items():
            nir_imgs = subject_data['nir_imgs']
            ppg_labels = subject_data['ppg_labels']
            face_locations = subject_data['face_locations']
            start_idx_array = []
            bpm_array = []
            spectrum_array = []
            for idx in range(0, len(nir_imgs) - self.config.test_window_size + 1, self.config.test_window_stride):
                ppg_labels_window = ppg_labels[idx : idx + self.config.test_window_size]
                if np.any(ppg_labels_window < 1):
                    pass  # NOTE: Currently, we don't skip window with corrupted data, since it's too easy to have corrupted data within the large test window.
                ppg_labels_window_detrend = sig.detrend(ppg_labels_window)
                ppg_labels_window_spectrum = np.abs(np.fft.rfft(ppg_labels_window_detrend))
                freq = np.fft.rfftfreq(len(ppg_labels_window_detrend), d=1./self.config.video_fps)
                freq_range = np.logical_and(freq <= self.config.max_heart_rate / 60, freq >= self.config.min_heart_rate / 60)
                max_idx = np.argmax(ppg_labels_window_spectrum[freq_range])
                max_freq = freq[freq_range][max_idx]
                bpm = max_freq * 60
                start_idx_array.append(idx)
                bpm_array.append(bpm)
                spectrum_array.append(ppg_labels_window_spectrum)
            start_idx_array = np.array(start_idx_array)
            bpm_array = np.array(bpm_array)
            spectrum_array = np.array(spectrum_array)
            test_data[subject_name] = (start_idx_array, bpm_array, spectrum_array)

        return test_data


    def __len__(self) -> int:
        return len(self.window_list)


    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # nir_imgs: (batch_size, window_size, 1, img_size_h, img_size_w)
        # ppg_labels: (batch_size, window_size, 1)
        subject_name, start_idx = self.window_list[idx]
        nir_imgs = []
        for i, nir_img in enumerate(self.data[subject_name]['nir_imgs'][start_idx : start_idx + self.config.window_size]):
            if self.config.crop_face_type == 'no':
                nir_img = pad_crop_resize(nir_img, resize_wh=(self.config.img_size_w, self.config.img_size_h),
                                          face_location=None, bbox_scale=None)
                nir_imgs.append(nir_img[np.newaxis, ...])
            elif self.config.crop_face_type == 'video_first':
                face_location = self.data[subject_name]['face_locations'][0]
                nir_img = pad_crop_resize(nir_img, resize_wh=(self.config.img_size_w, self.config.img_size_h),
                                          face_location=face_location, bbox_scale=self.config.bbox_scale)
                nir_imgs.append(nir_img[np.newaxis, ...])
            elif self.config.crop_face_type == 'window_first':
                face_location = self.data[subject_name]['face_locations'][start_idx]
                nir_img = pad_crop_resize(nir_img, resize_wh=(self.config.img_size_w, self.config.img_size_h),
                                          face_location=face_location, bbox_scale=self.config.bbox_scale)
                nir_imgs.append(nir_img[np.newaxis, ...])
            elif self.config.crop_face_type == 'every':
                face_location = self.data[subject_name]['face_locations'][start_idx + i]
                nir_img = pad_crop_resize(nir_img, resize_wh=(self.config.img_size_w, self.config.img_size_h),
                                          face_location=face_location, bbox_scale=self.config.bbox_scale)
                nir_imgs.append(nir_img[np.newaxis, ...])
            else:
                raise NotImplementedError
        nir_imgs = np.stack(nir_imgs, axis=0)
        # TODO: Add augmentation
        # NOTE: torch.from_numpy does not copy the data, be aware of this if you don't want to modify the data in-place.
        nir_imgs = torch.from_numpy(nir_imgs).float()
        ppg_labels = torch.from_numpy(self.data[subject_name]['ppg_labels'][start_idx : start_idx + self.config.window_size, np.newaxis]).float()
        return nir_imgs, ppg_labels
