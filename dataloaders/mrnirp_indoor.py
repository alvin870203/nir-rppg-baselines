import os
import glob
import numpy as np
import cv2
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform

from transforms.frame_transforms import FrameTransformConfig, FrameTransform


@dataclass
class MRNIRPIndoorDatasetConfig:
    dataset_root_path: str = '/mnt/Data/MR-NIRP_Indoor/'
    window_size: int = 2  # unit: frames
    window_stride: int = 1  # unit: frames
    img_h: int = 128  # input image height of the model
    img_w: int = 128  # input image width of the model
    video_fps: float = 30.
    ppg_fps: float = 60.
    train_list: tuple[str] = ()
    val_list: tuple[str] = ()
    test_list: tuple[str] = ()
    test_window_size: int = 900  # unit: frames (30 seconds)
    test_window_stride: int = 900  # unit: frames (non-overlapping)
    max_heart_rate: int = 250  # unit: bpm
    min_heart_rate: int = 40  # unit: bpm
    crop_face_type: str = 'no'  # 'no', 'video_first', 'window_first', 'every'
    bbox_scale: float = 1.6


class MRNIRPIndoorDataset(Dataset):
    def __init__(self, config: MRNIRPIndoorDatasetConfig, split: str,
                 video_transform: None | nn.Module = None, window_transform: None | nn.Module = None, frame_transform: None | nn.Module = None) -> None:
        assert split in ['train', 'val', 'test']
        self.config = config
        self.split = split
        self.video_transform = video_transform
        self.window_transform = window_transform
        self.frame_transform = frame_transform
        self.data = self.load_data()  # dict of {subject_name: {nir_img_array, ppg_labels, face_locations}}
        self.window_list = self.get_window_list()  # list of (subject_name, start_idx)
        self.test_list = self.get_test_list()  # dict of {subject_name: (start_idxs, bpm_labels, spectrum_labels)}
        self.window_transform = window_transform
        if frame_transform is not None:
            self.frame_transform = frame_transform
        else:
            self.frame_transform = FrameTransform(FrameTransformConfig(img_h=config.img_h,
                                                                       img_w=config.img_w,
                                                                       bbox_scale=config.bbox_scale))
        self.test_frame_transform = FrameTransform(FrameTransformConfig(img_h=config.img_h,
                                                                        img_w=config.img_w,
                                                                        bbox_scale=config.bbox_scale))


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


    def get_test_list(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        test_data = {}
        for subject_name, subject_data in self.data.items():
            nir_imgs = subject_data['nir_imgs']
            ppg_labels = subject_data['ppg_labels']
            face_locations = subject_data['face_locations']
            start_idxs = []
            bpm_labels = []
            spectrum_labels = []
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
                start_idxs.append(idx)
                bpm_labels.append(bpm)
                spectrum_labels.append(ppg_labels_window_spectrum)
            start_idxs = np.array(start_idxs)
            bpm_labels = np.array(bpm_labels)
            spectrum_labels = np.array(spectrum_labels)
            test_data[subject_name] = (start_idxs, bpm_labels, spectrum_labels)

        return test_data


    def get_test_data(self, subject_name: str, start_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # nir_imgs: (batch_size=test_window_size, window_size, 1, img_h, img_w)
        # ppg_labels: (batch_size=test_window_size, window_size, 1)
        # There's only (test_window_size - window_size + 1) windows of window_size in a test_window
        nir_imgs = []
        ppg_labels = []
        for start_idx_t0 in range(start_idx, start_idx + self.config.test_window_size - self.config.window_size + 1):
            nir_imgs_window = []
            ppg_labels_window = []
            for idx in range(start_idx_t0, start_idx_t0 + self.config.window_size):
                nir_img = self.data[subject_name]['nir_imgs'][idx]
                ppg_label = self.data[subject_name]['ppg_labels'][idx]
                if self.config.crop_face_type == 'no':
                    face_location = None
                elif self.config.crop_face_type == 'video_first':
                    face_location = self.data[subject_name]['face_locations'][0]
                elif self.config.crop_face_type == 'window_first':
                    face_location = self.data[subject_name]['face_locations'][start_idx_t0]
                elif self.config.crop_face_type == 'every':
                    face_location = self.data[subject_name]['face_locations'][start_idx_t0 + idx]
                else:
                    raise NotImplementedError
                nir_img = torch.from_numpy(nir_img[np.newaxis, ...]).float()
                ppg_label = torch.tensor([ppg_label]).float()
                nir_img, ppg_label = self.test_frame_transform(nir_img, ppg_label, face_location=face_location)
                nir_imgs_window.append(nir_img)
                ppg_labels_window.append(ppg_label)
            nir_imgs.append(torch.stack(nir_imgs_window, axis=0))
            ppg_labels.append(torch.stack(ppg_labels_window, axis=0))
        nir_imgs = torch.stack(nir_imgs, axis=0)
        ppg_labels = torch.stack(ppg_labels, axis=0)
        return nir_imgs, ppg_labels


    def __len__(self) -> int:
        return len(self.window_list)


    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # nir_imgs: (batch_size, window_size, 1, img_h, img_w)
        # ppg_labels: (batch_size, window_size, 1)
        subject_name, start_idx = self.window_list[idx]
        nir_imgs = []
        ppg_labels = []

        if self.video_transform is not None:
            data_nir_imgs = torch.from_numpy(self.data[subject_name]['nir_imgs'][start_idx:]).unsqueeze(1).float()
            data_ppg_labels = torch.from_numpy(self.data[subject_name]['ppg_labels'][start_idx:]).unsqueeze(1).float()
            data_nir_imgs, data_ppg_labels = self.video_transform(data_nir_imgs, data_ppg_labels)
        else:
            data_nir_imgs = torch.from_numpy(self.data[subject_name]['nir_imgs'][start_idx : start_idx + self.config.window_size]).unsqueeze(1).float()
            data_ppg_labels = torch.from_numpy(self.data[subject_name]['ppg_labels'][start_idx : start_idx + self.config.window_size]).unsqueeze(1).float()

        for i, (nir_img, ppg_label) in enumerate(zip(data_nir_imgs, data_ppg_labels)):
            if self.config.crop_face_type == 'no':
                face_location = None
            elif self.config.crop_face_type == 'video_first':
                face_location = self.data[subject_name]['face_locations'][0]
            elif self.config.crop_face_type == 'window_first':
                face_location = self.data[subject_name]['face_locations'][start_idx]
            elif self.config.crop_face_type == 'every':
                face_location = self.data[subject_name]['face_locations'][start_idx + i]
            else:
                raise NotImplementedError
            nir_img, ppg_label = self.frame_transform(nir_img, ppg_label, face_location=face_location)
            nir_imgs.append(nir_img)
            ppg_labels.append(ppg_label)
        # NOTE: torch.from_numpy does not copy the data, be aware of this if you don't want to modify the data in-place.
        nir_imgs = torch.stack(nir_imgs, axis=0)
        ppg_labels = torch.stack(ppg_labels, axis=0)

        if self.window_transform is not None:
            nir_imgs, ppg_labels = self.window_transform(nir_imgs, ppg_labels)

        # Visualization for debug
        # if self.split == 'val':
        #     for i, nir_img in enumerate(nir_imgs[:, 0]):
        #         cv2.imshow(f'nir_imgs{i}', nir_img.numpy())
        #     cv2.waitKey(0)

        return nir_imgs, ppg_labels
