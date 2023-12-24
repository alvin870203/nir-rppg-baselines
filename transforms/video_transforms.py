from dataclasses import dataclass
from fractions import Fraction
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
import cv2


@dataclass
class VideoTransformConfig:
    window_size: int = 2  # unit: frames
    video_fps: float = 30.
    video_freq_scale_range: tuple[float, float] = (1.0, 1.0)  # augmented freq ~= freq * random.uniform(min, max), e.g., (0.7, 1.4)
    video_freq_scale_p: float = 0.0  # probability of applying random video freq scale
    video_freq_scale_dt: int = 10  # max number of intervals to resampled between two originally consecutive frames


class VideoTransform(nn.Module):
    def __init__(self, config: VideoTransformConfig):
        super().__init__()
        self.config = config
        self.transform = []
        if config.video_freq_scale_range != (1.0, 1.0):
            self.transform.append("random_freq_scale")


    def forward(self, all_nir_imgs: torch.Tensor, all_ppg_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input:
        #    all_nir_imgs: (video_len - start_idx, 1, img_h, img_w)
        #    all_ppg_labels: (video_len - start_idx, 1)
        # Output:
        #    nir_imgs: (window_size, 1, img_h, img_w)
        #    ppg_labels: (window_size, 1)
        img_h, img_w = all_nir_imgs.shape[2:]
        if "random_freq_scale" in self.transform:  # This is not an accurate way to perform freq scale, but it's simple
            freq_scale = np.random.choice([1.0, np.random.uniform(*self.config.video_freq_scale_range)],
                                          p=[1 - self.config.video_freq_scale_p, self.config.video_freq_scale_p])
            if freq_scale == 1.0:
                nir_imgs = all_nir_imgs[:self.config.window_size]
                ppg_labels = all_ppg_labels[:self.config.window_size]
            else:
                freq_scale_numerator = Fraction(freq_scale).limit_denominator(self.config.video_freq_scale_dt).numerator
                freq_scale_denominator = Fraction(freq_scale).limit_denominator(self.config.video_freq_scale_dt).denominator
                required_timesteps = np.ceil((self.config.window_size - 1) * freq_scale_numerator / freq_scale_denominator).astype(int).item() + 1
                if required_timesteps > all_nir_imgs.shape[0]:  # no enough timesteps to perform freq scale
                    nir_imgs = all_nir_imgs[:self.config.window_size]
                    ppg_labels = all_ppg_labels[:self.config.window_size]
                else:  # FUTURE: How to deal with face_locations when freq scale?
                    required_nir_imgs = all_nir_imgs[:required_timesteps].permute(1, 0, 2, 3).unsqueeze(0)  # (1, 1, required_timesteps, img_h, img_w)
                    required_ppg_labels = all_ppg_labels[:required_timesteps].permute(1, 0).unsqueeze(0)  # (1, 1, required_timesteps)
                    resampled_timesteps = (required_timesteps - 1) * freq_scale_denominator + 1
                    nir_imgs = nn.functional.interpolate(required_nir_imgs, size=(resampled_timesteps, img_h, img_w),
                                                         mode='trilinear', align_corners=True).reshape(resampled_timesteps, 1, img_h, img_w)
                    ppg_labels = nn.functional.interpolate(required_ppg_labels, size=(resampled_timesteps),
                                                           mode='linear', align_corners=True).reshape(resampled_timesteps, 1)
                    window_resampled_timesteps = (self.config.window_size - 1) * freq_scale_numerator + 1
                    rand_start_timestep = np.random.randint(0, resampled_timesteps - (window_resampled_timesteps - 1))
                    nir_imgs = nir_imgs[rand_start_timestep : rand_start_timestep + window_resampled_timesteps : freq_scale_numerator]
                    ppg_labels = ppg_labels[rand_start_timestep : rand_start_timestep + window_resampled_timesteps : freq_scale_numerator]
        else:
            nir_imgs = all_nir_imgs[:self.config.window_size]
            ppg_labels = all_ppg_labels[:self.config.window_size]


        return nir_imgs, ppg_labels
