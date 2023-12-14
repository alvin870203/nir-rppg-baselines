import numpy as np
import cv2

def pad_crop_resize(nir_img: np.ndarray, resize_wh: None | tuple[int, int], face_location: None | np.ndarray, bbox_scale: None | float) -> np.ndarray:
    if face_location is not None:
        assert bbox_scale is not None, 'bbox_scale must be specified when face_location is specified'
        left, top, right, bottom = face_location
        center_x, center_y = (left + right) // 2, (top + bottom) // 2
        face_size = max(right - left, bottom - top) * bbox_scale
        left = int(center_x - face_size // 2)
        top = int(center_y - face_size // 2)
        right = int(center_x + face_size // 2)
        bottom = int(center_y + face_size // 2)
        left_pad = max(0, -left)
        top_pad = max(0, -top)
        right_pad = max(0, right - nir_img.shape[1])
        bottom_pad = max(0, bottom - nir_img.shape[0])
        nir_img = np.pad(nir_img, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=0)
        nir_img = nir_img[top + top_pad : bottom + top_pad, left + left_pad : right + left_pad]
    if resize_wh is not None:
        nir_img = cv2.resize(nir_img, resize_wh, interpolation=cv2.INTER_AREA)
    return nir_img