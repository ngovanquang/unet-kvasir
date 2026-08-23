"""Biến đổi đồng bộ cho cặp (ảnh, mask).

Tự viết thay vì dùng albumentations vì hai lý do: không phụ thuộc phiên bản
thư viện ngoài, và kiểm soát được hai cái bẫy quan trọng nhất của bài toán
phân đoạn:

  1. Mask phải resize bằng NEAREST. Dùng bilinear sẽ sinh giá trị trung gian
     (ví dụ 0.37) ở biên polyp, làm hỏng nhãn nhị phân.
  2. Mask Kvasir-SEG lưu dạng JPEG nên có nhiễu nén: pixel lẽ ra bằng 0 lại
     bằng 3, lẽ ra 255 lại 251. Phải nhị phân hoá bằng ngưỡng ngay khi nạp.

Mọi phép biến đổi hình học áp dụng ĐỒNG THỜI lên ảnh và mask. Phép biến đổi
màu chỉ áp lên ảnh.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MASK_BINARY_THRESHOLD = 127  # bẫy số 2 ở trên


def resize_pair(image: Image.Image, mask: Image.Image, size: int
                ) -> Tuple[Image.Image, Image.Image]:
    image = image.resize((size, size), Image.BILINEAR)
    mask = mask.resize((size, size), Image.NEAREST)  # KHÔNG đổi thành BILINEAR
    return image, mask


def random_geometric(image: Image.Image, mask: Image.Image
                     ) -> Tuple[Image.Image, Image.Image]:
    """Lật và xoay 90 độ. An toàn với ảnh nội soi vì không có chiều ưu tiên."""
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    k = random.randint(0, 3)
    if k:
        angle = 90 * k
        image = image.rotate(angle, resample=Image.BILINEAR)
        mask = mask.rotate(angle, resample=Image.NEAREST)
    return image, mask


def random_color_jitter(image: Image.Image, strength: float = 0.2) -> Image.Image:
    """Chỉ áp lên ảnh. Mask giữ nguyên."""
    from PIL import ImageEnhance

    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast,
                     ImageEnhance.Color):
        factor = 1.0 + random.uniform(-strength, strength)
        image = enhancer(image).enhance(factor)
    return image


def to_tensor_pair(image: Image.Image, mask: Image.Image
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    img = np.asarray(image, dtype=np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()

    msk = np.asarray(mask, dtype=np.uint8)
    if msk.ndim == 3:
        msk = msk[..., 0]
    msk = (msk > MASK_BINARY_THRESHOLD).astype(np.float32)
    msk = torch.from_numpy(msk).unsqueeze(0).contiguous()  # (1, H, W)
    return img, msk


def denormalize(img: torch.Tensor) -> np.ndarray:
    """Đảo chuẩn hoá để vẽ hình. Trả về mảng HxWx3 trong [0, 1]."""
    arr = img.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr, 0.0, 1.0)


class SegTransform:
    """Gộp toàn bộ pipeline. train=True mới bật augmentation."""

    def __init__(self, size: int = 256, train: bool = False):
        self.size = size
        self.train = train

    def __call__(self, image: Image.Image, mask: Image.Image
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = resize_pair(image, mask, self.size)
        if self.train:
            image, mask = random_geometric(image, mask)
            image = random_color_jitter(image)
        return to_tensor_pair(image, mask)
