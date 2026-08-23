"""Tiện ích dùng chung."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Cố định seed cho Python, NumPy và PyTorch.

    Đề bài yêu cầu "cùng mã nguồn, cùng seed, cùng con số". Gọi hàm này ở
    đầu MỌI lượt chạy, trước khi tạo DataLoader và model.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Truyền vào DataLoader(worker_init_fn=...) để augmentation cũng tái lập."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class AverageMeter:
    """Trung bình động có trọng số theo số mẫu."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0
