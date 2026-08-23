"""Cấu hình thí nghiệm.

Mọi siêu tham số nằm ở đúng một chỗ. Notebook chỉ tạo Config rồi truyền đi,
không hardcode số ở bất kỳ đâu khác. Config.to_dict() được ghi thẳng vào
logs/runs.csv nên báo cáo và log không bao giờ lệch nhau.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    # --- Dữ liệu ---
    data_root: str = "data/Kvasir-SEG"
    split_dir: str = "splits"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 2
    augment: bool = True

    # --- Kiến trúc (hai trục ablation của đề bài) ---
    in_channels: int = 3
    num_classes: int = 1
    base_channels: int = 64
    depth: int = 4
    up_mode: str = "transpose"   # transpose | bilinear | unpool
    skip_mode: str = "full"      # full | half | none

    # --- Hàm mất mát ---
    loss_name: str = "bce_dice"  # xem src/losses.py
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.3   # phạt false positive
    tversky_beta: float = 0.7    # phạt false negative (bỏ sót polyp) nặng hơn
    dice_smooth: float = 1e-6
    pos_weight: Optional[float] = None  # None = tự tính từ tập train

    # --- Huấn luyện ---
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    amp: bool = True
    early_stop_patience: int = 10
    threshold: float = 0.5       # ngưỡng nhị phân hoá CỐ ĐỊNH cho mọi cấu hình

    # --- Đầu ra ---
    ckpt_dir: str = "outputs/checkpoints"
    fig_dir: str = "outputs/figures"
    log_csv: str = "logs/runs.csv"
    tag: str = ""                # ghi chú tự do, ví dụ "loss_sweep"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def run_id(self) -> str:
        """Định danh ổn định: cùng cấu hình -> cùng run_id -> dễ dò lại trong log."""
        key = json.dumps(
            {k: v for k, v in self.to_dict().items()
             if k not in {"ckpt_dir", "fig_dir", "log_csv", "num_workers", "tag"}},
            sort_keys=True,
        )
        digest = hashlib.md5(key.encode()).hexdigest()[:6]
        return f"{self.loss_name}-{self.up_mode}-{self.skip_mode}-s{self.seed}-{digest}"

    @property
    def ckpt_path(self) -> Path:
        return Path(self.ckpt_dir) / f"{self.run_id}.pt"

    def replace(self, **kwargs: Any) -> "Config":
        """Tạo bản sao đổi vài trường. Dùng để quét thí nghiệm trong notebook."""
        data = self.to_dict()
        unknown = set(kwargs) - set(data)
        if unknown:
            raise KeyError(f"Trường không tồn tại trong Config: {sorted(unknown)}")
        data.update(kwargs)
        return Config(**data)
