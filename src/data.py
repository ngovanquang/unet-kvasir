"""Nạp dữ liệu Kvasir-SEG.

Cấu trúc thư mục sau khi giải nén:

    data/Kvasir-SEG/
        images/  cju0qkwl35piu0993l0dewei2.jpg  ...
        masks/   cju0qkwl35piu0993l0dewei2.jpg  ...

Tên file ảnh và mask trùng nhau, đó là cách ghép cặp.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import MASK_BINARY_THRESHOLD, SegTransform
from .utils import ensure_dir, seed_worker

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"

DOWNLOAD_HINT = f"""Chưa thấy dữ liệu. Chạy trong Colab:

    !wget -q {KVASIR_URL} -O kvasir-seg.zip
    !mkdir -p data && unzip -q kvasir-seg.zip -d data/
"""


class KvasirSegDataset(Dataset):
    def __init__(self, root: str | Path, names: Sequence[str],
                 transform: SegTransform | None = None):
        self.root = Path(root)
        self.names = list(names)
        self.transform = transform or SegTransform(train=False)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(DOWNLOAD_HINT)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx]
        image = Image.open(self.image_dir / name).convert("RGB")
        mask = Image.open(self.mask_dir / name).convert("L")
        return self.transform(image, mask)


def make_splits(root: str | Path, split_dir: str | Path, seed: int = 42,
                ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                overwrite: bool = False) -> dict:
    """Chia train/val/test MỘT LẦN rồi ghi ra file và commit vào repo.

    Không bao giờ chia lại ngẫu nhiên giữa các lượt chạy, nếu không thì mọi
    so sánh giữa các cấu hình đều vô nghĩa.
    """
    split_dir = ensure_dir(split_dir)
    paths = {name: split_dir / f"{name}.txt" for name in ("train", "val", "test")}
    if all(p.exists() for p in paths.values()) and not overwrite:
        return load_splits(split_dir)

    names = sorted(p.name for p in (Path(root) / "images").glob("*.jpg"))
    if not names:
        raise FileNotFoundError(DOWNLOAD_HINT)

    rng = random.Random(seed)
    rng.shuffle(names)
    n = len(names)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    splits = {
        "train": names[:n_train],
        "val": names[n_train:n_train + n_val],
        "test": names[n_train + n_val:],
    }
    for key, path in paths.items():
        path.write_text("\n".join(splits[key]), encoding="utf-8")
    return splits


def load_splits(split_dir: str | Path) -> dict:
    split_dir = Path(split_dir)
    return {
        name: (split_dir / f"{name}.txt").read_text(encoding="utf-8").split()
        for name in ("train", "val", "test")
    }


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    splits = make_splits(cfg.data_root, cfg.split_dir, seed=cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    def make(names, train: bool) -> DataLoader:
        dataset = KvasirSegDataset(
            cfg.data_root, names,
            SegTransform(size=cfg.image_size, train=train and cfg.augment),
        )
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=train,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=train,
            worker_init_fn=seed_worker,
            generator=generator if train else None,
        )

    return make(splits["train"], True), make(splits["val"], False), make(splits["test"], False)


def foreground_ratio(root: str | Path, names: Sequence[str],
                     sample: int | None = 200, seed: int = 42) -> float:
    """Tỉ lệ pixel polyp trên toàn bộ pixel.

    Con số này là lý do tồn tại của cả nghiên cứu về hàm mất mát, và cũng là
    giá trị pos_weight cho Weighted BCE. Đưa vào báo cáo ở mục Dữ liệu.
    """
    names = list(names)
    if sample and len(names) > sample:
        names = random.Random(seed).sample(names, sample)
    fg = total = 0
    for name in names:
        mask = np.asarray(Image.open(Path(root) / "masks" / name).convert("L"))
        fg += int((mask > MASK_BINARY_THRESHOLD).sum())
        total += mask.size
    return fg / total


def pos_weight_from_ratio(ratio: float) -> float:
    """pos_weight = số pixel nền / số pixel polyp."""
    return (1.0 - ratio) / max(ratio, 1e-8)
