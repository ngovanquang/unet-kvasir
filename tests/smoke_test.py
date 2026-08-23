"""Kiểm tra toàn bộ pipeline chạy được, dùng dữ liệu giả.

Chạy trên CPU trong khoảng một phút. Mục đích không phải đo chất lượng mà là
bắt lỗi lắp ráp: sai shape, sai kiểu, quên .to(device), CSV hỏng header.

    python -m tests.smoke_test
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (Config, LOSS_NAMES, RunLogger, SKIP_MODES, UNet, UP_MODES,  # noqa: E402
                 build_loss, foreground_ratio, run_experiment)


def make_fake_dataset(root: Path, n: int = 24, size: int = 64) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        cy, cx = rng.integers(16, size - 16, 2)
        r = int(rng.integers(6, 14))
        yy, xx = np.ogrid[:size, :size]
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 255
        Image.fromarray(img).save(root / "images" / f"{i:03d}.jpg")
        Image.fromarray(mask).save(root / "masks" / f"{i:03d}.jpg")


def test_architecture_grid() -> None:
    x = torch.randn(2, 3, 64, 64)
    for up_mode in UP_MODES:
        for skip_mode in SKIP_MODES:
            model = UNet(base_channels=8, up_mode=up_mode, skip_mode=skip_mode)
            y = model(x)
            assert y.shape == (2, 1, 64, 64), (up_mode, skip_mode, y.shape)
    print("[ok] 9 tổ hợp up_mode x skip_mode đều ra đúng shape")


def test_losses() -> None:
    logits = torch.randn(4, 1, 32, 32, requires_grad=True)
    targets = (torch.rand(4, 1, 32, 32) > 0.85).float()
    for name in LOSS_NAMES:
        cfg = Config(loss_name=name, pos_weight=5.0)
        loss = build_loss(cfg)(logits, targets)
        assert torch.isfinite(loss), name
        loss.backward(retain_graph=True)
    # Trường hợp biên: ảnh hoàn toàn không có polyp không được sinh NaN
    empty = torch.zeros(2, 1, 32, 32)
    for name in ("dice", "tversky", "bce_dice"):
        loss = build_loss(Config(loss_name=name))(torch.randn(2, 1, 32, 32) - 8, empty)
        assert torch.isfinite(loss), f"{name} sinh NaN với mask rỗng"
    print(f"[ok] {len(LOSS_NAMES)} hàm mất mát hữu hạn, gradient chạy, không NaN")


def test_end_to_end(tmp: Path) -> None:
    root = tmp / "data"
    make_fake_dataset(root)
    ratio = foreground_ratio(root, [f"{i:03d}.jpg" for i in range(24)])
    assert 0.0 < ratio < 1.0
    print(f"[ok] tỉ lệ pixel foreground = {ratio:.4f}")

    cfg = Config(
        data_root=str(root), split_dir=str(tmp / "splits"),
        log_csv=str(tmp / "runs.csv"), ckpt_dir=str(tmp / "ckpt"),
        image_size=64, base_channels=8, batch_size=4, epochs=2,
        num_workers=0, amp=False, early_stop_patience=0,
    )
    result = run_experiment(cfg, verbose=False)
    for key in ("test_dice", "test_iou", "test_precision", "test_recall"):
        assert 0.0 <= result[key] <= 1.0, (key, result[key])
    assert len(result["history"]) == 2
    print(f"[ok] chạy trọn 2 epoch, test Dice = {result['test_dice']:.4f}")

    # Chạy lại phải bị bỏ qua nhờ run_id trùng
    again = run_experiment(cfg, verbose=False)
    assert again.get("skipped") is True
    print("[ok] chống chạy trùng theo run_id hoạt động")

    rows = RunLogger(cfg.log_csv).read_all()
    assert len(rows) == 1 and rows[0]["run_id"] == cfg.run_id
    print(f"[ok] nhật ký CSV có {len(rows[0])} cột, ghi đúng một dòng")

    # Thêm một cấu hình khác để chắc chắn phần mở rộng header không hỏng
    run_experiment(cfg.replace(loss_name="tversky"), verbose=False)
    assert len(RunLogger(cfg.log_csv).read_all()) == 2
    print("[ok] append dòng thứ hai vào CSV")


if __name__ == "__main__":
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp())
    try:
        test_architecture_grid()
        test_losses()
        test_end_to_end(tmp)
        print("\nTẤT CẢ KIỂM TRA ĐỀU QUA.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
