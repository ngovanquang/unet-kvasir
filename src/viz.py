"""Hình cho báo cáo.

Đề bài yêu cầu tối thiểu 10 ảnh so sánh: ảnh gốc / mask thật / mask dự đoán
của 2-3 cấu hình tốt nhất và 1 cấu hình tệ nhất.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .metrics import dice_per_image
from .transforms import denormalize
from .utils import ensure_dir


def plot_history(history: List[Dict], title: str = "",
                 save_path: str | Path | None = None):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["val_dice"] for h in history], label="val Dice")
    axes[1].plot(epochs, [h["val_iou"] for h in history], label="val IoU")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_loss_curves(histories: Dict[str, List[Dict]], key: str = "val_dice",
                     save_path: str | Path | None = None):
    """Chồng nhiều lượt chạy lên một hình để so sánh trực tiếp."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, history in histories.items():
        ax.plot([h["epoch"] for h in history], [h[key] for h in history], label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel(key)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@torch.no_grad()
def predict_batch(model, images: torch.Tensor, device, threshold: float = 0.5):
    model.eval()
    logits = model(images.to(device))
    return (torch.sigmoid(logits) > threshold).float().cpu(), logits.cpu()


@torch.no_grad()
def comparison_grid(dataset, models: Dict[str, torch.nn.Module], indices: Sequence[int],
                    device, threshold: float = 0.5,
                    save_path: str | Path | None = None):
    """Mỗi hàng một ảnh: gốc | mask thật | dự đoán của từng cấu hình.

    models là dict {tên cấu hình: model đã nạp checkpoint tốt nhất}.
    """
    names = list(models)
    n_cols = 2 + len(names)
    fig, axes = plt.subplots(len(indices), n_cols,
                             figsize=(2.6 * n_cols, 2.6 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        axes[row, 0].imshow(denormalize(image))
        axes[row, 1].imshow(mask[0], cmap="gray")
        batch = image.unsqueeze(0)
        for col, name in enumerate(names, start=2):
            pred, logits = predict_batch(models[name], batch, device, threshold)
            dice = dice_per_image(logits, mask.unsqueeze(0), threshold).item()
            axes[row, col].imshow(pred[0, 0], cmap="gray")
            axes[row, col].set_xlabel(f"Dice {dice:.3f}", fontsize=8)
        for col in range(n_cols):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    for col, title in enumerate(["Ảnh gốc", "Mask thật"] + names):
        axes[0, col].set_title(title, fontsize=9)

    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@torch.no_grad()
def worst_case_indices(model, loader, device, threshold: float = 0.5,
                       k: int = 6) -> List[int]:
    """Trả về chỉ số k ảnh có Dice thấp nhất — nguyên liệu cho phân tích lỗi."""
    model.eval()
    scores, offset = [], 0
    for images, masks in loader:
        logits = model(images.to(device)).cpu()
        for i, d in enumerate(dice_per_image(logits, masks, threshold).tolist()):
            scores.append((d, offset + i))
        offset += images.size(0)
    scores.sort()
    return [idx for _, idx in scores[:k]]


def zoom_on_border(image_np: np.ndarray, pred: np.ndarray, box=(64, 64, 96, 96),
                   save_path: str | Path | None = None):
    """Cắt sát biên polyp để soi checkerboard artifact.

    Không phóng to thì gần như không nhìn thấy artifact, và phần bình luận
    trong báo cáo sẽ thành nói suông.
    """
    x, y, w, h = box
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(image_np[y:y + h, x:x + w])
    axes[0].set_title("Ảnh gốc (phóng to)", fontsize=9)
    axes[1].imshow(pred[y:y + h, x:x + w], cmap="gray", interpolation="nearest")
    axes[1].set_title("Dự đoán (phóng to)", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
