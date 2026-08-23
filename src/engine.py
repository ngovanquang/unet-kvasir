"""Vòng huấn luyện và đánh giá.

Tách khỏi model và data để notebook chỉ việc gọi fit(). Toàn bộ logic
early stopping, checkpoint và AMP nằm gọn ở đây.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import MetricResult, SegMetrics
from .utils import AverageMeter, ensure_dir


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    scaler: Optional[torch.amp.GradScaler] = None) -> float:
    model.train()
    meter = AverageMeter()
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()

        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, threshold: float = 0.5):
    model.eval()
    meter = AverageMeter()
    metrics = SegMetrics(threshold)
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        meter.update(criterion(logits, masks).item(), images.size(0))
        metrics.update(logits, masks)
    return meter.avg, metrics.compute()


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        criterion: nn.Module, optimizer: torch.optim.Optimizer,
        device: torch.device, epochs: int, ckpt_path: str | Path,
        scheduler=None, threshold: float = 0.5, patience: int = 10,
        amp: bool = True, verbose: bool = True) -> Dict:
    """Huấn luyện, chọn checkpoint theo val Dice cao nhất.

    Chọn theo val Dice chứ không theo val loss: các hàm mất mát khác nhau cho
    thang loss khác nhau, không so sánh chéo được, còn Dice thì có.
    """
    ckpt_path = Path(ckpt_path)
    ensure_dir(ckpt_path.parent)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history: List[Dict] = []
    best_dice, best_epoch, bad_epochs = -1.0, -1, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, scaler)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, threshold)
        if scheduler is not None:
            scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics.to_dict("val_"),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_metrics.dice > best_dice:
            best_dice, best_epoch, bad_epochs = val_metrics.dice, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_dice": best_dice}, ckpt_path)
        else:
            bad_epochs += 1

        if verbose:
            flag = " *" if best_epoch == epoch else ""
            print(f"epoch {epoch:3d}/{epochs} | train {train_loss:.4f} | "
                  f"val {val_loss:.4f} | val dice {val_metrics.dice:.4f} | "
                  f"val iou {val_metrics.iou:.4f}{flag}")

        if patience and bad_epochs >= patience:
            if verbose:
                print(f"Dừng sớm ở epoch {epoch} (không cải thiện {patience} epoch).")
            break

    return {
        "history": history,
        "best_val_dice": best_dice,
        "best_epoch": best_epoch,
        "train_time_min": (time.time() - start) / 60.0,
        "epochs_run": len(history),
    }


def load_best(model: nn.Module, ckpt_path: str | Path,
              device: torch.device) -> nn.Module:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model.to(device)
