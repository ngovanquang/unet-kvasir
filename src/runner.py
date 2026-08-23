"""Chạy trọn một thí nghiệm từ Config.

Đây là điểm vào duy nhất mà notebook cần biết:

    from src import Config, run_experiment
    result = run_experiment(Config(loss_name="dice"))

Hàm này lo hết: cố định seed, dựng dữ liệu, dựng model, huấn luyện, nạp lại
checkpoint tốt nhất, đánh giá trên tập test, ghi một dòng vào logs/runs.csv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .config import Config
from .data import build_dataloaders, foreground_ratio, load_splits, pos_weight_from_ratio
from .engine import evaluate, fit, load_best
from .logger import RunLogger
from .losses import build_loss
from .models import UNet
from .utils import count_parameters, get_device, set_seed


def history_path(cfg: Config) -> Path:
    return Path(cfg.ckpt_dir) / f"{cfg.run_id}_history.json"


def save_history(cfg: Config, history: List[Dict]) -> None:
    path = history_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=1), encoding="utf-8")


def load_history(cfg: Config) -> List[Dict]:
    """Đọc lại đường cong của một lượt đã chạy.

    Nhờ hàm này mà mất kết nối Colab không làm mất đường cong: chỉ cần
    checkpoint và file history còn trên Drive là vẽ lại được.
    """
    path = history_path(cfg)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def build_model(cfg: Config, device: Optional[torch.device] = None) -> UNet:
    device = device or get_device()
    model = UNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        up_mode=cfg.up_mode,
        skip_mode=cfg.skip_mode,
    )
    return model.to(device)


def run_experiment(cfg: Config, verbose: bool = True,
                   skip_if_logged: bool = True) -> Dict:
    """Chạy một cấu hình. Trả về dict gồm history và toàn bộ chỉ số test."""
    logger = RunLogger(cfg.log_csv)

    if skip_if_logged and any(r.get("run_id") == cfg.run_id for r in logger.read_all()):
        if verbose:
            print(f"[bỏ qua] {cfg.run_id} đã có trong log.")
        # Vẫn trả về history đọc từ đĩa, để vẽ lại đường cong không cần train lại.
        return {"run_id": cfg.run_id, "loss_name": cfg.loss_name,
                "up_mode": cfg.up_mode, "skip_mode": cfg.skip_mode,
                "skipped": True, "history": load_history(cfg)}

    set_seed(cfg.seed)
    device = get_device()
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    pos_weight = None
    if cfg.loss_name == "weighted_bce" and cfg.pos_weight is None:
        ratio = foreground_ratio(cfg.data_root, load_splits(cfg.split_dir)["train"])
        pos_weight = pos_weight_from_ratio(ratio)
        if verbose:
            print(f"pos_weight tự tính = {pos_weight:.2f} (tỉ lệ polyp {ratio:.4f})")

    model = build_model(cfg, device)
    criterion = build_loss(cfg, pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    if verbose:
        print(f"=== {cfg.run_id} === {count_parameters(model):,} tham số, "
              f"thiết bị {device}")

    train_out = fit(model, train_loader, val_loader, criterion, optimizer, device,
                    epochs=cfg.epochs, ckpt_path=cfg.ckpt_path, scheduler=scheduler,
                    threshold=cfg.threshold, patience=cfg.early_stop_patience,
                    amp=cfg.amp, verbose=verbose)

    load_best(model, cfg.ckpt_path, device)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device,
                                       cfg.threshold)

    record = {
        "run_id": cfg.run_id,
        **{k: v for k, v in cfg.to_dict().items()
           if k not in {"ckpt_dir", "fig_dir", "log_csv"}},
        "n_params": count_parameters(model),
        "best_epoch": train_out["best_epoch"],
        "epochs_run": train_out["epochs_run"],
        "best_val_dice": round(train_out["best_val_dice"], 5),
        "test_loss": round(test_loss, 5),
        **test_metrics.to_dict("test_"),
        "train_time_min": round(train_out["train_time_min"], 2),
        "ckpt": str(cfg.ckpt_path),
    }
    logger.append(record)
    save_history(cfg, train_out["history"])

    if verbose:
        print(f"TEST  Dice {test_metrics.dice:.4f} | IoU {test_metrics.iou:.4f} | "
              f"P {test_metrics.precision:.4f} | R {test_metrics.recall:.4f}")

    return {**record, "history": train_out["history"], "model": model, "skipped": False}


def run_sweep(configs, verbose: bool = True) -> Dict[str, Dict]:
    """Chạy một danh sách Config tuần tự. Trả về dict theo run_id."""
    results = {}
    for i, cfg in enumerate(configs, 1):
        if verbose:
            print(f"\n----- [{i}/{len(configs)}] -----")
        results[cfg.run_id] = run_experiment(cfg, verbose=verbose)
    return results
