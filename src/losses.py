"""Sáu hàm mất mát cho phân đoạn nhị phân mất cân bằng lớp.

QUY ƯỚC CHUNG: mọi hàm ở đây nhận LOGITS (đầu ra thô của UNet, chưa qua
sigmoid). Hàm nào cần xác suất thì tự gọi sigmoid bên trong.

Ba cái bẫy đã được xử lý sẵn:

  1. BCE dùng BCEWithLogitsLoss chứ không phải sigmoid + BCELoss. Bản gộp
     dùng thủ thuật log-sum-exp nên không tràn số khi logit lớn.
  2. Dice/Tversky đều có smooth ở CẢ tử và mẫu. Thiếu nó thì ảnh không có
     polyp sẽ cho 0/0.
  3. Focal tính qua logsigmoid thay vì log(sigmoid(x)) để không log(0).

Dice ở đây là DICE LOSS (soft, dùng xác suất liên tục, có gradient).
Đừng nhầm với DICE METRIC trong metrics.py (nhị phân hoá ở ngưỡng 0.5).
Hai thứ này khác nhau và phải báo cáo tách bạch.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten(logits: torch.Tensor, targets: torch.Tensor):
    """Gộp về (B, N) để tính Dice theo từng ảnh rồi mới lấy trung bình."""
    b = logits.shape[0]
    return logits.reshape(b, -1), targets.reshape(b, -1)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, tgt = _flatten(torch.sigmoid(logits), targets)
        inter = (probs * tgt).sum(dim=1)
        denom = probs.sum(dim=1) + tgt.sum(dim=1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """Tổng quát hoá của Dice. alpha phạt false positive, beta phạt false negative.

    alpha = beta = 0.5 thì Tversky trùng Dice. Đặt beta > alpha nghĩa là phạt
    việc bỏ sót polyp nặng hơn việc báo thừa — hợp lý về mặt lâm sàng và nên
    nêu rõ lý do này trong báo cáo.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, tgt = _flatten(torch.sigmoid(logits), targets)
        tp = (probs * tgt).sum(dim=1)
        fp = (probs * (1 - tgt)).sum(dim=1)
        fn = ((1 - probs) * tgt).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class FocalLoss(nn.Module):
    """BCE có trọng số (1-p)^gamma, làm mạng tập trung vào pixel khó."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()


class BCEDiceLoss(nn.Module):
    """Tổng có trọng số của BCE và Dice — thường là cấu hình mạnh nhất."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth)
        self.w = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w * self.bce(logits, targets) + (1 - self.w) * self.dice(logits, targets)


LOSS_NAMES = ("bce", "weighted_bce", "dice", "bce_dice", "focal", "tversky")


def build_loss(cfg, pos_weight: Optional[float] = None) -> nn.Module:
    """Tạo hàm mất mát từ Config. pos_weight chỉ dùng cho weighted_bce."""
    name = cfg.loss_name
    if name not in LOSS_NAMES:
        raise ValueError(f"loss_name phải thuộc {LOSS_NAMES}, nhận được {name!r}")

    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        w = cfg.pos_weight if cfg.pos_weight is not None else pos_weight
        if w is None:
            raise ValueError("weighted_bce cần pos_weight (tính từ tập train)")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(w)))
    if name == "dice":
        return DiceLoss(cfg.dice_smooth)
    if name == "bce_dice":
        return BCEDiceLoss(smooth=cfg.dice_smooth)
    if name == "focal":
        return FocalLoss(cfg.focal_alpha, cfg.focal_gamma)
    return TverskyLoss(cfg.tversky_alpha, cfg.tversky_beta, cfg.dice_smooth)
