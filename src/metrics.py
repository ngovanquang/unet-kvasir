"""Chỉ số đánh giá mức pixel.

Khác biệt quan trọng so với losses.py: ở đây dự đoán đã được NHỊ PHÂN HOÁ ở
ngưỡng cố định (mặc định 0.5) trước khi tính. Ngưỡng phải giống nhau cho mọi
cấu hình, nếu mỗi hàm mất mát lại dùng ngưỡng tối ưu riêng thì bảng so sánh
mất ý nghĩa.

Báo cáo hai kiểu tổng hợp vì chúng trả lời hai câu hỏi khác nhau:

  macro  trung bình Dice/IoU tính riêng từng ảnh rồi lấy trung bình.
         Mỗi ảnh có trọng số bằng nhau. Đây là con số hay gặp trong các bài
         báo về Kvasir-SEG, nên dùng làm chỉ số chính.
  micro  gộp TP/FP/FN của toàn bộ tập rồi mới tính một lần.
         Ảnh có polyp lớn ảnh hưởng nhiều hơn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

EPS = 1e-7


@dataclass
class MetricResult:
    dice: float          # macro
    iou: float           # macro
    precision: float     # micro
    recall: float        # micro
    dice_micro: float
    iou_micro: float

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        return {f"{prefix}{k}": round(v, 5) for k, v in self.__dict__.items()}


class SegMetrics:
    """Cộng dồn qua các batch rồi gọi compute() một lần ở cuối."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._dice_sum = 0.0
        self._iou_sum = 0.0
        self._n_images = 0
        self._tp = self._fp = self._fn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = (torch.sigmoid(logits) > self.threshold).float()
        b = preds.shape[0]
        p = preds.reshape(b, -1)
        t = targets.reshape(b, -1)

        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)

        # Macro: từng ảnh một. Ảnh không có polyp và dự đoán cũng rỗng -> Dice = 1.
        self._dice_sum += ((2 * tp + EPS) / (2 * tp + fp + fn + EPS)).sum().item()
        self._iou_sum += ((tp + EPS) / (tp + fp + fn + EPS)).sum().item()
        self._n_images += b

        self._tp += tp.sum().item()
        self._fp += fp.sum().item()
        self._fn += fn.sum().item()

    def compute(self) -> MetricResult:
        n = max(self._n_images, 1)
        tp, fp, fn = self._tp, self._fp, self._fn
        return MetricResult(
            dice=self._dice_sum / n,
            iou=self._iou_sum / n,
            precision=tp / (tp + fp + EPS),
            recall=tp / (tp + fn + EPS),
            dice_micro=2 * tp / (2 * tp + fp + fn + EPS),
            iou_micro=tp / (tp + fp + fn + EPS),
        )


@torch.no_grad()
def dice_per_image(logits: torch.Tensor, targets: torch.Tensor,
                   threshold: float = 0.5) -> torch.Tensor:
    """Dice của từng ảnh, dùng để tìm ảnh tệ nhất cho phần phân tích lỗi."""
    preds = (torch.sigmoid(logits) > threshold).float()
    b = preds.shape[0]
    p, t = preds.reshape(b, -1), targets.reshape(b, -1)
    tp = (p * t).sum(dim=1)
    return (2 * tp + EPS) / (p.sum(dim=1) + t.sum(dim=1) + EPS)
