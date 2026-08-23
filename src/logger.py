"""Nhật ký thí nghiệm.

Đề bài trừ 10 điểm nếu thiếu log, và xử như gian lận nếu số liệu trong báo
cáo lệch với log. Cách chắc chắn nhất để không bao giờ lệch: không bao giờ
gõ số bằng tay. Mọi lượt chạy tự append một dòng vào logs/runs.csv, mọi bảng
trong báo cáo đọc lại từ chính file đó.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .utils import ensure_dir


class RunLogger:
    def __init__(self, path: str | Path = "logs/runs.csv"):
        self.path = Path(path)
        ensure_dir(self.path.parent)

    def append(self, record: Dict[str, Any]) -> None:
        row = {"datetime": datetime.now().isoformat(timespec="seconds"), **record}
        existing = self._read_header()

        # Cảnh báo sớm nếu run_id đã có: hai dòng cùng run_id nhưng khác kết quả
        # là dấu hiệu có biến ẩn ngoài hash (num_workers là thủ phạm hay gặp,
        # vì nó đổi trình tự tăng cường dữ liệu). Chúng cũng dùng chung một
        # đường dẫn checkpoint, nên lượt sau ghi đè lượt trước và một trong hai
        # dòng CSV không còn khớp file nào trên đĩa.
        rid = row.get("run_id")
        if rid:
            old = [r for r in self.read_all() if r.get("run_id") == rid]
            if old:
                prev = old[-1].get("test_dice", "?")
                now = row.get("test_dice", "?")
                print(f"[CẢNH BÁO] run_id {rid} đã có trong log "
                      f"(test_dice cũ {prev}, mới {now}). Checkpoint đã bị ghi đè. "
                      f"Chạy merge_logs() để khử trùng trước khi lập bảng.")

        if existing is None:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            return

        new_cols = [k for k in row if k not in existing]
        if new_cols:
            # Có cột mới: đọc lại toàn bộ, mở rộng header, ghi lại.
            rows = self.read_all()
            fieldnames = existing + new_cols
            rows.append(row)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return

        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=existing).writerow(row)

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.read_all())

    def _read_header(self) -> List[str] | None:
        if not self.path.exists():
            return None
        with self.path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        return header


def merge_logs(paths, out_path: str | Path = "logs/runs.csv") -> int:
    """Gộp nhiều runs.csv (từ các tài khoản Colab khác nhau) thành một.

    Khử trùng lặp theo run_id, giữ bản ghi mới nhất. Trả về số dòng cuối cùng.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    fieldnames: List[str] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"[bỏ qua] không thấy {path}")
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
                run_id = row.get("run_id", "")
                if run_id not in seen or row.get("datetime", "") > seen[run_id].get("datetime", ""):
                    seen[run_id] = row

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(seen.values())
    return len(seen)
