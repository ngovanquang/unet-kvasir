#!/usr/bin/env python3
"""Gom toàn bộ kết quả thí nghiệm thành MỘT file markdown để review.

    python scripts/export_review.py                 # phần chạy trên CPU
    python scripts/export_review.py --with-gpu      # thêm phân tích theo ảnh

Kết quả ghi ra outputs/review_report.md. File này tự chứa: cấu hình môi
trường, thống kê dữ liệu, toàn bộ bảng kết quả, tóm tắt hình dạng từng đường
cong, kiểm tra nhất quán, và danh sách cảnh báo tự động.

Đường cong được TÓM TẮT chứ không xuất thô: 15 lượt x 120 epoch là 1.800 dòng,
không ai đọc nổi. Thay vào đó script tính các chỉ số mô tả hình dạng (số lần
sụp, biên độ sụp lớn nhất, độ dốc 20 epoch cuối, độ phẳng) — đủ để chẩn đoán
mà chỉ tốn một dòng mỗi lượt.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import Config, RunLogger  # noqa: E402

OUT: list[str] = []
NUMERIC_PREFIX = ("test_", "best_val", "n_params", "train_time", "focal_",
                  "tversky_", "dice_smooth", "pos_weight", "lr", "weight_decay")
INT_COLS = ("epochs", "seed", "best_epoch", "epochs_run", "batch_size",
            "image_size", "base_channels", "depth", "num_workers")


def w(line: str = "") -> None:
    OUT.append(line)


def table(df: pd.DataFrame, floatfmt: str = ".4f") -> None:
    if df.empty:
        w("_(không có dòng nào)_")
    else:
        w(df.to_markdown(index=False, floatfmt=floatfmt))
    w()


# ----------------------------------------------------------------- môi trường
def section_env(cfg: Config) -> None:
    w("## 1. Môi trường")
    w()
    rows = [("Python", platform.python_version()),
            ("Hệ điều hành", platform.platform())]
    try:
        import torch
        rows.append(("PyTorch", torch.__version__))
        rows.append(("CUDA khả dụng", str(torch.cuda.is_available())))
        if torch.cuda.is_available():
            rows.append(("GPU", torch.cuda.get_device_name(0)))
    except Exception as exc:  # pragma: no cover
        rows.append(("PyTorch", f"không import được: {exc}"))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                        stderr=subprocess.DEVNULL, text=True).strip()
        rows.append(("Git commit", sha + (" (có thay đổi chưa commit)" if dirty else "")))
    except Exception:
        rows.append(("Git commit", "không xác định"))
    table(pd.DataFrame(rows, columns=["Mục", "Giá trị"]), floatfmt="")

    w("Giá trị mặc định trong `Config` hiện tại:")
    w()
    d = cfg.to_dict()
    keys = ["image_size", "batch_size", "epochs", "lr", "weight_decay", "seed",
            "early_stop_patience", "threshold", "base_channels", "depth",
            "focal_alpha", "focal_gamma", "tversky_alpha", "tversky_beta", "augment"]
    # str() để pandas không ép cả cột về float (256 -> 256.0, True -> 1.0)
    table(pd.DataFrame([(k, str(d[k])) for k in keys if k in d],
                       columns=["Tham số", "Giá trị"]), floatfmt="")


# ------------------------------------------------------------------- dữ liệu
def section_data(cfg: Config, dup_threshold: float) -> None:
    from PIL import Image

    from src import load_splits

    w("## 2. Dữ liệu")
    w()
    root = Path(cfg.data_root)
    if not (root / "images").is_dir():
        w(f"_Không tìm thấy `{root}/images`, bỏ qua mục này._")
        w()
        return

    splits = load_splits(cfg.split_dir)
    rows, areas_by_split = [], {}
    for name in ("train", "val", "test"):
        a = np.array([(np.asarray(Image.open(root / "masks" / n).convert("L")) > 127).mean()
                      for n in splits[name]])
        areas_by_split[name] = a
        rows.append((name, len(a), a.mean(), np.median(a), a.min(), a.max()))
    table(pd.DataFrame(rows, columns=["split", "số ảnh", "tỉ lệ polyp TB",
                                      "trung vị", "nhỏ nhất", "lớn nhất"]))

    all_a = np.concatenate(list(areas_by_split.values()))
    trivial_micro = 2 * all_a.mean() / (1 + all_a.mean())
    trivial_macro = np.mean(2 * all_a / (1 + all_a))
    w(f"- Tỉ lệ pixel polyp toàn bộ: **{all_a.mean():.4f}** ({all_a.mean()*100:.2f}%)")
    w(f"- `pos_weight` tương ứng: **{(1 - all_a.mean()) / all_a.mean():.2f}** "
      f"(mất cân bằng 1:{(1 - all_a.mean()) / all_a.mean():.1f})")
    w(f"- Đường cơ sở tầm thường (đoán tất cả là polyp): "
      f"Dice macro **{trivial_macro:.4f}**, micro {trivial_micro:.4f}")
    w()

    bins = pd.cut(all_a, [0, .05, .15, .35, 1.0],
                  labels=["rất nhỏ (<5%)", "nhỏ (5-15%)", "vừa (15-35%)", "lớn (>35%)"])
    dist = pd.crosstab(bins, np.concatenate([[k] * len(v) for k, v in areas_by_split.items()]))
    w("Phân bố kích thước polyp theo split:")
    w()
    w(dist.to_markdown())
    w()

    # Thành phần liên thông: ảnh có nhiều hơn một polyp
    try:
        from scipy import ndimage
        multi = 0
        for n in splits["test"]:
            m = np.asarray(Image.open(root / "masks" / n).convert("L")) > 127
            lab, k = ndimage.label(m)
            if k and (ndimage.sum(m, lab, range(1, k + 1)) > 100).sum() > 1:
                multi += 1
        w(f"- Ảnh test có nhiều hơn một polyp: **{multi} / {len(splits['test'])}**")
    except ImportError:
        w("- _scipy chưa cài, bỏ qua đếm thành phần liên thông._")

    # Ảnh gần trùng
    def thumb(p: Path, size: int = 32) -> np.ndarray:
        a = np.asarray(Image.open(p).convert("L").resize((size, size), Image.BILINEAR),
                       dtype=np.float32)
        return ((a - a.mean()) / (a.std() + 1e-8)).ravel()

    names = splits["train"] + splits["val"] + splits["test"]
    origin = (["train"] * len(splits["train"]) + ["val"] * len(splits["val"])
              + ["test"] * len(splits["test"]))
    X = np.stack([thumb(root / "images" / n) for n in names])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = np.triu(X @ X.T, k=1)
    i, j = np.where(S > dup_threshold)
    w(f"- Cặp ảnh tương đồng > {dup_threshold} trong toàn bộ {len(names)} ảnh: **{len(i)}**")
    if len(i):
        dup = pd.DataFrame({
            "similarity": S[i, j], "ảnh A": [names[a] for a in i],
            "split A": [origin[a] for a in i], "ảnh B": [names[b] for b in j],
            "split B": [origin[b] for b in j],
        }).sort_values("similarity", ascending=False).head(15)
        dup["vắt chéo split"] = dup["split A"] != dup["split B"]
        w()
        table(dup)
    te = np.array([k for k, o in enumerate(origin) if o == "test"])
    tr = np.array([k for k, o in enumerate(origin) if o == "train"])
    best = (X[te] @ X[tr].T).max(axis=1)
    w("Phân vị của độ tương đồng cao nhất giữa mỗi ảnh test và tập train "
      "(để biết ngưỡng nào là bất thường):")
    w()
    w(" | ".join(f"p{q}={np.percentile(best, q):.3f}" for q in (50, 75, 90, 95, 99)))
    w()


# ------------------------------------------------------------- tóm tắt đường cong
def curve_stats(history: list[dict], key: str = "val_dice") -> dict:
    """Mô tả hình dạng đường cong bằng vài con số thay vì xuất toàn bộ."""
    v = np.array([h[key] for h in history], dtype=float)
    n = len(v)
    if n == 0:
        return {}
    peak = v.max()
    drops = np.diff(v)
    tail = v[-20:] if n >= 20 else v
    slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] if len(tail) > 1 else 0.0
    return {
        "n_ep": n,
        "đỉnh": round(float(peak), 4),
        "ep đỉnh": int(v.argmax()) + 1,
        "cuối": round(float(v[-1]), 4),
        "sụp <50% đỉnh": int((v < 0.5 * peak).sum()),
        # max(0, ...) vì nếu đường chỉ đi lên thì -min(diff) ra số âm, vô nghĩa
        "sụp mạnh nhất": round(max(0.0, float(-drops.min())) if len(drops) else 0.0, 4),
        "TB 20ep cuối": round(float(tail.mean()), 4),
        "SD 20ep cuối": round(float(tail.std()), 4),
        "dốc 20ep cuối": round(float(slope), 5),
    }


def section_curves(cfg: Config, df: pd.DataFrame) -> None:
    w("## 4. Tóm tắt hình dạng đường cong")
    w()
    w("`sụp <50% đỉnh` đếm số epoch mà val Dice rơi xuống dưới nửa giá trị đỉnh — "
      "chỉ số bất ổn định. `dốc 20ep cuối` dương rõ nghĩa là còn đang lên, tức "
      "chưa hội tụ. `SD 20ep cuối` lớn nghĩa là dao động chưa tắt.")
    w()
    ckpt_dir = Path(cfg.ckpt_dir)
    rows, missing = [], []
    for _, r in df.iterrows():
        path = ckpt_dir / f"{r.run_id}_history.json"
        if not path.exists():
            missing.append(r.run_id)
            continue
        st = curve_stats(json.loads(path.read_text(encoding="utf-8")))
        rows.append({"loss": r.loss_name, "up": r.up_mode, "skip": r.skip_mode,
                     "lr": r.lr, "ep": r.epochs, "seed": r.seed, **st})
    if rows:
        table(pd.DataFrame(rows).sort_values("đỉnh", ascending=False))
    if missing:
        w(f"_Thiếu file history cho {len(missing)} lượt "
          f"(chạy bằng bản code trước khi có `save_history`):_")
        w("`" + "`, `".join(missing) + "`")
        w()


def section_noise(df: pd.DataFrame) -> None:
    """So sánh cỡ hiệu ứng với mức nhiễu đo được.

    Đây là mục quan trọng nhất của báo cáo: không có nó thì mọi bảng xếp hạng
    chỉ là liệt kê con số mà không biết con số nào đáng tin.
    """
    w("## 5b. Nhiễu và cỡ hiệu ứng")
    w()

    # --- Nguồn nhiễu 1: cùng run_id, chạy nhiều lần ---
    dup = df.groupby("run_id")["test_dice"].agg(["min", "max", "count"])
    dup = dup[dup["count"] > 1]
    noise_same = 0.0
    if len(dup):
        dup["biên độ"] = (dup["max"] - dup["min"]).round(5)
        noise_same = float(dup["biên độ"].max())
        w("**Nguồn nhiễu 1 — cùng cấu hình, chạy lại.** Cùng `run_id` nghĩa là "
          "cùng seed và cùng mọi siêu tham số trong hash. Biên độ khác 0 là dấu "
          "hiệu có biến ẩn ngoài hash (thường là `num_workers`, vì nó đổi trình "
          "tự tăng cường dữ liệu).")
        w()
        table(dup.reset_index()[["run_id", "count", "min", "max", "biên độ"]], ".5f")
        if "num_workers" in df.columns:
            nw = df[df.run_id.isin(dup.index)][["run_id", "num_workers", "test_dice"]]
            w("Số worker của các lượt trùng:")
            w()
            table(nw.sort_values(["run_id", "num_workers"]), ".5f")

    # --- Nguồn nhiễu 2: đổi seed ---
    key = ["loss_name", "up_mode", "skip_mode", "epochs", "lr"]
    agg = (df.groupby(key)
             .agg(n_seed=("seed", "nunique"), n_run=("test_dice", "size"),
                  mean=("test_dice", "mean"), std=("test_dice", "std"),
                  lo=("test_dice", "min"), hi=("test_dice", "max"))
             .reset_index())
    multi = agg[agg.n_seed > 1].copy()
    noise_seed = 0.0
    if len(multi):
        multi["biên độ"] = (multi.hi - multi.lo).round(4)
        noise_seed = float(multi["biên độ"].max())
        w("**Nguồn nhiễu 2 — đổi seed khởi tạo.** Đây là mức nhiễu cần dùng làm "
          "chuẩn để phán xét mọi khác biệt giữa các cấu hình.")
        w()
        table(multi[key + ["n_seed", "mean", "std", "biên độ"]]
              .sort_values("biên độ", ascending=False))
        w(f"- Biên độ giữa seed lớn nhất: **{noise_seed:.4f}**")
        w(f"- Trung bình: **{multi['biên độ'].mean():.4f}**")
        w()
    else:
        w("_Chưa có cấu hình nào chạy từ 2 seed trở lên — không đánh giá được "
          "khác biệt nào là thật._")
        w()

    if noise_seed == 0 and noise_same == 0:
        return
    floor_typ = float(multi["biên độ"].mean()) if len(multi) else noise_same
    floor_max = max(noise_seed, noise_same)

    # --- Cỡ hiệu ứng trên từng trục ablation ---
    w(f"**Hai ngưỡng phán xét**: điển hình `{floor_typ:.4f}` (trung bình biên độ "
      f"giữa seed) và thận trọng `{floor_max:.4f}` (biên độ lớn nhất quan sát "
      f"được). Khác biệt nhỏ hơn ngưỡng điển hình thì chắc chắn là nhiễu; nằm "
      f"giữa hai ngưỡng thì chưa kết luận được.")
    w()
    w("Mỗi trục dưới đây giữ cố định mọi thứ khác và chỉ đổi một biến. Giá trị "
      "là trung bình qua các seed có sẵn.")
    w()

    def axis(name: str, axis_col: str, fixed: dict) -> dict | None:
        sub = agg.copy()
        for k, v in fixed.items():
            sub = sub[sub[k] == v]
        sub = sub.groupby(axis_col)["mean"].mean()
        if len(sub) < 2:
            return None
        return {"trục": name, "số mức": len(sub),
                "thấp nhất": round(float(sub.min()), 4),
                "cao nhất": round(float(sub.max()), 4),
                "biên độ": round(float(sub.max() - sub.min()), 4),
                "chi tiết": ", ".join(f"{k}={v:.4f}" for k, v in
                                      sub.sort_values(ascending=False).items())}

    ep = int(agg.epochs.mode().iloc[0])
    lr = float(agg.lr.mode().iloc[0])
    # Hàm mất mát "nền" của ablation = cái được khảo sát trên nhiều biến thể
    # kiến trúc nhất. KHÔNG lấy cái có Dice cao nhất: dòng đó thường là một
    # lượt chẩn đoán ở learning rate khác, chỉ có duy nhất một biến thể.
    fam = agg[(agg.epochs == ep) & (agg.lr == lr)]
    if fam.empty:
        return
    base_loss = (fam.groupby("loss_name")
                    .apply(lambda g: len(g.drop_duplicates(["up_mode", "skip_mode"])),
                           include_groups=False)
                    .idxmax())
    w(f"Hàm mất mát nền của ablation: `{base_loss}`, tại {ep} epoch, lr={lr:g}.")
    w()
    rows = [r for r in [
        axis("hàm mất mát", "loss_name",
             {"up_mode": "transpose", "skip_mode": "full", "epochs": ep, "lr": lr}),
        axis("cách tăng mẫu", "up_mode",
             {"loss_name": base_loss, "skip_mode": "full", "epochs": ep, "lr": lr}),
        axis("skip connection", "skip_mode",
             {"loss_name": base_loss, "up_mode": "transpose", "epochs": ep, "lr": lr}),
    ] if r]
    if not rows:
        return
    eff = pd.DataFrame(rows)
    eff["kết luận"] = np.where(
        eff["biên độ"] > floor_max, "VƯỢT nhiễu",
        np.where(eff["biên độ"] > floor_typ, "chưa kết luận", "trong nhiễu"))
    table(eff[["trục", "số mức", "thấp nhất", "cao nhất", "biên độ", "kết luận"]])
    for r in rows:
        w(f"- {r['trục']}: {r['chi tiết']}")
    w()
    w("Trục nào có biên độ nhỏ hơn ngưỡng nhiễu thì **không được xếp hạng** — "
      "phát biểu đúng là khác biệt nằm trong dao động giữa các lần khởi tạo.")
    w()


# ------------------------------------------------------------------ cảnh báo
def section_checks(cfg: Config, df: pd.DataFrame) -> None:
    w("## 5. Kiểm tra tự động")
    w()
    issues: list[str] = []

    # Nhất quán Dice/IoU: với từng ảnh IoU = D/(2-D); sau khi lấy TB thì IoU >= f(D)
    bad = df[df.test_iou < df.test_dice / (2 - df.test_dice) - 1e-6]
    if len(bad):
        issues.append(f"**{len(bad)} dòng vi phạm quan hệ Dice/IoU** — nghi lỗi metric: "
                      + ", ".join(bad.run_id))
    else:
        w("- Quan hệ Dice/IoU: **tất cả các dòng đều nhất quán** (IoU ≥ Dice/(2−Dice)).")

    # Dice có phải trung bình điều hoà của precision/recall micro
    hm = 2 * df.test_precision * df.test_recall / (df.test_precision + df.test_recall)
    gap = (df.test_dice - hm).abs()
    w(f"- Chênh lệch Dice macro so với F1 micro: trung vị {gap.median():.4f}, "
      f"lớn nhất {gap.max():.4f} (`{df.loc[gap.idxmax(), 'run_id']}`). "
      "Chênh lớn nghĩa là mô hình thất bại chọn lọc trên một nhóm nhỏ ảnh.")

    # Lượt bị cắt sớm
    cut = df[df.epochs_run < 0.5 * df.epochs]
    if len(cut):
        issues.append("**Lượt bị early stopping cắt trước nửa ngân sách** — "
                      "phải chú thích trong báo cáo:\n\n"
                      + cut[["run_id", "loss_name", "best_epoch", "epochs_run",
                             "test_dice"]].to_markdown(index=False))

    # Thiếu seed thứ hai
    if df.seed.nunique() < 2:
        issues.append("**Toàn bộ log chỉ có một seed.** Không thể phát biểu khác biệt "
                      "nào là thật hay nhiễu.")
    else:
        grp = df.groupby(["loss_name", "up_mode", "skip_mode", "epochs", "lr"])
        rep = grp.filter(lambda g: g.seed.nunique() > 1)
        w(f"- Số cấu hình có từ 2 seed trở lên: "
          f"**{rep.groupby(['loss_name','up_mode','skip_mode','epochs','lr']).ngroups}**")

    # Checkpoint còn không
    gone = [r.run_id for _, r in df.iterrows()
            if not (Path(cfg.ckpt_dir) / f"{r.run_id}.pt").exists()]
    if gone:
        issues.append(f"**{len(gone)}/{len(df)} checkpoint không còn trên đĩa** — "
                      "notebook 04 sẽ không nạp được: `" + "`, `".join(gone[:10]) + "`")
    else:
        w(f"- Checkpoint: **đủ cả {len(df)} lượt**.")

    # Cấu hình trùng nhau chỉ khác lr
    dup = (df.groupby(["loss_name", "up_mode", "skip_mode", "epochs", "seed"])["lr"]
             .nunique().reset_index())
    dup = dup[dup.lr > 1]
    if len(dup):
        w("- Cấu hình có nhiều learning rate (nhớ lọc `lr` khi lập bảng chính):")
        w()
        table(dup.rename(columns={"lr": "số lr khác nhau"}), floatfmt="")

    w()
    if issues:
        w("### Cần xử lý")
        w()
        for it in issues:
            w(f"- {it}")
            w()
    else:
        w("_Không phát hiện vấn đề nào._")
        w()


# ------------------------------------------------------------- phân tích GPU
def section_gpu(cfg: Config, df: pd.DataFrame, top_k: int) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    from src import (KvasirSegDataset, SegTransform, build_model, dice_per_image,
                     get_device, load_best, load_splits)

    w("## 6. Phân tích theo ảnh (cần GPU)")
    w()
    splits = load_splits(cfg.split_dir)
    ds = KvasirSegDataset(cfg.data_root, splits["test"],
                          SegTransform(cfg.image_size, train=False))
    areas = np.array([(np.asarray(Image.open(Path(cfg.data_root) / "masks" / n)
                                  .convert("L")) > 127).mean() for n in splits["test"]])
    bins = pd.cut(areas, [0, .05, .15, .35, 1.0],
                  labels=["rất nhỏ", "nhỏ", "vừa", "lớn"])
    device = get_device()

    PROTO = Config().to_dict()

    def cfg_from_row(row):
        d = {}
        for k, default in PROTO.items():
            if k not in row or pd.isna(row[k]):
                continue
            v = row[k]
            if isinstance(default, bool):
                d[k] = str(v).strip().lower() in ("true", "1")
            elif default is None:
                d[k] = None if str(v).strip() in ("", "None", "nan") else float(v)
            else:
                d[k] = type(default)(v)
        return Config(**d)

    scores = {}
    for _, row in df.nlargest(top_k, "test_dice").iterrows():
        c = cfg_from_row(row)
        if c.run_id != row.run_id or not c.ckpt_path.exists():
            w(f"_Bỏ qua `{row.run_id}`: không dựng lại được Config hoặc mất checkpoint._")
            continue
        model = load_best(build_model(c, device), c.ckpt_path, device)
        s = []
        for images, masks in DataLoader(ds, batch_size=8, shuffle=False):
            with torch.no_grad():
                lg = model(images.to(device)).cpu()
            s += dice_per_image(lg, masks, c.threshold).tolist()
        scores[f"{row.loss_name}/{row.up_mode}/{row.skip_mode}"] = np.array(s)
        del model
        torch.cuda.empty_cache()

    if not scores:
        w("_Không nạp được mô hình nào._")
        w()
        return

    tbl = pd.DataFrame(scores)
    tbl["nhóm"] = bins
    w("Dice trung bình theo kích thước polyp:")
    w()
    w(tbl.groupby("nhóm", observed=False).mean().round(4).to_markdown())
    w()
    w(f"Số ảnh mỗi nhóm: {bins.value_counts().sort_index().to_dict()}")
    w()

    mat = np.stack(list(scores.values()))
    spread = mat.std(axis=0)
    worst = np.argsort(mat.mean(axis=0))[:8]
    cont = np.argsort(-spread)[:8]
    w("8 ảnh mọi cấu hình đều làm tệ nhất:")
    w()
    table(pd.DataFrame({"idx": worst, "tên": [splits["test"][i] for i in worst],
                        "tỉ lệ polyp": areas[worst],
                        "Dice TB": mat.mean(axis=0)[worst]}))
    w("8 ảnh các cấu hình bất đồng nhất:")
    w()
    table(pd.DataFrame({"idx": cont, "tên": [splits["test"][i] for i in cont],
                        "tỉ lệ polyp": areas[cont], "SD giữa cấu hình": spread[cont],
                        "Dice TB": mat.mean(axis=0)[cont]}))


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=None, help="đường dẫn runs.csv")
    ap.add_argument("--out", default="outputs/review_report.md")
    ap.add_argument("--with-gpu", action="store_true",
                    help="thêm mục phân tích theo từng ảnh (cần checkpoint + GPU)")
    ap.add_argument("--top-k", type=int, default=4,
                    help="số cấu hình tốt nhất đem phân tích theo ảnh")
    ap.add_argument("--dup-threshold", type=float, default=0.95)
    ap.add_argument("--skip-data", action="store_true", help="bỏ qua mục dữ liệu")
    args = ap.parse_args()

    cfg = Config()
    log_path = args.log or cfg.log_csv
    df = RunLogger(log_path).to_dataframe()
    if df.empty:
        print(f"Không có dữ liệu trong {log_path}", file=sys.stderr)
        sys.exit(1)

    for c in df.columns:
        if c.startswith(NUMERIC_PREFIX):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    w("# Báo cáo tổng hợp kết quả — U-Net Kvasir-SEG")
    w()
    w(f"Sinh tự động bởi `scripts/export_review.py` từ `{log_path}` "
      f"({len(df)} lượt chạy).")
    w()
    section_env(cfg)
    if not args.skip_data:
        try:
            section_data(cfg, args.dup_threshold)
        except Exception as exc:
            w(f"## 2. Dữ liệu\n\n_Lỗi khi phân tích: {exc}_\n")

    w("## 3. Toàn bộ lượt chạy")
    w()
    cols = [c for c in ["loss_name", "up_mode", "skip_mode", "epochs", "lr", "seed",
                        "focal_alpha", "tversky_alpha", "tversky_beta", "n_params",
                        "best_epoch", "epochs_run", "best_val_dice", "test_dice",
                        "test_iou", "test_precision", "test_recall", "train_time_min",
                        "tag"] if c in df.columns]
    table(df[cols].sort_values(["epochs", "test_dice"], ascending=[True, False]))

    for label, sub in [
        ("So sánh hàm mất mát (kiến trúc và lr cố định)",
         df[(df.up_mode == "transpose") & (df.skip_mode == "full")]),
        ("Ablation cách tăng mẫu", df[df.skip_mode == "full"]),
        ("Ablation skip connection", df[df.up_mode == "transpose"]),
    ]:
        w(f"### {label}")
        w()
        keep = ["loss_name", "up_mode", "skip_mode", "epochs", "lr", "seed",
                "n_params", "test_dice", "test_iou", "test_precision", "test_recall"]
        table(sub[[c for c in keep if c in sub.columns]]
              .sort_values("test_dice", ascending=False))

    section_curves(cfg, df)
    section_noise(df)
    section_checks(cfg, df)

    if args.with_gpu:
        try:
            section_gpu(cfg, df, args.top_k)
        except Exception as exc:
            w(f"## 6. Phân tích theo ảnh\n\n_Lỗi: {exc}_\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(OUT), encoding="utf-8")
    size = out.stat().st_size
    print(f"Đã ghi {out}  ({size/1024:.1f} KB, {len(OUT)} dòng)")
    if size > 200_000:
        print("CẢNH BÁO: file khá lớn, cân nhắc thêm --skip-data")


if __name__ == "__main__":
    main()
