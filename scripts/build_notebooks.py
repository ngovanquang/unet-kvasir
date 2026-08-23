"""Sinh các notebook trong notebooks/.

Viết JSON .ipynb bằng tay rất dễ sai, nên định nghĩa cell ở đây rồi sinh ra.
Sửa nội dung notebook thì sửa file này và chạy lại:

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"

BOOTSTRAP = """# Chạy được cả trên Colab lẫn máy cá nhân
import sys, os
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not Path("unet-kvasir").exists():
    # !git clone <repo cua nhom> unet-kvasir
    pass
ROOT = Path("unet-kvasir") if Path("unet-kvasir").exists() else Path("..")
os.chdir(ROOT.resolve())
sys.path.insert(0, str(Path.cwd()))

%load_ext autoreload
%autoreload 2

from src import *
print("Thư mục làm việc:", Path.cwd())
print("Thiết bị:", get_device())"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


def notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "00_setup_and_data.ipynb": [
        md("""# 00 · Môi trường và dữ liệu

Chạy notebook này **một lần duy nhất** khi bắt đầu. Nó tải Kvasir-SEG, chia
train/val/test cố định, và tính tỉ lệ mất cân bằng lớp — con số dùng cho
mục Dữ liệu trong báo cáo.

Người phụ trách: **SV B**."""),
        code(BOOTSTRAP),
        md("## 1. Tải dữ liệu\n\nBỏ qua ô này nếu `data/Kvasir-SEG` đã có."),
        code("""from src.data import KVASIR_URL

if not Path("data/Kvasir-SEG/images").is_dir():
    !wget -q {KVASIR_URL} -O kvasir-seg.zip
    !mkdir -p data && unzip -q kvasir-seg.zip -d data/ && rm kvasir-seg.zip

n_images = len(list(Path("data/Kvasir-SEG/images").glob("*.jpg")))
print(f"Số ảnh: {n_images}")"""),
        md("""## 2. Chia tập cố định

Chia **một lần** rồi ghi ra `splits/*.txt` và commit vào git. Không bao giờ
chia lại ngẫu nhiên, nếu không mọi so sánh giữa các cấu hình sẽ vô nghĩa."""),
        code("""cfg = Config()
splits = make_splits(cfg.data_root, cfg.split_dir, seed=cfg.seed)
for name, names in splits.items():
    print(f"{name:5s}: {len(names):4d} ảnh")"""),
        md("""## 3. Mức mất cân bằng lớp

Đây là lý do tồn tại của cả nghiên cứu về hàm mất mát. Đưa con số này vào
báo cáo, và dùng `pos_weight` cho Weighted BCE."""),
        code("""ratio = foreground_ratio(cfg.data_root, splits["train"], sample=None)
pw = pos_weight_from_ratio(ratio)
print(f"Tỉ lệ pixel polyp : {ratio:.4f}  ({ratio*100:.2f}%)")
print(f"pos_weight        : {pw:.2f}")"""),
        md("## 4. Xem thử vài mẫu\n\nKiểm mắt thường rằng ảnh và mask khớp nhau."),
        code("""import matplotlib.pyplot as plt

ds = KvasirSegDataset(cfg.data_root, splits["train"][:6],
                      SegTransform(cfg.image_size, train=False))
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for i in range(6):
    img, msk = ds[i]
    axes[0, i].imshow(denormalize(img)); axes[0, i].axis("off")
    axes[1, i].imshow(msk[0], cmap="gray"); axes[1, i].axis("off")
axes[0, 0].set_title("ảnh", loc="left"); axes[1, 0].set_title("mask", loc="left")
plt.tight_layout()"""),
        md("""## 5. Kiểm chứng mask đã nhị phân

Mask Kvasir lưu dạng JPEG nên có nhiễu nén. Sau khi qua `SegTransform`,
tensor mask chỉ được chứa đúng hai giá trị 0 và 1."""),
        code("""import torch
vals = torch.unique(torch.cat([ds[i][1].flatten() for i in range(6)]))
print("Các giá trị có trong mask:", vals.tolist())
assert set(vals.tolist()) <= {0.0, 1.0}, "Mask chưa được nhị phân hoá!"
print("Mask đã nhị phân đúng.")"""),
    ],

    "01_train_baseline.ipynb": [
        md("""# 01 · Huấn luyện cấu hình gốc

Một lượt chạy end-to-end với cấu hình nền: `transpose` + `full skip` +
`BCE+Dice`. Mọi thí nghiệm sau đều so với cấu hình này.

Người phụ trách: **SV A**."""),
        code(BOOTSTRAP),
        code("""cfg = Config(
    loss_name="bce_dice",
    up_mode="transpose",
    skip_mode="full",
    epochs=40,
    batch_size=8,
    seed=42,
)
print("run_id:", cfg.run_id)"""),
        md("""Nếu Colab báo hết VRAM, hạ `base_channels=32` hoặc `batch_size=4`.
Nhớ ghi lại thay đổi đó trong báo cáo vì nó ảnh hưởng tới mọi so sánh."""),
        code("""result = run_experiment(cfg, skip_if_logged=False)"""),
        md("## Đường cong huấn luyện"),
        code("""from src.viz import plot_history

fig = plot_history(result["history"], title=cfg.run_id,
                   save_path=f"{cfg.fig_dir}/history_{cfg.run_id}.png")"""),
        md("""## Đọc gì từ hình này

- `train_loss` giảm nhưng `val_loss` tăng dần từ một epoch nào đó → quá khớp.
- `val_dice` dao động mạnh giữa các epoch → learning rate còn cao.
- `val_dice` phẳng ngay từ đầu → nghi ngờ lỗi dữ liệu, kiểm lại notebook 00."""),
        md("## Xem thử dự đoán"),
        code("""from src.viz import comparison_grid

splits = load_splits(cfg.split_dir)
test_ds = KvasirSegDataset(cfg.data_root, splits["test"],
                           SegTransform(cfg.image_size, train=False))
fig = comparison_grid(test_ds, {"baseline": result["model"]},
                      indices=range(4), device=get_device())"""),
    ],

    "02_loss_comparison.ipynb": [
        md("""# 02 · So sánh 6 hàm mất mát

Yêu cầu bắt buộc của đề bài. Kiến trúc giữ **cố định** (`transpose` +
`full skip`), chỉ đổi đúng một biến là hàm mất mát.

Sáu lượt chạy, mỗi lượt 20–30 phút trên T4. Nên chia hai tài khoản Colab:
tài khoản 1 chạy `bce`, `weighted_bce`, `dice`; tài khoản 2 chạy `bce_dice`,
`focal`, `tversky`. Cuối cùng gộp hai file `logs/runs.csv` lại.

Người phụ trách: **SV A** chạy, **SV B** tổng hợp."""),
        code(BOOTSTRAP),
        code("""base = Config(up_mode="transpose", skip_mode="full", epochs=40, seed=42)

# Đổi lát cắt này khi chia việc giữa hai tài khoản Colab
MY_LOSSES = LOSS_NAMES            # hoặc LOSS_NAMES[:3] / LOSS_NAMES[3:]
configs = [base.replace(loss_name=name, tag="loss_sweep") for name in MY_LOSSES]

for c in configs:
    print(c.run_id)"""),
        code("""results = run_sweep(configs)"""),
        md("""## Bảng so sánh

Đọc thẳng từ `logs/runs.csv` chứ không gõ số bằng tay — đây là cách duy nhất
đảm bảo báo cáo không lệch với log."""),
        code("""import pandas as pd

df = RunLogger(base.log_csv).to_dataframe()
df = df[df["tag"] == "loss_sweep"]
cols = ["loss_name", "test_dice", "test_iou", "test_precision", "test_recall",
        "best_epoch", "train_time_min"]
table = df[cols].astype({c: float for c in cols[1:]}).sort_values("test_dice",
                                                                  ascending=False)
table.round(4)"""),
        code("""print(table.round(4).to_markdown(index=False))   # dán thẳng vào báo cáo"""),
        md("## Đường cong val Dice của các hàm mất mát"),
        code("""from src.viz import plot_loss_curves

histories = {r["loss_name"]: r["history"] for r in results.values()
             if not r.get("skipped")}
fig = plot_loss_curves(histories, key="val_dice",
                       save_path=f"{base.fig_dir}/loss_sweep_valdice.png")"""),
        md("""## Câu hỏi cần trả lời trong báo cáo

1. Dice-based loss có thắng BCE không, và chênh bao nhiêu? Chênh đó có lớn
   hơn dao động giữa các seed không?
2. Weighted BCE và Focal đều nhắm vào mất cân bằng lớp — cái nào hiệu quả hơn
   ở đây, và tại sao?
3. Tversky với `beta > alpha` có làm recall tăng và precision giảm đúng như
   kỳ vọng lý thuyết không? Kiểm bằng đúng hai cột đó trong bảng."""),
        md("""## Lặp seed cho 3 cấu hình tốt nhất

Không bắt buộc với đề tài 5, nhưng đây là chỗ ăn điểm mục *Thí nghiệm và
phân tích*: nó cho phép nói câu "khác biệt này nằm trong nhiễu"."""),
        code("""top3 = table["loss_name"].head(3).tolist()
seed_configs = [base.replace(loss_name=name, seed=1337, tag="loss_seed2")
                for name in top3]
# results_seed2 = run_sweep(seed_configs)"""),
    ],

    "03_ablation_architecture.ipynb": [
        md("""# 03 · Ablation kiến trúc

Hai thí nghiệm bắt buộc còn lại, đều dùng hàm mất mát tốt nhất tìm được ở
notebook 02:

1. **Cách tăng mẫu**: `transpose` / `bilinear` / `unpool`
2. **Skip connection**: `full` / `half` / `none`

Cấu hình gốc đã chạy rồi nên chỉ cần thêm 4 lượt mới.

Người phụ trách: **SV A**."""),
        code(BOOTSTRAP),
        code("""BEST_LOSS = "bce_dice"   # cập nhật theo kết quả notebook 02
base = Config(loss_name=BEST_LOSS, epochs=40, seed=42)

up_configs = [base.replace(up_mode=m, skip_mode="full", tag="ablation_up")
              for m in UP_MODES]
skip_configs = [base.replace(up_mode="transpose", skip_mode=m, tag="ablation_skip")
                for m in SKIP_MODES]

for c in up_configs + skip_configs:
    print(f"{c.run_id:50s}")"""),
        md("""Số tham số **không bằng nhau** giữa ba cách tăng mẫu và ba mức skip.
Nêu rõ điểm này trong báo cáo như một hạn chế của thiết kế thí nghiệm — giám
khảo hay hỏi đúng chỗ này."""),
        code("""for c in up_configs + skip_configs:
    m = build_model(c)
    print(f"{c.up_mode:10s} {c.skip_mode:5s} {count_parameters(m):>12,} tham số")
    del m"""),
        code("""results = run_sweep(up_configs + skip_configs)"""),
        md("## Hai bảng ablation"),
        code("""import pandas as pd

df = RunLogger(base.log_csv).to_dataframe()
num = ["test_dice", "test_iou", "test_precision", "test_recall", "n_params"]
df[num] = df[num].astype(float)

print("=== Cách tăng mẫu ===")
print(df[df.tag == "ablation_up"][["up_mode"] + num].round(4).to_markdown(index=False))
print("\\n=== Skip connection ===")
print(df[df.tag == "ablation_skip"][["skip_mode"] + num].round(4).to_markdown(index=False))"""),
        md("""## Soi checkerboard artifact

Phải phóng to sát biên polyp mới thấy. Không zoom thì phần bình luận trong
báo cáo sẽ thành nói suông."""),
        code("""from src.viz import zoom_on_border, predict_batch
import numpy as np

splits = load_splits(base.split_dir)
test_ds = KvasirSegDataset(base.data_root, splits["test"],
                           SegTransform(base.image_size, train=False))
img, msk = test_ds[0]

for cfg_ in up_configs:
    model = load_best(build_model(cfg_), cfg_.ckpt_path, get_device())
    pred, _ = predict_batch(model, img.unsqueeze(0), get_device())
    fig = zoom_on_border(denormalize(img), pred[0, 0].numpy(), box=(90, 90, 70, 70),
                         save_path=f"{base.fig_dir}/zoom_{cfg_.up_mode}.png")
    fig.suptitle(cfg_.up_mode)"""),
        md("""## Câu hỏi cần trả lời

1. `bilinear` có ít artifact hơn `transpose` không? Có bằng chứng hình ảnh chưa?
2. Bỏ skip làm Dice tụt bao nhiêu? Tụt nhiều nhất ở loại ảnh nào — polyp nhỏ
   hay polyp có biên phức tạp?
3. `half skip` nằm ở đâu giữa `full` và `none`? Nếu nó gần `full` thì kết luận
   là gì về vai trò của các tầng nông?"""),
    ],

    "04_figures_and_report.ipynb": [
        md("""# 04 · Hình và bảng cho báo cáo

Gom mọi thứ đã chạy thành sản phẩm nộp. Không huấn luyện gì thêm ở đây.

Người phụ trách: **SV B**."""),
        code(BOOTSTRAP),
        code("""import pandas as pd

cfg = Config()
df = RunLogger(cfg.log_csv).to_dataframe()
num = [c for c in df.columns if c.startswith(("test_", "best_val", "n_params"))]
df[num] = df[num].astype(float)
print(f"Tổng số lượt chạy trong log: {len(df)}")
df[["run_id", "loss_name", "up_mode", "skip_mode", "seed", "test_dice"]].round(4)"""),
        md("""## Bảng tổng hợp cho báo cáo"""),
        code("""summary = (df[["loss_name", "up_mode", "skip_mode", "seed", "n_params",
                "test_dice", "test_iou", "test_precision", "test_recall",
                "train_time_min"]]
           .sort_values("test_dice", ascending=False).round(4))
print(summary.to_markdown(index=False))"""),
        md("""## Lưới ảnh so sánh (yêu cầu tối thiểu 10 ảnh)

Đề bài yêu cầu: ảnh gốc / mask thật / dự đoán của 2–3 cấu hình tốt nhất và
1 cấu hình tệ nhất."""),
        code("""from src.viz import comparison_grid

best_rows = df.nlargest(3, "test_dice")
worst_row = df.nsmallest(1, "test_dice").iloc[0]

def cfg_from_row(row):
    return Config(loss_name=row["loss_name"], up_mode=row["up_mode"],
                  skip_mode=row["skip_mode"], seed=int(row["seed"]))

models = {}
for _, row in best_rows.iterrows():
    c = cfg_from_row(row)
    models[f"{row.loss_name}/{row.up_mode}/{row.skip_mode}"] = \\
        load_best(build_model(c), c.ckpt_path, get_device())
c = cfg_from_row(worst_row)
models[f"TỆ NHẤT: {worst_row.skip_mode}"] = \\
    load_best(build_model(c), c.ckpt_path, get_device())

splits = load_splits(cfg.split_dir)
test_ds = KvasirSegDataset(cfg.data_root, splits["test"],
                           SegTransform(cfg.image_size, train=False))"""),
        code("""fig = comparison_grid(test_ds, models, indices=range(10), device=get_device(),
                      save_path=f"{cfg.fig_dir}/comparison_grid.png")"""),
        md("""## Ảnh mô hình làm tệ nhất

Nguyên liệu cho mục *Phân tích và thảo luận*. Nhìn kỹ xem chúng có điểm chung
gì: polyp quá nhỏ, biên mờ, có bọt khí, ánh sáng chói, hay nhiều polyp."""),
        code("""from src.viz import worst_case_indices
from torch.utils.data import DataLoader

best_name = list(models)[0]
loader = DataLoader(test_ds, batch_size=8, shuffle=False)
worst = worst_case_indices(models[best_name], loader, get_device(), k=6)
print("Chỉ số 6 ảnh tệ nhất:", worst)

fig = comparison_grid(test_ds, {best_name: models[best_name]}, indices=worst,
                      device=get_device(),
                      save_path=f"{cfg.fig_dir}/worst_cases.png")"""),
        md("""## Danh sách kiểm trước khi nộp

- [ ] `logs/runs.csv` có đủ mọi lượt chạy, kể cả lượt hỏng
- [ ] Mọi số trong báo cáo đều đọc từ CSV, không gõ tay
- [ ] `README.md` chạy lại được từ đầu trên máy sạch
- [ ] `requirements.txt` khớp với môi trường thật
- [ ] Bảng phân công có tỉ lệ đóng góp cộng lại đúng 100%
- [ ] Mục *Khai báo sử dụng công cụ AI* ở cuối báo cáo
- [ ] Hình nào cũng có chú thích và được nhắc tới trong nội dung"""),
    ],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = OUT / name
        path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"đã ghi {path.relative_to(OUT.parent)}  ({len(cells)} cell)")


if __name__ == "__main__":
    main()
