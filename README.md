# U-Net cho phân đoạn polyp trên Kvasir-SEG

Bài tập lớn môn Học sâu — **Đề tài 5**: Phân đoạn ảnh y sinh với U-Net và so
sánh giữa các hàm mất mát.

| | |
|---|---|
| Thành viên | SV A (MSSV), SV B (MSSV) |
| Dữ liệu | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) — 1.000 ảnh nội soi + mask polyp |
| Framework | PyTorch, U-Net **tự cài** (không dùng thư viện segmentation dựng sẵn) |

## Chạy lại kết quả

### Trên Google Colab

```python
!git clone <URL repo của nhóm> unet-kvasir
%cd unet-kvasir
!pip install -q -r requirements.txt
```

Rồi mở lần lượt các notebook trong `notebooks/` theo thứ tự 00 → 04.

### Trên máy cá nhân

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

wget https://datasets.simula.no/downloads/kvasir-seg.zip
mkdir -p data && unzip -q kvasir-seg.zip -d data/

python -m tests.smoke_test          # kiểm tra pipeline, ~1 phút, chạy trên CPU
jupyter lab notebooks/
```

Chạy một cấu hình đơn lẻ không cần notebook:

```bash
python scripts/run_experiment.py --loss_name tversky --up_mode bilinear
```

## Cấu trúc

```
src/
  config.py       Dataclass Config — nguồn duy nhất của mọi siêu tham số
  data.py         Dataset, chia tập cố định, thống kê mất cân bằng lớp
  transforms.py   Biến đổi đồng bộ ảnh–mask (resize NEAREST, nhị phân hoá mask)
  models/unet.py  U-Net tham số hoá theo up_mode và skip_mode
  losses.py       6 hàm mất mát: BCE, Weighted BCE, Dice, BCE+Dice, Focal, Tversky
  metrics.py      Dice, IoU, precision, recall mức pixel ở ngưỡng cố định 0.5
  engine.py       Vòng train/eval, early stopping, checkpoint
  logger.py       Ghi logs/runs.csv
  viz.py          Đường cong, lưới ảnh so sánh, ảnh tệ nhất, zoom biên
  runner.py       run_experiment(cfg) — điểm vào duy nhất cho notebook

notebooks/
  00_setup_and_data.ipynb          Tải dữ liệu, chia tập, thống kê    [SV B]
  01_train_baseline.ipynb          Một lượt chạy end-to-end            [SV A]
  02_loss_comparison.ipynb         Quét 6 hàm mất mát                  [SV A/B]
  03_ablation_architecture.ipynb   Ablation upsampling và skip         [SV A]
  04_figures_and_report.ipynb      Bảng và hình cho báo cáo            [SV B]

scripts/build_notebooks.py   Sinh lại notebook từ định nghĩa cell
scripts/run_experiment.py    Chạy một cấu hình từ dòng lệnh
tests/smoke_test.py          Kiểm tra toàn bộ pipeline trên dữ liệu giả
splits/                      Danh sách train/val/test cố định (đã commit)
logs/runs.csv                Nhật ký thí nghiệm — sinh tự động
```

## Nguyên tắc thiết kế

**Notebook mỏng, module dày.** Toàn bộ logic nằm trong `src/`. Notebook chỉ
tạo `Config` rồi gọi `run_experiment`. Nhờ vậy code chạy được cả trong
notebook lẫn từ dòng lệnh, và không bao giờ xảy ra chuyện hai notebook có hai
phiên bản khác nhau của cùng một hàm.

**Config là nguồn sự thật duy nhất.** Không hardcode siêu tham số ở bất kỳ
đâu ngoài `config.py`. `Config.run_id` là hash của cấu hình, nên cùng cấu
hình luôn cho cùng `run_id`, và `run_experiment` tự bỏ qua nếu đã chạy rồi.

**Không gõ số bằng tay.** Mọi lượt chạy tự append một dòng vào
`logs/runs.csv`; mọi bảng trong báo cáo đọc lại từ chính file đó. Đề bài xử
việc số liệu lệch giữa báo cáo và log như gian lận học thuật.

**Tái lập được.** `set_seed` được gọi ở đầu mỗi lượt chạy, `splits/*.txt`
được commit vào repo, `cudnn.deterministic = True`.

## Ba bẫy kỹ thuật đã xử lý sẵn

1. **Mask resize bằng NEAREST** (`transforms.resize_pair`). Bilinear sẽ sinh
   giá trị trung gian ở biên polyp làm hỏng nhãn nhị phân.
2. **Mask Kvasir là JPEG nên có nhiễu nén** — nhị phân hoá bằng ngưỡng 127
   ngay khi nạp (`transforms.to_tensor_pair`).
3. **Dice loss có smooth ở cả tử và mẫu** (`losses.DiceLoss`), tránh 0/0 với
   ảnh không có polyp.

Ngoài ra: model trả về **logits**, BCE dùng `BCEWithLogitsLoss`; và Dice
*loss* (`losses.py`, dùng xác suất liên tục) tách bạch với Dice *metric*
(`metrics.py`, nhị phân hoá ở ngưỡng 0.5).

## Kết quả chính

*(điền sau khi chạy — đọc từ `logs/runs.csv`, không gõ tay)*

| Hàm mất mát | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| BCE | | | | |
| Weighted BCE | | | | |
| Dice | | | | |
| BCE + Dice | | | | |
| Focal | | | | |
| Tversky | | | | |

## Trích dẫn nguồn

- Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical
  Image Segmentation.* MICCAI 2015.
- Jha et al. *Kvasir-SEG: A Segmented Polyp Dataset.* MMM 2020.
- Salehi et al. *Tversky loss function for image segmentation.* MLMI 2017.
- Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017.

Phần nào tham khảo mã nguồn công khai thì ghi rõ ở đây, kèm phạm vi dòng.

## Khai báo sử dụng công cụ AI

*(bắt buộc theo quy định môn học — nêu rõ dùng công cụ nào cho việc gì)*
# unet-kvasir
