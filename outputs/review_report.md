# Báo cáo tổng hợp kết quả — U-Net Kvasir-SEG

Sinh tự động bởi `scripts/export_review.py` từ `logs/runs.csv` (23 lượt chạy).

## 1. Môi trường

| Mục           | Giá trị                                 |
|:--------------|:----------------------------------------|
| Python        | 3.10.12                                 |
| Hệ điều hành  | Linux-6.18.20-sm1-x86_64-with-glibc2.35 |
| PyTorch       | 2.13.0+cu130                            |
| CUDA khả dụng | True                                    |
| GPU           | NVIDIA GeForce RTX 5060 Ti              |
| Git commit    | 83154dd (có thay đổi chưa commit)       |

Giá trị mặc định trong `Config` hiện tại:

| Tham số             |   Giá trị |
|:--------------------|----------:|
| image_size          |   256.0   |
| batch_size          |     8.0   |
| epochs              |   120.0   |
| lr                  |     0.001 |
| weight_decay        |     1e-05 |
| seed                |    42.0   |
| early_stop_patience |    20.0   |
| threshold           |     0.5   |
| base_channels       |    64.0   |
| depth               |     4.0   |
| focal_alpha         |     0.75  |
| focal_gamma         |     2.0   |
| tversky_alpha       |     0.3   |
| tversky_beta        |     0.7   |
| augment             |  True     |

## 2. Dữ liệu

| split   |   số ảnh |   tỉ lệ polyp TB |   trung vị |   nhỏ nhất |   lớn nhất |
|:--------|---------:|-----------------:|-----------:|-----------:|-----------:|
| train   |      700 |           0.1472 |     0.1082 |     0.0047 |     0.8118 |
| val     |      150 |           0.1777 |     0.1387 |     0.0122 |     0.6601 |
| test    |      150 |           0.1616 |     0.1215 |     0.0065 |     0.7634 |

- Tỉ lệ pixel polyp toàn bộ: **0.1539** (15.39%)
- `pos_weight` tương ứng: **5.50** (mất cân bằng 1:5.5)
- Đường cơ sở tầm thường (đoán tất cả là polyp): Dice macro **0.2477**, micro 0.2668

Phân bố kích thước polyp theo split:

| row_0         |   test |   train |   val |
|:--------------|-------:|--------:|------:|
| rất nhỏ (<5%) |     23 |     157 |    20 |
| nhỏ (5-15%)   |     62 |     300 |    61 |
| vừa (15-35%)  |     54 |     186 |    53 |
| lớn (>35%)    |     11 |      57 |    16 |

- Ảnh test có nhiều hơn một polyp: **4 / 150**
- Cặp ảnh tương đồng > 0.95 trong toàn bộ 1000 ảnh: **8**

|   similarity | ảnh A                         | split A   | ảnh B                         | split B   | vắt chéo split   |
|-------------:|:------------------------------|:----------|:------------------------------|:----------|:-----------------|
|       1.0000 | cju76lsehyia10987u54vn8rb.jpg | train     | cju772304yw5t0818vbw8kkjf.jpg | test      | True             |
|       0.9999 | cju8amfdtqi4x09871tygrgqe.jpg | train     | cju886ryxnsl50801r93jai7q.jpg | val       | True             |
|       0.9999 | cju30bmab08bi0835mvlr6e0r.jpg | train     | cju77k828z46w0871r0avuoo9.jpg | val       | True             |
|       0.9947 | cju5fydrud94708507vo6oy21.jpg | train     | cju7avvi51iox0817ym55y6tt.jpg | test      | True             |
|       0.9706 | cju3x9lttikfb0818a0g104zn.jpg | train     | cju1hhj6mxfp90835n3wofrap.jpg | train     | False            |
|       0.9584 | cju5uget8krjy0818kvywd0zu.jpg | train     | cju1gi7jlwyld0835cdf6g6qz.jpg | train     | False            |
|       0.9556 | cju88rl5eo94l0850kf5wtrm1.jpg | val       | cju6wuojavt740818b5qcv3iw.jpg | test      | True             |
|       0.9533 | cju7b9vcs1luz0987ta60j1dy.jpg | train     | cju83mki1jv5w0817kubxm31r.jpg | val       | True             |

Phân vị của độ tương đồng cao nhất giữa mỗi ảnh test và tập train (để biết ngưỡng nào là bất thường):

p50=0.860 | p75=0.888 | p90=0.918 | p95=0.926 | p99=0.965

## 3. Toàn bộ lượt chạy

| loss_name    | up_mode   | skip_mode    |   epochs |     lr |   seed |   focal_alpha |   tversky_alpha |   tversky_beta |   n_params |   best_epoch |   epochs_run |   best_val_dice |   test_dice |   test_iou |   test_precision |   test_recall |   train_time_min | tag           |
|:-------------|:----------|:-------------|---------:|-------:|-------:|--------------:|----------------:|---------------:|-----------:|-------------:|-------------:|----------------:|------------:|-----------:|-----------------:|--------------:|-----------------:|:--------------|
| bce_dice     | transpose | full         |       40 | 0.0010 |     42 |        0.2500 |          0.3000 |         0.7000 |   31037633 |           39 |           40 |          0.7413 |      0.7219 |     0.6194 |           0.8562 |        0.6380 |           6.9400 |               |
| weighted_bce | transpose | full         |       40 | 0.0010 |     42 |        0.2500 |          0.3000 |         0.7000 |   31037633 |           37 |           40 |          0.6149 |      0.5768 |     0.4427 |           0.5242 |        0.8372 |           6.9200 | loss_sweep    |
| dice         | transpose | full         |       40 | 0.0010 |     42 |        0.2500 |          0.3000 |         0.7000 |   31037633 |           38 |           40 |          0.5934 |      0.5672 |     0.4344 |           0.5706 |        0.6484 |           6.9300 | loss_sweep    |
| bce          | transpose | full         |       40 | 0.0010 |     42 |        0.2500 |          0.3000 |         0.7000 |   31037633 |           27 |           37 |          0.6082 |      0.5578 |     0.4255 |           0.5314 |        0.7668 |           6.4000 | loss_sweep    |
| tversky      | transpose | full         |       40 | 0.0010 |     42 |        0.2500 |          0.3000 |         0.7000 |   31037633 |           38 |           40 |          0.5636 |      0.5375 |     0.3993 |           0.4501 |        0.7775 |           7.0200 | loss_sweep    |
| focal        | transpose | full         |       40 | 0.0010 |     42 |        0.2500 |          0.3000 |         0.7000 |   31037633 |            1 |           11 |          0.0808 |      0.0766 |     0.0439 |           0.3775 |        0.0508 |           1.9000 | loss_sweep    |
| bce_dice     | bilinear  | full         |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   34520193 |           94 |          114 |          0.8811 |      0.8607 |     0.7890 |           0.8831 |        0.8293 |          37.3300 | ablation_up   |
| bce          | transpose | full         |      120 | 0.0003 |     42 |        0.7500 |          0.3000 |         0.7000 |   31037633 |           90 |          110 |          0.8656 |      0.8533 |     0.7752 |           0.8648 |        0.8508 |          38.8700 | diag_bce_lr   |
| bce_dice     | transpose | half         |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   30853313 |           96 |          116 |          0.8637 |      0.8519 |     0.7744 |           0.8713 |        0.8475 |          27.4000 | ablation_skip |
| bce_dice     | transpose | half_shallow |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   28088513 |          109 |          120 |          0.8544 |      0.8513 |     0.7745 |           0.8839 |        0.8343 |          20.0500 |               |
| bce_dice     | transpose | full         |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   31037633 |           86 |          106 |          0.8537 |      0.8507 |     0.7727 |           0.8995 |        0.7952 |          26.9400 |               |
| bce          | transpose | full         |      120 | 0.0010 |   1337 |        0.7500 |          0.3000 |         0.7000 |   31037633 |          106 |          120 |          0.8557 |      0.8463 |     0.7701 |           0.8630 |        0.8453 |          20.7700 | loss_seed2    |
| bce_dice     | transpose | full         |      120 | 0.0010 |   1337 |        0.7500 |          0.3000 |         0.7000 |   31037633 |           82 |          102 |          0.8619 |      0.8444 |     0.7641 |           0.8966 |        0.7794 |          18.1200 |               |
| bce_dice     | unpool    | full         |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   32083073 |           87 |          107 |          0.8534 |      0.8433 |     0.7638 |           0.8486 |        0.8401 |          42.4000 | ablation_up   |
| bce_dice     | bilinear  | full         |      120 | 0.0010 |   1337 |        0.7500 |          0.3000 |         0.7000 |   34520193 |           82 |          102 |          0.8707 |      0.8387 |     0.7681 |           0.9004 |        0.7900 |          24.1000 |               |
| bce          | transpose | full         |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   31037633 |          109 |          120 |          0.8606 |      0.8385 |     0.7655 |           0.8942 |        0.8122 |          20.7700 | loss_sweep    |
| bce_dice     | transpose | none         |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   27904193 |           71 |           91 |          0.8498 |      0.8381 |     0.7569 |           0.8675 |        0.7988 |          13.6900 | ablation_skip |
| weighted_bce | transpose | full         |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   31037633 |           91 |          111 |          0.8476 |      0.8246 |     0.7387 |           0.8239 |        0.8658 |          19.2100 | loss_sweep    |
| weighted_bce | transpose | full         |      120 | 0.0010 |   1337 |        0.7500 |          0.3000 |         0.7000 |   31037633 |           95 |          115 |          0.8109 |      0.7946 |     0.6945 |           0.7738 |        0.8494 |          19.9800 | loss_seed2    |
| tversky      | transpose | full         |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   31037633 |          119 |          120 |          0.8003 |      0.7769 |     0.6801 |           0.8258 |        0.7521 |          20.8000 | loss_sweep    |
| dice         | transpose | full         |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   31037633 |          117 |          120 |          0.7922 |      0.7672 |     0.6729 |           0.8365 |        0.6938 |          20.7900 | loss_sweep    |
| focal        | transpose | full         |      120 | 0.0030 |     42 |        0.7500 |          0.3000 |         0.7000 |   31037633 |          113 |          120 |          0.6945 |      0.6462 |     0.5249 |           0.6600 |        0.7551 |          42.7200 | diag_focal_lr |
| focal        | transpose | full         |      120 | 0.0010 |     42 |        0.7500 |          0.3000 |         0.7000 |   31037633 |           15 |           35 |          0.5259 |      0.5002 |     0.3608 |           0.4093 |        0.7310 |           6.0600 | loss_sweep    |

### So sánh hàm mất mát (kiến trúc và lr cố định)

| loss_name    | up_mode   | skip_mode   |   epochs |     lr |   seed |   n_params |   test_dice |   test_iou |   test_precision |   test_recall |
|:-------------|:----------|:------------|---------:|-------:|-------:|-----------:|------------:|-----------:|-----------------:|--------------:|
| bce          | transpose | full        |      120 | 0.0003 |     42 |   31037633 |      0.8533 |     0.7752 |           0.8648 |        0.8508 |
| bce_dice     | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.8507 |     0.7727 |           0.8995 |        0.7952 |
| bce          | transpose | full        |      120 | 0.0010 |   1337 |   31037633 |      0.8463 |     0.7701 |           0.8630 |        0.8453 |
| bce_dice     | transpose | full        |      120 | 0.0010 |   1337 |   31037633 |      0.8444 |     0.7641 |           0.8966 |        0.7794 |
| bce          | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.8385 |     0.7655 |           0.8942 |        0.8122 |
| weighted_bce | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.8246 |     0.7387 |           0.8239 |        0.8658 |
| weighted_bce | transpose | full        |      120 | 0.0010 |   1337 |   31037633 |      0.7946 |     0.6945 |           0.7738 |        0.8494 |
| tversky      | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.7769 |     0.6801 |           0.8258 |        0.7521 |
| dice         | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.7672 |     0.6729 |           0.8365 |        0.6938 |
| bce_dice     | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.7219 |     0.6194 |           0.8562 |        0.6380 |
| focal        | transpose | full        |      120 | 0.0030 |     42 |   31037633 |      0.6462 |     0.5249 |           0.6600 |        0.7551 |
| weighted_bce | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.5768 |     0.4427 |           0.5242 |        0.8372 |
| dice         | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.5672 |     0.4344 |           0.5706 |        0.6484 |
| bce          | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.5578 |     0.4255 |           0.5314 |        0.7668 |
| tversky      | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.5375 |     0.3993 |           0.4501 |        0.7775 |
| focal        | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.5002 |     0.3608 |           0.4093 |        0.7310 |
| focal        | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.0766 |     0.0439 |           0.3775 |        0.0508 |

### Ablation cách tăng mẫu

| loss_name    | up_mode   | skip_mode   |   epochs |     lr |   seed |   n_params |   test_dice |   test_iou |   test_precision |   test_recall |
|:-------------|:----------|:------------|---------:|-------:|-------:|-----------:|------------:|-----------:|-----------------:|--------------:|
| bce_dice     | bilinear  | full        |      120 | 0.0010 |     42 |   34520193 |      0.8607 |     0.7890 |           0.8831 |        0.8293 |
| bce          | transpose | full        |      120 | 0.0003 |     42 |   31037633 |      0.8533 |     0.7752 |           0.8648 |        0.8508 |
| bce_dice     | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.8507 |     0.7727 |           0.8995 |        0.7952 |
| bce          | transpose | full        |      120 | 0.0010 |   1337 |   31037633 |      0.8463 |     0.7701 |           0.8630 |        0.8453 |
| bce_dice     | transpose | full        |      120 | 0.0010 |   1337 |   31037633 |      0.8444 |     0.7641 |           0.8966 |        0.7794 |
| bce_dice     | unpool    | full        |      120 | 0.0010 |     42 |   32083073 |      0.8433 |     0.7638 |           0.8486 |        0.8401 |
| bce_dice     | bilinear  | full        |      120 | 0.0010 |   1337 |   34520193 |      0.8387 |     0.7681 |           0.9004 |        0.7900 |
| bce          | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.8385 |     0.7655 |           0.8942 |        0.8122 |
| weighted_bce | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.8246 |     0.7387 |           0.8239 |        0.8658 |
| weighted_bce | transpose | full        |      120 | 0.0010 |   1337 |   31037633 |      0.7946 |     0.6945 |           0.7738 |        0.8494 |
| tversky      | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.7769 |     0.6801 |           0.8258 |        0.7521 |
| dice         | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.7672 |     0.6729 |           0.8365 |        0.6938 |
| bce_dice     | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.7219 |     0.6194 |           0.8562 |        0.6380 |
| focal        | transpose | full        |      120 | 0.0030 |     42 |   31037633 |      0.6462 |     0.5249 |           0.6600 |        0.7551 |
| weighted_bce | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.5768 |     0.4427 |           0.5242 |        0.8372 |
| dice         | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.5672 |     0.4344 |           0.5706 |        0.6484 |
| bce          | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.5578 |     0.4255 |           0.5314 |        0.7668 |
| tversky      | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.5375 |     0.3993 |           0.4501 |        0.7775 |
| focal        | transpose | full        |      120 | 0.0010 |     42 |   31037633 |      0.5002 |     0.3608 |           0.4093 |        0.7310 |
| focal        | transpose | full        |       40 | 0.0010 |     42 |   31037633 |      0.0766 |     0.0439 |           0.3775 |        0.0508 |

### Ablation skip connection

| loss_name    | up_mode   | skip_mode    |   epochs |     lr |   seed |   n_params |   test_dice |   test_iou |   test_precision |   test_recall |
|:-------------|:----------|:-------------|---------:|-------:|-------:|-----------:|------------:|-----------:|-----------------:|--------------:|
| bce          | transpose | full         |      120 | 0.0003 |     42 |   31037633 |      0.8533 |     0.7752 |           0.8648 |        0.8508 |
| bce_dice     | transpose | half         |      120 | 0.0010 |     42 |   30853313 |      0.8519 |     0.7744 |           0.8713 |        0.8475 |
| bce_dice     | transpose | half_shallow |      120 | 0.0010 |     42 |   28088513 |      0.8513 |     0.7745 |           0.8839 |        0.8343 |
| bce_dice     | transpose | full         |      120 | 0.0010 |     42 |   31037633 |      0.8507 |     0.7727 |           0.8995 |        0.7952 |
| bce          | transpose | full         |      120 | 0.0010 |   1337 |   31037633 |      0.8463 |     0.7701 |           0.8630 |        0.8453 |
| bce_dice     | transpose | full         |      120 | 0.0010 |   1337 |   31037633 |      0.8444 |     0.7641 |           0.8966 |        0.7794 |
| bce          | transpose | full         |      120 | 0.0010 |     42 |   31037633 |      0.8385 |     0.7655 |           0.8942 |        0.8122 |
| bce_dice     | transpose | none         |      120 | 0.0010 |     42 |   27904193 |      0.8381 |     0.7569 |           0.8675 |        0.7988 |
| weighted_bce | transpose | full         |      120 | 0.0010 |     42 |   31037633 |      0.8246 |     0.7387 |           0.8239 |        0.8658 |
| weighted_bce | transpose | full         |      120 | 0.0010 |   1337 |   31037633 |      0.7946 |     0.6945 |           0.7738 |        0.8494 |
| tversky      | transpose | full         |      120 | 0.0010 |     42 |   31037633 |      0.7769 |     0.6801 |           0.8258 |        0.7521 |
| dice         | transpose | full         |      120 | 0.0010 |     42 |   31037633 |      0.7672 |     0.6729 |           0.8365 |        0.6938 |
| bce_dice     | transpose | full         |       40 | 0.0010 |     42 |   31037633 |      0.7219 |     0.6194 |           0.8562 |        0.6380 |
| focal        | transpose | full         |      120 | 0.0030 |     42 |   31037633 |      0.6462 |     0.5249 |           0.6600 |        0.7551 |
| weighted_bce | transpose | full         |       40 | 0.0010 |     42 |   31037633 |      0.5768 |     0.4427 |           0.5242 |        0.8372 |
| dice         | transpose | full         |       40 | 0.0010 |     42 |   31037633 |      0.5672 |     0.4344 |           0.5706 |        0.6484 |
| bce          | transpose | full         |       40 | 0.0010 |     42 |   31037633 |      0.5578 |     0.4255 |           0.5314 |        0.7668 |
| tversky      | transpose | full         |       40 | 0.0010 |     42 |   31037633 |      0.5375 |     0.3993 |           0.4501 |        0.7775 |
| focal        | transpose | full         |      120 | 0.0010 |     42 |   31037633 |      0.5002 |     0.3608 |           0.4093 |        0.7310 |
| focal        | transpose | full         |       40 | 0.0010 |     42 |   31037633 |      0.0766 |     0.0439 |           0.3775 |        0.0508 |

## 4. Tóm tắt hình dạng đường cong

`sụp <50% đỉnh` đếm số epoch mà val Dice rơi xuống dưới nửa giá trị đỉnh — chỉ số bất ổn định. `dốc 20ep cuối` dương rõ nghĩa là còn đang lên, tức chưa hội tụ. `SD 20ep cuối` lớn nghĩa là dao động chưa tắt.

| loss         | up        | skip         |     lr |   ep |   seed |   n_ep |   đỉnh |   ep đỉnh |   cuối |   sụp <50% đỉnh |   sụp mạnh nhất |   TB 20ep cuối |   SD 20ep cuối |   dốc 20ep cuối |
|:-------------|:----------|:-------------|-------:|-----:|-------:|-------:|-------:|----------:|-------:|----------------:|----------------:|---------------:|---------------:|----------------:|
| bce_dice     | bilinear  | full         | 0.0010 |  120 |     42 |    114 | 0.8811 |        94 | 0.8721 |               4 |          0.2660 |         0.8722 |         0.0024 |          0.0001 |
| bce_dice     | bilinear  | full         | 0.0010 |  120 |   1337 |    102 | 0.8707 |        82 | 0.8620 |               1 |          0.1731 |         0.8612 |         0.0058 |          0.0001 |
| bce          | transpose | full         | 0.0003 |  120 |     42 |    110 | 0.8656 |        90 | 0.8621 |              14 |          0.2786 |         0.8583 |         0.0041 |          0.0003 |
| bce_dice     | transpose | half         | 0.0010 |  120 |     42 |    116 | 0.8637 |        96 | 0.8605 |               3 |          0.1576 |         0.8568 |         0.0043 |          0.0004 |
| bce_dice     | transpose | full         | 0.0010 |  120 |   1337 |    102 | 0.8619 |        82 | 0.8538 |               4 |          0.2349 |         0.8494 |         0.0043 |          0.0003 |
| bce          | transpose | full         | 0.0010 |  120 |     42 |    120 | 0.8606 |       109 | 0.8568 |              17 |          0.3623 |         0.8548 |         0.0036 |          0.0003 |
| bce          | transpose | full         | 0.0010 |  120 |   1337 |    120 | 0.8557 |       106 | 0.8541 |              25 |          0.3799 |         0.8523 |         0.0027 |          0.0003 |
| bce_dice     | transpose | half_shallow | 0.0010 |  120 |     42 |    120 | 0.8544 |       109 | 0.8506 |               6 |          0.1365 |         0.8491 |         0.0027 |          0.0002 |
| bce_dice     | transpose | full         | 0.0010 |  120 |     42 |    106 | 0.8537 |        86 | 0.8458 |               4 |          0.1606 |         0.8415 |         0.0071 |          0.0007 |
| bce_dice     | unpool    | full         | 0.0010 |  120 |     42 |    107 | 0.8534 |        87 | 0.8435 |              11 |          0.1831 |         0.8417 |         0.0045 |         -0.0001 |
| bce_dice     | transpose | none         | 0.0010 |  120 |     42 |     91 | 0.8498 |        71 | 0.8465 |               5 |          0.2856 |         0.8389 |         0.0092 |          0.0006 |
| weighted_bce | transpose | full         | 0.0010 |  120 |     42 |    111 | 0.8476 |        91 | 0.8330 |               1 |          0.1345 |         0.8242 |         0.0098 |          0.0008 |
| weighted_bce | transpose | full         | 0.0010 |  120 |   1337 |    115 | 0.8109 |        95 | 0.8005 |               1 |          0.1067 |         0.7964 |         0.0099 |          0.0007 |
| tversky      | transpose | full         | 0.0010 |  120 |     42 |    120 | 0.8003 |       119 | 0.7995 |               3 |          0.2202 |         0.7979 |         0.0017 |          0.0002 |
| dice         | transpose | full         | 0.0010 |  120 |     42 |    120 | 0.7922 |       117 | 0.7858 |               1 |          0.0830 |         0.7827 |         0.0062 |          0.0008 |
| focal        | transpose | full         | 0.0030 |  120 |     42 |    120 | 0.6945 |       113 | 0.6936 |              13 |          0.4192 |         0.6897 |         0.0044 |          0.0005 |
| focal        | transpose | full         | 0.0010 |  120 |     42 |     35 | 0.5259 |        15 | 0.1364 |              10 |          0.4468 |         0.3566 |         0.1788 |          0.0100 |

_Thiếu file history cho 6 lượt (chạy bằng bản code trước khi có `save_history`):_
`bce_dice-transpose-full-s42-3790b4`, `bce-transpose-full-s42-ef40ae`, `weighted_bce-transpose-full-s42-c8d926`, `dice-transpose-full-s42-b12a50`, `focal-transpose-full-s42-c03c02`, `tversky-transpose-full-s42-f6e06a`

## 5b. Nhiễu và cỡ hiệu ứng

**Nguồn nhiễu 2 — đổi seed khởi tạo.** Đây là mức nhiễu cần dùng làm chuẩn để phán xét mọi khác biệt giữa các cấu hình.

| loss_name    | up_mode   | skip_mode   |   epochs |     lr |   n_seed |   mean |    std |   biên độ |
|:-------------|:----------|:------------|---------:|-------:|---------:|-------:|-------:|----------:|
| weighted_bce | transpose | full        |      120 | 0.0010 |        2 | 0.8096 | 0.0212 |    0.0300 |
| bce_dice     | bilinear  | full        |      120 | 0.0010 |        2 | 0.8497 | 0.0155 |    0.0220 |
| bce          | transpose | full        |      120 | 0.0010 |        2 | 0.8424 | 0.0055 |    0.0078 |
| bce_dice     | transpose | full        |      120 | 0.0010 |        2 | 0.8476 | 0.0045 |    0.0063 |

- Biên độ giữa seed lớn nhất: **0.0300**
- Trung bình: **0.0165**

**Hai ngưỡng phán xét**: điển hình `0.0165` (trung bình biên độ giữa seed) và thận trọng `0.0300` (biên độ lớn nhất quan sát được). Khác biệt nhỏ hơn ngưỡng điển hình thì chắc chắn là nhiễu; nằm giữa hai ngưỡng thì chưa kết luận được.

Mỗi trục dưới đây giữ cố định mọi thứ khác và chỉ đổi một biến. Giá trị là trung bình qua các seed có sẵn.

Hàm mất mát nền của ablation: `bce_dice`, tại 120 epoch, lr=0.001.

| trục            |   số mức |   thấp nhất |   cao nhất |   biên độ | kết luận    |
|:----------------|---------:|------------:|-----------:|----------:|:------------|
| hàm mất mát     |        6 |      0.5002 |     0.8476 |    0.3474 | VƯỢT nhiễu  |
| cách tăng mẫu   |        3 |      0.8433 |     0.8497 |    0.0064 | trong nhiễu |
| skip connection |        4 |      0.8381 |     0.8519 |    0.0138 | trong nhiễu |

- hàm mất mát: bce_dice=0.8476, bce=0.8424, weighted_bce=0.8096, tversky=0.7769, dice=0.7672, focal=0.5002
- cách tăng mẫu: bilinear=0.8497, transpose=0.8476, unpool=0.8433
- skip connection: half=0.8519, half_shallow=0.8513, full=0.8476, none=0.8381

Trục nào có biên độ nhỏ hơn ngưỡng nhiễu thì **không được xếp hạng** — phát biểu đúng là khác biệt nằm trong dao động giữa các lần khởi tạo.

## 5. Kiểm tra tự động

- Quan hệ Dice/IoU: **tất cả các dòng đều nhất quán** (IoU ≥ Dice/(2−Dice)).
- Chênh lệch Dice macro so với F1 micro: trung vị 0.0103, lớn nhất 0.0699 (`bce-transpose-full-s42-ef40ae`). Chênh lớn nghĩa là mô hình thất bại chọn lọc trên một nhóm nhỏ ảnh.
- Số cấu hình có từ 2 seed trở lên: **4**
- Checkpoint: **đủ cả 23 lượt**.
- Cấu hình có nhiều learning rate (nhớ lọc `lr` khi lập bảng chính):

| loss_name   | up_mode   | skip_mode   |   epochs |   seed |   số lr khác nhau |
|:------------|:----------|:------------|---------:|-------:|------------------:|
| bce         | transpose | full        |      120 |     42 |                 2 |
| focal       | transpose | full        |      120 |     42 |                 2 |


### Cần xử lý

- **Lượt bị early stopping cắt trước nửa ngân sách** — phải chú thích trong báo cáo:

| run_id                          | loss_name   |   best_epoch |   epochs_run |   test_dice |
|:--------------------------------|:------------|-------------:|-------------:|------------:|
| focal-transpose-full-s42-c03c02 | focal       |            1 |           11 |     0.07657 |
| focal-transpose-full-s42-83feaf | focal       |           15 |           35 |     0.50017 |

## 6. Phân tích theo ảnh (cần GPU)

Dice trung bình theo kích thước polyp:

| nhóm    |   bce_dice/bilinear/full |   bce/transpose/full |   bce_dice/transpose/half |   bce_dice/transpose/half_shallow |   bce_dice/transpose/full |   bce_dice/unpool/full |
|:--------|-------------------------:|---------------------:|--------------------------:|----------------------------------:|--------------------------:|-----------------------:|
| rất nhỏ |                   0.8579 |               0.794  |                    0.8009 |                            0.8011 |                    0.8042 |                 0.8196 |
| nhỏ     |                   0.8581 |               0.8413 |                    0.8693 |                            0.8566 |                    0.8692 |                 0.8286 |
| vừa     |                   0.8814 |               0.8819 |                    0.884  |                            0.8851 |                    0.8636 |                 0.8818 |
| lớn     |                   0.7729 |               0.7952 |                    0.7652 |                            0.764  |                    0.7072 |                 0.791  |

Số ảnh mỗi nhóm: {'rất nhỏ': 23, 'nhỏ': 62, 'vừa': 54, 'lớn': 11}

8 ảnh mọi cấu hình đều làm tệ nhất:

|   idx | tên                           |   tỉ lệ polyp |   Dice TB |
|------:|:------------------------------|--------------:|----------:|
|    65 | cju1dg44i4z3w0801nyz4p6zf.jpg |        0.0088 |    0.0000 |
|    46 | cju76o55nymqd0871h31sph9w.jpg |        0.0065 |    0.2699 |
|    14 | cju7d8m3b2e210755l8fj1yph.jpg |        0.2706 |    0.2858 |
|   130 | cju1ewnoh5z030855vpex9uzt.jpg |        0.1019 |    0.3824 |
|    68 | cju2tpfa5uyx408359datxqqj.jpg |        0.1016 |    0.4560 |
|   144 | cju2uzabhs6er0993x3aaf87p.jpg |        0.1012 |    0.4586 |
|     4 | ck2bxiswtxuw80838qkisqjwz.jpg |        0.4424 |    0.4612 |
|    17 | cju1cdxvz48hw0801i0fjwcnk.jpg |        0.0643 |    0.4674 |

8 ảnh các cấu hình bất đồng nhất:

|   idx | tên                           |   tỉ lệ polyp |   SD giữa cấu hình |   Dice TB |
|------:|:------------------------------|--------------:|-------------------:|----------:|
|   111 | cju0s690hkp960855tjuaqvv0.jpg |        0.0366 |             0.3164 |    0.6890 |
|   126 | cju5yimthmlv80850zhoc90c2.jpg |        0.0520 |             0.2389 |    0.6634 |
|    67 | cju7fbndk2sl608015ravktum.jpg |        0.1105 |             0.2164 |    0.6936 |
|    68 | cju2tpfa5uyx408359datxqqj.jpg |        0.1016 |             0.2159 |    0.4560 |
|   113 | cju84hibuktj80871u519o71q.jpg |        0.0628 |             0.1956 |    0.8628 |
|    50 | cju7ae7bq1f820987toc8si1d.jpg |        0.0367 |             0.1903 |    0.7983 |
|    46 | cju76o55nymqd0871h31sph9w.jpg |        0.0065 |             0.1487 |    0.2699 |
|    82 | cju5thdbrjp1108715xdfx356.jpg |        0.0238 |             0.1341 |    0.8882 |
