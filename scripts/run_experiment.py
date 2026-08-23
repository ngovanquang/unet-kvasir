"""Chạy một thí nghiệm từ dòng lệnh (thay thế cho notebook khi chạy nền).

    python scripts/run_experiment.py --loss_name dice --up_mode bilinear
    python scripts/run_experiment.py --loss_name bce --force   # chạy lại dù đã có

Mặc định BỎ QUA lượt đã có trong log. Trước đây script luôn chạy lại, và điều
đó sinh ra hai dòng cùng run_id với hai kết quả khác nhau — cùng đường dẫn
checkpoint nên lượt sau ghi đè lượt trước, làm một dòng CSV không còn khớp file
nào trên đĩa. Dùng --force nếu thực sự muốn chạy lại.

CẢNH BÁO VỀ num_workers: trường này KHÔNG nằm trong hash run_id nhưng CÓ ảnh
hưởng tới kết quả, vì worker_init_fn gieo seed theo chỉ số worker nên đổi số
worker là đổi trình tự tăng cường dữ liệu. Giữ nguyên một giá trị cho toàn bộ
thí nghiệm, nếu không hai lượt "cùng cấu hình" sẽ cho hai con số khác nhau.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import Config, RunLogger, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="chạy lại kể cả khi run_id đã có trong log")
    for field, value in Config().to_dict().items():
        if isinstance(value, bool):
            parser.add_argument(f"--{field}", type=lambda s: s.lower() == "true",
                                default=value)
        elif value is None:
            parser.add_argument(f"--{field}", type=float, default=None)
        else:
            parser.add_argument(f"--{field}", type=type(value), default=value)
    args = vars(parser.parse_args())
    force = args.pop("force")
    cfg = Config(**args)

    logged = [r for r in RunLogger(cfg.log_csv).read_all()
              if r.get("run_id") == cfg.run_id]
    if logged and not force:
        r = logged[-1]
        print(f"[bỏ qua] {cfg.run_id} đã chạy ngày {r.get('datetime')}, "
              f"test_dice = {r.get('test_dice')}.")
        print("Thêm --force nếu thực sự muốn chạy lại (sẽ ghi đè checkpoint).")
        return
    if logged and force:
        print(f"[--force] Chạy lại {cfg.run_id}, checkpoint cũ sẽ bị ghi đè.")
        print(f"          num_workers hiện tại = {cfg.num_workers}. Nếu khác lần "
              f"trước, kết quả sẽ khác dù cùng seed.")

    run_experiment(cfg, skip_if_logged=False)


if __name__ == "__main__":
    main()
