"""Chạy một thí nghiệm từ dòng lệnh (thay thế cho notebook khi chạy nền).

    python scripts/run_experiment.py --loss_name dice --up_mode bilinear
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import Config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    for field, value in Config().to_dict().items():
        if isinstance(value, bool):
            parser.add_argument(f"--{field}", type=lambda s: s.lower() == "true",
                                default=value)
        elif value is None:
            parser.add_argument(f"--{field}", type=float, default=None)
        else:
            parser.add_argument(f"--{field}", type=type(value), default=value)
    args = parser.parse_args()
    run_experiment(Config(**vars(args)), skip_if_logged=False)


if __name__ == "__main__":
    main()
