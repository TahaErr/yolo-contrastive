"""CSV logging mixin — per-step loss tracking."""

from __future__ import annotations
import csv
import os


def _is_main_process(trainer) -> bool:
    try:
        from ultralytics.utils import RANK
        r = getattr(trainer, "rank", RANK)
    except Exception:
        r = getattr(trainer, "rank", 0)
    try:
        r = int(r)
    except Exception:
        r = 0
    return r in (-1, 0)


class CSVLoggerMixin:

    def _csv_init_if_needed(self) -> None:
        if not _is_main_process(self):
            return
        path = getattr(self, "_cl_csv_path", None)
        if not path or os.path.exists(path):
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow([
                "step", "epoch", "iter_tag", "iter_a", "iter_b", "source",
                "det_loss", "cl_loss", "rot_loss", "total_loss",
                "lambda_cl", "lambda_rot", "temp",
                "tap_layer", "two_view", "aug_preset", "rot_acc",
            ])

    def _csv_append(self, row: list) -> None:
        if not _is_main_process(self):
            return
        path = getattr(self, "_cl_csv_path", None)
        if not path:
            return
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)
