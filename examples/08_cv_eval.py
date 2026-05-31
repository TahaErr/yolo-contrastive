"""Example 08 — Cross-validation training + evaluation across backbones.

For every backbone x every CV fold (from example 07), fine-tune YOLO and record
the validation mAP, then report per-backbone mean +/- std. A thin wrapper over
RunMatrix: resumable (re-run to continue after a Colab disconnect — completed
cells are skipped) with append-mode CSV logging.

Run (logo folds -> one run per source per backbone; 14 backbones x 10 = 140 runs):
    python examples/08_cv_eval.py \\
        --backbones backbones.txt \\
        --folds datasets/splits/cv/logo \\
        --out runs/cv_results.csv \\
        --epochs 5 --imgsz 320

`backbones.txt`: one "name /path/to/backbone.pt" per line (see load_backbones
for accepted formats). `--folds` is the directory written by example 07
(datasets/splits/cv/logo or .../cv/group_kfold).

`--freeze 0` is full fine-tune; `--freeze 10` freezes the backbone (probe-like).
"""

from __future__ import annotations

import argparse

from yolo_contrastive.eval.cross_val import run_cv_eval


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-validation eval across backbones")
    p.add_argument("--backbones", required=True, help="backbone registry (.txt/.yaml)")
    p.add_argument("--folds", required=True, help="CV fold dir from example 07")
    p.add_argument("--out", default="runs/cv_results.csv", help="results CSV (append/resume)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=0)
    p.add_argument("--freeze", type=int, default=0,
                   help="0 = full fine-tune; 10 = frozen backbone (probe-like)")
    p.add_argument("--baselines", nargs="*", default=[], choices=["coco", "scratch"],
                   help="control methods run through the same folds (e.g. --baselines coco scratch)")
    p.add_argument("--metric", default="mAP50", choices=["mAP50", "metric_value"],
                   help="metric to aggregate (metric_value = mAP50-95)")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    hp = {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
          "device": args.device, "freeze": args.freeze}
    run_cv_eval(args.backbones, args.folds, args.out, seed=args.seed, hp=hp,
                baselines=args.baselines, resume=not args.no_resume, metric=args.metric)


if __name__ == "__main__":
    main()
