"""Example 12 — roadrecon pretraining pipeline (pure, self-contained: B2 → M2 + M3 dataset).

Two roadrecon-specific stages, no COCO and no external model anywhere:

    1. Train the B2 road-reconstruction net from scratch on the unlabeled RGB pool
       → saves the encoder (the **M2** init) and the full net (for mining).
    2. Mine reconstruction-anomaly boxes over the pool → a YOLO detection dataset
       (``images/`` + ``labels/`` + ``data.yaml``) — the **M3** replay data.

Run the kill-gate (example 12b) on a LABELED set BEFORE trusting the mined signal.

Invocations
-----------
    # train B2 + mine (Colab A100)
    python examples/12_roadrecon_pretrain.py \
        --pool /content/pool_images --out runs/roadrecon \
        --imgsz 640 --epochs 50 --batch 32 --device 0

    # reuse an already-trained full-net checkpoint, just re-mine
    python examples/12_roadrecon_pretrain.py --pool /content/pool_images \
        --out runs/roadrecon --skip-train --z-thresh 3.0

M3 anchored pretraining + LOSO eval (after this script) — pure, no COCO:

    from yolo_contrastive import AnchoredJointTrainer, RoadReconChannel, run_cv_eval
    from yolo_contrastive.roadrecon import build_scratch_detector, full_transplant_detection_runner

    trainer = AnchoredJointTrainer(
        model=build_scratch_detector(nc=1),         # scratch, nc=1 (matches mined + downstream)
        replay_data="runs/roadrecon/mined/data.yaml",  # mined potholes ride the replay slot
        channels=[RoadReconChannel("/content/pool_images", imgsz=640)],
        lambda_aux=1.0, epochs=12, imgsz=640, batch=24,
        # CRITICAL for a FROM-SCRATCH backbone: the trainer's COCO-tuned defaults
        # (warmup_steps=300 freezes backbone+neck; backbone_lr=1e-4) would leave the
        # scratch backbone almost untrained. Disable warmup and raise the backbone LR.
        warmup_steps=0, backbone_lr=1e-2,
        output_dir="runs/anchored_roadrecon")
    ckpt = trainer.train()                          # -> runs/.../anchored_full.pt (whole detector)

    run_cv_eval(
        [{"name": "roadrecon_m3", "backbone_ckpt": ckpt, "base_model": "yolov8n.yaml"}],
        "datasets/splits/cv/logo", "runs/cv_roadrecon.csv",
        baselines=("coco", "scratch"), fractions=(0.1, 0.5, 1.0),
        runners={"detection": full_transplant_detection_runner})   # R8 whole-detector transplant
"""

from __future__ import annotations

import argparse
from pathlib import Path

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _pool_images(pool: Path):
    return [(p.stem, str(p)) for p in sorted(pool.rglob("*"))
            if p.suffix.lower() in _IMG_EXTS and p.is_file()]


def main() -> None:
    ap = argparse.ArgumentParser(description="roadrecon B2 pretrain + M3 anomaly mining")
    ap.add_argument("--pool", required=True, help="unlabeled pool image directory")
    ap.add_argument("--out", default="runs/roadrecon", help="output directory")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tap-level", default="P3", choices=["P3", "P4", "P5"])
    ap.add_argument("--device", default=0)
    ap.add_argument("--z-thresh", type=float, default=3.0, help="mining robust-z anomaly threshold")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse an existing roadrecon_full.pt and only re-mine")
    args = ap.parse_args()

    pool = Path(args.pool)
    if not pool.exists():
        raise SystemExit(f"--pool not found: {pool}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full_ckpt = out / "roadrecon_full.pt"
    backbone_ckpt = out / "roadrecon_backbone.pt"

    from yolo_contrastive.roadrecon import (
        AnomalyMineConfig, RoadReconstructor, load_reconstructor, mine_anomaly_labels,
    )

    # ── Stage 1: train B2 (or reload) ────────────────────────────────────────
    if args.skip_train:
        if not full_ckpt.exists():
            raise SystemExit(f"--skip-train set but {full_ckpt} does not exist; train first")
        rec = load_reconstructor(str(full_ckpt), device=args.device)
        print(f"[12] reloaded reconstructor from {full_ckpt}")
    else:
        rec = RoadReconstructor(model="yolov8n.yaml", imgsz=args.imgsz,
                                tap_level=args.tap_level, device=args.device)
        rec.train(str(pool), epochs=args.epochs, batch_size=args.batch,
                  num_workers=args.num_workers, output=str(backbone_ckpt))
        rec.save(str(full_ckpt))
        print(f"[12] M2 backbone: {backbone_ckpt}\n[12] full net:    {full_ckpt}")

    # ── Stage 2: mine anomaly labels → YOLO dataset (M3 replay data) ──────────
    mined_root = out / "mined"
    stats = mine_anomaly_labels(
        rec, _pool_images(pool), str(mined_root),
        cfg=AnomalyMineConfig(z_thresh=args.z_thresh), imgsz=args.imgsz,
    )
    rec.cleanup()

    print(f"[12] mining: scanned={stats['scanned']} with_boxes={stats['with_boxes']} "
          f"boxes={stats['boxes']} skipped={stats['skipped']}")
    print(f"[12] M3 dataset: {mined_root / 'data.yaml'}")
    print("[12] NEXT: run examples/12b_roadrecon_killgate.py on a LABELED set before training M3.")


if __name__ == "__main__":
    main()
