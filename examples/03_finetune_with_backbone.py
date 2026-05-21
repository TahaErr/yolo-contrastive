"""Example 03 — Fine-tune YOLO detection with a pretrained backbone.

Loads a self-supervised pretrained backbone (from example 01 or 02) into a
YOLO model and fine-tunes it on a labeled detection dataset. The pretrained
weights are loaded backbone-only — the detection head is trained from scratch.

Run:
    python examples/03_finetune_with_backbone.py \\
        --backbone dt_saps_backbone.pt \\
        --data dataset.yaml

`dataset.yaml` is a standard Ultralytics data config (train/val paths, nc,
names).
"""

from __future__ import annotations

import argparse

from yolo_contrastive import load_backbone


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune with a pretrained backbone")
    parser.add_argument("--backbone", required=True,
                        help="pretrained backbone checkpoint")
    parser.add_argument("--data", required=True,
                        help="Ultralytics dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO model spec")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    n_loaded = load_backbone(
        model.model, args.backbone,
        strict=False, backbone_only=True,
    )
    print(f"Loaded {n_loaded} pretrained backbone params from {args.backbone}")

    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
