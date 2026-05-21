"""Example 04 — Linear probe evaluation of a pretrained backbone.

Measures how good the frozen pretrained features are: a single linear head is
trained on top of a frozen backbone for multi-label classification, and the
validation mAP is reported. No backbone weights are updated.

This is the standard "is my SSL pretraining any good?" probe — run it on
backbones from examples 01 / 02 to compare them.

Run:
    python examples/04_linear_probe_eval.py \\
        --backbone dt_saps_backbone.pt \\
        --data dataset.yaml
"""

from __future__ import annotations

import argparse

from yolo_contrastive import LinearProbeTrainer
from yolo_contrastive.data import loaders_from_yolo_data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe evaluation")
    parser.add_argument("--backbone", required=True,
                        help="pretrained backbone checkpoint")
    parser.add_argument("--data", required=True,
                        help="Ultralytics dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO backbone spec")
    parser.add_argument("--feat-level", default="P5",
                        choices=["P3", "P4", "P5"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Build multi-label loaders from the YOLO dataset.yaml.
    train_loader, val_loader, info = loaders_from_yolo_data_yaml(
        args.data, batch_size=args.batch_size,
    )
    print(f"Dataset: {info['n_train']} train / {info['n_val']} val, "
          f"{info['nc']} classes")

    probe = LinearProbeTrainer(
        backbone=args.model,
        backbone_ckpt=args.backbone,
        num_classes=info["nc"],
        feat_level=args.feat_level,
    )
    try:
        # train(...) and fit(...) are equivalent for the linear probe.
        result = probe.train(
            train_loader, val_loader, epochs=args.epochs,
        )
        print(f"\nBest validation mAP: {result['best_val_mAP']:.4f} "
              f"(epoch {result['best_epoch']})")
    finally:
        probe.cleanup()


if __name__ == "__main__":
    main()
