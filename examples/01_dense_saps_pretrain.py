"""Example 01 — Dense SAPS self-supervised pretraining.

Pretrains a YOLOv8n backbone on a directory of unlabeled images using dense,
scale-aware contrastive learning (SAPS), then saves the backbone checkpoint.

Run:
    python examples/01_dense_saps_pretrain.py --images /data/unlabeled_pool

The saved checkpoint can be loaded into a YOLO fine-tuning run — see
examples/03_finetune_with_backbone.py.
"""

from __future__ import annotations

import argparse

from yolo_contrastive import DenseSSLPretrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense SAPS pretraining")
    parser.add_argument("--images", required=True,
                        help="directory of unlabeled training images")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO backbone spec")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="saps_backbone.pt")
    args = parser.parse_args()

    trainer = DenseSSLPretrainer(model=args.model, imgsz=args.imgsz)
    try:
        backbone_path = trainer.train(
            images_dir=args.images,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output=args.output,
        )
        print(f"\nPretrained backbone saved to: {backbone_path}")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
