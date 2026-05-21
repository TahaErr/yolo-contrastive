"""Example 02 — DT-SAPS dual-teacher pretraining.

Pretrains a YOLOv8n backbone with dense SAPS *plus* distillation from two
frozen teachers:

  - COCO teacher  — a supervised YOLOv8x backbone (general semantic priors).
  - SSL teacher   — a pure-SAPS pretrained backbone (domain-specific priors),
                    typically the winner of a prior SAPS-only ablation.

The two teachers are fused by ConsensusLoss; per-position disagreement
weighting (DisagreementWeighter) is enabled by default inside the trainer.

Run:
    python examples/02_dt_saps_dual_teacher.py \\
        --images /data/unlabeled_pool \\
        --ssl-teacher saps_backbone.pt

Prerequisite: a pure-SAPS backbone to act as the SSL teacher — produce one
with examples/01_dense_saps_pretrain.py.
"""

from __future__ import annotations

import argparse

from yolo_contrastive import CocoTeacher, DualTeacherTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="DT-SAPS dual-teacher pretraining")
    parser.add_argument("--images", required=True,
                        help="directory of unlabeled training images")
    parser.add_argument("--ssl-teacher", required=True,
                        help="pure-SAPS backbone checkpoint (SSL teacher)")
    parser.add_argument("--coco-weights", default="yolov8x.pt",
                        help="COCO-pretrained YOLO weights (COCO teacher)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="student backbone spec")
    parser.add_argument("--teacher-combo", default="both",
                        choices=["none", "coco_only", "ssl_only", "both"])
    parser.add_argument("--distill-form", default="B+C",
                        choices=["B", "C", "B+C"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="dt_saps_backbone.pt")
    args = parser.parse_args()

    # COCO teacher — YOLOv8x; its features are adapted to the YOLOv8n student
    # channel widths via a trainable per-scale adapter.
    coco_teacher = CocoTeacher(
        weights=args.coco_weights,
        student_channels={"P3": 64, "P4": 128, "P5": 256},
    )
    # SSL teacher — same architecture as the student, so no adapter is needed.
    ssl_teacher = CocoTeacher(weights=args.ssl_teacher)

    trainer = DualTeacherTrainer(
        model=args.model,
        teacher_combo=args.teacher_combo,
        coco_teacher=coco_teacher,
        ssl_teacher=ssl_teacher,
        distill_form=args.distill_form,
        imgsz=args.imgsz,
    )
    try:
        backbone_path = trainer.train(
            images_dir=args.images,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output=args.output,
        )
        print(f"\nDT-SAPS backbone saved to: {backbone_path}")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
