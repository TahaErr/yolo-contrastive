# yolo-contrastive

> Self-supervised pretraining + dual-teacher distillation for Ultralytics YOLOv8+

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

A research library for pretraining YOLO detection backbones on **unlabeled**
traffic-scene images, then transferring to downstream detection with few
labels. Its centerpiece is **DT-SAPS** — Scale-Aware Dense Contrastive
Pretraining with Dual-Teacher Distillation — which augments dense
self-supervised pretraining with knowledge distilled from a supervised
(COCO) teacher and a self-supervised teacher.

## Highlights

- **DT-SAPS dual-teacher framework** — dense SAPS pretraining + distillation
  from a frozen COCO teacher and a frozen SSL teacher, fused via consensus
  and per-position disagreement weighting.
- **Dense SAPS pretraining** — scale-aware, per-position contrastive
  pretraining over P3/P4/P5 feature maps (not global-pooled).
- **External SSL baselines** — SimCLR-YOLO, MoCo-v3-YOLO, CoMAD-YOLO, ready
  for fair "vs SOTA" comparison.
- **Evaluation tooling** — frozen-backbone linear probe, YAML-driven ablation
  grids, and cross-set leakage checking.
- **Drop-in fine-tuning** — load a pretrained backbone into a YOLO
  fine-tuning trainer with differential LR and layer freezing.

## Installation

```bash
pip install -e ".[all]"        # everything (yolo + pretrain + dev extras)
```

Granular extras: `yolo` (ultralytics), `pretrain` (opencv, imagehash),
`dev` (pytest, ruff). `torch` and `pyyaml` are hard dependencies.

The top-level package imports lazily — `import yolo_contrastive` is
lightweight and does not require ultralytics; a class is only loaded on
first use.

## Quick start

### Dense SAPS pretraining

```python
from yolo_contrastive import DenseSSLPretrainer

trainer = DenseSSLPretrainer(model="yolov8n.pt", imgsz=640)
backbone_path = trainer.train(
    images_dir="/data/unlabeled_pool",
    epochs=100, batch_size=32,
    output="saps_backbone.pt",
)
trainer.cleanup()
```

### DT-SAPS dual-teacher pretraining

```python
from yolo_contrastive import DualTeacherTrainer, CocoTeacher

# COCO teacher — frozen YOLOv8x backbone, per-scale adapter to student channels
coco_teacher = CocoTeacher(
    weights="yolov8x.pt",
    student_channels={"P3": 64, "P4": 128, "P5": 256},
)
# SSL teacher — the pure-SAPS pretraining winner (student architecture)
ssl_teacher = CocoTeacher(weights="saps_backbone.pt")

trainer = DualTeacherTrainer(
    model="yolov8n.pt",
    teacher_combo="both",          # none | coco_only | ssl_only | both
    coco_teacher=coco_teacher,
    ssl_teacher=ssl_teacher,
    distill_form="B+C",            # B | C | B+C
    imgsz=640,
)
backbone_path = trainer.train(images_dir="/data/unlabeled_pool", epochs=100)
trainer.cleanup()
```

### Fine-tuning with a pretrained backbone

```python
from yolo_contrastive import load_backbone
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
load_backbone(model.model, "saps_backbone.pt", backbone_only=True)
model.train(data="dataset.yaml", epochs=50)
```

### Linear probe evaluation

```python
from yolo_contrastive import LinearProbeTrainer

probe = LinearProbeTrainer(
    backbone="yolov8n.pt",
    backbone_ckpt="saps_backbone.pt",
    num_classes=4,
)
result = probe.train(train_loader, val_loader, epochs=10)   # train() or fit()
print(result["best_val_mAP"])
probe.cleanup()
```

### Cross-set leakage check

The SSL pretrain pool must not overlap downstream evaluation data:

```bash
python -m yolo_contrastive.eval.leakage_check \
    --pool-phash pool_phash.parquet \
    --downstream /data/eval/train /data/eval/valid \
    --hamming-threshold 5
```

## Package layout

| Package | What it provides |
|---|---|
| `pretrain/` | `DenseSSLPretrainer`, `SSLPretrainer`, backbone I/O |
| `dual_teacher/` | DT-SAPS — `DualTeacherTrainer`, `CocoTeacher`, `TeacherCache`, `ConsensusLoss`, `DisagreementWeighter` |
| `baselines/` | `SimCLRYOLOTrainer`, `MoCoV3YOLOTrainer`, `CoMADYOLOTrainer` |
| `dense/` | dense primitives — feature tap, momentum encoder, multi-scale loss |
| `eval/` | `LinearProbeTrainer`, `RunMatrix`, leakage check |
| `finetune/` | `FinetuneDetectionTrainer` — YOLO fine-tuning with pretrained backbone |
| `data/` | SSL pool ingestion, deduplication, downstream loaders |
| `contrastive/`, `augmentations/` | NT-Xent loss, augmentation presets |

Runnable examples live in [`examples/`](examples/).

## Trainer conventions

Every trainer follows the same shape: a constructor, a `train(...)` method
returning the saved backbone path, and a `cleanup()` that releases feature-tap
hooks. `LinearProbeTrainer` additionally exposes `fit(...)` (the scikit-learn
name) — `train` and `fit` are equivalent there.

## Testing

```bash
pytest                    # full suite
pytest -m "not slow"      # skip the real-YOLO end-to-end tests
```

## Legacy modules

Earlier versions centered on a registry-based **pretext-task system**
(rotation, solarization, color permutation, jigsaw) and a **composite
multi-task** trainer, plus an experimental `FrequencyBandPrediction`
frequency-domain pretext task. These modules remain in the codebase for
reproducibility but are not part of the DT-SAPS pipeline; the
frequency-domain pretext task is deferred to a separate follow-up study.

## License

MIT — see [LICENSE](LICENSE).
