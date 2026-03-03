# yolo-contrastive

> Contrastive learning + multi-task self-supervised pretraining for Ultralytics YOLOv8+

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

## Features

- **Drop-in trainer** — subclasses Ultralytics `DetectionTrainer`
- **Auto feature tap** — automatically selects the best backbone layer
- **NT-Xent loss** (InfoNCE / SimCLR) with configurable temperature
- **Multi-task pretext system** — 6 pluggable pretext tasks with composite training
- **FrequencyBandPrediction** — novel frequency domain pretext task (first in image SSL for detection)
- **SSL pretraining** — pretrain on unlabeled images, fine-tune with labels
- **Pluggable augmentations** — registry-based, preset pipelines (simclr_v1/v2, byol, aggressive)
- **Fine-tuning** — differential LR, layer freezing, pretrained backbone support
- **CSV logging** — per-step loss tracking

## Pretext Task System

The library provides a registry-based pretext task system inspired by IE-Rot (Yamaguchi et al. 2019) and extended with novel contributions. Tasks can be used individually or combined via `CompositeTask` for multi-task SSL.

### Available Tasks

| Task | Classes | What it learns | Difficulty |
|---|---|---|---|
| `rotation` | 4 | Shape/orientation (0°/90°/180°/270°) | Trivial for pretrained |
| `solarization` | 4 | Color/brightness texture | Medium |
| `color_perm` | 6 | Color-object relationships (RGB permutations) | Hard |
| `patch_shuffle` | 24 | Spatial coherence (2×2 jigsaw) | Hard |
| `blur` | 4 | Frequency/detail level (Gaussian blur) | Medium |
| `freq_band` | 4 | Frequency structure (FFT band masking) | Hard |

### FrequencyBandPrediction (Novel Contribution)

Frequency domain pretext tasks have been used in time series SSL (TF-C, TRLS, FreMixer) but **this is the first application to image SSL for object detection**.

The task applies 2D FFT to the image, masks a random frequency band, and reconstructs via IFFT. The model predicts which band was removed:

- **Low frequency removed** → shape/contour information lost
- **Mid frequency removed** → texture/pattern information lost
- **High frequency removed** → edge/fine detail information lost

Pipeline: `img → FFT2D → band mask (smooth sigmoid) → IFFT2D → predict`

### Recommended Combination

The following combination covers three orthogonal feature axes and provides a non-trivial learning signal even for pretrained backbones:
```python
from yolo_contrastive.pretext import CompositeTask

composite = CompositeTask.from_names(
    ["freq_band", "solarization", "patch_shuffle"],
    feat_dim=256,
    weights=[1.0, 0.8, 0.5],
)
# freq_band    → frequency structure (shape/texture/edge)
# solarization → color/brightness texture
# patch_shuffle → spatial coherence
```

## Installation
```bash
git clone https://github.com/TahaErr/yolo-contrastive.git
cd yolo-contrastive
pip install -e ".[yolo,dev]"
```

## Quick Start

### Contrastive Training (with labels)
```python
import os
os.environ["YCL_LAMBDA"] = "0.1"
os.environ["YCL_TEMP"] = "0.2"

from ultralytics import YOLO
from yolo_contrastive.trainer import ContrastiveDetectionTrainer

model = YOLO("yolov8n.pt")
model.train(data="coco128.yaml", epochs=10, trainer=ContrastiveDetectionTrainer)
```

### Multi-Task Pretext Training (with labels)
```python
import os
os.environ["YCL_LAMBDA"] = "0.1"
os.environ["YCL_PRETEXT_TASKS"] = "freq_band,solarization,patch_shuffle"
os.environ["YCL_PRETEXT_WEIGHTS"] = "1.0,0.8,0.5"
os.environ["YCL_LAMBDA_PRETEXT"] = "0.3"

from ultralytics import YOLO
from yolo_contrastive.trainer import ContrastiveDetectionTrainer

model = YOLO("yolov8n.pt")
model.train(data="coco128.yaml", epochs=10, trainer=ContrastiveDetectionTrainer)
```

### SSL Pretraining (without labels)
```python
from yolo_contrastive.pretrain import SSLPretrainer

# Multi-task pretext (recommended)
pretrainer = SSLPretrainer(
    model="yolov8n.pt",
    aug_preset="simclr_v2",
    lambda_cl=1.0,
    pretext_tasks=["freq_band", "solarization", "patch_shuffle"],
    pretext_weights=[1.0, 0.8, 0.5],
    lambda_pretext=0.5,
)
pretrainer.train(images_dir="path/to/images", epochs=100, output="backbone.pt")

# Legacy rotation (backward compatible)
pretrainer = SSLPretrainer(
    model="yolov8n.pt",
    aug_preset="simclr_v2",
    lambda_cl=1.0,
    lambda_rot=0.5,
)
pretrainer.train(images_dir="path/to/images", epochs=100, output="backbone.pt")
```

### Fine-tuning with Pretrained Backbone
```python
import os
os.environ["YCL_PRETRAINED"] = "backbone.pt"
os.environ["YCL_FREEZE_BACKBONE"] = "10"
os.environ["YCL_UNFREEZE_EPOCH"] = "5"

from ultralytics import YOLO
from yolo_contrastive.finetune import FinetuneDetectionTrainer

model = YOLO("yolov8n.pt")
model.train(data="dataset.yaml", epochs=50, trainer=FinetuneDetectionTrainer)
```

## Configuration

### Contrastive Learning

| Env Var | Default | Description |
|---|---|---|
| `YCL_LAMBDA` | `0.0` | CL loss weight (0=disabled) |
| `YCL_LOSS` | `ntxent` | Loss: `ntxent`/`infonce`/`simclr` |
| `YCL_TEMP` | `0.2` | NT-Xent temperature |
| `YCL_TWO_VIEW` | `0` | Real two-view augmentation |
| `YCL_AUG_PRESET` | `` | Preset: `simclr_v1`, `simclr_v2`, `byol`, `aggressive` |

### Multi-Task Pretext

| Env Var | Default | Description |
|---|---|---|
| `YCL_PRETEXT_TASKS` | `` | Comma-separated task names (e.g. `freq_band,solarization,blur`) |
| `YCL_PRETEXT_WEIGHTS` | `` | Per-task loss weights (e.g. `1.0,0.8,0.5`), defaults to all 1.0 |
| `YCL_LAMBDA_PRETEXT` | `0.0` | Total pretext loss multiplier |
| `YCL_LAMBDA_ROT` | `0.0` | Legacy rotation-only weight (backward compat) |

### Fine-tuning

| Env Var | Default | Description |
|---|---|---|
| `YCL_PRETRAINED` | `` | Pretrained backbone path |
| `YCL_FREEZE_BACKBONE` | `10` | Layers to freeze |
| `YCL_UNFREEZE_EPOCH` | `0` | Epoch to unfreeze (0=never) |
| `YCL_BACKBONE_LR_SCALE` | `0.1` | Backbone LR multiplier |

### Loss Formula
```
total = det_loss + λ_cl × cl_loss + λ_pretext × Σ(wᵢ × taskᵢ_loss)
```

## Architecture
```
┌──────────────────────────────────────────────────┐
│                  Input Image                      │
├────────────┬─────────────────┬───────────────────┤
│            │                 │                    │
│    Detection Path     CL Path          Pretext Path
│    (YOLO head)     (NT-Xent)      (CompositeTask)
│            │                 │                    │
│         det_loss         cl_loss    ┌────────────┤
│            │                 │      │  freq_band │
│            │                 │      │  solarize  │
│            │                 │      │  patch_shuf│
│            │                 │      └────────────┤
│            │                 │       pretext_loss │
├────────────┴─────────────────┴───────────────────┤
│  total = det + λ_cl·cl + λ_pretext·Σ(wᵢ·taskᵢ) │
└──────────────────────────────────────────────────┘
```

## License

MIT — see [LICENSE](LICENSE).
