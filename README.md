# yolo-contrastive

> Contrastive learning + self-supervised pretraining for Ultralytics YOLOv8+

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

## Features

- **Drop-in trainer** — subclasses Ultralytics `DetectionTrainer`
- **Auto feature tap** — automatically selects the best backbone layer
- **NT-Xent loss** (InfoNCE / SimCLR) with configurable temperature
- **Rotation pretext task** — RotNet-style auxiliary loss
- **SSL pretraining** — pretrain on unlabeled images, fine-tune with labels
- **Pluggable augmentations** — registry-based, preset pipelines (simclr_v1/v2, byol, aggressive)
- **Fine-tuning** — differential LR, layer freezing, pretrained backbone support
- **CSV logging** — per-step loss tracking

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

### SSL Pretraining (without labels)
```python
from yolo_contrastive.pretrain import SSLPretrainer

pretrainer = SSLPretrainer(model="yolov8n.pt", aug_preset="simclr_v2",
                           lambda_cl=1.0, lambda_rot=0.5)
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

| Env Var | Default | Description |
|---|---|---|
| `YCL_LAMBDA` | `0.0` | CL loss weight (0=disabled) |
| `YCL_LOSS` | `ntxent` | Loss: `ntxent`/`infonce`/`simclr` |
| `YCL_TEMP` | `0.2` | NT-Xent temperature |
| `YCL_TWO_VIEW` | `0` | Real two-view augmentation |
| `YCL_AUG_PRESET` | `` | Preset: `simclr_v1`, `simclr_v2`, `byol`, `aggressive` |
| `YCL_LAMBDA_ROT` | `0.0` | Rotation loss weight |
| `YCL_PRETRAINED` | `` | Pretrained backbone path |
| `YCL_FREEZE_BACKBONE` | `10` | Layers to freeze |
| `YCL_UNFREEZE_EPOCH` | `0` | Epoch to unfreeze (0=never) |
| `YCL_BACKBONE_LR_SCALE` | `0.1` | Backbone LR multiplier |

## License

MIT — see [LICENSE](LICENSE).
