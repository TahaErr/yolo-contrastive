# Nature's Labels — Colab A100 Quickstart

Three teacher channels on the shared COCO-anchored joint trainer
(`AnchoredJointTrainer`, R3: replay detection loss in every optimizer step):

| Method    | Module         | Channel              | Supervision source                            |
|-----------|----------------|----------------------|-----------------------------------------------|
| TERRA     | `geoteach/`    | `TerraChannel`       | road-plane residuals from monocular depth     |
| REVISIT   | `persistence/` | `PersistenceChannel` | cross-traversal Mapillary pairs               |
| GASP-Real | `scalereal/`   | `ScaleRealChannel`   | real scale ratios from METRIC monocular depth |

All commands below assume a Colab A100 (40 GB) with the repo cloned at
`/content/yolo-contrastive` and a pool image directory at `/content/pool_images`.
Nothing requires a GPU except the depth/embedding passes and training itself.

```bash
cd /content/yolo-contrastive
pip install -e ".[yolo,pretrain]"        # base: ultralytics + manifest deps
```

---

## 1. TERRA (geoteach)

### Install

```bash
pip install -e ".[yolo,pretrain,geo]" accelerate    # geo = transformers + opencv
```

### Stage 0a — depth cache (resumable, idempotent)

```python
from pathlib import Path
from yolo_contrastive.geoteach import DepthCache, run_depth_anything

pool = Path("/content/pool_images")
images = [(p.stem, str(p)) for p in sorted(pool.glob("*.jpg"))]
cache = DepthCache("/content/cache", tag="depth_anything_v2_small")
run_depth_anything(images, cache, batch_size=16, fp16=True)   # skips existing
```

### Stage 0b — E1 fidelity kill-gate (run BEFORE any training GPU)

On a labeled YOLO dataset (e.g. your pothole val split), sweep |z| thresholds.
E1 needs inverse depth for the LABELED dataset images too — run the Stage-0a
cache over them first (same cache root + tag), or add `--compute` to the E1
command to run the depth model on the fly:

```python
val = Path("/content/datasets/pothole/val/images")
run_depth_anything([(p.stem, str(p)) for p in sorted(val.rglob("*.jpg"))], cache)
```

```bash
python examples/09_terra_e1_fidelity.py \
    --dataset /content/datasets/pothole/val \
    --depth-cache /content/cache --tag depth_anything_v2_small \
    --class-map "0:depression,3:elevation" --out runs/terra_e1
```

GO: `recall_near` >= 50% (the close-range column in
`runs/terra_e1/e1_fidelity.csv`), wrong-polarity < 10% at the chosen operating
point (+ overlay PNGs for visual audit). KILL -> stop here.
Re-calibrate `ResidualLabelConfig.z_anomaly` from the CSV if needed.

### Stage 0c — label factory -> training pool layout

```python
import shutil, cv2
from yolo_contrastive.geoteach import labels_from_inverse_depth, write_yolo_txt

root = Path("/content/terra_pool")
for sub in ("images", "labels", "boxes"):
    (root / sub).mkdir(parents=True, exist_ok=True)
for image_id, path in images:
    if image_id not in cache:
        continue
    inv_depth, _ = cache.load(image_id)
    geo = labels_from_inverse_depth(inv_depth)        # plane fit + bins + gates
    if not geo.use_dense:
        continue                                      # trust gate: no supervision
    shutil.copy(path, root / "images" / f"{image_id}.jpg")
    cv2.imwrite(str(root / "labels" / f"{image_id}.png"), geo.label_map.labels)
    if geo.use_boxes and geo.boxes:
        write_yolo_txt(root / "boxes" / f"{image_id}.txt", geo.boxes)
```

### Stage 1 — anchored joint training

```python
from yolo_contrastive import AnchoredJointTrainer, TerraChannel

trainer = AnchoredJointTrainer(
    model="yolov8n.pt",                 # COCO init (R3 anchor)
    replay_data="coco128.yaml",         # use a larger COCO subset for real runs
    channels=[TerraChannel(root="/content/terra_pool", beta=1.0)],
    lambda_aux=1.0, epochs=12, imgsz=512, batch=24,
    output_dir="runs/anchored_terra",
)
ckpt = trainer.train()                  # -> runs/anchored_terra/anchored_full.pt
```

---

## 2. REVISIT (persistence)

### Install

```bash
pip install -e ".[yolo,pretrain,revisit]"    # revisit = requests + opencv
export MAPILLARY_TOKEN="MLY|..."
```

### Pool factory (mine -> download -> align -> propose -> label; all resumable)

```bash
python examples/10_revisit_prepare.py --root /content/revisit_pool --mine
```

(Offline alternative: `--local-dir my_pairs` with two-image subdirectories.
Stage-by-stage re-runs: `python -m yolo_contrastive.persistence.cli
{mine,download,align,propose,label,stats} --pairs /content/revisit_pool/pairs.parquet ...`)

GO/NO-GO before training: >= 10K aligned pairs (`cli stats`) and >= 80%
plausible labels over the 200 audit renders (`cli label --audit-dir ...`).

### Training (16 pairs + 32 replay images per step)

```python
from yolo_contrastive import AnchoredJointTrainer, PersistenceChannel

ch = PersistenceChannel(
    pairs_path="/content/revisit_pool/pairs.parquet",
    labels_path="/content/revisit_pool/persistence_labels.parquet",
    batch_pairs=16,
)
trainer = AnchoredJointTrainer(
    model="yolov8n.pt", channels=[ch], lambda_aux=1.0,
    epochs=12, imgsz=512, batch=32, output_dir="runs/anchored_revisit",
)
ckpt = trainer.train()
```

`trainer.train()` invokes `ch.on_epoch_end(epoch)` every epoch (R9 —
structural): the SimSiam-collapse / class-frequency / skip-rate sentinels are
logged into `trainer.history` as `sentinel/persistence/*` and the
accumulators reset. Manual loops can still call `ch.sentinel_metrics()` +
`ch.reset_epoch_stats()` directly.

---

## 3. GASP-Real (scalereal)

### Install

```bash
pip install -e ".[yolo,pretrain,geo]"   # transformers for METRIC depth; DINOv2 via torch.hub
```

### Stage 0a — METRIC depth cache

Must be metric (never relative) — the affine-ambiguity guard rejects
non-metric sidecars with `ValueError`.

```python
from pathlib import Path
from yolo_contrastive.geoteach import DepthCache, run_depth_anything

pool = Path("/content/pool_images")
images = [(p.stem, str(p)) for p in sorted(pool.glob("*.jpg"))]
# geoteach writes {root}/{tag}; scalereal reads {root}/depth/{variant} —
# rooting the writer at /content/cache/depth makes both see the same files.
cache = DepthCache("/content/cache/depth", tag="dav2_metric_outdoor_small")
run_depth_anything(
    images, cache, batch_size=16, fp16=True,
    model_name="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
)   # metric checkpoints are auto-inverted to 1/Z, sidecar gets metric=True
```

### Stage 0b — offline pair mining + human audit

```bash
python examples/11_gasp_real_pairs.py \
    --images /content/pool_images \
    --depth-cache /content/cache --variant dav2_metric_outdoor_small \
    --out runs/scalereal --embedder dinov2_vits14 --audit 50
```

Gates printed by the script: image yield >= 40%, all four |log_r| bins
populated, >= 60% of audit mosaics judged same-content/plausible-scale.
The geoteach-written sidecar dialect is normalized automatically; non-metric
sidecars raise `ValueError`. (Manifest-driven production CLI:
`python -m yolo_contrastive.scalereal.mine_pairs`.)

### Stage 1 — anchored joint training (lambda_aux = 0.3 is the spec'd arm)

```python
import pandas as pd
from pathlib import Path
from yolo_contrastive import AnchoredJointTrainer, ScaleRealChannel

pool = Path("/content/pool_images")          # image_id -> materialized_path
manifest = pd.DataFrame([{"image_id": p.stem, "materialized_path": str(p)}
                         for p in sorted(pool.glob("*.jpg"))])
ch = ScaleRealChannel(pairs_path="runs/scalereal/pairs.parquet",
                      pool_manifest=manifest)
trainer = AnchoredJointTrainer(
    model="yolov8n.pt", channels=[ch], lambda_aux=0.3,
    epochs=12, imgsz=512, batch=32, output_dir="runs/anchored_scalereal",
)
# the trainer calls ch.on_epoch_end(epoch) every epoch (R9): the fixed
# probe batch is built lazily from the 1% holdout and the row/size-shortcut
# + collapse sentinels land in trainer.history as sentinel/scalereal/*.
ckpt = trainer.train()
print(ch.sentinel_records[-1])          # or pre-build: ch.build_probe_batch(512)
```

---

## 4. Export + fine-tune handoff (R8: whole detector, never backbone-only)

`trainer.train()` already exports `output_dir/anchored_full.pt`
(EMA weights, backbone + neck + head). Direct fine-tune:

```python
from yolo_contrastive import load_for_finetune

yolo = load_for_finetune("runs/anchored_terra/anchored_full.pt", base="yolov8n.pt")
yolo.train(data="/content/datasets/pothole/data.yaml",
           epochs=50, imgsz=640, batch=32, device=0)
```

CV evaluation against the COCO baseline through the existing harness
(folds from example 07). Convert first so the FULL detector is the init —
do NOT put `anchored_full.pt` alone in a `backbones.txt` for
`examples/08_cv_eval.py`: that path loads backbone-only under a COCO neck,
the documented R8 failure mode.

```python
from yolo_contrastive import load_for_finetune, run_cv_eval

load_for_finetune("runs/anchored_terra/anchored_full.pt").save("backbones/terra_full.pt")
run_cv_eval(
    [{"name": "terra_full", "backbone_ckpt": "runs/anchored_terra/anchored_full.pt"}],
    "datasets/splits/cv/logo", "runs/cv_results.csv",
    hp={"base_model": "backbones/terra_full.pt",   # full-detector init (R8)
        "epochs": 5, "imgsz": 320, "batch": 16, "device": 0, "freeze": 0},
    baselines=["coco"],                            # COCO control arm, same folds
)
```

(The `backbone_ckpt` re-load on top of the same `base_model` is a value-identical
no-op; the baseline methods carry their own `base_model="yolov8n.pt"`.)

---

## 5. Per-epoch sentinels to watch (R9)

Trainer (`output_dir/sentinels.csv`, one row per epoch; warn -> `warnings`,
abort -> `SentinelAbort`):

| Metric              | Warn        | Abort  | Meaning                                  |
|---------------------|-------------|--------|------------------------------------------|
| `eff_rank`          | < 30        | < 20   | P5 rank collapse (failed SSL arms: 3-10) |
| `replay_cls_drift`  | > +15%      | > +30% | COCO forgetting — the anchor is slipping |
| `cka_prev_epoch`    | < 0.30      | —      | representation churn between epochs      |
| `head_norm/{name}`  | > 3x initial| —      | a head is laundering the loss            |

Channel-specific — invoked automatically by `trainer.train()` via each
channel's `on_epoch_end(epoch)` hook (logged into `trainer.history` as
`sentinel/{channel}/*`; manual run loops can call the methods directly):

* REVISIT — `on_epoch_end` -> `sentinel_metrics()` + reset: abort guidance
  `proj_std < 0.044` (SimSiam collapse), any `class_freq_* < 0.01` after
  epoch 2, `skip_rate > 0.30`.
* GASP-Real — `on_epoch_end` -> `epoch_sentinels(epoch)` (probe batch built
  lazily from the 1% holdout): `flag_pred_collapse` (`pred_std < 0.05`) and
  `flag_row_shortcut` (head R^2 must beat BOTH the row-only and the box-size-
  only regressor by epoch 4); also track `sign_acc` and `spearman` rising.
* TERRA — watch `terra/ordinal` and `terra/geobox` staying finite and falling;
  trainer sentinels cover the rest.

Always run the replay-only control arm (`channels=[]`, same schedule) — a
channel only wins if it beats BOTH the COCO init and that control.
