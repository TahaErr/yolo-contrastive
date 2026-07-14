# roadrecon — Colab A100 Quickstart (pure, self-contained B2 → M3 → M2)

Train our OWN road-reconstruction net from scratch on the unlabeled pool, mine
reconstruction-anomaly pothole boxes, detection-pretrain a scratch YOLOv8n on them,
and evaluate the whole detector under LOSO against the `coco` and `scratch` baselines.
**No COCO weights, no external pretrained model anywhere.**

All commands assume a Colab A100 (40 GB) with the repo cloned at
`/content/yolo-contrastive`. See [`METHOD_OPTIONS.md`](METHOD_OPTIONS.md) for the design.

```bash
cd /content/yolo-contrastive
pip install -e ".[yolo,pretrain]"        # ultralytics + opencv/pandas/pyarrow. NO transformers/geo needed.
```

---

## 0. Data

### Unlabeled pool (~181K, or ~150K without A2D2)
From the Drive `SSL_POOL_PARTS` folder (processed ~10 GB), extract the materialized JPEGs
into one directory — the pretrainer/miner just `rglob`s it:

```bash
mkdir -p /content/pool_images
for t in bdd100k a2d2 mapillary cityscapes; do
    tar -xf "/content/drive/MyDrive/SSL_POOL_PARTS/$t.tar" -C /content/pool_images
done
find /content/pool_images -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l   # sanity: ~181K
```

### Labeled Pothole-5000 (kill-gate + LOSO eval)
Use the deduped **Pothole-5000** with LOGO 10-fold splits (`splits/cv/logo/fold_0..9`,
built by `build_pothole5k.py`, 0 cross-source leaks). A single fold's `val` split (with GT
boxes) is the kill-gate set; the fold dir is the LOSO input.

---

## 1. Stage 1 — train the B2 reconstructor + mine (`examples/12`)

`examples/12_roadrecon_pretrain.py` trains B2 (saves the **M2** backbone and a full-net
checkpoint) and mines the anomaly boxes (the **M3** replay dataset) in one run.

**Trial first (recommended): `--limit 30000`** runs B2 training AND mining on the same
seeded 30K subset — a fast dry run before committing to the full pool:

```bash
python examples/12_roadrecon_pretrain.py \
    --pool /content/pool_images --out runs/roadrecon_trial \
    --limit 30000 --imgsz 384 --epochs 10 --batch 64 --device 0 --z-thresh 3.0
```

Then the full pool (drop `--limit`, longer/larger):

```bash
python examples/12_roadrecon_pretrain.py \
    --pool /content/pool_images --out runs/roadrecon \
    --imgsz 512 --epochs 30 --batch 64 --tap-level P3 --device 0 --z-thresh 3.0
```

Outputs (per `--out`): `roadrecon_backbone.pt` (M2), `roadrecon_full.pt` (mining-ready),
`mined/data.yaml` (M3 dataset). Watch the recon loss fall. *(A100 rough budget: the 30K
trial @ 384px is well under an hour; ~30 epochs on ~181K @ 512px ≈ several GPU-hours.)*

---

## 2. THE KILL-GATE — run BEFORE trusting M3 (`examples/12b`)

The decisive cheap test: are the mined anomalies actually potholes (purity) and do they
cover the small/hard tail? Run on a **labeled** fold's val split:

```bash
python examples/12b_roadrecon_killgate.py \
    --reconstructor runs/roadrecon/roadrecon_full.pt \
    --dataset datasets/splits/cv/logo/fold_0/data.yaml \
    --z-thresh 3.0 --iou 0.3 --min-box-area 128 --out runs/roadrecon_killgate
```

**GO if** `precision(purity) >= 0.5` **and** `small_recall >= 0.3`. Otherwise **KILL** —
the appearance-anomaly signal is impure or misses the hard cases, so M3 will tie at best.

**Sweep `--min-box-area` (64 / 128 / 256)** and read `small_recall` vs `precision`: the
default 256 is inherited from the depth factory and may exclude the small-pothole tail that
*is* the beat-vs-tie margin. Also sanity-check the incrementality (does the signal add over
a `scratch` init?) — if the mined set only recovers big obvious potholes a fine-tune already
gets, that is a silent tie.

---

## 3. Stage 3 — M3 anchored detection-pretraining (pure, no COCO)

The mined pothole boxes ride the **replay slot** (real Detect head, TAL/DFL); the
`RoadReconChannel` keeps reconstruction alive as a content-pressuring aux in the same step.
**Critical for a from-scratch backbone:** disable the COCO-tuned warmup-freeze and raise the
backbone LR (defaults would leave the scratch backbone almost untrained).

```python
from yolo_contrastive import AnchoredJointTrainer, RoadReconChannel
from yolo_contrastive.roadrecon import build_scratch_detector

trainer = AnchoredJointTrainer(
    model=build_scratch_detector(nc=1),                # scratch, nc=1 (matches mined + downstream)
    replay_data="runs/roadrecon/mined/data.yaml",      # mined potholes = the detection signal
    channels=[RoadReconChannel("/content/pool_images", imgsz=512)],
    lambda_aux=1.0, epochs=12, imgsz=512, batch=24,
    warmup_steps=0, backbone_lr=1e-2,                  # scratch: no warmup-freeze, higher backbone LR
    output_dir="runs/anchored_roadrecon")
ckpt = trainer.train()                                 # -> runs/anchored_roadrecon/anchored_full.pt (whole detector)
```

Sentinels to watch (per epoch): `eff_rank` of P5 rising (not collapsing), `replay/det_loss`
falling, and — since there is no COCO to forget — the replay-drift sentinel is informational.
Run a **replay-only control** (`channels=[]`) too: the reconstruction channel only earns its
keep if the full run beats it.

---

## 4. Stage 4 — LOSO eval vs the COCO & scratch baselines

Whole-detector transplant (R8) via the roadrecon runner; the `coco` / `scratch` baselines run
through the stock path automatically.

```python
from yolo_contrastive import run_cv_eval
from yolo_contrastive.roadrecon import full_transplant_detection_runner

run_cv_eval(
    [{"name": "roadrecon_m3",
      "backbone_ckpt": "runs/anchored_roadrecon/anchored_full.pt",
      "base_model": "yolov8n.yaml"}],                  # pure: random-init arch, then transplant
    "datasets/splits/cv/logo", "runs/cv_roadrecon.csv",
    baselines=("coco", "scratch"),                     # the bar to clear + the floor
    fractions=(0.1, 0.5, 1.0),                         # headline claim lives at 10%
    runners={"detection": full_transplant_detection_runner})
```

The headline comparison is **roadrecon_m3 vs coco at 10% labels**, paired per-fold Wilcoxon
across seeds. `roadrecon_m3` must clear `scratch` decisively and reach/pass `coco` to count.

---

## 5. Order of operations (fail fast)

1. **Extract pool + verify count** (§0).
2. **30K TRIAL B2 pretrain** (`--limit 30000 --epochs 10 --imgsz 384`, §1) → `roadrecon_full.pt`.
3. **KILL-GATE** (§2) on a labeled fold, sweeping `--min-box-area`. **KILL here if impure /
   small-tail missing** — before any long GPU run.
4. If GO: **full-pool B2 pretrain** (§1, drop `--limit`) → **M3 anchored** (§3, +replay-only
   control) → **LOSO eval** (§4).
5. Later phases: **B1** (own depth net from A2D2 LiDAR + Cityscapes stereo — needs net-new
   sensor-GT ingestion) → **M1** geometry → **M4** geometry∧appearance consensus.
