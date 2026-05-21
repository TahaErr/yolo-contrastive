# Examples

Runnable scripts for the main yolo-contrastive workflows. Each is a
standalone CLI — pass `--help` to any of them for options.

| Script | What it does |
|---|---|
| `01_dense_saps_pretrain.py` | Dense SAPS self-supervised pretraining of a YOLOv8n backbone |
| `02_dt_saps_dual_teacher.py` | DT-SAPS — dual-teacher (COCO + SSL) distillation pretraining |
| `03_finetune_with_backbone.py` | Fine-tune YOLO detection with a pretrained backbone |
| `04_linear_probe_eval.py` | Linear-probe evaluation of frozen pretrained features |
| `05_leakage_check.py` | Cross-set leakage check between the SSL pool and eval data |

## Typical sequence

A full run, from raw images to an evaluated detector:

```bash
# 0. Make sure the pool doesn't overlap the eval data
python examples/05_leakage_check.py \
    --pool-phash pool_phash.parquet \
    --downstream /data/eval/train /data/eval/valid

# 1. Pure-SAPS pretraining — produces the SSL teacher for step 2
python examples/01_dense_saps_pretrain.py \
    --images /data/unlabeled_pool \
    --output saps_backbone.pt

# 2. DT-SAPS dual-teacher pretraining
python examples/02_dt_saps_dual_teacher.py \
    --images /data/unlabeled_pool \
    --ssl-teacher saps_backbone.pt \
    --output dt_saps_backbone.pt

# 3a. Evaluate the frozen features (quick signal)
python examples/04_linear_probe_eval.py \
    --backbone dt_saps_backbone.pt --data dataset.yaml

# 3b. Fine-tune for the actual detection task
python examples/03_finetune_with_backbone.py \
    --backbone dt_saps_backbone.pt --data dataset.yaml
```

## Notes

- All scripts import from the top-level package (`from yolo_contrastive
  import ...`); `import yolo_contrastive` is lightweight and lazy.
- Pretraining is GPU-bound — the epoch counts shown are paper-scale
  defaults. For a quick smoke run, lower `--epochs` and point `--images` at
  a small directory.
- `dataset.yaml` is a standard Ultralytics data config (`train`, `val`,
  `nc`, `names`).
