"""full_transplant_detection_runner — LOSO eval runner for whole-detector checkpoints.

The default detection runner (``eval.run_matrix._run_detection``) loads a method's
``backbone_ckpt`` **backbone-only** (layers 0-9), re-randomizing the neck+head. That
is correct for backbone-transfer methods (M2), but an M3 ``anchored_full.pt`` is a
**whole detector** (backbone+neck+head) that must transplant intact (design rule R8) —
otherwise its trained neck/head are thrown away.

This runner plugs into ``eval.cross_val.run_cv_eval(runners={"detection": ...})``:

    * a method WITH a ``backbone_ckpt`` → whole-detector transplant via
      ``anchored.export.load_for_finetune`` (base ``"yolov8n.yaml"`` by default so no
      COCO weight is ever touched), then a standard ``YOLO.train`` fine-tune;
    * a method WITHOUT one (the ``coco`` / ``scratch`` baselines) → delegated to the
      stock ``_run_detection`` so the comparison arms run exactly as usual.

Returns the same metric dict shape as ``_run_detection``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional


def _read_nc(data_yaml: str) -> int:
    """Read the class count from an ultralytics data.yaml (default 1)."""
    import yaml as _yaml
    with open(data_yaml) as f:
        return int((_yaml.safe_load(f) or {}).get("nc", 1))


def _load_full_ncmatched(ckpt_path: str, base: str, nc: Optional[int], device: Any = "cpu",
                         verbose: bool = False):
    """Whole-detector transplant (R8) into an ``nc``-matched scratch architecture.

    ``anchored.export.load_for_finetune`` hardwires the base head (nc=80 for
    ``yolov8n.yaml``), so an ``nc``=1 checkpoint's head tensors would shape-mismatch
    and be dropped. Here we rebuild the ``DetectionModel`` at the downstream ``nc`` so
    every tensor — backbone, neck AND head — transplants intact. No COCO weights.

    ``device`` is passed through unchanged (int index, ``"cpu"``, or ``torch.device`` —
    NOT stringified, which would make ``.to("0")`` raise on CUDA).
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ..pretrain.backbone_utils import _extract_state_dict, _safe_torch_load, transplant_full

    yolo = YOLO(base, task="detect")
    head = yolo.model.model[-1]
    if nc is not None and int(getattr(head, "nc", -1)) != int(nc):
        cfg = base if str(base).endswith(".yaml") else "yolov8n.yaml"
        yolo.model = DetectionModel(cfg, nc=int(nc), verbose=False)
    yolo.model.to(device)

    state = _extract_state_dict(_safe_torch_load(ckpt_path))
    # S1: refuse a silent R8-violating partial transplant. If the checkpoint's Detect
    # head shape-mismatches the nc-matched arch, transplant_full would drop those head
    # tensors and leave a random head under a pretrained backbone — exactly what R8
    # forbids. Require the pretraining nc to equal the downstream nc.
    model_state = yolo.model.state_dict()
    idxs = [int(k.split(".")[1]) for k in model_state
            if k.startswith("model.") and k.split(".")[1].isdigit()]
    head_prefix = f"model.{max(idxs)}." if idxs else "model."
    bad = [k for k in state if k.startswith(head_prefix) and k in model_state
           and state[k].shape != model_state[k].shape]
    if bad:
        raise ValueError(
            f"nc mismatch: checkpoint Detect head is incompatible with the nc={nc} "
            f"architecture ({len(bad)} head tensors would be dropped). The M3 pretraining "
            f"nc must equal the downstream nc.")

    n = transplant_full(yolo.model, state, verbose=verbose)
    if n == 0:
        raise RuntimeError(f"_load_full_ncmatched: zero tensors transplanted from {ckpt_path!r}")
    if not getattr(yolo, "ckpt", None):
        yolo.ckpt = {"epoch": -1}   # so .train() carries the transplanted weights
    return yolo


def full_transplant_detection_runner(cell: Dict[str, Any], hp: Dict[str, Any]) -> Dict[str, Any]:
    """Detection runner that transplants whole-detector checkpoints (R8).

    See module docstring. Baselines (no ``backbone_ckpt``) fall back to the stock
    detection runner so they are evaluated identically to a normal ``run_cv_eval``.
    """
    backbone_ckpt = cell["method"].get("backbone_ckpt")
    if not backbone_ckpt:
        # coco / scratch baselines — stock path (identical to a normal run_cv_eval).
        from ..eval.run_matrix import _run_detection
        return _run_detection(cell, hp)

    from ..eval.run_matrix import _fraction_train_yaml

    data_yaml = cell["dataset"].get("data_yaml")
    if not data_yaml:
        raise ValueError("full_transplant_detection_runner requires cell['dataset']['data_yaml']")

    # Pure by default: build the bare architecture and transplant our detector over it.
    # Do NOT fall back to hp['base_model'] — run_cv_eval merges DEFAULT_HP['base_model']
    # = "yolov8n.pt" (COCO), which load_backbones does not override per method, so an
    # hp fallback would silently pull COCO weights and break purity.
    base_model = cell["method"].get("base_model") or "yolov8n.yaml"
    if not str(base_model).endswith(".yaml"):
        raise ValueError(
            f"full_transplant path requires a random-init .yaml base (pure, no COCO), "
            f"got base_model={base_model!r}")
    epochs = int(hp.get("epochs", 30))
    imgsz = int(hp.get("imgsz", 640))
    batch = int(hp.get("batch", 16))
    freeze = int(hp.get("freeze", 0))
    device = hp.get("device", 0)
    project = hp.get("project", "/content/runs/eval_matrix")
    fraction = float(cell.get("fraction", 1.0))
    seed = int(cell.get("seed", 42))

    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    cell_id = cell.get("cell_id", "")
    run_name = (
        f"cell_{cell_id[:8]}" if cell_id
        else f"{cell['method']['name']}_{cell['dataset']['name']}_seed{seed}"
    )

    if fraction < 1.0:
        data_yaml = _fraction_train_yaml(
            data_yaml, fraction, seed,
            os.path.join(str(project), "_frac_data"),
            prefix=f"{cell['dataset']['name']}_")

    # Whole-detector transplant into an nc-matched arch, then a standard ultralytics
    # fine-tune. We do NOT route through FinetuneDetectionTrainer: its setup_model
    # would rebuild from base_model and (via YCL_PRETRAINED) reload backbone-only,
    # discarding the transplant. _load_full_ncmatched sets yolo.ckpt so .train()
    # carries the transplanted weights.
    nc = _read_nc(data_yaml)
    yolo = _load_full_ncmatched(backbone_ckpt, base_model, nc, device)  # device passed through (M2)
    try:
        train_kwargs = dict(
            data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
            project=project, name=run_name, exist_ok=True, verbose=False, plots=False,
            freeze=freeze,
        )
        for _k in ("optimizer", "lr0", "lrf", "patience", "cos_lr", "weight_decay"):
            if _k in hp:
                train_kwargs[_k] = hp[_k]
        results = yolo.train(**train_kwargs)
    finally:
        del yolo
        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass

    if not hasattr(results, "box"):
        raise RuntimeError(
            f"full_transplant_detection_runner: results has no .box (got {type(results).__name__})"
        )
    return {
        "metric": "mAP50-95",
        "metric_value": float(results.box.map),
        "mAP50": float(results.box.map50),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }


def detection_runners() -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Convenience: the ``runners=`` mapping for ``run_cv_eval`` that evaluates
    whole-detector (M3) methods with full transplant while leaving baselines stock."""
    return {"detection": full_transplant_detection_runner}
