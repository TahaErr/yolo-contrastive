"""YAML-driven eval matrix orchestrator.

Faz 4.6 — Eval infrastructure (WORK_PLAN_v5 §5).

Why this exists:
    Faz 5 runs (4 methods) × (9 label fractions) × (2 datasets) × (1-3 seeds)
    = up to 200+ training runs. Manual orchestration is infeasible. This
    module reads a YAML describing the matrix, expands the cartesian
    product, and runs each cell with append-mode CSV logging + resume.

YAML schema (full example):

    task: linear_probe         # or 'detection'
    output_csv: results.csv

    methods:
      - name: ours_a_d
        backbone_ckpt: /path/to/ours.pt
      - name: mocov3
        backbone_ckpt: /path/to/mocov3.pt

    datasets:
      - name: pothole
        data_yaml: pothole.yaml
        num_classes: 1

    fractions: [0.1, 0.25, 0.5, 1.0]
    seeds: [42, 43]

    # Optional task-specific hyperparameters
    hp:
      epochs: 10
      lr: 0.01
      batch_size: 16

    # Optional excludes (skip specific cells)
    exclude:
      - {method: ours_a_d, fraction: 0.01}      # too few samples, skip

Cell expansion:
    Cartesian product of methods × datasets × fractions × seeds, minus
    cells matching any `exclude` filter. ~size = M×D×F×S.

Resume:
    On startup, if output_csv exists, read it. For each cell already
    present with status="ok", skip it. Re-run failed/missing cells.

Error handling:
    on_error="continue" (default): catch exception, write row with
        status="failed" and error message, proceed to next cell.
    on_error="raise": let exception propagate, halt the matrix.

Single-process, in-process: each cell runs sequentially in the calling
Python process. No multi-GPU orchestration. Designed for Colab/single-machine.
"""

from __future__ import annotations

import csv
import os
import time
import traceback
from itertools import product
from typing import Any, Callable, Dict, List, Optional

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "run_matrix requires pyyaml. Install with: pip install pyyaml"
    ) from e


# CSV columns — fixed schema so resume works deterministically
CSV_COLUMNS = [
    "method", "dataset", "fraction", "seed", "task",
    "metric", "metric_value",
    "status",          # "ok" | "failed" | "skipped"
    "elapsed_s", "error", "started_at",
]


# ─────────────────────────────────────────────────────────────────────────
# Runner interface
# ─────────────────────────────────────────────────────────────────────────


def _run_linear_probe(cell: Dict[str, Any], hp: Dict[str, Any]) -> Dict[str, Any]:
    """Linear probe runner: Faz 4.5's LinearProbeTrainer.

    Expects cell to contain:
        method.backbone_ckpt
        dataset.data_yaml | dataset.train_loader | dataset.val_loader
        dataset.num_classes
        seed, fraction
    Plus hp from YAML: epochs, lr, batch_size, weight_decay, feat_level.

    Returns:
        {"metric": "mAP", "metric_value": float, ...details...}

    NOTE: full implementation requires a YOLO-format → multi-label dataset
    converter, which lives in Faz 4.3 (data/unified_loader.py — pending
    Pothole 5K dataset arrival). For now this runner accepts pre-built
    DataLoaders passed via cell["_train_loader"] / cell["_val_loader"]
    keys (programmatic use), and otherwise raises NotImplementedError
    when only data_yaml paths are given.
    """
    from ..pretrain import DenseSSLPretrainer  # noqa: F401  (sanity import)
    from .linear_probe import LinearProbeTrainer

    backbone = cell["method"].get("backbone")
    backbone_ckpt = cell["method"].get("backbone_ckpt")
    num_classes = cell["dataset"]["num_classes"]
    feat_level = hp.get("feat_level", "P5")
    epochs = int(hp.get("epochs", 10))
    lr = float(hp.get("lr", 1e-2))
    weight_decay = float(hp.get("weight_decay", 0.0))
    seed = int(cell.get("seed", 42))

    # Apply seed
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # Get loaders — programmatic injection only for now
    train_loader = cell.get("_train_loader")
    val_loader = cell.get("_val_loader")
    if train_loader is None or val_loader is None:
        raise NotImplementedError(
            "linear_probe runner needs DataLoaders via cell['_train_loader'] "
            "and cell['_val_loader']. Building from data_yaml is deferred to "
            "Faz 4.3 (data/unified_loader.py) once Pothole 5K is available."
        )

    probe = LinearProbeTrainer(
        backbone=backbone if backbone is not None else "yolov8n.pt",
        num_classes=num_classes,
        backbone_ckpt=backbone_ckpt,
        feat_level=feat_level,
        device=cell.get("device"),
    )
    try:
        result = probe.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=False,
        )
    finally:
        probe.cleanup()

    return {
        "metric": "mAP",
        "metric_value": float(result["best_val_mAP"]),
        "best_epoch": result["best_epoch"],
    }


def _run_detection(cell: Dict[str, Any], hp: Dict[str, Any]) -> Dict[str, Any]:
    """Detection runner — STUB. Will be filled in Faz 5 with Ultralytics call.

    Faz 5 implementation: load backbone via env vars (YCL_PRETRAINED + freeze
    settings), call YOLO(...).train(data=..., trainer=FinetuneDetectionTrainer).
    Capture final mAP50 from results object.
    """
    raise NotImplementedError(
        "Detection runner is a stub. Faz 5 will integrate Ultralytics YOLO.train."
    )


DEFAULT_RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "linear_probe": _run_linear_probe,
    "detection": _run_detection,
}


# ─────────────────────────────────────────────────────────────────────────
# Run matrix
# ─────────────────────────────────────────────────────────────────────────


class RunMatrix:
    """Orchestrate a grid of (method × dataset × fraction × seed) runs.

    Args:
        config_path: path to YAML config (see module docstring for schema).
                     OR pass `config` dict directly to skip file I/O.
        output_csv: path to CSV results file (created/appended).
        runners: optional dict of task_name → runner_fn for testing or
                 custom tasks. Default uses DEFAULT_RUNNERS.
        config: optional pre-loaded config dict (overrides config_path).

    Usage:
        rm = RunMatrix("eval_config.yaml", "results.csv")
        cells = rm.expand()
        print(f"Will run {len(cells)} cells")
        df_summary = rm.run(resume=True, on_error="continue")
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_csv: str = "results.csv",
        runners: Optional[Dict[str, Callable]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("must provide either config_path or config dict")

        self.output_csv = output_csv
        self.runners = runners if runners is not None else DEFAULT_RUNNERS
        self._validate_config()

    # ── validation ────────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        cfg = self.config
        for required in ("task", "methods", "datasets", "fractions", "seeds"):
            if required not in cfg:
                raise ValueError(f"config missing required key: {required!r}")

        if cfg["task"] not in self.runners:
            raise ValueError(
                f"task {cfg['task']!r} has no runner. "
                f"Available: {sorted(self.runners.keys())}"
            )

        for m in cfg["methods"]:
            if "name" not in m:
                raise ValueError(f"method entry missing 'name': {m}")
        for d in cfg["datasets"]:
            if "name" not in d:
                raise ValueError(f"dataset entry missing 'name': {d}")
        for f in cfg["fractions"]:
            if not 0.0 < float(f) <= 1.0:
                raise ValueError(f"fraction {f} not in (0, 1]")

    # ── expansion ────────────────────────────────────────────────────────

    def expand(self) -> List[Dict[str, Any]]:
        """Cartesian product of methods × datasets × fractions × seeds.

        Returns list of cell dicts with keys:
            method, dataset, fraction, seed, task, hp
        """
        cfg = self.config
        excludes = cfg.get("exclude", [])
        hp = cfg.get("hp", {})

        cells: List[Dict[str, Any]] = []
        for m, d, f, s in product(
            cfg["methods"], cfg["datasets"], cfg["fractions"], cfg["seeds"]
        ):
            cell = {
                "method": m,
                "dataset": d,
                "fraction": float(f),
                "seed": int(s),
                "task": cfg["task"],
                "hp": hp,
            }
            if self._is_excluded(cell, excludes):
                continue
            cells.append(cell)
        return cells

    def _is_excluded(
        self, cell: Dict[str, Any], excludes: List[Dict[str, Any]]
    ) -> bool:
        """A cell matches an exclude filter if all the filter's fields match."""
        for exc in excludes:
            match = True
            for key, val in exc.items():
                if key == "method":
                    if cell["method"]["name"] != val:
                        match = False
                        break
                elif key == "dataset":
                    if cell["dataset"]["name"] != val:
                        match = False
                        break
                elif key == "fraction":
                    if abs(cell["fraction"] - float(val)) > 1e-9:
                        match = False
                        break
                elif key == "seed":
                    if cell["seed"] != int(val):
                        match = False
                        break
                else:
                    # Unknown filter key — treat as no-match
                    match = False
                    break
            if match:
                return True
        return False

    # ── resume ──────────────────────────────────────────────────────────

    def _completed_cells(self) -> set:
        """Read CSV and return set of (method, dataset, fraction, seed) tuples
        that already have status='ok'."""
        if not os.path.exists(self.output_csv):
            return set()
        completed = set()
        try:
            with open(self.output_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") != "ok":
                        continue
                    key = (
                        row.get("method", ""),
                        row.get("dataset", ""),
                        float(row.get("fraction", 0)),
                        int(row.get("seed", 0)),
                    )
                    completed.add(key)
        except (OSError, ValueError):
            return set()
        return completed

    @staticmethod
    def _cell_key(cell: Dict[str, Any]) -> tuple:
        return (
            cell["method"]["name"],
            cell["dataset"]["name"],
            cell["fraction"],
            cell["seed"],
        )

    # ── csv writing ─────────────────────────────────────────────────────

    def _ensure_csv_header(self) -> None:
        if not os.path.exists(self.output_csv):
            os.makedirs(os.path.dirname(self.output_csv) or ".", exist_ok=True)
            with open(self.output_csv, "w", newline="") as f:
                csv.writer(f).writerow(CSV_COLUMNS)

    def _append_row(self, row: Dict[str, Any]) -> None:
        with open(self.output_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})

    # ── execution ────────────────────────────────────────────────────────

    def run(
        self,
        resume: bool = True,
        on_error: str = "continue",
        verbose: bool = True,
        cells: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run all cells (or a pre-expanded subset).

        Args:
            resume: if True, skip cells already in output_csv with status=ok.
            on_error: "continue" → log error and proceed; "raise" → halt.
            verbose: print progress per cell.
            cells: optional list of pre-expanded cells (e.g. from expand()
                with extra programmatic injection like _train_loader).

        Returns:
            list of result dicts (one per cell attempted).
        """
        if on_error not in ("continue", "raise"):
            raise ValueError(f"on_error must be 'continue' or 'raise', got {on_error!r}")

        if cells is None:
            cells = self.expand()

        self._ensure_csv_header()
        already_done = self._completed_cells() if resume else set()

        runner = self.runners[self.config["task"]]
        results: List[Dict[str, Any]] = []

        for i, cell in enumerate(cells, start=1):
            key = self._cell_key(cell)
            row = {
                "method": cell["method"]["name"],
                "dataset": cell["dataset"]["name"],
                "fraction": cell["fraction"],
                "seed": cell["seed"],
                "task": cell["task"],
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            if resume and key in already_done:
                row["status"] = "skipped"
                row["error"] = "already in CSV"
                if verbose:
                    self._print(f"[{i}/{len(cells)}] SKIP {key} (resume)")
                results.append(row)
                continue

            t0 = time.time()
            try:
                out = runner(cell, cell["hp"])
                row["metric"] = out.get("metric", "")
                row["metric_value"] = out.get("metric_value", "")
                row["status"] = "ok"
                row["elapsed_s"] = round(time.time() - t0, 2)
                if verbose:
                    self._print(
                        f"[{i}/{len(cells)}] OK   {key} → "
                        f"{row['metric']}={row['metric_value']}"
                    )
            except Exception as e:
                if on_error == "raise":
                    raise
                row["status"] = "failed"
                row["error"] = f"{type(e).__name__}: {e}"
                row["elapsed_s"] = round(time.time() - t0, 2)
                if verbose:
                    self._print(f"[{i}/{len(cells)}] FAIL {key}: {row['error']}")
                # Optionally also include traceback for debugging
                # (kept short — full traceback is in stderr if escalated)

            self._append_row(row)
            results.append(row)

        return results

    # ── helpers ──────────────────────────────────────────────────────────

    def _print(self, msg: str) -> None:
        try:
            from ultralytics.utils import LOGGER
            LOGGER.info(msg)
        except Exception:
            print(msg)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"RunMatrix(task={cfg.get('task')!r}, "
            f"methods={len(cfg.get('methods', []))}, "
            f"datasets={len(cfg.get('datasets', []))}, "
            f"fractions={len(cfg.get('fractions', []))}, "
            f"seeds={len(cfg.get('seeds', []))})"
        )
