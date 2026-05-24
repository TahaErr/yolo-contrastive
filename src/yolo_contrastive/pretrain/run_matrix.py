"""YAML-driven pretrain matrix orchestrator.

Faz 5 prep — ablation orchestration for ``DenseSSLPretrainer``.

Why this exists:
    Plan §5 calls for a four-axis ablation grid (saps_mode × saps_both_lambda
    × queue_update_strategy × saps_t_scale = up to 192 cells). Manual loop is
    infeasible. This module reads a YAML describing fixed-vs-varying
    parameters, expands the cartesian product (with optional exclusion
    rules), and runs each cell with append-mode CSV logging + resume.

Sister to :mod:`yolo_contrastive.eval.run_matrix`. The two are deliberately
near-identical in skeleton — same resume logic, same on_error contract,
same CSV-as-source-of-truth pattern — so anyone familiar with one can read
the other without surprise. Differences are scoped to:

  * Grid is parametric (any number of named axes) rather than the fixed
    methods/datasets/fractions/seeds tetrad. This fits SSL pretraining
    where ablations vary trainer kwargs.
  * Exclude DSL accepts a list of allowed values per field, not just a
    scalar — so common patterns like "λ is redundant when mode≠both" can
    be expressed as one YAML stanza instead of N enumerated rows.
  * CSV records axes as a JSON column, keeping the schema deterministic
    regardless of which axes a given grid varies. This makes pandas-side
    analysis of multiple matrix runs straightforward.

YAML schema:

    output_dir: /content/drive/.../pretrain_runs
    output_csv: pretrain_results.csv

    base:                                # fixed across every cell
      images_dir: /content/ssl_pool_local
      model: yolov8n.pt
      imgsz: 640
      epochs: 100
      batch_size: 64
      lr: 1.0e-3
      warmup_epochs: 5
      queue_size: 65536
      momentum: 0.999
      temperature: 0.2

    grid:                                # varying ablation axes
      saps_mode: [none, within, cross, both]
      saps_both_lambda: [0.0, 0.5, 1.0, 2.0]
      queue_update_strategy: [pooled, per_position, subsample]
      saps_t_scale: [0.5, 1.0, 2.0, .inf]

    seeds: [42]

    exclude:
      # λ redundant when mode≠both — keep only λ=0.0 representative.
      - saps_mode: [none, within, cross]
        saps_both_lambda: [0.5, 1.0, 2.0]

Single-process, sequential. Designed for Colab/single-machine usage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from itertools import product
from typing import Any, Callable, Dict, List, Optional

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "run_matrix requires pyyaml. Install with: pip install pyyaml"
    ) from e


#: CSV columns — fixed schema across runs so resume + cross-run analysis
#: stay deterministic. ``axes_json`` holds the per-cell grid values as a
#: JSON string so we don't need a dynamic header per YAML.
CSV_COLUMNS = [
    "cell_id",        # short deterministic id derived from axes+seed
    "seed",
    "axes_json",      # JSON-encoded {axis: value, ...}
    "metric",         # name of the captured metric (e.g. "final_loss")
    "metric_value",
    "status",         # "ok" | "failed" | "skipped"
    "elapsed_s",
    "error",
    "started_at",
    "backbone_path",  # produced backbone file, if any
]


# ─────────────────────────────────────────────────────────────────────────
# Runner interface
# ─────────────────────────────────────────────────────────────────────────


def _run_pretrain(cell: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """Default runner: instantiate ``DenseSSLPretrainer`` and call ``.train``.

    Imports ``DenseSSLPretrainer`` lazily so the matrix module stays
    importable in environments without ultralytics/torch installed
    (e.g. CI containers that only test orchestration logic).

    Returns a dict with at minimum ``metric``, ``metric_value``, and
    ``backbone_path``. ``cell["axes"]`` overrides any same-named key in
    ``base`` — that's how the grid varies trainer kwargs.
    """
    from .dense_trainer import DenseSSLPretrainer  # local import — see docstring

    import inspect

    merged = {**base, **cell["axes"]}
    # train()-time kwargs vs. constructor-time kwargs are split here so the
    # caller doesn't need to remember which is which. A third class —
    # orchestrator *meta* keys (e.g. output_dir) — belongs to neither: they
    # configure run_matrix itself, not the trainer. The split is therefore
    # signature-driven: init_kwargs is filtered to what __init__ actually
    # accepts, so any meta/unknown key is dropped instead of crashing the
    # constructor. (Pre-fix: a bare "not in train_keys" rule routed
    # output_dir into init_kwargs → TypeError. Mock-runner unit tests never
    # exercised this real split — see plan §10.33.)
    train_keys = {
        "images_dir", "epochs", "batch_size", "lr", "warmup_epochs",
        "weight_decay", "num_workers", "output", "save_every", "print_every",
    }
    train_kwargs = {k: merged[k] for k in train_keys if k in merged}

    init_accepted = set(
        inspect.signature(DenseSSLPretrainer.__init__).parameters
    ) - {"self"}
    init_kwargs = {
        k: v for k, v in merged.items()
        if k not in train_keys and k in init_accepted
    }
    # Keys that are neither a train kwarg, an __init__ param, nor a known
    # meta key — surface them so a YAML typo isn't silently swallowed.
    meta_keys = {"output_dir", "output_csv", "task"}
    unknown = set(merged) - train_keys - init_accepted - meta_keys
    if unknown:
        DEFAULT_LOGGER_PRINT = print
        DEFAULT_LOGGER_PRINT(
            f"[run_matrix] WARN: YAML key(s) {sorted(unknown)} match no "
            f"trainer arg or known meta key — ignored. Check for typos."
        )

    backbone_path = train_kwargs.get("output") or os.path.join(
        merged.get("output_dir", "."), f"{cell['cell_id']}.pt"
    )
    train_kwargs["output"] = backbone_path

    pretrainer = DenseSSLPretrainer(**init_kwargs)
    try:
        out_path = pretrainer.train(**train_kwargs)
    finally:
        if hasattr(pretrainer, "cleanup"):
            pretrainer.cleanup()

    return {
        "metric": "final_backbone",
        "metric_value": out_path,
        "backbone_path": out_path,
    }


DEFAULT_RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "pretrain": _run_pretrain,
}


# ─────────────────────────────────────────────────────────────────────────
# PretrainMatrix
# ─────────────────────────────────────────────────────────────────────────


class PretrainMatrix:
    """Orchestrate a grid of pretrain configurations.

    Args:
        config_path: path to YAML config. OR pass ``config`` dict directly.
        output_csv: CSV path (created/appended). If absent in YAML, must be
            provided here.
        runners: optional ``{task_name: runner_fn}``; defaults to
            ``DEFAULT_RUNNERS`` ({"pretrain": _run_pretrain}).
        config: pre-loaded config dict (overrides ``config_path``).

    Usage:
        pm = PretrainMatrix("pretrain.yaml", "results.csv")
        cells = pm.expand()
        print(f"Will run {len(cells)} cells")
        pm.run(resume=True, on_error="continue")
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_csv: Optional[str] = None,
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

        self.output_csv = output_csv or self.config.get("output_csv")
        if not self.output_csv:
            raise ValueError("output_csv must be set (arg or config)")

        self.runners = runners if runners is not None else DEFAULT_RUNNERS
        self.task = self.config.get("task", "pretrain")

        self._validate_config()

    # ── validation ────────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        cfg = self.config
        for required in ("base", "grid", "seeds"):
            if required not in cfg:
                raise ValueError(f"config missing required key: {required!r}")

        if not isinstance(cfg["base"], dict):
            raise ValueError("'base' must be a mapping")

        grid = cfg["grid"]
        if not isinstance(grid, dict) or not grid:
            raise ValueError("'grid' must be a non-empty mapping of axis → values")
        for axis, values in grid.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"grid axis {axis!r} must be a non-empty list, got {type(values).__name__}"
                )

        seeds = cfg["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("'seeds' must be a non-empty list")

        if self.task not in self.runners:
            raise ValueError(
                f"task {self.task!r} has no runner. "
                f"Available: {sorted(self.runners.keys())}"
            )

        # Excludes: each entry must be a mapping; values can be scalar or list
        for exc in cfg.get("exclude", []):
            if not isinstance(exc, dict):
                raise ValueError(f"exclude entry must be a mapping, got {exc!r}")

    # ── expansion ────────────────────────────────────────────────────────

    def expand(self) -> List[Dict[str, Any]]:
        """Cartesian product of grid × seeds, with exclude filtering.

        Returns a list of cells, each with keys:
            axes      — {axis_name: chosen_value, ...} for this cell
            seed      — int
            cell_id   — short deterministic hash (axes + seed)
            base      — reference to the base config dict
        """
        cfg = self.config
        axis_names = list(cfg["grid"].keys())
        axis_values = [cfg["grid"][n] for n in axis_names]
        excludes = cfg.get("exclude", [])

        cells: List[Dict[str, Any]] = []
        for combo in product(*axis_values):
            axes = dict(zip(axis_names, combo))
            for seed in cfg["seeds"]:
                cell = {
                    "axes": axes,
                    "seed": int(seed),
                    "base": cfg["base"],
                    "cell_id": self._cell_id(axes, int(seed)),
                }
                if self._is_excluded(cell, excludes):
                    continue
                cells.append(cell)
        return cells

    @staticmethod
    def _cell_id(axes: Dict[str, Any], seed: int) -> str:
        """Deterministic short id from axes+seed. Used for backbone filenames
        and CSV cell keys. Stable across runs because we sort axis keys."""
        payload = json.dumps(
            {"axes": axes, "seed": seed}, sort_keys=True, default=str
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _is_excluded(
        self, cell: Dict[str, Any], excludes: List[Dict[str, Any]]
    ) -> bool:
        """A cell matches an exclude entry if every field in the entry matches.

        Each field's value in the exclude entry may be:
          * a scalar  — cell's axis value must equal it
          * a list    — cell's axis value must be in the list (the DSL extension)
          * "seed"    — special key matching the seed (scalar or list)

        Unknown axis keys in an exclude entry are treated as "no match" for
        safety (typos shouldn't silently drop cells).
        """
        for exc in excludes:
            match = True
            for key, allowed in exc.items():
                if key == "seed":
                    actual = cell["seed"]
                elif key in cell["axes"]:
                    actual = cell["axes"][key]
                else:
                    # Unknown key — be conservative and treat as no-match
                    match = False
                    break

                if isinstance(allowed, list):
                    if actual not in allowed:
                        match = False
                        break
                else:
                    if actual != allowed:
                        match = False
                        break
            if match:
                return True
        return False

    # ── resume ──────────────────────────────────────────────────────────

    def _completed_cells(self) -> set:
        """Set of cell_ids with status='ok' in the CSV (or empty if no CSV)."""
        if not os.path.exists(self.output_csv):
            return set()
        completed = set()
        try:
            with open(self.output_csv) as f:
                for row in csv.DictReader(f):
                    if row.get("status") == "ok":
                        completed.add(row.get("cell_id", ""))
        except (OSError, ValueError):
            return set()
        return completed

    # ── csv writing ─────────────────────────────────────────────────────

    def _ensure_csv_header(self) -> None:
        if not os.path.exists(self.output_csv):
            d = os.path.dirname(self.output_csv) or "."
            os.makedirs(d, exist_ok=True)
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

        Mirrors :meth:`yolo_contrastive.eval.run_matrix.RunMatrix.run` —
        same semantics for ``resume``, ``on_error``, ``verbose``.
        """
        if on_error not in ("continue", "raise"):
            raise ValueError(
                f"on_error must be 'continue' or 'raise', got {on_error!r}"
            )

        if cells is None:
            cells = self.expand()

        self._ensure_csv_header()
        already_done = self._completed_cells() if resume else set()

        runner = self.runners[self.task]
        results: List[Dict[str, Any]] = []

        for i, cell in enumerate(cells, start=1):
            row = {
                "cell_id": cell["cell_id"],
                "seed": cell["seed"],
                "axes_json": json.dumps(cell["axes"], sort_keys=True, default=str),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            if resume and cell["cell_id"] in already_done:
                row["status"] = "skipped"
                row["error"] = "already in CSV"
                if verbose:
                    self._print(f"[{i}/{len(cells)}] SKIP {cell['cell_id']} (resume)")
                results.append(row)
                continue

            t0 = time.time()
            try:
                out = runner(cell, cell["base"])
                row["metric"] = out.get("metric", "")
                row["metric_value"] = out.get("metric_value", "")
                row["backbone_path"] = out.get("backbone_path", "")
                row["status"] = "ok"
                row["elapsed_s"] = round(time.time() - t0, 2)
                if verbose:
                    self._print(
                        f"[{i}/{len(cells)}] OK   {cell['cell_id']} → "
                        f"{row['metric']}={row['metric_value']}"
                    )
            except Exception as e:
                if on_error == "raise":
                    raise
                row["status"] = "failed"
                row["error"] = f"{type(e).__name__}: {e}"
                row["elapsed_s"] = round(time.time() - t0, 2)
                if verbose:
                    self._print(
                        f"[{i}/{len(cells)}] FAIL {cell['cell_id']}: {row['error']}"
                    )

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
        g = self.config.get("grid", {})
        return (
            f"PretrainMatrix(task={self.task!r}, "
            f"axes={list(g.keys())}, "
            f"seeds={self.config.get('seeds', [])})"
        )
