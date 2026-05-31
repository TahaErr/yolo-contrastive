"""Cross-validation training + evaluation across backbones.

WORK_PLAN_v9 §5 / §13.4 — downstream eval. Fine-tunes every backbone on every
CV fold and reports per-backbone mean +/- std of the validation metric.

This is a THIN layer over ``RunMatrix``: each CV fold (from ``build_cv_splits``)
is treated as one "dataset" — its own ``data.yaml`` — so the matrix is

    methods   = backbones        (N backbones)
    datasets  = folds            (K folds)
    fractions = [1.0]            (the fold's data.yaml IS the train/val set;
                                  ``fraction`` is metadata-only for detection)
    seeds     = [seed]

giving N x K detection fine-tunes. RunMatrix supplies append-mode CSV logging
and **resume** — essential, since e.g. 14 backbones x 10 logo folds = 140 runs
will not finish in one Colab session; re-running skips completed cells.
``aggregate_cv_results`` then groups the per-cell rows by backbone and computes
the across-fold statistics.

Fine-tune defaults (``DEFAULT_HP``) follow the project's protocol (5 epochs,
imgsz 320, batch 16, full fine-tune to match the COCO "direct fine-tune"
baseline). Override any of them via ``hp=...``; ``freeze`` in particular is a
methodology choice (0 = full fine-tune, 10 = frozen backbone / probe-like).
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import yaml

from .run_matrix import RunMatrix

DEFAULT_HP = {
    "base_model": "yolov8n.pt",
    "epochs": 5,
    "imgsz": 320,
    "batch": 16,
    "freeze": 0,          # 0 = full fine-tune (matches the COCO direct-fine-tune baseline)
    "device": 0,
    "project": "runs/cv_eval",
}
DEFAULT_SEED = 0


# --------------------------------------------------------------------------- backbones
def _auto_name(i: int, n: int) -> str:
    return f"bb_{i:0{max(2, len(str(n)))}d}"


def _coerce_backbones(spec) -> list[dict]:
    if isinstance(spec, dict):
        return [{"name": k, "backbone_ckpt": v} for k, v in spec.items()]
    if isinstance(spec, (list, tuple)):
        items = list(spec)
        if items and isinstance(items[0], dict):
            return [dict(x) for x in items]
        n = len(items)
        return [{"name": _auto_name(i, n), "backbone_ckpt": p} for i, p in enumerate(items, 1)]
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"backbones file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _coerce_backbones(yaml.safe_load(path.read_text()) or [])
    raw_items: list = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        raw_items.append({"name": parts[0], "backbone_ckpt": parts[1].strip()}
                         if len(parts) == 2 else parts[0])
    n = len(raw_items)
    return [it if isinstance(it, dict) else {"name": _auto_name(i, n), "backbone_ckpt": it}
            for i, it in enumerate(raw_items, 1)]


def load_backbones(spec) -> list[dict]:
    """Normalize a backbone spec to ``[{'name', 'backbone_ckpt'}]``.

    spec: dict ``{name: ckpt}`` | list of ``{name, ckpt}`` | list of ckpt paths
    (auto-named ``bb_01``..) | path to a ``.yaml``/``.yml`` or a ``.txt`` with
    ``name <whitespace> /path.pt`` per line ('#' comments allowed).
    """
    out = []
    for it in _coerce_backbones(spec):
        ckpt = it.get("backbone_ckpt") or it.get("ckpt") or it.get("path")
        if not it.get("name") or not ckpt:
            raise ValueError(f"backbone entry needs name + ckpt: {it}")
        out.append({"name": str(it["name"]), "backbone_ckpt": str(ckpt)})
    if not out:
        raise ValueError("no backbones in spec")
    names = [b["name"] for b in out]
    if len(set(names)) != len(names):
        raise ValueError("duplicate backbone names")
    return out


# --------------------------------------------------------------------------- folds
def _fold_datasets(fold_dir: str | Path) -> list[dict]:
    """Enumerate CV folds as RunMatrix dataset entries (name + data_yaml)."""
    fold_dir = Path(fold_dir)
    summ = fold_dir / "summary.json"
    datasets: list[dict] = []
    if summ.is_file():
        for f in json.loads(summ.read_text())["folds"]:
            dy = f.get("data_yaml") or str(fold_dir / f"fold_{f['fold']}" / "data.yaml")
            datasets.append({"name": f"fold_{f['fold']}", "data_yaml": dy})
    else:
        for d in sorted(fold_dir.glob("fold_*")):
            if (d / "data.yaml").is_file():
                datasets.append({"name": d.name, "data_yaml": str(d / "data.yaml")})
    if not datasets:
        raise FileNotFoundError(
            f"no folds under {fold_dir} — run build_cv_splits first")
    return datasets


# --------------------------------------------------------------------------- matrix
BASELINE_METHODS = {
    # COCO-pretrained yolov8n, fine-tuned with NO SSL backbone load (referans taban).
    "coco": {"name": "coco_baseline", "base_model": "yolov8n.pt"},
    # Random init from the architecture config — lower bound (alt sınır).
    "scratch": {"name": "scratch", "base_model": "yolov8n.yaml"},
}


def _baseline_methods(names) -> list[dict]:
    out = []
    for n in names:
        if n not in BASELINE_METHODS:
            raise ValueError(f"unknown baseline {n!r}; choose from {sorted(BASELINE_METHODS)}")
        out.append(dict(BASELINE_METHODS[n]))
    return out


def build_cv_matrix(backbones, fold_dir, *, task: str = "detection",
                    seed: int = DEFAULT_SEED, hp: dict | None = None,
                    baselines=()) -> dict:
    """Build the RunMatrix config: (backbones + baselines) x folds, fraction=[1.0].

    baselines: iterable of "coco" / "scratch" — control methods run through the
    SAME folds as the SSL backbones, so they get per-fold mean/std on equal footing.
    They carry a ``base_model`` and no ``backbone_ckpt``; ``_run_detection`` then
    initialises from the base model and skips SSL loading.
    """
    methods = load_backbones(backbones) + _baseline_methods(baselines)
    names = [m["name"] for m in methods]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate method names (backbone/baseline clash): {names}")
    return {
        "task": task,
        "methods": methods,
        "datasets": _fold_datasets(fold_dir),
        "fractions": [1.0],
        "seeds": [int(seed)],
        "hp": {**DEFAULT_HP, **(hp or {})},
    }


# --------------------------------------------------------------------------- aggregate
def aggregate_cv_results(csv_path: str | Path, *, metric: str = "mAP50",
                         out_path: str | None = None) -> dict:
    """Group RunMatrix rows by backbone; report across-fold mean/std/min/max."""
    rows = list(csv.DictReader(open(csv_path, newline="")))
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") == "failed"]

    def _val(r) -> float:
        v = r.get(metric, "")
        if v in ("", None):
            v = r.get("metric_value", "")  # fall back to mAP50-95
        return float(v)

    all_folds = sorted({r["dataset"] for r in ok})
    per: dict[str, dict[str, float]] = {}
    for r in ok:
        per.setdefault(r["method"], {})[r["dataset"]] = _val(r)

    backbones = []
    for name, fold_vals in per.items():
        vals = [fold_vals[d] for d in sorted(fold_vals)]
        n = len(vals)
        backbones.append({
            "name": name,
            "n_folds": n,
            "mean": round(statistics.fmean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if n > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "missing_folds": [d for d in all_folds if d not in fold_vals],
            "n_failed": sum(1 for r in failed if r["method"] == name),
            "per_fold": {d: round(fold_vals[d], 4) for d in sorted(fold_vals)},
        })
    backbones.sort(key=lambda b: b["mean"], reverse=True)

    summary = {"metric": metric, "n_folds_total": len(all_folds), "backbones": backbones}
    if out_path is None:
        out_path = str(Path(csv_path).with_suffix("")) + "_summary.json"
    Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== CV leaderboard ({metric}, {len(all_folds)} folds) ===")
    print(f"  {'backbone':<24}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}{'folds':>7}")
    for b in backbones:
        incomplete = b["n_folds"] != len(all_folds) or b["n_failed"]
        flag = "  (incomplete)" if incomplete else ""
        print(f"  {b['name']:<24}{b['mean']:>8.4f}{b['std']:>8.4f}"
              f"{b['min']:>8.4f}{b['max']:>8.4f}{b['n_folds']:>7}{flag}")
    print(f"  -> {out_path}")
    return summary


# --------------------------------------------------------------------------- orchestrator
def run_cv_eval(backbones, fold_dir, output_csv: str = "runs/cv_results.csv", *,
                seed: int = DEFAULT_SEED, hp: dict | None = None, baselines=(),
                resume: bool = True, on_error: str = "continue",
                metric: str = "mAP50", runners=None) -> dict:
    """End-to-end: build matrix -> run (resumable) -> aggregate. Returns the summary."""
    config = build_cv_matrix(backbones, fold_dir, seed=seed, hp=hp, baselines=baselines)
    rm = RunMatrix(config=config, output_csv=output_csv, runners=runners)
    cells = rm.expand()
    print(f"[cv-eval] {len(config['methods'])} backbones x {len(config['datasets'])} "
          f"folds = {len(cells)} runs -> {output_csv}")
    rm.run(resume=resume, on_error=on_error)
    return aggregate_cv_results(output_csv, metric=metric)
