"""Source-disjoint holdout and cross-validation splits from a selection manifest.

WORK_PLAN_v9 §13.4 — downstream eval. Consumes the ``selection_manifest.json``
produced by ``prepare``.

WHY SOURCE-DISJOINT: the sources are road-scene datasets, each shot with one
camera, so consecutive frames within a source are near-duplicates. If a source
were split across train and val (image-level), those near-duplicate frames leak
across the boundary and inflate validation scores. So EVERY regime here keeps a
whole source on one side only — train and val never share a source. This is the
grouped-data setting scikit-learn warns about (the i.i.d. assumption breaks),
for which GroupKFold / LeaveOneGroupOut are the correct tools.

Regimes:

Holdout (no CV) — train/val/test:
    Whole sources are assigned to train OR val OR test (never split). The ratios
    are TARGET image proportions, approximated by a largest-deficit greedy over
    whole sources. With few sources the achieved ratios quantize to source
    boundaries (e.g. ~10 equal sources at 70/15/15 -> ~7 / 1-2 / 1-2 sources),
    so the realised split is reported. Default 70/15/15.

Cross-validation — (train, val) folds, no separate test:
    - "group_kfold" (default, k=5): the sources are partitioned into k
      size-balanced GROUPS; fold i's val = group i (whole sources), train = the
      rest. With 10 sources and k=5 each fold validates on 2 held-out sources.
    - "logo" (leave-one-source-out): fold i holds out one source as val, the
      rest as train (k = number of sources). One training run per source.

    NOTE: image-level k-fold is intentionally rejected here (it leaks, see above).

Output is YOLO-native and copy-free, mirroring ``LabelFractionSplitter``: each
split/fold is a ``.txt`` of absolute image paths plus a ``data.yaml`` pointing
train/val[/test] at those txts. Labels are found via the ``images`` -> ``labels``
convention, so the original per-source folders are referenced in place.

Determinism: a single ``seed`` (default 42) drives a deterministic shuffle
before the greedy assignment, so runs reproduce while differing seeds give
different (still source-disjoint) partitions.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import yaml

HOLDOUT_DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_K = 5
DEFAULT_SEED = 42


# --------------------------------------------------------------------------- helpers
def _derive_seed(seed: int, key: str) -> int:
    """Deterministic per-key seed, independent of PYTHONHASHSEED / process."""
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16)


def _as_manifest(manifest: dict | str | Path) -> dict:
    return manifest if isinstance(manifest, dict) else json.loads(Path(manifest).read_text())


def _source_images(manifest: dict) -> dict[str, list[Path]]:
    """source name -> absolute paths of its selected images (from the manifest)."""
    out: dict[str, list[Path]] = {}
    for name, info in manifest["sources"].items():
        idir = Path(info["images_dir"])
        out[name] = [idir / fn for fn in info["selected_images"]]
    return out


def _infer_names(manifest: dict) -> list[str]:
    """Read class names from a source's data.yaml (images_dir -> dataset root)."""
    first = next(iter(manifest["sources"].values()))
    root = Path(first["images_dir"]).parent.parent  # ds_XX/train/images -> ds_XX
    dy = root / "data.yaml"
    if dy.is_file():
        cfg = yaml.safe_load(dy.read_text()) or {}
        names = cfg.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names:
            return [str(n).strip() for n in names]
    raise ValueError("Could not infer class names; pass names=[...] explicitly.")


def _shuffled_keys(sizes: dict[str, int], seed: int) -> list[str]:
    """Source names shuffled deterministically, then sorted by size (desc, stable)."""
    keys = list(sizes)
    random.Random(_derive_seed(seed, "partition")).shuffle(keys)
    keys.sort(key=lambda s: sizes[s], reverse=True)
    return keys


def _greedy_partition_to_targets(sizes: dict[str, int], targets: dict[str, float],
                                 seed: int) -> dict[str, list[str]]:
    """Assign whole sources to labelled bins, filling the largest deficit first."""
    assign: dict[str, list[str]] = {label: [] for label in targets}
    current: dict[str, float] = {label: 0.0 for label in targets}
    for src in _shuffled_keys(sizes, seed):
        label = max(targets, key=lambda lb: targets[lb] - current[lb])
        assign[label].append(src)
        current[label] += sizes[src]
    return assign


def _balanced_groups(sizes: dict[str, int], k: int, seed: int) -> list[list[str]]:
    """Partition sources into k groups with near-equal total image counts."""
    groups: list[list[str]] = [[] for _ in range(k)]
    totals = [0] * k
    for src in _shuffled_keys(sizes, seed):
        j = min(range(k), key=lambda i: totals[i])
        groups[j].append(src)
        totals[j] += sizes[src]
    return groups


def _validate_ratios(ratios) -> None:
    if len(ratios) != 3:
        raise ValueError("ratios must be a (train, val, test) triple")
    if any(r < 0 for r in ratios):
        raise ValueError("ratios must be non-negative")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")


def _write_split_dir(out_dir: Path, splits: dict[str, list[Path]], names: list[str]) -> Path:
    """Write one txt per split + a data.yaml pointing at them. Returns the data.yaml path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, str] = {}
    for split, paths in splits.items():
        txt = out_dir / f"{split}.txt"
        txt.write_text("".join(f"{p}\n" for p in paths))
        entries[split] = str(txt)
    data: dict = {"train": entries["train"], "val": entries["val"]}
    if "test" in entries:
        data["test"] = entries["test"]
    data["nc"] = len(names)
    data["names"] = list(names)
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return out_dir / "data.yaml"


# --------------------------------------------------------------------------- holdout
def build_holdout_split(manifest: dict | str | Path, out_dir: str | Path,
                        ratios: tuple[float, float, float] = HOLDOUT_DEFAULT_RATIOS,
                        seed: int = DEFAULT_SEED, names: list[str] | None = None) -> dict:
    """Source-disjoint train/val/test holdout (whole sources per split)."""
    man = _as_manifest(manifest)
    names = names or _infer_names(man)
    _validate_ratios(ratios)

    per_source = _source_images(man)
    sizes = {s: len(imgs) for s, imgs in per_source.items()}
    total = sum(sizes.values()) or 1

    labels = ["train", "val"] + (["test"] if ratios[2] > 0 else [])
    if len(sizes) < len(labels):
        raise ValueError(
            f"source-disjoint split needs >= {len(labels)} sources, got {len(sizes)}")

    targets = {"train": ratios[0] * total, "val": ratios[1] * total, "test": ratios[2] * total}
    targets = {lb: targets[lb] for lb in labels}
    assign = _greedy_partition_to_targets(sizes, targets, seed)

    empty = [lb for lb in labels if not assign[lb]]
    if empty:
        raise ValueError(
            f"split(s) {empty} received no source; adjust ratios or add sources")

    splits = {lb: [p for s in assign[lb] for p in per_source[s]] for lb in labels}
    out = Path(out_dir) / "holdout"
    data_yaml = _write_split_dir(out, splits, names)

    counts = {lb: len(splits[lb]) for lb in labels}
    summary = {
        "regime": "holdout", "mode": "source_disjoint", "ratios": list(ratios),
        "seed": seed, "names": names, "assignment": {lb: assign[lb] for lb in labels},
        "counts": counts, "data_yaml": str(data_yaml),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[holdout] source-disjoint (target ratios %s):" % (tuple(ratios),))
    for lb in labels:
        print(f"  {lb:<5}: {len(assign[lb])} sources, {counts[lb]} imgs "
              f"({100 * counts[lb] / total:.1f}%)  {assign[lb]}")
    print(f"  -> {data_yaml}")
    return summary


# --------------------------------------------------------------------------- cross-validation
def build_cv_splits(manifest: dict | str | Path, out_dir: str | Path, *,
                    scheme: str = "group_kfold", k: int = DEFAULT_K, seed: int = DEFAULT_SEED,
                    names: list[str] | None = None) -> dict:
    """Build source-disjoint CV folds. scheme: 'group_kfold' (default) or 'logo'."""
    man = _as_manifest(manifest)
    names = names or _infer_names(man)
    per_source = _source_images(man)
    sizes = {s: len(imgs) for s, imgs in per_source.items()}
    sources = list(per_source)

    if scheme == "kfold":
        raise ValueError(
            "Image-level 'kfold' splits a source across train/val, leaking "
            "frame-adjacent near-duplicates for same-camera data. Use "
            "scheme='group_kfold' (or 'logo') for source-disjoint folds.")
    elif scheme == "logo":
        if len(sources) < 2:
            raise ValueError("logo needs at least 2 sources")
        val_groups = [[s] for s in sources]
    elif scheme == "group_kfold":
        if k < 2:
            raise ValueError("k must be >= 2")
        if k > len(sources):
            raise ValueError(
                f"group_kfold k={k} exceeds the number of sources ({len(sources)})")
        val_groups = _balanced_groups(sizes, k, seed)
    else:
        raise ValueError(f"unknown scheme: {scheme!r} (use 'group_kfold' or 'logo')")

    base = Path(out_dir) / "cv" / scheme
    fold_records = []
    print(f"[cv:{scheme}] {len(val_groups)} source-disjoint folds (seed={seed})")
    for i, vg in enumerate(val_groups):
        vg_set = set(vg)
        val = [p for s in vg for p in per_source[s]]
        train = [p for s in sources if s not in vg_set for p in per_source[s]]
        data_yaml = _write_split_dir(base / f"fold_{i}", {"train": train, "val": val}, names)
        fold_records.append({"fold": i, "val_sources": list(vg),
                             "train": len(train), "val": len(val), "data_yaml": str(data_yaml)})
        print(f"  fold_{i}: val_sources={vg}  train={len(train)}  val={len(val)}")

    summary = {"regime": "cv", "scheme": scheme, "mode": "source_disjoint",
               "n_folds": len(val_groups), "seed": seed, "names": names, "folds": fold_records}
    base.mkdir(parents=True, exist_ok=True)
    (base / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  -> {base}/summary.json")
    return summary
