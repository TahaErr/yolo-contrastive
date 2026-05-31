"""Download, verify, consolidate, and balance Roboflow sources into a selection.

WORK_PLAN_v9 §13.4 — downstream "Pothole-5K" assembly. Steps 1-4 of the
pipeline (cross-validation fold assembly is a separate step that consumes the
selection manifest produced here).

  1. download     — fetch each source zip into its OWN folder under ``root``
  2. verify        — assert every source has the SAME single class (else raise)
  3. consolidate   — move valid/test image+label pairs into each source's train
  4. select        — water-fill to the target total, recording the kept files in
                     a JSON selection manifest. NON-DESTRUCTIVE: no image is
                     deleted; the manifest is the authoritative selection that
                     the cross-validation step reads.

Determinism: a single ``seed`` (default 42, the project constant) drives the
random downsampling. Per-source seeds are derived deterministically (sha256,
not PYTHONHASHSEED-dependent) so a run is reproducible regardless of source
processing order.

Network is required only for step 1; steps 2-4 are pure local file operations.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import yaml

from .allocate import resolve_target, water_fill_allocation
from .sources import load_sources

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SEED = 42


# --------------------------------------------------------------------------- helpers
def _count_images(d: Path | str) -> int:
    d = Path(d)
    if not d.is_dir():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def _nonclobber(path: Path) -> Path:
    """Return ``path``, or ``path`` with a ``_dup{n}`` suffix if it already exists."""
    if not path.exists():
        return path
    i = 1
    while (cand := path.with_name(f"{path.stem}_dup{i}{path.suffix}")).exists():
        i += 1
    return cand


def find_dataset_root(base: Path | str) -> Path:
    """Return the directory containing ``data.yaml`` (zips extract flat OR nested)."""
    base = Path(base)
    if (base / "data.yaml").is_file():
        return base
    hits = sorted(base.rglob("data.yaml"))
    if not hits:
        raise FileNotFoundError(f"data.yaml not found under: {base}")
    return hits[0].parent


def _derive_seed(seed: int, name: str) -> int:
    """Deterministic per-source seed, independent of PYTHONHASHSEED / process."""
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- 1) download
def download_source(name: str, url: str, root: Path | str, force: bool = False) -> Path:
    """Download + extract one Roboflow export into ``root/name``. Returns dataset root."""
    root = Path(root)
    dest = root / name
    if dest.exists() and not force:
        try:
            return find_dataset_root(dest)  # already downloaded
        except FileNotFoundError:
            pass
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "_roboflow.zip"
    print(f"[download] {name} <- {url.split('?')[0]}?key=...")
    subprocess.run(["curl", "-sL", url, "-o", str(zip_path)], check=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile as e:
        raise RuntimeError(
            f"{name}: could not open zip — the link may have expired or the key is invalid."
        ) from e
    finally:
        zip_path.unlink(missing_ok=True)
    return find_dataset_root(dest)


# --------------------------------------------------------------------------- 2) verify
def _read_names(data_yaml: Path) -> list[str]:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n).strip() for n in names]


def verify_single_label(roots: dict[str, Path]) -> list[str]:
    """Assert every source has the SAME single class. Raise on any divergence."""
    print("\n[verify] data.yaml class check")
    per = {name: _read_names(root / "data.yaml") for name, root in roots.items()}
    for name, names in per.items():
        print(f"  {name}: nc={len(names)} names={names}")

    multi = {n: v for n, v in per.items() if len(v) != 1}
    if multi:
        raise ValueError(f"Expected a single label but these sources have nc != 1: {multi}")

    uniques = {tuple(v) for v in per.values()}
    if len(uniques) != 1:
        raise ValueError(f"Sources do not share the same label: {per}")

    label = list(next(iter(uniques)))
    print(f"  OK — all {len(roots)} sources share one label: {label}")
    return label


# --------------------------------------------------------------------------- 3) consolidate
def consolidate_to_train(root: Path) -> int:
    """Move valid/val/test image+label pairs into ``train``. Return train image count."""
    train_img = root / "train" / "images"
    train_lbl = root / "train" / "labels"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)

    for split in ("valid", "val", "test"):
        s_img = root / split / "images"
        if not s_img.is_dir():
            continue
        s_lbl = root / split / "labels"
        for img in list(s_img.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            dst_img = _nonclobber(train_img / img.name)
            shutil.move(str(img), str(dst_img))
            lbl = s_lbl / (img.stem + ".txt")
            if lbl.is_file():  # background images may have no label — fine
                shutil.move(str(lbl), str(train_lbl / (dst_img.stem + ".txt")))
        shutil.rmtree(root / split, ignore_errors=True)  # drop the emptied split dir

    return _count_images(train_img)


# --------------------------------------------------------------------------- 4) select
def select_kept_files(root: Path, keep: int, seed: int = DEFAULT_SEED,
                      source_name: str = "") -> list[str]:
    """Deterministically pick ``keep`` image filenames from ``root/train/images``."""
    img_dir = root / "train" / "images"
    imgs = sorted(f.name for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if keep >= len(imgs):
        return imgs
    rng = random.Random(_derive_seed(seed, source_name))
    return sorted(rng.sample(imgs, keep))


def build_selection_manifest(roots: dict[str, Path], *, total: int | None = None,
                             per_dataset: int | None = None, seed: int = DEFAULT_SEED,
                             out_path: str = "selection_manifest.json") -> dict:
    """Water-fill to the resolved target and write a non-destructive selection manifest."""
    target = resolve_target(len(roots), total, per_dataset)
    counts = {name: _count_images(root / "train" / "images") for name, root in roots.items()}
    alloc = water_fill_allocation(counts, target)

    print(f"\n[select] water-filling (target = {target}, seed = {seed})")
    print(f"  {'source':<10}{'available':>11}{'keep':>7}{'drop':>7}")

    manifest = {"target": target, "seed": seed, "n_sources": len(roots),
                "total_selected": 0, "sources": {}}
    total_sel = 0
    for name, root in roots.items():
        kept = select_kept_files(root, alloc[name], seed, name)
        keep = len(kept)
        total_sel += keep
        manifest["sources"][name] = {
            "available": counts[name],
            "keep": keep,
            "drop": counts[name] - keep,
            "images_dir": str(root / "train" / "images"),
            "labels_dir": str(root / "train" / "labels"),
            "selected_images": kept,
        }
        print(f"  {name:<10}{counts[name]:>11}{keep:>7}{counts[name] - keep:>7}")

    manifest["total_selected"] = total_sel
    print(f"  {'TOTAL':<10}{sum(counts.values()):>11}{total_sel:>7}")
    if total_sel < target:
        print(f"  WARNING: supply {total_sel} < target {target} (all sources taken whole)")

    Path(out_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  -> manifest written: {out_path}")
    return manifest


def read_selection_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- orchestrator
def prepare_downstream(sources: dict | list | tuple | str | Path, root: str = "datasets", *,
                       total: int | None = None, per_dataset: int | None = None,
                       seed: int = DEFAULT_SEED, force_download: bool = False,
                       manifest_path: str | None = None) -> dict:
    """End-to-end steps 1-4: download -> verify -> consolidate -> select.

    Pass exactly one of ``total`` (fixed budget) or ``per_dataset`` (total scales
    with the number of sources). Returns the selection manifest dict.
    """
    src = load_sources(sources)
    roots = {name: download_source(name, url, root, force_download)
             for name, url in src.items()}
    verify_single_label(roots)
    print("\n[consolidate] valid/test -> train")
    for name, r in roots.items():
        print(f"  {name}: train -> {consolidate_to_train(r)} images")
    if manifest_path is None:
        manifest_path = str(Path(root) / "selection_manifest.json")
    return build_selection_manifest(roots, total=total, per_dataset=per_dataset,
                                    seed=seed, out_path=manifest_path)
