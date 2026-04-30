"""Unified data loaders for SSL pretrain pool + YOLO eval datasets.

Faz 4.3 — Eval infrastructure (WORK_PLAN_v5 §5).

Three responsibilities, kept in one module to minimize coupling:

1. SSL pretrain pool — `build_ssl_manifest()`
   Reads a YAML listing of dataset roots + glob patterns. Walks each
   dataset, dedupes, writes a single manifest txt with one image path
   per line. UnlabeledImageDataset (Faz 1) consumes this file directly.

2. Eval dataset class — `MultiLabelImageDataset`
   Walks YOLO image/label pairs, derives image-level multi-hot labels
   from the SET of class ids present in each label file. Background
   images (no label or empty label) → all-zero vector.

3. Convenience builder — `loaders_from_yolo_data_yaml()`
   Reads a YOLO `data.yaml`, resolves train/val image lists, optionally
   restricts train to a label-fraction subset (txt manifest from
   LabelFractionSplitter), returns ready-to-use DataLoaders.

Design choices:
- Manifest file (txt of image paths) is the universal interchange format.
- Declarative YAML for the SSL pool (no per-dataset Python adapter classes).
- Multi-label by construction (works for nc=1 and nc=N with same code).
- Minimal preprocessing: resize + /255. No augmentation in this layer —
  Ultralytics handles training augmentation; eval is deterministic anyway.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "unified_loader requires pyyaml. Install with: pip install pyyaml"
    ) from e


# ─────────────────────────────────────────────────────────────────────────
# 1. SSL pretrain pool — manifest builder
# ─────────────────────────────────────────────────────────────────────────


_DEFAULT_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_ssl_manifest(
    config: Any,                            # path to YAML or pre-loaded dict
    output_path: str,
    verify_exists: bool = True,
    dedupe: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Build a unified manifest txt from per-dataset roots+globs.

    YAML schema:
        datasets:
          - name: bdd100k
            root: /data/bdd100k/images
            image_glob: "**/*.jpg"           # standard glob pattern
            recursive: true                  # default true
          - name: a2d2
            root: /data/a2d2
            image_glob: "**/cam_front_center/*.png"

    Args:
        config: path to YAML file OR a dict with the above schema.
        output_path: where to write the merged manifest txt.
        verify_exists: drop paths that don't exist on disk (warns about count).
        dedupe: remove duplicate paths (e.g. same file globbed by 2 datasets).
        verbose: print summary per dataset.

    Returns:
        dict with stats: {
            "total": int,
            "per_dataset": {name: count, ...},
            "dropped_missing": int,
            "dropped_dupes": int,
            "output_path": str,
        }
    """
    if isinstance(config, (str, os.PathLike)):
        with open(config) as f:
            cfg = yaml.safe_load(f)
    elif isinstance(config, dict):
        cfg = config
    else:
        raise TypeError(f"config must be path or dict, got {type(config).__name__}")

    if "datasets" not in cfg or not cfg["datasets"]:
        raise ValueError("config must have non-empty 'datasets' list")

    all_paths: List[str] = []
    per_dataset: Dict[str, int] = {}
    seen: Set[str] = set()
    n_dropped_missing = 0
    n_dropped_dupes = 0

    for ds in cfg["datasets"]:
        if "name" not in ds or "root" not in ds:
            raise ValueError(f"dataset entry missing name/root: {ds}")
        name = ds["name"]
        root = os.path.expanduser(ds["root"])
        pattern = ds.get("image_glob", "**/*.jpg")
        recursive = bool(ds.get("recursive", True))

        if not os.path.isdir(root):
            if verify_exists:
                if verbose:
                    _safe_print(f"[ssl-manifest] {name}: root not found: {root}")
                continue

        full_glob = os.path.join(root, pattern)
        matched = glob.glob(full_glob, recursive=recursive)
        # Filter to known image extensions to avoid pulling in non-image files
        matched = [
            p for p in matched
            if Path(p).suffix.lower() in _DEFAULT_IMG_EXTENSIONS
        ]

        kept = 0
        for p in matched:
            p_abs = os.path.abspath(p)
            if verify_exists and not os.path.isfile(p_abs):
                n_dropped_missing += 1
                continue
            if dedupe:
                if p_abs in seen:
                    n_dropped_dupes += 1
                    continue
                seen.add(p_abs)
            all_paths.append(p_abs)
            kept += 1

        per_dataset[name] = kept
        if verbose:
            _safe_print(
                f"[ssl-manifest] {name}: {kept} images "
                f"(matched {len(matched)} via {pattern})"
            )

    # Write manifest
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for p in all_paths:
            f.write(f"{p}\n")

    stats = {
        "total": len(all_paths),
        "per_dataset": per_dataset,
        "dropped_missing": n_dropped_missing,
        "dropped_dupes": n_dropped_dupes,
        "output_path": output_path,
    }
    if verbose:
        _safe_print(
            f"[ssl-manifest] DONE: {stats['total']} images → {output_path}"
            f" (missing={n_dropped_missing}, dupes={n_dropped_dupes})"
        )
    return stats


# ─────────────────────────────────────────────────────────────────────────
# 2. Multi-label image dataset (eval)
# ─────────────────────────────────────────────────────────────────────────


def _read_class_set(label_path: str) -> Set[int]:
    """Return set of class ids appearing in a YOLO label file. Empty if missing."""
    if not os.path.isfile(label_path):
        return set()
    classes: Set[int] = set()
    try:
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tok = line.split()
                if not tok:
                    continue
                try:
                    classes.add(int(float(tok[0])))
                except ValueError:
                    continue
    except OSError:
        return set()
    return classes


def _label_path_for_image(image_path: str, labels_dir: Optional[str] = None) -> str:
    """Standard YOLO convention: swap 'images' segment with 'labels'."""
    if labels_dir is not None:
        stem = Path(image_path).stem
        return os.path.join(labels_dir, f"{stem}.txt")
    p = Path(image_path)
    parts = list(p.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return str(Path(*parts).with_suffix(".txt"))
    return str(p.with_suffix(".txt"))


class MultiLabelImageDataset(Dataset):
    """YOLO-format dataset returning (image_tensor, multi_hot_label).

    Each sample:
        image:  [3, imgsz, imgsz] float in [0, 1]
        label:  [num_classes] float multi-hot

    Args:
        image_paths: list of absolute image file paths.
        labels_dir: optional override for label location. If None, infer
            via 'images' → 'labels' swap (standard YOLO convention).
        num_classes: total number of classes (multi-hot vector dim).
        imgsz: target image size (square resize). Matches Roboflow export.
        drop_classes: optional set of class ids to mask out (set their
            multi-hot bit to 0 always — useful when downgrading from
            multi-class to subset task without re-exporting the dataset).

    Loading:
        Uses torchvision.io.read_image when available, falls back to PIL.
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        num_classes: int,
        labels_dir: Optional[str] = None,
        imgsz: int = 640,
        drop_classes: Optional[Sequence[int]] = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if imgsz <= 0:
            raise ValueError(f"imgsz must be positive, got {imgsz}")
        if not image_paths:
            raise ValueError("image_paths is empty")

        self.image_paths = list(image_paths)
        self.num_classes = int(num_classes)
        self.labels_dir = labels_dir
        self.imgsz = int(imgsz)
        self.drop_classes: Set[int] = set(drop_classes or ())

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load image → [3, H, W] uint8 → resize → [3, imgsz, imgsz] float in [0,1]."""
        try:
            from torchvision.io import read_image, ImageReadMode
            from torchvision.transforms.functional import resize
            img = read_image(path, mode=ImageReadMode.RGB)
            # Stretch to imgsz×imgsz to match Roboflow export preprocessing
            img = resize(img, [self.imgsz, self.imgsz], antialias=True)
            return img.float() / 255.0
        except Exception:
            # PIL fallback
            from PIL import Image
            import numpy as np
            with Image.open(path) as pil:
                pil = pil.convert("RGB").resize(
                    (self.imgsz, self.imgsz), Image.BILINEAR,
                )
                arr = np.array(pil, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        img = self._load_image(img_path)

        # Derive multi-hot label
        lbl_path = _label_path_for_image(img_path, self.labels_dir)
        cls_set = _read_class_set(lbl_path)
        cls_set -= self.drop_classes  # filter dropped classes

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for c in cls_set:
            if 0 <= c < self.num_classes:
                target[c] = 1.0

        return img, target


# ─────────────────────────────────────────────────────────────────────────
# 3. Convenience: build train/val loaders from a YOLO data.yaml
# ─────────────────────────────────────────────────────────────────────────


def _resolve_path(p: str, base_dir: str) -> str:
    """Resolve a possibly-relative path against base_dir (data.yaml's dir)."""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def _list_images_in_dir(directory: str) -> List[str]:
    """Return all image files under directory, recursive."""
    if not os.path.isdir(directory):
        return []
    out = []
    for ext in _DEFAULT_IMG_EXTENSIONS:
        out.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
    return sorted(out)


def _read_image_list_txt(path: str) -> List[str]:
    """Read a manifest txt (one path per line)."""
    paths = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                paths.append(s)
    return paths


def _strip_leading_dotdot(spec: str) -> str:
    """Drop one or more leading '../' segments from a relative path.

    Roboflow exports often write `train: ../train/images` even when the
    dataset is one level below where the `..` would land. This helper
    yields a candidate stripped of leading `..` segments so we can retry
    resolution against base_dir.
    """
    parts = list(Path(spec).parts)
    while parts and parts[0] == "..":
        parts.pop(0)
    if not parts:
        return ""
    return str(Path(*parts))


def _resolve_split(spec: str, base_dir: str) -> List[str]:
    """A YOLO data.yaml split entry can be:
        - a directory of images
        - a txt file listing images
        - relative or absolute
    Returns absolute image paths.

    Resolution strategy:
        1. Try resolving `spec` directly against base_dir (standard YOLO).
        2. If that fails AND spec uses '../' (Roboflow quirk), retry
           after stripping leading '..' segments (treats path as
           dataset-root-relative). This handles the common pattern where
           Roboflow exports `train: ../train/images` but the dataset is
           a sibling of data.yaml.
    """
    # Attempt 1: standard resolution
    abs_spec = _resolve_path(spec, base_dir)
    if os.path.isdir(abs_spec):
        return _list_images_in_dir(abs_spec)
    if os.path.isfile(abs_spec) and abs_spec.lower().endswith(".txt"):
        return _read_image_list_txt(abs_spec)

    # Attempt 2: Roboflow `..` pattern fallback
    if not os.path.isabs(spec) and ".." in Path(spec).parts:
        stripped = _strip_leading_dotdot(spec)
        if stripped:
            alt_abs = _resolve_path(stripped, base_dir)
            if os.path.isdir(alt_abs):
                return _list_images_in_dir(alt_abs)
            if os.path.isfile(alt_abs) and alt_abs.lower().endswith(".txt"):
                return _read_image_list_txt(alt_abs)

    raise FileNotFoundError(
        f"data.yaml split entry {spec!r} resolved to {abs_spec!r}, "
        f"which is neither a directory nor a manifest txt"
    )


def loaders_from_yolo_data_yaml(
    data_yaml: str,
    train_subset: Optional[Sequence[str]] = None,
    batch_size: int = 16,
    imgsz: int = 640,
    num_workers: int = 0,
    drop_classes: Optional[Sequence[int]] = None,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Build (train_loader, val_loader, info) from a YOLO data.yaml.

    Args:
        data_yaml: path to a YOLO data.yaml (with `train`, `val`, `nc`,
            `names` keys; `train` and `val` may be directories or txt files).
        train_subset: optional explicit list of train image paths
            (e.g. from LabelFractionSplitter). Overrides the yaml's train.
        batch_size, imgsz, num_workers, pin_memory: passed to DataLoader.
        drop_classes: optional class ids to mask (passed to dataset).

    Returns:
        (train_loader, val_loader, info_dict)
        info_dict keys: nc, names, n_train, n_val, data_yaml.
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    for required in ("train", "val", "nc"):
        if required not in cfg:
            raise ValueError(f"data.yaml missing required key: {required!r}")

    base_dir = os.path.dirname(os.path.abspath(data_yaml))

    if train_subset is not None:
        train_paths = list(train_subset)
    else:
        train_paths = _resolve_split(cfg["train"], base_dir)

    val_paths = _resolve_split(cfg["val"], base_dir)
    num_classes = int(cfg["nc"])

    train_ds = MultiLabelImageDataset(
        image_paths=train_paths, num_classes=num_classes,
        imgsz=imgsz, drop_classes=drop_classes,
    )
    val_ds = MultiLabelImageDataset(
        image_paths=val_paths, num_classes=num_classes,
        imgsz=imgsz, drop_classes=drop_classes,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )

    info = {
        "nc": num_classes,
        "names": cfg.get("names", []),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "data_yaml": os.path.abspath(data_yaml),
        "drop_classes": list(drop_classes or ()),
    }
    return train_loader, val_loader, info


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _safe_print(msg: str) -> None:
    try:
        from ultralytics.utils import LOGGER
        LOGGER.info(msg)
    except Exception:
        print(msg)
