"""Deterministic, nested, stratified label-fraction subset generator.

Faz 4.4 — Eval infrastructure (WORK_PLAN_v5 §5).

Why this exists:
    Faz 5 eval matrix runs each pretraining method against a fixed set of
    label fractions (1%, 5%, 10%, ...). Every method MUST see the same
    subset images at each fraction, otherwise comparisons are confounded
    by random sampling noise. This module produces those subsets once,
    deterministically, and writes them as YOLO-compatible image-list txt
    files.

Design choices:

1. Stratification: dominant class.
   Each YOLO label file lists 0-N bboxes with class ids. We pick the
   class id that appears MOST OFTEN in the file as the image's "dominant
   class" and stratify on that with sklearn's train_test_split(stratify=).
   For images with no labels (background / negatives), we use a sentinel
   class -1. Falls back to non-stratified random shuffle if any class
   has too few samples for the requested fraction.

2. Nested fractions.
   subset(p1) ⊂ subset(p2) for p1 < p2. Means as label fraction grows,
   we ADD images, never replace. Cleaner for label-efficiency curves
   ("at 10% the model saw exactly the 1% subset PLUS 9% extra").
   Achieved by sorting all images deterministically once, then taking
   prefixes.

3. Output: image-list txt files.
   YOLO data.yaml supports `train: path/to/train_pct10.txt`. No image
   copying — disk efficient. We write one txt per fraction.

4. Determinism: single seed controls everything (np.random + sklearn).
"""

from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _label_path_for(image_path: str, labels_dir: Optional[str] = None) -> str:
    """Convert a YOLO image path to its expected label file path.

    Default convention: replace 'images/' segment with 'labels/' and
    extension with '.txt'. If labels_dir is given, place the .txt there
    using the image stem.
    """
    if labels_dir is not None:
        stem = Path(image_path).stem
        return os.path.join(labels_dir, f"{stem}.txt")
    p = Path(image_path)
    # Walk up looking for 'images' segment to swap with 'labels'
    parts = list(p.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return str(Path(*parts).with_suffix(".txt"))
    # Fallback: sibling 'labels' folder next to image
    return str(p.with_suffix(".txt"))


def _read_dominant_class(label_path: str) -> int:
    """Return dominant class id from a YOLO label file. -1 if empty/missing."""
    if not os.path.isfile(label_path):
        return -1
    classes: List[int] = []
    try:
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # YOLO format: "class cx cy w h ..."
                tok = line.split()
                if not tok:
                    continue
                try:
                    classes.append(int(float(tok[0])))
                except ValueError:
                    continue
    except OSError:
        return -1
    if not classes:
        return -1
    return Counter(classes).most_common(1)[0][0]


class LabelFractionSplitter:
    """Generate deterministic, nested, stratified label fractions.

    Args:
        fractions: list of fractions in (0, 1], e.g. [0.01, 0.05, 0.1, 0.25, 0.5, 1.0].
        seed: integer RNG seed. Same seed → same subsets across runs.
        stratify_mode: "dominant" (use dominant class per image) or
            "none" (uniform random, no class balancing).
        min_per_class: classes with fewer images than this are folded into
            a single bucket for stratification fallback. Default 2.

    Usage:
        splitter = LabelFractionSplitter([0.1, 0.25, 0.5, 1.0], seed=42)
        subsets = splitter.split(
            image_paths=image_paths,
            labels_dir="dataset/labels/train",
            output_dir="dataset/splits/",
        )
        # subsets[0.1] is a list of image paths (10% of input)
        # output_dir contains train_pct10.txt, train_pct25.txt, etc.
    """

    def __init__(
        self,
        fractions: Sequence[float],
        seed: int = 42,
        stratify_mode: str = "dominant",
        min_per_class: int = 2,
    ) -> None:
        if not fractions:
            raise ValueError("fractions list is empty")
        for f in fractions:
            if not 0.0 < f <= 1.0:
                raise ValueError(f"fraction {f} not in (0, 1]")
        if stratify_mode not in ("dominant", "none"):
            raise ValueError(
                f"stratify_mode must be 'dominant' or 'none', got {stratify_mode!r}"
            )
        if min_per_class < 1:
            raise ValueError(f"min_per_class must be >= 1, got {min_per_class}")

        # Sort + dedupe fractions for nested ordering
        self.fractions: List[float] = sorted(set(float(f) for f in fractions))
        self.seed = int(seed)
        self.stratify_mode = stratify_mode
        self.min_per_class = int(min_per_class)

    # ── public API ────────────────────────────────────────────────────────

    def split(
        self,
        image_paths: Sequence[str],
        labels_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[float, List[str]]:
        """Compute nested fraction subsets.

        Args:
            image_paths: full list of image paths (from train set).
            labels_dir: if given, look for label .txt files here using image
                        stems. Else infer from path (swap images→labels).
            output_dir: if given, write one txt per fraction here.

        Returns:
            {fraction: [image_paths]} — sorted by fraction ascending. Each
            list is a prefix of the larger ones (nested).
        """
        if not image_paths:
            raise ValueError("image_paths is empty")

        image_list = list(image_paths)

        # 1) Determine class label per image (or -1)
        if self.stratify_mode == "dominant":
            classes = [
                _read_dominant_class(_label_path_for(p, labels_dir))
                for p in image_list
            ]
        else:
            classes = [-1] * len(image_list)

        # 2) Compute deterministic ordering of images
        # Stratified mode: sort within each class, then interleave so the
        # final order respects class proportions in any prefix.
        # 'none' mode: just deterministic shuffle.
        if self.stratify_mode == "dominant":
            ordering = self._stratified_ordering(image_list, classes)
        else:
            ordering = self._uniform_ordering(image_list)

        # 3) Take prefixes for each fraction
        N = len(ordering)
        subsets: Dict[float, List[str]] = {}
        for f in self.fractions:
            k = max(1, int(round(f * N)))
            subsets[f] = ordering[:k]

        # 4) Write txt files if requested
        if output_dir is not None:
            self._write_txt_files(subsets, output_dir)

        return subsets

    # ── internals ────────────────────────────────────────────────────────

    def _uniform_ordering(self, image_list: List[str]) -> List[str]:
        """Deterministic shuffle, no class balancing."""
        rng = random.Random(self.seed)
        idx = list(range(len(image_list)))
        rng.shuffle(idx)
        return [image_list[i] for i in idx]

    def _stratified_ordering(
        self, image_list: List[str], classes: List[int]
    ) -> List[str]:
        """Build a globally-ordered list where any prefix is class-balanced.

        Strategy: round-robin across classes, in proportion to class frequency.
        Within each class, deterministic shuffle.
        Classes with fewer than min_per_class samples get merged into a
        single 'tiny' bucket.
        """
        # Group images by class
        by_class: Dict[int, List[str]] = {}
        for img, cls in zip(image_list, classes):
            by_class.setdefault(cls, []).append(img)

        # Merge tiny classes into a fallback bucket (key = -999) so they
        # don't break the stratified prefix property
        merged: Dict[int, List[str]] = {}
        tiny_bucket: List[str] = []
        for cls, imgs in by_class.items():
            if len(imgs) < self.min_per_class:
                tiny_bucket.extend(imgs)
            else:
                merged[cls] = imgs
        if tiny_bucket:
            merged[-999] = tiny_bucket

        # Deterministic shuffle within each class
        rng = random.Random(self.seed)
        for cls in merged:
            cls_rng = random.Random(self.seed * 1000 + cls + 999)
            cls_rng.shuffle(merged[cls])

        # Interleave: at each step, pop one item from the class with the
        # highest "remaining proportion" deficit. This makes any prefix
        # respect global class proportions.
        total = sum(len(v) for v in merged.values())
        target_props = {cls: len(v) / total for cls, v in merged.items()}

        # Track how many we've taken from each class
        taken = {cls: 0 for cls in merged}
        order: List[str] = []
        # Use a cursor per class to pop from the front of the (already shuffled) list
        cursors = {cls: 0 for cls in merged}

        for step in range(total):
            # Pick class with largest deficit relative to its target proportion
            # deficit = target * step_total - taken
            best_cls = None
            best_def = -float("inf")
            for cls, prop in target_props.items():
                if cursors[cls] >= len(merged[cls]):
                    continue
                # Deficit at next step's total
                deficit = prop * (step + 1) - taken[cls]
                # Deterministic tie-break: smaller class id wins
                if deficit > best_def or (deficit == best_def and (best_cls is None or cls < best_cls)):
                    best_def = deficit
                    best_cls = cls
            if best_cls is None:
                break  # shouldn't happen
            order.append(merged[best_cls][cursors[best_cls]])
            cursors[best_cls] += 1
            taken[best_cls] += 1

        # Sanity: we used every image
        assert len(order) == total, f"ordering size {len(order)} != total {total}"
        return order

    def _write_txt_files(
        self, subsets: Dict[float, List[str]], output_dir: str,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for f, imgs in subsets.items():
            pct = int(round(f * 100))
            # Use 3-digit pct for sub-1% fractions; clean names otherwise
            if f < 0.01:
                fname = f"train_pct{f:.4f}.txt".replace("0.", "")
            else:
                fname = f"train_pct{pct:03d}.txt"
            path = os.path.join(output_dir, fname)
            with open(path, "w") as fh:
                for img in imgs:
                    fh.write(f"{img}\n")

    # ── repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"LabelFractionSplitter(fractions={self.fractions}, "
            f"seed={self.seed}, stratify={self.stratify_mode!r})"
        )


# ─────────────────────────────────────────────────────────────────────────
# Helpers (public)
# ─────────────────────────────────────────────────────────────────────────


def class_distribution(
    image_paths: Sequence[str],
    labels_dir: Optional[str] = None,
) -> Dict[int, int]:
    """Return {class_id: count} from a list of YOLO images. -1 = no labels.

    Diagnostic helper to inspect dataset balance before splitting.
    """
    counts: Dict[int, int] = {}
    for p in image_paths:
        cls = _read_dominant_class(_label_path_for(p, labels_dir))
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def verify_nested(
    subsets: Dict[float, List[str]],
) -> bool:
    """Return True iff smaller subsets are prefixes of larger ones."""
    fracs = sorted(subsets.keys())
    for i in range(len(fracs) - 1):
        small = subsets[fracs[i]]
        large = subsets[fracs[i + 1]]
        if len(small) > len(large):
            return False
        if small != large[: len(small)]:
            return False
    return True
