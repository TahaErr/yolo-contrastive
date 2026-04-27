"""Tests for LabelFractionSplitter."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pytest

from yolo_contrastive.data import LabelFractionSplitter
from yolo_contrastive.data.label_fraction import (
    class_distribution,
    verify_nested,
    _read_dominant_class,
    _label_path_for,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _make_yolo_dataset(
    n_images: int = 100,
    class_dist: dict = None,  # {class_id: count}; rest get class 0
    extra_unlabeled: int = 0,
) -> Tuple[str, List[str]]:
    """Create a tmp YOLO-style directory with empty image files + label .txts.

    Returns:
        (tmp_dir, image_paths)
    """
    if class_dist is None:
        class_dist = {0: n_images}

    # Verify counts
    total_labeled = sum(class_dist.values())
    assert total_labeled <= n_images, (
        f"class_dist sum {total_labeled} > n_images {n_images}"
    )

    tmp = tempfile.mkdtemp(prefix="ycl_lf_test_")
    img_dir = os.path.join(tmp, "images", "train")
    lbl_dir = os.path.join(tmp, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    image_paths: List[str] = []
    image_idx = 0

    for cls_id, count in class_dist.items():
        for _ in range(count):
            img_name = f"img_{image_idx:04d}.jpg"
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, f"img_{image_idx:04d}.txt")
            # Create empty image file (we don't actually read images)
            Path(img_path).touch()
            # Write label file with the dominant class
            with open(lbl_path, "w") as f:
                # Single bbox: class cx cy w h (random-ish coords)
                f.write(f"{cls_id} 0.5 0.5 0.2 0.2\n")
            image_paths.append(img_path)
            image_idx += 1

    # Unlabeled images (no .txt)
    for _ in range(extra_unlabeled):
        img_name = f"img_{image_idx:04d}.jpg"
        img_path = os.path.join(img_dir, img_name)
        Path(img_path).touch()
        image_paths.append(img_path)
        image_idx += 1

    return tmp, image_paths


# ── label path / dominant class helpers ──────────────────────────────────


class TestDominantClass:
    def test_single_class_label_file(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "x.txt")
            with open(p, "w") as f:
                f.write("3 0.5 0.5 0.1 0.1\n")
            assert _read_dominant_class(p) == 3
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_multi_bbox_picks_majority(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "x.txt")
            with open(p, "w") as f:
                f.write("0 0.1 0.1 0.1 0.1\n")
                f.write("2 0.2 0.2 0.1 0.1\n")
                f.write("2 0.3 0.3 0.1 0.1\n")  # class 2 appears twice
            assert _read_dominant_class(p) == 2
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_file_returns_minus_one(self):
        assert _read_dominant_class("/nonexistent/path/x.txt") == -1

    def test_empty_file_returns_minus_one(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "x.txt")
            Path(p).touch()
            assert _read_dominant_class(p) == -1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_label_path_swaps_images_to_labels(self):
        p = "/data/dataset/images/train/img_001.jpg"
        result = _label_path_for(p)
        assert result == "/data/dataset/labels/train/img_001.txt"

    def test_label_path_explicit_dir(self):
        p = "/data/dataset/images/train/img_001.jpg"
        result = _label_path_for(p, labels_dir="/somewhere/labels/")
        assert result.endswith("img_001.txt")
        assert "/somewhere/labels/" in result


# ── construction ─────────────────────────────────────────────────────────


class TestConstruction:
    def test_basic(self):
        s = LabelFractionSplitter([0.1, 0.5, 1.0])
        # sorted + dedup
        assert s.fractions == [0.1, 0.5, 1.0]

    def test_unsorted_input_sorted(self):
        s = LabelFractionSplitter([0.5, 0.1, 0.25])
        assert s.fractions == [0.1, 0.25, 0.5]

    def test_dedup(self):
        s = LabelFractionSplitter([0.1, 0.1, 0.5])
        assert s.fractions == [0.1, 0.5]

    def test_invalid_fractions(self):
        with pytest.raises(ValueError, match="empty"):
            LabelFractionSplitter([])
        with pytest.raises(ValueError, match="not in"):
            LabelFractionSplitter([0.0])
        with pytest.raises(ValueError, match="not in"):
            LabelFractionSplitter([1.5])
        with pytest.raises(ValueError, match="not in"):
            LabelFractionSplitter([-0.1])

    def test_invalid_stratify_mode(self):
        with pytest.raises(ValueError, match="stratify_mode"):
            LabelFractionSplitter([0.5], stratify_mode="bogus")

    def test_invalid_min_per_class(self):
        with pytest.raises(ValueError, match="min_per_class"):
            LabelFractionSplitter([0.5], min_per_class=0)

    def test_repr(self):
        s = LabelFractionSplitter([0.5], seed=7)
        r = repr(s)
        assert "LabelFractionSplitter" in r
        assert "seed=7" in r


# ── basic split sizes ────────────────────────────────────────────────────


class TestSplitSizes:
    def test_subset_sizes_match_fractions(self):
        tmp, paths = _make_yolo_dataset(n_images=100,
                                          class_dist={0: 50, 1: 50})
        try:
            s = LabelFractionSplitter([0.1, 0.5, 1.0])
            subsets = s.split(paths)
            assert len(subsets[0.1]) == 10
            assert len(subsets[0.5]) == 50
            assert len(subsets[1.0]) == 100
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sub_one_percent_rounds_to_at_least_1(self):
        tmp, paths = _make_yolo_dataset(n_images=10)
        try:
            # 1% of 10 = 0.1 → rounds to 0 → bumped to 1
            s = LabelFractionSplitter([0.01])
            subsets = s.split(paths)
            assert len(subsets[0.01]) >= 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_empty_paths_raises(self):
        s = LabelFractionSplitter([0.5])
        with pytest.raises(ValueError, match="empty"):
            s.split([])


# ── determinism ──────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_seed_same_subsets(self):
        tmp, paths = _make_yolo_dataset(n_images=50,
                                          class_dist={0: 20, 1: 20, 2: 10})
        try:
            s1 = LabelFractionSplitter([0.2, 0.5], seed=42)
            s2 = LabelFractionSplitter([0.2, 0.5], seed=42)
            sub1 = s1.split(paths)
            sub2 = s2.split(paths)
            for f in (0.2, 0.5):
                assert sub1[f] == sub2[f]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_different_seeds_differ(self):
        tmp, paths = _make_yolo_dataset(n_images=50,
                                          class_dist={0: 20, 1: 20, 2: 10})
        try:
            s1 = LabelFractionSplitter([0.5], seed=42)
            s2 = LabelFractionSplitter([0.5], seed=123)
            sub1 = s1.split(paths)
            sub2 = s2.split(paths)
            assert sub1[0.5] != sub2[0.5]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── nested property ──────────────────────────────────────────────────────


class TestNested:
    def test_subsets_are_nested(self):
        tmp, paths = _make_yolo_dataset(n_images=100,
                                          class_dist={0: 50, 1: 30, 2: 20})
        try:
            s = LabelFractionSplitter([0.1, 0.25, 0.5, 1.0])
            subsets = s.split(paths)
            assert verify_nested(subsets)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_nested_in_uniform_mode(self):
        tmp, paths = _make_yolo_dataset(n_images=80)
        try:
            s = LabelFractionSplitter([0.1, 0.5, 1.0],
                                        stratify_mode="none")
            subsets = s.split(paths)
            assert verify_nested(subsets)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── stratification balance ───────────────────────────────────────────────


class TestStratification:
    def test_subset_class_balance_close_to_global(self):
        """Stratified subset should preserve class proportions roughly."""
        tmp, paths = _make_yolo_dataset(
            n_images=200, class_dist={0: 100, 1: 60, 2: 40},
        )
        try:
            s = LabelFractionSplitter([0.1, 0.25, 0.5], seed=42)
            subsets = s.split(paths)

            # Global proportions: 50%, 30%, 20%
            for f in (0.25, 0.5):  # at 0.1 = 20 imgs balance is jittery
                imgs = subsets[f]
                # Recompute classes for this subset
                cls_counts = class_distribution(imgs)
                total = sum(cls_counts.values())
                p0 = cls_counts.get(0, 0) / total
                p1 = cls_counts.get(1, 0) / total
                p2 = cls_counts.get(2, 0) / total
                # Each within 8 percentage points of global
                assert abs(p0 - 0.5) < 0.08, f"p0={p0:.3f} at f={f}"
                assert abs(p1 - 0.3) < 0.08, f"p1={p1:.3f} at f={f}"
                assert abs(p2 - 0.2) < 0.08, f"p2={p2:.3f} at f={f}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unlabeled_images_get_minus_one_class(self):
        tmp, paths = _make_yolo_dataset(
            n_images=20, class_dist={0: 10}, extra_unlabeled=10,
        )
        try:
            counts = class_distribution(paths)
            assert counts[-1] == 10  # unlabeled images
            assert counts[0] == 10
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_stratified_with_unlabeled(self):
        """Splitter should handle a mix of labeled + unlabeled images."""
        tmp, paths = _make_yolo_dataset(
            n_images=20, class_dist={0: 10}, extra_unlabeled=10,
        )
        try:
            s = LabelFractionSplitter([0.5], seed=42)
            subsets = s.split(paths)
            assert len(subsets[0.5]) == 10
            # Should have ~5 of each (-1 and 0)
            cls_in_subset = class_distribution(subsets[0.5])
            assert cls_in_subset.get(-1, 0) + cls_in_subset.get(0, 0) == 10
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── tiny-class fallback ──────────────────────────────────────────────────


class TestTinyClassFallback:
    def test_classes_below_min_are_merged(self):
        """A class with 1 image (below min_per_class=2) should be merged
        into the fallback bucket and not crash."""
        tmp, paths = _make_yolo_dataset(
            n_images=21, class_dist={0: 10, 1: 10, 2: 1},
        )
        try:
            s = LabelFractionSplitter([0.5], seed=42, min_per_class=2)
            subsets = s.split(paths)
            # Should run without error and return 10-11 images
            assert 9 <= len(subsets[0.5]) <= 12
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── uniform mode (no stratification) ────────────────────────────────────


class TestUniformMode:
    def test_uniform_mode_runs(self):
        tmp, paths = _make_yolo_dataset(n_images=50,
                                          class_dist={0: 25, 1: 25})
        try:
            s = LabelFractionSplitter([0.5], seed=42, stratify_mode="none")
            subsets = s.split(paths)
            assert len(subsets[0.5]) == 25
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── output txt files ────────────────────────────────────────────────────


class TestOutputFiles:
    def test_writes_txt_files(self):
        tmp, paths = _make_yolo_dataset(n_images=100,
                                          class_dist={0: 50, 1: 50})
        out_dir = tempfile.mkdtemp(prefix="ycl_lf_out_")
        try:
            s = LabelFractionSplitter([0.1, 0.5, 1.0], seed=42)
            s.split(paths, output_dir=out_dir)
            for pct in (10, 50, 100):
                fname = f"train_pct{pct:03d}.txt"
                fpath = os.path.join(out_dir, fname)
                assert os.path.exists(fpath), f"missing {fname}"
                with open(fpath) as f:
                    lines = [l.strip() for l in f if l.strip()]
                expected = {10: 10, 50: 50, 100: 100}[pct]
                assert len(lines) == expected
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_txt_files_match_returned_dict(self):
        tmp, paths = _make_yolo_dataset(n_images=40, class_dist={0: 40})
        out_dir = tempfile.mkdtemp(prefix="ycl_lf_match_")
        try:
            s = LabelFractionSplitter([0.5], seed=42)
            subsets = s.split(paths, output_dir=out_dir)
            with open(os.path.join(out_dir, "train_pct050.txt")) as f:
                file_paths = [l.strip() for l in f if l.strip()]
            assert file_paths == subsets[0.5]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)


# ── public helpers ──────────────────────────────────────────────────────


class TestHelpers:
    def test_class_distribution(self):
        tmp, paths = _make_yolo_dataset(
            n_images=30, class_dist={0: 10, 1: 5, 2: 15},
        )
        try:
            counts = class_distribution(paths)
            assert counts == {0: 10, 1: 5, 2: 15}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_verify_nested_true(self):
        subsets = {
            0.1: ["a", "b"],
            0.5: ["a", "b", "c", "d"],
            1.0: ["a", "b", "c", "d", "e"],
        }
        assert verify_nested(subsets) is True

    def test_verify_nested_false(self):
        subsets = {
            0.5: ["a", "b"],
            1.0: ["c", "d", "e"],  # not a superset
        }
        assert verify_nested(subsets) is False
