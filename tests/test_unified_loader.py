"""Tests for unified_loader (Faz 4.3)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
import yaml
from PIL import Image

from yolo_contrastive.data import (
    build_ssl_manifest,
    MultiLabelImageDataset,
    loaders_from_yolo_data_yaml,
)
from yolo_contrastive.data.unified_loader import (
    _read_class_set,
    _label_path_for_image,
    _resolve_split,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _make_image_file(path: str, size: int = 32, color=(128, 64, 32)) -> None:
    """Create a small RGB JPEG/PNG at the given path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (size, size), color=color).save(path)


def _make_label_file(path: str, classes: List[int]) -> None:
    """Create a YOLO label .txt with given class ids (cx=cy=0.5, w=h=0.1)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for c in classes:
            f.write(f"{c} 0.5 0.5 0.1 0.1\n")


def _make_yolo_split(
    base_dir: str, split: str, n: int, num_classes: int = 2,
    extra_unlabeled: int = 0,
) -> List[str]:
    """Create images/{split} + labels/{split} dirs with n labeled images
    cycling through classes 0..num_classes-1, plus optional unlabeled.
    Returns list of image absolute paths."""
    img_dir = os.path.join(base_dir, "images", split)
    lbl_dir = os.path.join(base_dir, "labels", split)
    paths = []
    for i in range(n):
        img_p = os.path.join(img_dir, f"img_{i:04d}.jpg")
        lbl_p = os.path.join(lbl_dir, f"img_{i:04d}.txt")
        _make_image_file(img_p)
        _make_label_file(lbl_p, [i % num_classes])
        paths.append(img_p)
    for i in range(n, n + extra_unlabeled):
        img_p = os.path.join(img_dir, f"img_{i:04d}.jpg")
        _make_image_file(img_p)  # no label
        paths.append(img_p)
    return paths


# ═════════════════════════════════════════════════════════════════════════
# A. SSL manifest builder
# ═════════════════════════════════════════════════════════════════════════


class TestBuildSSLManifest:
    def test_single_dataset(self):
        tmp = tempfile.mkdtemp(prefix="ycl_ssl_")
        out_dir = tempfile.mkdtemp(prefix="ycl_out_")
        try:
            # Create 5 jpgs in a dataset root
            ds_root = os.path.join(tmp, "ds1", "images")
            for i in range(5):
                _make_image_file(os.path.join(ds_root, f"img_{i}.jpg"))

            cfg = {
                "datasets": [
                    {"name": "ds1", "root": ds_root, "image_glob": "*.jpg"},
                ]
            }
            out_path = os.path.join(out_dir, "manifest.txt")
            stats = build_ssl_manifest(cfg, out_path, verbose=False)

            assert stats["total"] == 5
            assert stats["per_dataset"] == {"ds1": 5}
            assert os.path.exists(out_path)
            with open(out_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) == 5
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_multiple_datasets_merged(self):
        tmp = tempfile.mkdtemp()
        out_dir = tempfile.mkdtemp()
        try:
            ds1 = os.path.join(tmp, "ds1")
            ds2 = os.path.join(tmp, "ds2")
            for i in range(3):
                _make_image_file(os.path.join(ds1, f"a_{i}.jpg"))
            for i in range(7):
                _make_image_file(os.path.join(ds2, f"b_{i}.png"))

            cfg = {
                "datasets": [
                    {"name": "ds1", "root": ds1, "image_glob": "*.jpg"},
                    {"name": "ds2", "root": ds2, "image_glob": "*.png"},
                ]
            }
            out_path = os.path.join(out_dir, "manifest.txt")
            stats = build_ssl_manifest(cfg, out_path, verbose=False)
            assert stats["total"] == 10
            assert stats["per_dataset"] == {"ds1": 3, "ds2": 7}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_recursive_glob(self):
        tmp = tempfile.mkdtemp()
        out_dir = tempfile.mkdtemp()
        try:
            root = os.path.join(tmp, "ds")
            # Nested directories
            _make_image_file(os.path.join(root, "a/b/c/img1.jpg"))
            _make_image_file(os.path.join(root, "x/y/img2.jpg"))
            _make_image_file(os.path.join(root, "img3.jpg"))

            cfg = {
                "datasets": [
                    {"name": "ds", "root": root, "image_glob": "**/*.jpg"},
                ]
            }
            out_path = os.path.join(out_dir, "m.txt")
            stats = build_ssl_manifest(cfg, out_path, verbose=False)
            assert stats["total"] == 3
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_dedupe_across_datasets(self):
        """If two datasets glob the same file, dedupe should count once."""
        tmp = tempfile.mkdtemp()
        out_dir = tempfile.mkdtemp()
        try:
            shared = os.path.join(tmp, "shared", "x.jpg")
            _make_image_file(shared)

            cfg = {
                "datasets": [
                    {"name": "ds1", "root": os.path.join(tmp, "shared"),
                     "image_glob": "*.jpg"},
                    {"name": "ds2", "root": os.path.join(tmp, "shared"),
                     "image_glob": "*.jpg"},
                ]
            }
            out_path = os.path.join(out_dir, "m.txt")
            stats = build_ssl_manifest(cfg, out_path, verbose=False, dedupe=True)
            assert stats["total"] == 1
            assert stats["dropped_dupes"] == 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_yaml_file_input(self):
        tmp = tempfile.mkdtemp()
        out_dir = tempfile.mkdtemp()
        try:
            ds = os.path.join(tmp, "ds")
            for i in range(3):
                _make_image_file(os.path.join(ds, f"a_{i}.jpg"))

            yaml_cfg = {
                "datasets": [
                    {"name": "ds", "root": ds, "image_glob": "*.jpg"}
                ]
            }
            yaml_path = os.path.join(tmp, "config.yaml")
            with open(yaml_path, "w") as f:
                yaml.safe_dump(yaml_cfg, f)
            out_path = os.path.join(out_dir, "m.txt")
            stats = build_ssl_manifest(yaml_path, out_path, verbose=False)
            assert stats["total"] == 3
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_missing_root_skipped_when_verify_exists(self):
        tmp = tempfile.mkdtemp()
        out_dir = tempfile.mkdtemp()
        try:
            ds_real = os.path.join(tmp, "real")
            for i in range(3):
                _make_image_file(os.path.join(ds_real, f"a_{i}.jpg"))

            cfg = {
                "datasets": [
                    {"name": "real", "root": ds_real, "image_glob": "*.jpg"},
                    {"name": "missing", "root": "/nonexistent/path",
                     "image_glob": "*.jpg"},
                ]
            }
            out_path = os.path.join(out_dir, "m.txt")
            stats = build_ssl_manifest(cfg, out_path, verbose=False)
            assert stats["total"] == 3
            assert "missing" not in stats["per_dataset"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_invalid_config_raises(self):
        out_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError, match="datasets"):
                build_ssl_manifest({}, os.path.join(out_dir, "m.txt"),
                                     verbose=False)
            with pytest.raises(ValueError, match="datasets"):
                build_ssl_manifest({"datasets": []},
                                     os.path.join(out_dir, "m.txt"),
                                     verbose=False)
            with pytest.raises(ValueError, match="name"):
                build_ssl_manifest(
                    {"datasets": [{"root": "/x", "image_glob": "*.jpg"}]},
                    os.path.join(out_dir, "m.txt"), verbose=False,
                )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_invalid_config_type(self):
        out_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(TypeError):
                build_ssl_manifest(12345,
                                     os.path.join(out_dir, "m.txt"),
                                     verbose=False)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_extension_filter(self):
        """Globbing **/* should not pull in non-image files."""
        tmp = tempfile.mkdtemp()
        out_dir = tempfile.mkdtemp()
        try:
            ds = os.path.join(tmp, "ds")
            os.makedirs(ds)
            _make_image_file(os.path.join(ds, "img.jpg"))
            # Non-image files
            with open(os.path.join(ds, "readme.txt"), "w") as f:
                f.write("not an image")
            with open(os.path.join(ds, "data.json"), "w") as f:
                f.write("{}")

            cfg = {"datasets": [{"name": "ds", "root": ds, "image_glob": "*"}]}
            out_path = os.path.join(out_dir, "m.txt")
            stats = build_ssl_manifest(cfg, out_path, verbose=False)
            assert stats["total"] == 1  # only the .jpg
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════
# B. MultiLabelImageDataset
# ═════════════════════════════════════════════════════════════════════════


class TestLabelHelpers:
    def test_class_set_single_bbox(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "x.txt")
            _make_label_file(p, [3])
            assert _read_class_set(p) == {3}
        finally:
            shutil.rmtree(tmp)

    def test_class_set_multi_bbox_unique(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "x.txt")
            _make_label_file(p, [0, 1, 1, 2, 0])
            # set, so duplicates removed
            assert _read_class_set(p) == {0, 1, 2}
        finally:
            shutil.rmtree(tmp)

    def test_class_set_empty_file(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "x.txt")
            Path(p).touch()
            assert _read_class_set(p) == set()
        finally:
            shutil.rmtree(tmp)

    def test_class_set_missing_file(self):
        assert _read_class_set("/nonexistent/x.txt") == set()

    def test_label_path_swap(self):
        p = "/data/dataset/images/train/img.jpg"
        assert _label_path_for_image(p) == "/data/dataset/labels/train/img.txt"

    def test_label_path_explicit_dir(self):
        p = "/data/dataset/images/train/img.jpg"
        result = _label_path_for_image(p, labels_dir="/labels/")
        assert result.endswith("img.txt")
        assert "/labels/" in result


class TestMultiLabelDataset:
    def test_basic_load(self):
        tmp = tempfile.mkdtemp()
        try:
            paths = _make_yolo_split(tmp, "train", n=10, num_classes=2)
            ds = MultiLabelImageDataset(
                image_paths=paths, num_classes=2, imgsz=32,
            )
            assert len(ds) == 10
            img, target = ds[0]
            assert img.shape == (3, 32, 32)
            assert img.dtype == torch.float32
            assert (img >= 0).all() and (img <= 1).all()
            assert target.shape == (2,)
            assert target.dtype == torch.float32
            # First image had class 0 (i % 2 == 0)
            assert target[0].item() == 1.0 and target[1].item() == 0.0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_class_cycling(self):
        """Even-index image → class 0; odd-index → class 1."""
        tmp = tempfile.mkdtemp()
        try:
            paths = _make_yolo_split(tmp, "train", n=4, num_classes=2)
            ds = MultiLabelImageDataset(
                image_paths=paths, num_classes=2, imgsz=16,
            )
            for i in range(4):
                _, target = ds[i]
                expected = i % 2
                assert target[expected].item() == 1.0
                assert target[1 - expected].item() == 0.0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unlabeled_image_zero_target(self):
        tmp = tempfile.mkdtemp()
        try:
            paths = _make_yolo_split(
                tmp, "train", n=2, num_classes=2, extra_unlabeled=2,
            )
            ds = MultiLabelImageDataset(
                image_paths=paths, num_classes=2, imgsz=16,
            )
            # First 2 have labels, last 2 don't
            _, t_unlabeled = ds[2]
            assert (t_unlabeled == 0.0).all()
            _, t_unlabeled2 = ds[3]
            assert (t_unlabeled2 == 0.0).all()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_drop_classes(self):
        tmp = tempfile.mkdtemp()
        try:
            paths = _make_yolo_split(tmp, "train", n=4, num_classes=2)
            # Drop class 1 → only class 0 remains active
            ds = MultiLabelImageDataset(
                image_paths=paths, num_classes=2, imgsz=16,
                drop_classes=[1],
            )
            for i in range(4):
                _, target = ds[i]
                assert target[1].item() == 0.0  # always 0 for dropped
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_multi_class_image(self):
        """Image with multiple bbox classes → multi-hot."""
        tmp = tempfile.mkdtemp()
        try:
            img_p = os.path.join(tmp, "images", "train", "x.jpg")
            lbl_p = os.path.join(tmp, "labels", "train", "x.txt")
            _make_image_file(img_p)
            _make_label_file(lbl_p, [0, 1, 0])  # both classes present

            ds = MultiLabelImageDataset(
                image_paths=[img_p], num_classes=2, imgsz=16,
            )
            _, target = ds[0]
            assert target[0].item() == 1.0
            assert target[1].item() == 1.0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_invalid_init(self):
        with pytest.raises(ValueError, match="num_classes"):
            MultiLabelImageDataset(["/x.jpg"], num_classes=0)
        with pytest.raises(ValueError, match="imgsz"):
            MultiLabelImageDataset(["/x.jpg"], num_classes=1, imgsz=0)
        with pytest.raises(ValueError, match="empty"):
            MultiLabelImageDataset([], num_classes=2)

    def test_class_id_out_of_range_ignored(self):
        """Class id >= num_classes should not crash; just doesn't set bit."""
        tmp = tempfile.mkdtemp()
        try:
            img_p = os.path.join(tmp, "images", "train", "x.jpg")
            lbl_p = os.path.join(tmp, "labels", "train", "x.txt")
            _make_image_file(img_p)
            _make_label_file(lbl_p, [0, 5])  # 5 is out of range for num_classes=2

            ds = MultiLabelImageDataset(
                image_paths=[img_p], num_classes=2, imgsz=16,
            )
            _, target = ds[0]
            assert target[0].item() == 1.0
            # Class 5 shouldn't break anything; target is only [2] so we
            # don't index out of bounds (the dataset code skips OOR ids)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════
# C. loaders_from_yolo_data_yaml
# ═════════════════════════════════════════════════════════════════════════


def _make_yolo_dataset_with_yaml(
    base_dir: str, n_train: int = 10, n_val: int = 4, num_classes: int = 2,
) -> str:
    """Create a complete YOLO dataset structure + data.yaml. Returns yaml path."""
    _make_yolo_split(base_dir, "train", n=n_train, num_classes=num_classes)
    _make_yolo_split(base_dir, "valid", n=n_val, num_classes=num_classes)
    yaml_path = os.path.join(base_dir, "data.yaml")
    cfg = {
        "train": "./images/train",
        "val": "./images/valid",
        "nc": num_classes,
        "names": [f"class_{i}" for i in range(num_classes)],
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return yaml_path


class TestLoadersFromYAML:
    def test_basic(self):
        tmp = tempfile.mkdtemp()
        try:
            yaml_path = _make_yolo_dataset_with_yaml(tmp, n_train=10, n_val=4)
            train, val, info = loaders_from_yolo_data_yaml(
                yaml_path, batch_size=4, imgsz=16, num_workers=0,
            )
            assert info["n_train"] == 10
            assert info["n_val"] == 4
            assert info["nc"] == 2
            assert info["names"] == ["class_0", "class_1"]

            # Iterate
            n_seen = 0
            for imgs, targets in train:
                assert imgs.shape[1:] == (3, 16, 16)
                assert targets.shape[1:] == (2,)
                n_seen += imgs.shape[0]
            assert n_seen == 10
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_train_subset_override(self):
        tmp = tempfile.mkdtemp()
        try:
            yaml_path = _make_yolo_dataset_with_yaml(tmp, n_train=20, n_val=4)
            # Only use first 5 train images
            all_train = sorted(
                Path(tmp, "images", "train").glob("*.jpg")
            )
            subset = [str(p) for p in all_train[:5]]

            train, val, info = loaders_from_yolo_data_yaml(
                yaml_path, train_subset=subset, batch_size=4, imgsz=16,
            )
            assert info["n_train"] == 5
            assert info["n_val"] == 4
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_txt_manifest_split(self):
        """data.yaml's `train` can point to a txt manifest file."""
        tmp = tempfile.mkdtemp()
        try:
            paths = _make_yolo_split(tmp, "train", n=8, num_classes=2)
            _make_yolo_split(tmp, "valid", n=3, num_classes=2)

            # Write train manifest
            train_txt = os.path.join(tmp, "train_list.txt")
            with open(train_txt, "w") as f:
                for p in paths:
                    f.write(f"{p}\n")

            yaml_path = os.path.join(tmp, "data.yaml")
            with open(yaml_path, "w") as f:
                yaml.safe_dump({
                    "train": "./train_list.txt",
                    "val": "./images/valid",
                    "nc": 2,
                    "names": ["a", "b"],
                }, f)

            train, val, info = loaders_from_yolo_data_yaml(
                yaml_path, batch_size=4, imgsz=16,
            )
            assert info["n_train"] == 8
            assert info["n_val"] == 3
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_drop_classes_propagated(self):
        tmp = tempfile.mkdtemp()
        try:
            yaml_path = _make_yolo_dataset_with_yaml(tmp, n_train=4, n_val=2)
            train, val, info = loaders_from_yolo_data_yaml(
                yaml_path, batch_size=2, imgsz=16, drop_classes=[1],
            )
            assert info["drop_classes"] == [1]
            for imgs, targets in train:
                # Class 1 should always be 0 in targets
                assert (targets[:, 1] == 0.0).all()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_data_yaml_fields(self):
        tmp = tempfile.mkdtemp()
        try:
            yaml_path = os.path.join(tmp, "bad.yaml")
            with open(yaml_path, "w") as f:
                yaml.safe_dump({"train": "/x", "val": "/y"}, f)  # no nc
            with pytest.raises(ValueError, match="nc"):
                loaders_from_yolo_data_yaml(yaml_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_split_path_resolution(self):
        """Test the helper that resolves split entries to image lists."""
        tmp = tempfile.mkdtemp()
        try:
            _make_yolo_split(tmp, "train", n=3, num_classes=2)
            paths = _resolve_split("./images/train", tmp)
            assert len(paths) == 3
            for p in paths:
                assert os.path.isabs(p)
                assert p.endswith(".jpg")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_split_invalid_raises(self):
        with pytest.raises(FileNotFoundError):
            _resolve_split("/nonexistent/path", "/")

    def test_split_resolves_roboflow_dotdot_pattern(self):
        """Roboflow exports often write `train: ../train/images` even when
        the dataset folder is a sibling of data.yaml (not parent's child).
        We should fall back to stripping the leading `..` and retry."""
        tmp = tempfile.mkdtemp()
        try:
            # Layout:
            #   tmp/data.yaml
            #   tmp/train/images/*.jpg   <-- HERE (sibling of data.yaml)
            #   tmp/valid/images/*.jpg
            train_imgs = os.path.join(tmp, "train", "images")
            os.makedirs(train_imgs)
            for i in range(3):
                _make_image_file(os.path.join(train_imgs, f"img_{i}.jpg"))

            # Roboflow-style spec: '../train/images' relative to data.yaml.
            # Standard resolution would land at parent-of-tmp/train/images
            # (which doesn't exist). Fallback should strip '..' and find it
            # at tmp/train/images.
            paths = _resolve_split("../train/images", tmp)
            assert len(paths) == 3
            for p in paths:
                assert "train/images" in p
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_split_dotdot_only_when_standard_fails(self):
        """If '../train/images' DOES correctly resolve (i.e. there genuinely
        is a parent-level directory), prefer the standard resolution."""
        tmp = tempfile.mkdtemp()
        try:
            # Layout:
            #   tmp/parent/train/images/*.jpg     <-- the standard target
            #   tmp/parent/yaml_dir/data.yaml
            #   tmp/parent/yaml_dir/train/images/*.jpg  <-- the fallback target
            # Standard `_resolve_path('../train/images', yaml_dir)` lands
            # at tmp/parent/train/images. We should use that.
            yaml_dir = os.path.join(tmp, "parent", "yaml_dir")
            os.makedirs(yaml_dir)
            std_target = os.path.join(tmp, "parent", "train", "images")
            os.makedirs(std_target)
            for i in range(2):
                _make_image_file(os.path.join(std_target, f"std_{i}.jpg"))
            fallback_target = os.path.join(yaml_dir, "train", "images")
            os.makedirs(fallback_target)
            for i in range(5):
                _make_image_file(os.path.join(fallback_target, f"fb_{i}.jpg"))

            paths = _resolve_split("../train/images", yaml_dir)
            # Should find the 2 std_*.jpg, NOT the 5 fb_*.jpg
            assert len(paths) == 2
            for p in paths:
                assert "std_" in os.path.basename(p)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_split_dotdot_fallback_still_raises_when_truly_missing(self):
        """Even with fallback, a fully-bogus path should still raise."""
        tmp = tempfile.mkdtemp()
        try:
            with pytest.raises(FileNotFoundError):
                _resolve_split("../nonexistent_xyz/images", tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
