"""Shared fixtures for integration smoke tests.

These fixtures back the test_*_*.py files under tests/integration/, providing
isolated tmp workspaces, dummy datasets, and seeded RNG state. They're
deliberately minimal — each test owns its own state, no global mutable
fixtures, no test ordering assumptions.

Design notes:
    - Every fixture that writes to disk uses tmp_path (pytest builtin) for
      auto-cleanup; we never use /tmp/* directly.
    - Images are 32-64px to keep the suite CPU-friendly. The point of
      integration smoke is to exercise public API plumbing, not to learn
      anything from the trained weights.
    - For ultralytics-dependent fixtures (tiny_backbone_pt) we accept a
      slower one-time YOLOv8n download — caching makes repeated runs fast.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────
# RNG seeding
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def seeded_torch():
    """Set torch + numpy + random seeds before every test for reproducibility.

    Autouse: applies to every test in tests/integration/ without explicit ask.
    No yield needed — we just set state, tests run, pytest's normal isolation
    handles teardown (we don't try to restore — that would mask state leaks
    from the test under inspection).
    """
    import random
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ─────────────────────────────────────────────────────────────────────────
# Tmp workspace
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Per-test scratch dir. pytest's tmp_path is per-test by design; we just
    rename for readability in test bodies."""
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────
# Dummy images / datasets
# ─────────────────────────────────────────────────────────────────────────


def _write_dummy_image(path: Path, size: int = 64, seed: int = None) -> Path:
    """Write a deterministic random image. Uses OpenCV if available, else PIL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed if seed is not None else int(path.stem.split("_")[-1]) if "_" in path.stem else 0)
    arr = (rng.rand(size, size, 3) * 255).astype(np.uint8)
    try:
        import cv2
        cv2.imwrite(str(path), arr)
    except Exception:
        from PIL import Image
        Image.fromarray(arr).save(str(path))
    return path


@pytest.fixture
def dummy_images_dir(tmp_workspace: Path):
    """Factory: returns a function that creates N dummy images in a subdir.

    Usage:
        def test_foo(dummy_images_dir):
            img_dir = dummy_images_dir(n=8, size=64, name="ssl_pool")
            # img_dir contains 8 .jpg files
    """

    def _factory(n: int = 8, size: int = 64, name: str = "imgs") -> Path:
        d = tmp_workspace / name
        d.mkdir(exist_ok=True)
        for i in range(n):
            _write_dummy_image(d / f"img_{i:04d}.jpg", size=size, seed=i)
        return d

    return _factory


def _write_yolo_label(path: Path, class_ids: List[int], boxes: List[tuple] = None):
    """Write a YOLO format label file. `boxes` defaults to a single 50% center box per class."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if boxes is None:
        boxes = [(0.5, 0.5, 0.2, 0.2)] * len(class_ids)
    with open(path, "w") as f:
        for cls, (cx, cy, w, h) in zip(class_ids, boxes):
            f.write(f"{cls} {cx} {cy} {w} {h}\n")


@pytest.fixture
def dummy_yolo_dataset(tmp_workspace: Path):
    """Factory: builds a complete YOLO-format dataset on disk.

    Returns a dict with paths to data.yaml, train images dir, val images dir,
    train labels dir, val labels dir. Class distribution is balanced across
    `num_classes` over `n_train` + `n_val` images.

    Usage:
        ds = dummy_yolo_dataset(n_train=10, n_val=4, num_classes=3, imgsz=32)
        # ds["data_yaml"], ds["train_dir"], ds["val_dir"], ...
    """
    import yaml

    def _factory(n_train: int = 10, n_val: int = 4, num_classes: int = 3,
                  imgsz: int = 32, name: str = "yolo_ds",
                  roboflow_dotdot: bool = False) -> dict:
        root = tmp_workspace / name
        if roboflow_dotdot:
            # Roboflow-style layout: data.yaml in yaml_dir, images sibling
            yaml_dir = root / "yaml_dir"
            yaml_dir.mkdir(parents=True)
            train_imgs = yaml_dir / "train" / "images"
            val_imgs = yaml_dir / "valid" / "images"
            train_lbls = yaml_dir / "train" / "labels"
            val_lbls = yaml_dir / "valid" / "labels"
        else:
            train_imgs = root / "images" / "train"
            val_imgs = root / "images" / "val"
            train_lbls = root / "labels" / "train"
            val_lbls = root / "labels" / "val"
            yaml_dir = root

        for d in (train_imgs, val_imgs, train_lbls, val_lbls):
            d.mkdir(parents=True, exist_ok=True)

        # Train: balanced class distribution
        for i in range(n_train):
            cls = i % num_classes
            _write_dummy_image(train_imgs / f"img_{i:04d}.jpg", size=imgsz, seed=i)
            _write_yolo_label(train_lbls / f"img_{i:04d}.txt", [cls])

        for i in range(n_val):
            cls = i % num_classes
            _write_dummy_image(val_imgs / f"img_{i:04d}.jpg", size=imgsz, seed=100 + i)
            _write_yolo_label(val_lbls / f"img_{i:04d}.txt", [cls])

        # data.yaml
        if roboflow_dotdot:
            yaml_content = {
                "train": "../train/images",  # Roboflow quirk
                "val": "../valid/images",
                "nc": num_classes,
                "names": [f"class_{c}" for c in range(num_classes)],
            }
            yaml_path = yaml_dir / "data.yaml"
        else:
            yaml_content = {
                "train": "./images/train",
                "val": "./images/val",
                "nc": num_classes,
                "names": [f"class_{c}" for c in range(num_classes)],
            }
            yaml_path = yaml_dir / "data.yaml"

        with open(yaml_path, "w") as f:
            yaml.safe_dump(yaml_content, f)

        return {
            "root": root,
            "data_yaml": str(yaml_path),
            "yaml_dir": yaml_dir,
            "train_dir": train_imgs,
            "val_dir": val_imgs,
            "train_lbl_dir": train_lbls,
            "val_lbl_dir": val_lbls,
            "num_classes": num_classes,
            "n_train": n_train,
            "n_val": n_val,
        }

    return _factory


# ─────────────────────────────────────────────────────────────────────────
# SSL pool fixtures (manifest parquet)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_ssl_pool(tmp_workspace: Path, dummy_images_dir):
    """Factory: builds a minimal SSL pool (images + manifest.parquet).

    Returns a dict with paths and counts. Mimics the layout produced by
    data/ssl_pool adapters (bdd100k.ingest, etc.) without running them.

    Usage:
        pool = dummy_ssl_pool(n=10)
        # pool["root"], pool["manifest_path"], pool["image_count"]
    """
    import pandas as pd

    def _factory(n: int = 10, dataset_name: str = "bdd100k") -> dict:
        pool_root = tmp_workspace / "ssl_pool"
        img_dir = pool_root / "images" / dataset_name / "train"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Use the actual MANIFEST_COLUMNS schema so read_manifest /
        # compute_pool_phashes consume this without "missing columns" errors.
        from yolo_contrastive.data.ssl_pool.manifest import MANIFEST_COLUMNS

        rows = []
        for i in range(n):
            img_path = img_dir / f"img_{i:04d}.jpg"
            _write_dummy_image(img_path, size=64, seed=i)
            rel_path = img_path.relative_to(pool_root).as_posix()
            rows.append({
                "image_id": f"{dataset_name}/train/img_{i:04d}",
                "dataset": dataset_name,
                "original_split": "train",
                "materialized_path": rel_path,
                "original_h": 64,
                "original_w": 64,
                "materialized_h": 64,
                "materialized_w": 64,
                "image_hash": f"{i:064x}",  # fake sha256
                "original_filename": f"img_{i:04d}.jpg",
            })

        manifest_path = pool_root / "manifest.parquet"
        # Write in canonical column order
        pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_parquet(manifest_path, index=False)

        return {
            "root": pool_root,
            "manifest_path": manifest_path,
            "image_dir": img_dir,
            "image_count": n,
            "dataset_name": dataset_name,
        }

    return _factory


# ─────────────────────────────────────────────────────────────────────────
# Tiny backbone .pt for finetune tests (Hat A, C)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def yolov8n_weights_path(tmp_path_factory):
    """Session-scoped: download YOLOv8n once, reuse across tests.

    Returns absolute path to yolov8n.pt. If ultralytics isn't available,
    skips dependent tests via fixture failure.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        pytest.skip("ultralytics not installed — skipping YOLO-dependent integration tests")

    # YOLO("yolov8n.pt") downloads to its own cache; we just trigger that
    cache_dir = tmp_path_factory.mktemp("yolo_cache")
    old_cwd = os.getcwd()
    try:
        os.chdir(cache_dir)
        model = YOLO("yolov8n.pt")  # triggers download
        # The .pt file should be in cwd or in ultralytics' weight cache
        pt_files = list(cache_dir.glob("*.pt"))
        if pt_files:
            return str(pt_files[0].absolute())
        # Fall back: yolov8n.pt is the spec string ultralytics resolves itself
        return "yolov8n.pt"
    finally:
        os.chdir(old_cwd)


@pytest.fixture
def tiny_dense_backbone(tmp_workspace: Path, yolov8n_weights_path):
    """Run 1-iter DenseSSLPretrainer to produce a real .pt backbone.

    Slow fixture — only depended on by tests that genuinely need a trained
    backbone (linear probe, finetune integration). Marked function-scope so
    each test gets a clean backbone, but DenseSSLPretrainer's setup is heavy
    so we keep iterations minimal.

    Usage:
        def test_finetune(tiny_dense_backbone):
            backbone_pt = tiny_dense_backbone(epochs=1, n_images=4)
            # backbone_pt is a path to a saved .pt
    """

    def _factory(epochs: int = 1, n_images: int = 4, imgsz: int = 64) -> str:
        from yolo_contrastive.pretrain import DenseSSLPretrainer
        import cv2

        img_dir = tmp_workspace / "tiny_ssl_imgs"
        img_dir.mkdir(exist_ok=True)
        for i in range(n_images):
            arr = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.jpg"), arr)

        out_path = tmp_workspace / "tiny_backbone.pt"
        trainer = DenseSSLPretrainer(
            model=yolov8n_weights_path,
            out_dim=16, queue_size=8, n_query=4,
            momentum=0.9, temperature=0.2, pos_radius=0.1,
            imgsz=imgsz, device="cpu",
        )
        try:
            trainer.train(
                images_dir=str(img_dir), epochs=epochs,
                batch_size=2, lr=1e-3, warmup_epochs=0,
                num_workers=0, output=str(out_path),
                save_every=0, print_every=0,
            )
        finally:
            trainer.cleanup()
        return str(out_path)

    return _factory


# ─────────────────────────────────────────────────────────────────────────
# Env var lifecycle helper
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def env_isolation():
    """Save + restore environment variables touched during a test.

    Tests that set YCL_* env vars should request this fixture; cleanup is
    automatic. Without it, env state can leak to subsequent tests.

    Usage:
        def test_finetune(env_isolation):
            os.environ["YCL_PRETRAINED"] = "/path/to/backbone.pt"
            # ... test body ...
            # env_isolation will restore on teardown
    """
    backup = dict(os.environ)
    yield
    # Remove keys added during test
    added = set(os.environ.keys()) - set(backup.keys())
    for k in added:
        os.environ.pop(k, None)
    # Restore modified keys
    for k, v in backup.items():
        if os.environ.get(k) != v:
            os.environ[k] = v
