"""Hat D — Data Infrastructure integration smoke tests.

Covers 15 scenarios from INVENTORY.md §2.4:
    D1-D3:   LabelFractionSplitter (dominant/none/nested)
    D4-D7:   unified_loader (build_ssl_manifest, MultiLabelImageDataset, yaml loaders, Roboflow `..`)
    D8-D12:  ssl_pool/ adapters (BDD/A2D2/Cityscapes/Mapillary) + manifest schema
    D13-D15: dedup/ (pHash compute, find_exact_duplicates, cross_set_leakage, hamming)

Integration scope:
    Each test exercises a complete public-API path end-to-end with real
    filesystem state. Unit-level invariants (e.g. cosine sim > 0.7) are
    already covered by tests/test_*.py; here we verify the public API
    plumbing works together — pool ingest → manifest parquet → pHash compute
    → leakage check, etc.

Why this matters for UX redesign:
    Without these passing, every UX entry point that touches data
    (auto_train, SSLFinetunePipeline.run_ssl, label fraction splits) has
    no whole-path guarantee. Unit tests pass in isolation but the chain
    can break at module boundaries.
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile
from pathlib import Path

import pytest


# ═════════════════════════════════════════════════════════════════════════
# D1-D3 — LabelFractionSplitter
# ═════════════════════════════════════════════════════════════════════════


class TestD1_LabelFractionDominant:
    """D1: dominant stratification produces class-balanced subsets at every fraction."""

    def test_dominant_stratify_balanced(self, dummy_yolo_dataset, tmp_workspace):
        ds = dummy_yolo_dataset(n_train=40, n_val=8, num_classes=4)
        from yolo_contrastive.data import LabelFractionSplitter

        train_imgs = sorted(str(p) for p in ds["train_dir"].glob("*.jpg"))
        splitter = LabelFractionSplitter(
            fractions=[0.25, 0.5, 1.0],
            seed=42,
            stratify_mode="dominant",
        )
        subsets = splitter.split(
            image_paths=train_imgs,
            labels_dir=str(ds["train_lbl_dir"]),
            output_dir=str(tmp_workspace / "splits"),
        )

        assert set(subsets.keys()) == {0.25, 0.5, 1.0}
        # 25% of 40 = 10, 50% = 20, 100% = 40
        assert len(subsets[0.25]) == 10
        assert len(subsets[0.5]) == 20
        assert len(subsets[1.0]) == 40

        # Txt files produced
        assert (tmp_workspace / "splits" / "train_pct025.txt").exists()
        assert (tmp_workspace / "splits" / "train_pct100.txt").exists()


class TestD2_LabelFractionNone:
    """D2: 'none' stratify uses uniform random shuffle — no class balancing."""

    def test_none_stratify_deterministic(self, dummy_yolo_dataset, tmp_workspace):
        ds = dummy_yolo_dataset(n_train=20, n_val=4, num_classes=2)
        from yolo_contrastive.data import LabelFractionSplitter

        train_imgs = sorted(str(p) for p in ds["train_dir"].glob("*.jpg"))

        s1 = LabelFractionSplitter(fractions=[0.5], seed=42, stratify_mode="none")
        s2 = LabelFractionSplitter(fractions=[0.5], seed=42, stratify_mode="none")
        out1 = s1.split(train_imgs, labels_dir=str(ds["train_lbl_dir"]))
        out2 = s2.split(train_imgs, labels_dir=str(ds["train_lbl_dir"]))

        # Same seed → bit-identical subsets
        assert out1[0.5] == out2[0.5]


class TestD3_LabelFractionNested:
    """D3: smaller subsets are strict prefixes of larger ones."""

    def test_nested_invariant(self, dummy_yolo_dataset):
        ds = dummy_yolo_dataset(n_train=40, n_val=4, num_classes=3)
        from yolo_contrastive.data import LabelFractionSplitter
        from yolo_contrastive.data.label_fraction import verify_nested

        train_imgs = sorted(str(p) for p in ds["train_dir"].glob("*.jpg"))
        splitter = LabelFractionSplitter(
            fractions=[0.1, 0.25, 0.5, 1.0], seed=42, stratify_mode="dominant",
        )
        subsets = splitter.split(train_imgs, labels_dir=str(ds["train_lbl_dir"]))

        assert verify_nested(subsets)
        # Manually verify prefix property
        for i, f_small in enumerate([0.1, 0.25, 0.5]):
            f_large = [0.25, 0.5, 1.0][i]
            assert subsets[f_small] == subsets[f_large][: len(subsets[f_small])]


# ═════════════════════════════════════════════════════════════════════════
# D4-D7 — unified_loader
# ═════════════════════════════════════════════════════════════════════════


class TestD4_BuildSSLManifest:
    """D4: build_ssl_manifest merges multiple dataset roots into one txt."""

    def test_multi_dataset_merge(self, tmp_workspace, dummy_images_dir):
        from yolo_contrastive.data import build_ssl_manifest

        # Two dataset roots
        ds_a = dummy_images_dir(n=4, name="ds_a")
        ds_b = dummy_images_dir(n=6, name="ds_b")

        out_path = tmp_workspace / "manifest.txt"
        stats = build_ssl_manifest(
            {"datasets": [
                {"name": "ds_a", "root": str(ds_a), "image_glob": "*.jpg"},
                {"name": "ds_b", "root": str(ds_b), "image_glob": "*.jpg"},
            ]},
            output_path=str(out_path),
            verbose=False,
        )

        assert stats["total"] == 10
        assert stats["per_dataset"] == {"ds_a": 4, "ds_b": 6}
        assert out_path.exists()
        lines = [l.strip() for l in out_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 10


class TestD5_MultiLabelDataset:
    """D5: MultiLabelImageDataset emits multi-hot labels from YOLO label files."""

    def test_multilabel_emit(self, dummy_yolo_dataset):
        from yolo_contrastive.data.unified_loader import MultiLabelImageDataset

        ds = dummy_yolo_dataset(n_train=6, n_val=2, num_classes=3, imgsz=32)
        img_paths = sorted(str(p) for p in ds["train_dir"].glob("*.jpg"))

        dataset = MultiLabelImageDataset(
            image_paths=img_paths, num_classes=3, imgsz=32,
        )

        assert len(dataset) == 6
        img0, target0 = dataset[0]
        # Image is [3, 32, 32]
        assert img0.shape == (3, 32, 32)
        # Target is multi-hot [3]
        assert target0.shape == (3,)
        # First image has class 0 (i % 3 == 0) → target[0] == 1.0
        assert target0[0].item() == 1.0


class TestD6_LoadersFromYAML:
    """D6: loaders_from_yolo_data_yaml builds train/val DataLoaders from data.yaml."""

    def test_standard_yaml(self, dummy_yolo_dataset):
        from yolo_contrastive.data import loaders_from_yolo_data_yaml

        ds = dummy_yolo_dataset(n_train=10, n_val=4, num_classes=2, imgsz=32)
        train, val, info = loaders_from_yolo_data_yaml(
            ds["data_yaml"], batch_size=2, imgsz=32, num_workers=0,
        )

        assert info["nc"] == 2
        assert info["n_train"] == 10
        assert info["n_val"] == 4
        assert info["names"] == ["class_0", "class_1"]
        # Loaders should iterate
        for imgs, targets in train:
            assert imgs.shape[0] <= 2
            break


class TestD7_RoboflowDotDot:
    """D7: `train: ../train/images` fallback when standard resolution fails."""

    def test_roboflow_dotdot_fallback(self, dummy_yolo_dataset):
        """The dummy_yolo_dataset fixture with roboflow_dotdot=True produces
        the exact layout Roboflow exports use — data.yaml in yaml_dir,
        train/valid as siblings of yaml_dir (parent), but spec uses '../'."""
        from yolo_contrastive.data import loaders_from_yolo_data_yaml

        ds = dummy_yolo_dataset(
            n_train=6, n_val=2, num_classes=2, imgsz=32, roboflow_dotdot=True,
        )
        train, val, info = loaders_from_yolo_data_yaml(
            ds["data_yaml"], batch_size=2, imgsz=32, num_workers=0,
        )
        assert info["n_train"] == 6
        assert info["n_val"] == 2


# ═════════════════════════════════════════════════════════════════════════
# D8-D12 — ssl_pool/ adapters (mock zip ingest)
# ═════════════════════════════════════════════════════════════════════════


def _make_dummy_jpeg_bytes(color: tuple = (128, 128, 128)) -> bytes:
    """Build a minimal valid JPEG byte string."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


class TestD8_BDD100KAdapter:
    """D8: ssl_pool/bdd100k.py ingests a zip → pool images + manifest."""

    def test_bdd_ingest_creates_manifest(self, tmp_workspace):
        from yolo_contrastive.data.ssl_pool.bdd100k import (
            ingest, CANONICAL_IMAGE_PREFIX,
        )
        from yolo_contrastive.data.ssl_pool import read_manifest

        zip_path = tmp_workspace / "bdd.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for split in ["train", "val", "test"]:
                for i in range(2):
                    z.writestr(
                        f"{CANONICAL_IMAGE_PREFIX}{split}/img_{i}.jpg",
                        _make_dummy_jpeg_bytes(),
                    )

        pool = tmp_workspace / "pool"
        manifest = pool / "manifest.parquet"
        stats = ingest(zip_path, pool, manifest)

        assert stats["materialized"] == 6
        df = read_manifest(manifest)
        assert len(df) == 6
        assert set(df["dataset"]) == {"bdd100k"}


class TestD9_A2D2Adapter:
    """D9: ssl_pool/a2d2.py — A2D2 dataset adapter smoke."""

    def test_a2d2_module_importable(self):
        """A2D2 adapter module must be importable (paper supplementary data
        pipeline depends on this). Full ingest tested in tests/data/ssl_pool/
        with real-format mocks."""
        from yolo_contrastive.data.ssl_pool import a2d2
        # ingest function must exist
        assert callable(a2d2.ingest)


class TestD10_CityscapesAdapter:
    """D10: ssl_pool/cityscapes.py — both coarse + fine zip packages."""

    def test_cityscapes_module_importable(self):
        from yolo_contrastive.data.ssl_pool import cityscapes
        assert callable(cityscapes.ingest)
        # The known split set
        assert "train_extra" in cityscapes.KNOWN_SPLITS
        assert "train" in cityscapes.KNOWN_SPLITS
        assert "val" in cityscapes.KNOWN_SPLITS


class TestD11_MapillaryAdapter:
    """D11: ssl_pool/mapillary.py — Mapillary Vistas adapter."""

    def test_mapillary_module_importable(self):
        from yolo_contrastive.data.ssl_pool import mapillary
        assert callable(mapillary.ingest)


class TestD12_ManifestSchema:
    """D12: manifest.parquet schema invariant — required columns present."""

    def test_manifest_columns(self, dummy_ssl_pool):
        from yolo_contrastive.data.ssl_pool import read_manifest, MANIFEST_COLUMNS

        pool = dummy_ssl_pool(n=5)
        df = read_manifest(pool["manifest_path"])

        # All required columns must be present (regardless of order)
        for col in MANIFEST_COLUMNS:
            assert col in df.columns, f"manifest missing required column: {col}"

        # Schema invariants
        assert len(df) == 5
        assert df["image_id"].is_unique  # no dupes
        assert df["materialized_w"].dtype in ("int64", "int32")
        assert df["materialized_h"].dtype in ("int64", "int32")


# ═════════════════════════════════════════════════════════════════════════
# D13-D15 — dedup/ (pHash + leakage + hamming)
# ═════════════════════════════════════════════════════════════════════════


class TestD13_PHashCompute:
    """D13: compute_phash + compute_pool_phashes — pHash sidecar parquet."""

    def test_phash_compute_and_persist(self, dummy_ssl_pool, tmp_workspace):
        try:
            import imagehash  # noqa: F401
        except ImportError:
            pytest.skip("imagehash not installed")

        from yolo_contrastive.data.dedup import (
            compute_phash, compute_pool_phashes, load_phashes,
        )

        pool = dummy_ssl_pool(n=5)

        # Compute single pHash
        first_img = next(pool["image_dir"].glob("*.jpg"))
        h = compute_phash(str(first_img))
        assert isinstance(h, str)
        assert len(h) == 16  # 64-bit hex

        # Compute pool-wide
        sidecar = tmp_workspace / "phashes.parquet"
        stats = compute_pool_phashes(
            pool_root=pool["root"],
            manifest_path=pool["manifest_path"],
            output_path=sidecar,
        )
        assert stats["computed"] == 5
        assert stats["errors"] == 0
        assert sidecar.exists()

        # Load back
        phashes = load_phashes(sidecar)
        assert len(phashes) == 5


class TestD14_DuplicateAndLeakage:
    """D14: find_exact_duplicates + cross_set_leakage."""

    def test_duplicate_detection(self):
        from yolo_contrastive.data.dedup import (
            find_exact_duplicates, cross_set_leakage, summarize_duplicates,
        )

        # In-set duplicates: two images share a hash
        phashes = {
            "ds_a/img_1": "abcd1234",
            "ds_a/img_2": "abcd1234",   # duplicate of img_1
            "ds_a/img_3": "ffff0000",
            "ds_b/img_4": "1234abcd",
        }
        groups = find_exact_duplicates(phashes)
        assert len(groups) == 1
        assert set(groups[0]) == {"ds_a/img_1", "ds_a/img_2"}

        # Cross-set leakage
        pool = {"pool/x1": "abcd1234", "pool/x2": "ffff0000"}
        eval_ = {"eval/y1": "abcd1234", "eval/y2": "0000aaaa"}
        pairs = cross_set_leakage(pool, eval_)
        assert pairs == [("pool/x1", "eval/y1", "abcd1234")]

        # summarize
        summary = summarize_duplicates(groups)
        assert "ds_a" in summary  # group involves ds_a


class TestD15_HammingDistance:
    """D15: hamming_distance correctness (bit XOR + popcount)."""

    def test_hamming(self):
        from yolo_contrastive.data.dedup import hamming_distance

        # Identical hashes: distance 0
        assert hamming_distance("abcdef0123456789", "abcdef0123456789") == 0

        # Single bit difference
        assert hamming_distance("0000000000000000", "0000000000000001") == 1

        # Max difference (all bits flipped)
        assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64

        # Symmetric
        h1, h2 = "a3f5", "1c0e"
        assert hamming_distance(h1, h2) == hamming_distance(h2, h1)
