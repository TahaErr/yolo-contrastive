"""Tests for the BDD100K adapter.

We build small fake zip archives in-process — the real 7 GB bundle is never
touched in CI.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from yolo_contrastive.data.ssl_pool.bdd100k import (
    CANONICAL_IMAGE_PREFIX,
    _is_canonical_image,
    _parse_entry,
    count_canonical_images,
    ingest,
)
from yolo_contrastive.data.ssl_pool.manifest import (
    existing_image_ids,
    read_manifest,
)


def _jpeg_bytes(size=(1280, 720), color=(50, 100, 150)) -> bytes:
    """Return JPEG-encoded bytes of a solid-color image."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _fake_bundle(
    zip_path: Path,
    *,
    n_train: int = 2,
    n_val: int = 1,
    n_test: int = 1,
    include_noise: bool = True,
) -> Path:
    """Build a fake BDD100K bundle zip with canonical + non-canonical entries."""
    with zipfile.ZipFile(zip_path, "w") as z:
        for i in range(n_train):
            z.writestr(
                f"{CANONICAL_IMAGE_PREFIX}train/img_t{i}.jpg",
                _jpeg_bytes(color=(min(i * 10, 255), 50, 50)),
            )
        for i in range(n_val):
            z.writestr(
                f"{CANONICAL_IMAGE_PREFIX}val/img_v{i}.jpg",
                _jpeg_bytes(color=(50, min(i * 10, 255), 50)),
            )
        for i in range(n_test):
            z.writestr(
                f"{CANONICAL_IMAGE_PREFIX}test/img_e{i}.jpg",
                _jpeg_bytes(color=(50, 50, min(i * 10, 255))),
            )
        if include_noise:
            # Things the adapter must ignore
            z.writestr("bdd100k_seg/images/some.jpg", _jpeg_bytes())
            z.writestr("bdd100k_labels_release/labels.json", b'{"x":1}')
            z.writestr(f"{CANONICAL_IMAGE_PREFIX}train/notes.json", b"{}")
    return zip_path


class TestIsCanonicalImage:
    def test_train_jpg(self):
        assert _is_canonical_image(f"{CANONICAL_IMAGE_PREFIX}train/x.jpg")

    def test_val_jpg(self):
        assert _is_canonical_image(f"{CANONICAL_IMAGE_PREFIX}val/x.jpg")

    def test_test_jpg(self):
        assert _is_canonical_image(f"{CANONICAL_IMAGE_PREFIX}test/x.jpg")

    def test_uppercase_extension(self):
        assert _is_canonical_image(f"{CANONICAL_IMAGE_PREFIX}train/x.JPG")

    def test_rejects_seg_tree(self):
        assert not _is_canonical_image("bdd100k_seg/images/x.jpg")

    def test_rejects_labels_tree(self):
        assert not _is_canonical_image("bdd100k_labels_release/labels.json")

    def test_rejects_unknown_split(self):
        assert not _is_canonical_image(f"{CANONICAL_IMAGE_PREFIX}weird/x.jpg")

    def test_accepts_nested_subdir(self):
        # Real BDD100K bundles sometimes group images into sub-buckets:
        # bdd100k/.../100k/test/testA/<id>.jpg
        assert _is_canonical_image(f"{CANONICAL_IMAGE_PREFIX}train/sub/x.jpg")
        assert _is_canonical_image(f"{CANONICAL_IMAGE_PREFIX}test/testA/x.jpg")

    def test_rejects_non_jpeg_extension(self):
        assert not _is_canonical_image(f"{CANONICAL_IMAGE_PREFIX}train/x.json")
        assert not _is_canonical_image(f"{CANONICAL_IMAGE_PREFIX}train/x.png")


class TestParseEntry:
    def test_train(self):
        assert _parse_entry(f"{CANONICAL_IMAGE_PREFIX}train/abc-123.jpg") == (
            "train",
            "abc-123.jpg",
        )

    def test_test_split(self):
        assert _parse_entry(f"{CANONICAL_IMAGE_PREFIX}test/x.jpg") == ("test", "x.jpg")


class TestCountCanonicalImages:
    def test_counts_only_canonical(self, tmp_path):
        zp = _fake_bundle(tmp_path / "b.zip", n_train=3, n_val=2, n_test=1)
        assert count_canonical_images(zp) == 6  # noise entries excluded


class TestIngest:
    def test_creates_files_and_manifest(self, tmp_path):
        zp = _fake_bundle(tmp_path / "b.zip", n_train=2, n_val=1, n_test=1)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(zp, pool, manifest)

        assert stats == {
            "scanned": 4,
            "skipped_existing": 0,
            "materialized": 4,
            "errors": 0,
        }

        train_jpgs = list((pool / "images" / "bdd100k" / "train").glob("*.jpg"))
        val_jpgs = list((pool / "images" / "bdd100k" / "val").glob("*.jpg"))
        test_jpgs = list((pool / "images" / "bdd100k" / "test").glob("*.jpg"))
        assert (len(train_jpgs), len(val_jpgs), len(test_jpgs)) == (2, 1, 1)

        df = read_manifest(manifest)
        assert len(df) == 4
        assert set(df["dataset"]) == {"bdd100k"}
        assert set(df["original_split"]) == {"train", "val", "test"}
        # All hashes 64-char hex (sha256)
        assert all(len(h) == 64 for h in df["image_hash"])

    def test_skips_non_canonical_entries(self, tmp_path):
        # Bundle has 3 canonical + 4 noise entries → only 3 scanned/materialized
        zp = _fake_bundle(
            tmp_path / "b.zip", n_train=1, n_val=1, n_test=1, include_noise=True
        )
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(zp, pool, manifest)
        assert stats["scanned"] == 3
        assert stats["materialized"] == 3
        # No nested/seg/label files leaked into the pool
        all_jpgs = list((pool / "images" / "bdd100k").rglob("*.jpg"))
        assert len(all_jpgs) == 3

    def test_idempotent_rerun_is_noop(self, tmp_path):
        zp = _fake_bundle(tmp_path / "b.zip", n_train=2, n_val=1, n_test=1)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        first = ingest(zp, pool, manifest)
        second = ingest(zp, pool, manifest)

        assert first["materialized"] == 4
        assert second["materialized"] == 0
        assert second["skipped_existing"] == 4
        assert len(read_manifest(manifest)) == 4

    def test_limit_caps_materialized_count(self, tmp_path):
        zp = _fake_bundle(tmp_path / "b.zip", n_train=5, n_val=0, n_test=0)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(zp, pool, manifest, limit=2)
        assert stats["materialized"] == 2
        assert len(read_manifest(manifest)) == 2

    def test_image_ids_follow_convention(self, tmp_path):
        zp = _fake_bundle(tmp_path / "b.zip", n_train=1, n_val=0, n_test=0)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        ingest(zp, pool, manifest)
        assert existing_image_ids(manifest) == {"bdd100k/train/img_t0"}

    def test_materialized_resolution_is_640_long_side(self, tmp_path):
        zp = _fake_bundle(tmp_path / "b.zip", n_train=1, n_val=0, n_test=0)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        ingest(zp, pool, manifest)
        jpg = next((pool / "images" / "bdd100k" / "train").glob("*.jpg"))
        with Image.open(jpg) as img:
            # Source is 1280x720, long-side resized to 640 → (640, 360)
            assert img.size == (640, 360)

    def test_manifest_records_original_and_materialized_dims(self, tmp_path):
        zp = _fake_bundle(tmp_path / "b.zip", n_train=1, n_val=0, n_test=0)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        ingest(zp, pool, manifest)
        row = read_manifest(manifest).iloc[0]
        assert (row["original_w"], row["original_h"]) == (1280, 720)
        assert (row["materialized_w"], row["materialized_h"]) == (640, 360)

    def test_corrupt_entry_is_skipped(self, tmp_path):
        zp = tmp_path / "b.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr(f"{CANONICAL_IMAGE_PREFIX}train/good.jpg", _jpeg_bytes())
            z.writestr(f"{CANONICAL_IMAGE_PREFIX}train/bad.jpg", b"definitely not a jpeg")

        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"
        stats = ingest(zp, pool, manifest)

        assert stats["scanned"] == 2
        assert stats["materialized"] == 1
        assert stats["errors"] == 1
        assert (pool / "images" / "bdd100k" / "train" / "good.jpg").exists()
        assert not (pool / "images" / "bdd100k" / "train" / "bad.jpg").exists()

    def test_flush_every_does_not_lose_rows(self, tmp_path):
        # 7 images with flush_every=3 → batches of 3, 3, 1
        zp = _fake_bundle(tmp_path / "b.zip", n_train=7, n_val=0, n_test=0)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(zp, pool, manifest, flush_every=3)
        assert stats["materialized"] == 7
        assert len(read_manifest(manifest)) == 7

    def test_nested_subdir_entries_preserve_path(self, tmp_path):
        # Mirrors the real BDD100K bundle layout where some splits group
        # images under sub-buckets like test/testA/, test/testB/, ...
        zp = tmp_path / "b.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr(f"{CANONICAL_IMAGE_PREFIX}test/testA/foo.jpg", _jpeg_bytes())
            z.writestr(f"{CANONICAL_IMAGE_PREFIX}test/testB/bar.jpg", _jpeg_bytes())
            z.writestr(f"{CANONICAL_IMAGE_PREFIX}test/baz.jpg", _jpeg_bytes())  # flat

        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"
        stats = ingest(zp, pool, manifest)

        assert stats["materialized"] == 3
        assert existing_image_ids(manifest) == {
            "bdd100k/test/testA/foo",
            "bdd100k/test/testB/bar",
            "bdd100k/test/baz",
        }
        # Subdir preserved on disk so flat and nested files with the same
        # name cannot overwrite each other.
        assert (pool / "images" / "bdd100k" / "test" / "testA" / "foo.jpg").exists()
        assert (pool / "images" / "bdd100k" / "test" / "testB" / "bar.jpg").exists()
        assert (pool / "images" / "bdd100k" / "test" / "baz.jpg").exists()

    def test_flat_and_nested_with_same_name_dont_collide(self, tmp_path):
        zp = tmp_path / "b.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr(f"{CANONICAL_IMAGE_PREFIX}test/foo.jpg", _jpeg_bytes(color=(10, 0, 0)))
            z.writestr(f"{CANONICAL_IMAGE_PREFIX}test/sub/foo.jpg", _jpeg_bytes(color=(200, 0, 0)))

        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"
        stats = ingest(zp, pool, manifest)

        assert stats["materialized"] == 2
        # Two distinct image_ids and two on-disk files — no overwrite.
        assert (pool / "images" / "bdd100k" / "test" / "foo.jpg").exists()
        assert (pool / "images" / "bdd100k" / "test" / "sub" / "foo.jpg").exists()
