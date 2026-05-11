"""Tests for the A2D2 adapter.

We build small fake tar archives in-process — the real 164 GB archive is
never touched in CI. Fakes include label/lidar branches and sibling cameras
so we can assert the filter rejects them.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pytest
from PIL import Image

from yolo_contrastive.data.ssl_pool.a2d2 import (
    DATASET_NAME,
    ORIGINAL_SPLIT,
    TAR_ROOT,
    TARGET_CAMERA,
    _is_canonical_image,
    _parse_entry,
    count_canonical_images,
    ingest,
)
from yolo_contrastive.data.ssl_pool.manifest import (
    existing_image_ids,
    read_manifest,
)


def _png_bytes(size: Tuple[int, int] = (1920, 1208), color=(60, 90, 120)) -> bytes:
    """Return PNG-encoded bytes of a solid-color image (mimics A2D2 1920×1208 frames)."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _add(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.type = tarfile.REGTYPE
    tar.addfile(info, io.BytesIO(payload))


def _fake_tar(
    tar_path: Path,
    *,
    scenes: Iterable[str] = ("20181016_125231", "20181107_132300"),
    per_scene: int = 2,
    include_other_cameras: bool = True,
    include_label: bool = True,
    include_lidar: bool = True,
    include_json_companions: bool = True,
) -> Path:
    """Build a fake A2D2 tar with the target camera plus realistic noise.

    Returns the path. Image count materialized by an ingest run equals
    ``len(scenes) * per_scene`` (target camera only).
    """
    scenes = list(scenes)
    with tarfile.open(tar_path, "w") as t:
        for scene in scenes:
            for i in range(per_scene):
                stem = f"{scene.replace('_', '')}_camera_frontcenter_{i:09d}"
                # The target — included
                _add(
                    t,
                    f"{TAR_ROOT}/{scene}/camera/{TARGET_CAMERA}/{stem}.png",
                    _png_bytes(color=(min(i * 20, 255), 80, 80)),
                )
                # JSON metadata companion alongside the PNG
                if include_json_companions:
                    _add(
                        t,
                        f"{TAR_ROOT}/{scene}/camera/{TARGET_CAMERA}/{stem}.json",
                        b'{"tstamp":0}',
                    )

            # Sibling cameras — must be skipped
            if include_other_cameras:
                for cam in ("cam_front_left", "cam_rear_center", "cam_side_right"):
                    cam_short = cam.replace("cam_", "").replace("_", "")
                    fname = f"{scene.replace('_','')}_camera_{cam_short}_000000001.png"
                    _add(t, f"{TAR_ROOT}/{scene}/camera/{cam}/{fname}", _png_bytes())

            # Label masks (PNGs in the /label/ branch) — must be skipped
            if include_label:
                fname = f"{scene.replace('_','')}_label_frontcenter_000000001.png"
                _add(t, f"{TAR_ROOT}/{scene}/label/{TARGET_CAMERA}/{fname}", _png_bytes())

            # Lidar npz — must be skipped
            if include_lidar:
                fname = f"{scene.replace('_','')}_lidar_frontcenter_000000001.npz"
                _add(t, f"{TAR_ROOT}/{scene}/lidar/{TARGET_CAMERA}/{fname}", b"\x00" * 16)
    return tar_path


class TestIsCanonicalImage:
    def _path(self, scene="20181016_125231", file="x.png") -> str:
        return f"{TAR_ROOT}/{scene}/camera/{TARGET_CAMERA}/{file}"

    def test_canonical_front_center_png(self):
        assert _is_canonical_image(self._path())

    def test_uppercase_extension_accepted(self):
        assert _is_canonical_image(self._path(file="x.PNG"))

    def test_rejects_label_branch(self):
        n = f"{TAR_ROOT}/20181016_125231/label/{TARGET_CAMERA}/x.png"
        assert not _is_canonical_image(n)

    def test_rejects_lidar_branch(self):
        n = f"{TAR_ROOT}/20181016_125231/lidar/{TARGET_CAMERA}/x.npz"
        assert not _is_canonical_image(n)

    def test_rejects_sibling_cameras(self):
        for cam in ("cam_front_left", "cam_front_right", "cam_rear_center",
                    "cam_side_left", "cam_side_right"):
            n = f"{TAR_ROOT}/20181016_125231/camera/{cam}/x.png"
            assert not _is_canonical_image(n), cam

    def test_rejects_json_companion(self):
        assert not _is_canonical_image(self._path(file="x.json"))

    def test_rejects_wrong_root(self):
        assert not _is_canonical_image(
            f"camera_lidar_other/20181016_125231/camera/{TARGET_CAMERA}/x.png"
        )

    def test_rejects_unexpected_depth(self):
        # 4 segments — missing the scene dir
        assert not _is_canonical_image(f"{TAR_ROOT}/camera/{TARGET_CAMERA}/x.png")
        # 6 segments — extra nesting
        assert not _is_canonical_image(
            f"{TAR_ROOT}/20181016_125231/camera/{TARGET_CAMERA}/sub/x.png"
        )


class TestParseEntry:
    def test_basic(self):
        n = f"{TAR_ROOT}/20181016_125231/camera/{TARGET_CAMERA}/foo.png"
        assert _parse_entry(n) == ("20181016_125231", "foo.png")


class TestCountCanonicalImages:
    def test_counts_only_target(self, tmp_path):
        tp = _fake_tar(
            tmp_path / "a.tar",
            scenes=("s1", "s2", "s3"),
            per_scene=4,
        )
        # 3 scenes × 4 target-cam images = 12. All noise must be excluded.
        assert count_canonical_images(tp) == 12


class TestIngest:
    def test_basic_ingest_creates_files_and_manifest(self, tmp_path):
        tp = _fake_tar(tmp_path / "a.tar", scenes=("s1", "s2"), per_scene=2)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(tp, pool, manifest)

        assert stats == {
            "scanned": 4,
            "skipped_existing": 0,
            "materialized": 4,
            "errors": 0,
        }

        # Files exist under per-scene subdirs
        s1 = list((pool / "images" / "a2d2" / "s1").glob("*.jpg"))
        s2 = list((pool / "images" / "a2d2" / "s2").glob("*.jpg"))
        assert (len(s1), len(s2)) == (2, 2)

        df = read_manifest(manifest)
        assert len(df) == 4
        assert set(df["dataset"]) == {"a2d2"}
        assert set(df["original_split"]) == {"unlabeled"}
        assert all(len(h) == 64 for h in df["image_hash"])

    def test_skips_labels_lidar_siblings_jsons(self, tmp_path):
        tp = _fake_tar(
            tmp_path / "a.tar",
            scenes=("s1",),
            per_scene=2,
            include_other_cameras=True,
            include_label=True,
            include_lidar=True,
            include_json_companions=True,
        )
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(tp, pool, manifest)

        # Only the 2 target images, despite many sibling entries in the tar
        assert stats["scanned"] == 2
        assert stats["materialized"] == 2
        assert len(list((pool / "images" / "a2d2").rglob("*.jpg"))) == 2

    def test_idempotent_rerun_is_noop(self, tmp_path):
        tp = _fake_tar(tmp_path / "a.tar", scenes=("s1", "s2"), per_scene=2)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        first = ingest(tp, pool, manifest)
        second = ingest(tp, pool, manifest)

        assert first["materialized"] == 4
        assert second["materialized"] == 0
        assert second["skipped_existing"] == 4
        assert len(read_manifest(manifest)) == 4

    def test_limit_caps_materialized(self, tmp_path):
        tp = _fake_tar(tmp_path / "a.tar", scenes=("s1",), per_scene=10)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(tp, pool, manifest, limit=3)
        assert stats["materialized"] == 3
        assert len(read_manifest(manifest)) == 3

    def test_image_id_includes_scene(self, tmp_path):
        tp = _fake_tar(tmp_path / "a.tar", scenes=("20181016_125231",), per_scene=1,
                       include_other_cameras=False, include_label=False,
                       include_lidar=False, include_json_companions=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        ingest(tp, pool, manifest)
        # image_id convention: <dataset>/<scene>/<stem>
        ids = existing_image_ids(manifest)
        assert len(ids) == 1
        only = next(iter(ids))
        assert only.startswith("a2d2/20181016_125231/")
        assert "frontcenter" in only

    def test_materialized_resolution_is_640_long_side(self, tmp_path):
        tp = _fake_tar(tmp_path / "a.tar", scenes=("s1",), per_scene=1,
                       include_other_cameras=False, include_label=False,
                       include_lidar=False, include_json_companions=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        ingest(tp, pool, manifest)
        jpg = next((pool / "images" / "a2d2" / "s1").glob("*.jpg"))
        with Image.open(jpg) as img:
            # Source PNG is 1920×1208, resize to 640 long side → (640, 402)
            assert img.size[0] == 640
            assert img.size[1] == round(640 * 1208 / 1920)

    def test_manifest_records_original_and_materialized_dims(self, tmp_path):
        tp = _fake_tar(tmp_path / "a.tar", scenes=("s1",), per_scene=1,
                       include_other_cameras=False, include_label=False,
                       include_lidar=False, include_json_companions=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        ingest(tp, pool, manifest)
        row = read_manifest(manifest).iloc[0]
        assert (row["original_w"], row["original_h"]) == (1920, 1208)
        assert row["materialized_w"] == 640
        assert row["dataset"] == "a2d2"
        assert row["original_split"] == "unlabeled"

    def test_corrupt_png_entry_is_skipped(self, tmp_path):
        tp = tmp_path / "a.tar"
        with tarfile.open(tp, "w") as t:
            _add(t,
                 f"{TAR_ROOT}/s1/camera/{TARGET_CAMERA}/good.png",
                 _png_bytes())
            _add(t,
                 f"{TAR_ROOT}/s1/camera/{TARGET_CAMERA}/bad.png",
                 b"not a real png")
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(tp, pool, manifest)
        assert stats["scanned"] == 2
        assert stats["materialized"] == 1
        assert stats["errors"] == 1

    def test_flush_every_does_not_lose_rows(self, tmp_path):
        # 7 images with flush_every=3 → batches of 3, 3, 1
        tp = _fake_tar(tmp_path / "a.tar", scenes=("s1",), per_scene=7,
                       include_other_cameras=False, include_label=False,
                       include_lidar=False, include_json_companions=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(tp, pool, manifest, flush_every=3)
        assert stats["materialized"] == 7
        assert len(read_manifest(manifest)) == 7

    def test_multiple_scenes_partitioned_correctly(self, tmp_path):
        tp = _fake_tar(tmp_path / "a.tar",
                       scenes=("s1", "s2", "s3"),
                       per_scene=2,
                       include_other_cameras=False, include_label=False,
                       include_lidar=False, include_json_companions=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        ingest(tp, pool, manifest)
        df = read_manifest(manifest)
        # Each scene's image_ids form a partition under <dataset>/<scene>/
        for scene in ("s1", "s2", "s3"):
            count = sum(1 for x in df["image_id"] if x.startswith(f"a2d2/{scene}/"))
            assert count == 2, f"scene {scene}"
