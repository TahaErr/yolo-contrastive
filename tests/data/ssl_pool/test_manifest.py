"""Tests for the SSL-pool manifest (parquet read/write + dedup)."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from yolo_contrastive.data.ssl_pool.manifest import (
    MANIFEST_COLUMNS,
    ManifestRow,
    append_rows,
    existing_image_ids,
    read_manifest,
    write_manifest,
)


def _row(image_id: str, dataset: str = "a2d2") -> ManifestRow:
    return ManifestRow(
        image_id=image_id,
        dataset=dataset,
        original_split="train",
        materialized_path=f"{dataset}/{image_id}.jpg",
        original_h=1080,
        original_w=1920,
        materialized_h=360,
        materialized_w=640,
        image_hash="a" * 64,
        original_filename=f"{image_id}.png",
    )


class TestReadManifest:
    def test_missing_file_returns_empty_with_schema(self, tmp_path):
        df = read_manifest(tmp_path / "nope.parquet")
        assert df.empty
        assert list(df.columns) == MANIFEST_COLUMNS

    def test_existing_file_roundtrip(self, tmp_path):
        rows = [_row("a/1"), _row("a/2")]
        df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
        path = tmp_path / "manifest.parquet"
        write_manifest(df, path)
        loaded = read_manifest(path)
        assert len(loaded) == 2
        assert set(loaded["image_id"]) == {"a/1", "a/2"}
        assert list(loaded.columns) == MANIFEST_COLUMNS


class TestWriteManifest:
    def test_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame([dataclasses.asdict(_row("a/1"))])
        path = tmp_path / "nested" / "deeper" / "m.parquet"
        write_manifest(df, path)
        assert path.exists()

    def test_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame({"image_id": ["x"]})
        with pytest.raises(ValueError, match="missing columns"):
            write_manifest(df, tmp_path / "m.parquet")

    def test_extra_columns_dropped(self, tmp_path):
        rows = [_row("a/1")]
        df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
        df["extra_col"] = "ignored"
        path = tmp_path / "m.parquet"
        write_manifest(df, path)
        loaded = read_manifest(path)
        assert "extra_col" not in loaded.columns


class TestAppendRows:
    def test_append_to_empty_manifest(self, tmp_path):
        path = tmp_path / "m.parquet"
        n = append_rows(path, [_row("a/1"), _row("a/2")])
        assert n == 2
        assert len(read_manifest(path)) == 2

    def test_append_dedups_on_image_id(self, tmp_path):
        path = tmp_path / "m.parquet"
        append_rows(path, [_row("a/1")])
        n = append_rows(path, [_row("a/1"), _row("a/2")])
        assert n == 1
        df = read_manifest(path)
        assert len(df) == 2
        assert set(df["image_id"]) == {"a/1", "a/2"}

    def test_append_empty_iterable_is_noop(self, tmp_path):
        path = tmp_path / "m.parquet"
        assert append_rows(path, []) == 0
        assert not path.exists()

    def test_append_all_duplicates_returns_zero(self, tmp_path):
        path = tmp_path / "m.parquet"
        append_rows(path, [_row("a/1"), _row("a/2")])
        n = append_rows(path, [_row("a/1"), _row("a/2")])
        assert n == 0
        assert len(read_manifest(path)) == 2

    def test_append_preserves_existing_rows(self, tmp_path):
        path = tmp_path / "m.parquet"
        append_rows(path, [_row("a/1", dataset="a2d2")])
        append_rows(path, [_row("b/1", dataset="bdd100k")])
        df = read_manifest(path)
        assert len(df) == 2
        assert set(df["dataset"]) == {"a2d2", "bdd100k"}


class TestExistingImageIds:
    def test_missing_manifest_returns_empty_set(self, tmp_path):
        assert existing_image_ids(tmp_path / "nope.parquet") == set()

    def test_returns_all_ids(self, tmp_path):
        path = tmp_path / "m.parquet"
        append_rows(path, [_row("a/1"), _row("a/2"), _row("a/3")])
        assert existing_image_ids(path) == {"a/1", "a/2", "a/3"}
