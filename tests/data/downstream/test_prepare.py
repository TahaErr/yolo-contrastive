"""Tests for the prepare pipeline steps — verify, consolidate, select, manifest.

Fixtures are tiny fake Roboflow YOLO exports built on disk (no network, no real
images — bytes are enough since we only count by extension and move files).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_contrastive.data.downstream.prepare import (
    build_selection_manifest,
    consolidate_to_train,
    read_selection_manifest,
    select_kept_files,
    verify_single_label,
)


def make_ds(root: Path, name: str, n_train: int, n_valid: int = 0, n_test: int = 0,
            names=("pothole",), labels: bool = True) -> Path:
    d = root / name
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        if n == 0:
            continue
        (d / split / "images").mkdir(parents=True)
        (d / split / "labels").mkdir(parents=True)
        for i in range(n):
            stem = f"{name}_{split}_{i:04d}"
            (d / split / "images" / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
            if labels:
                (d / split / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (d / "data.yaml").write_text(yaml.safe_dump(
        {"train": "../train/images", "val": "../valid/images", "test": "../test/images",
         "nc": len(names), "names": list(names)}))
    return d


# --------------------------------------------------------------------------- verify
def test_verify_ok(tmp_path):
    roots = {n: make_ds(tmp_path, n, 4) for n in ("a", "b", "c")}
    assert verify_single_label(roots) == ["pothole"]


def test_verify_name_mismatch_raises(tmp_path):
    roots = {"a": make_ds(tmp_path, "a", 4),
             "b": make_ds(tmp_path, "b", 4, names=("crack",))}
    with pytest.raises(ValueError):
        verify_single_label(roots)


def test_verify_multilabel_raises(tmp_path):
    roots = {"a": make_ds(tmp_path, "a", 4),
             "b": make_ds(tmp_path, "b", 4, names=("pothole", "crack"))}
    with pytest.raises(ValueError):
        verify_single_label(roots)


# --------------------------------------------------------------------------- consolidate
def test_consolidate_merges_all_splits(tmp_path):
    d = make_ds(tmp_path, "a", n_train=5, n_valid=3, n_test=2)
    assert consolidate_to_train(d) == 10
    assert len(list((d / "train" / "labels").glob("*.txt"))) == 10
    assert not (d / "valid").exists() and not (d / "test").exists()


def test_consolidate_tolerates_missing_label(tmp_path):
    d = make_ds(tmp_path, "a", n_train=2, n_valid=2)
    next((d / "valid").glob("labels/*.txt")).unlink()  # a background image
    assert consolidate_to_train(d) == 4  # image still moves, no error


# --------------------------------------------------------------------------- select / manifest
def test_selection_deterministic(tmp_path):
    d = make_ds(tmp_path, "a", n_train=100)
    s1 = select_kept_files(d, 30, seed=42, source_name="a")
    s2 = select_kept_files(d, 30, seed=42, source_name="a")
    assert s1 == s2 and len(s1) == 30 and len(set(s1)) == 30


def test_manifest_end_to_end_total(tmp_path):
    roots = {"s0": make_ds(tmp_path, "s0", 450),
             "s1": make_ds(tmp_path, "s1", 450),
             **{f"L{i}": make_ds(tmp_path, f"L{i}", 800) for i in range(8)}}
    out = tmp_path / "manifest.json"
    m = build_selection_manifest(roots, total=5000, seed=42, out_path=str(out))
    assert m["total_selected"] == 5000 and m["n_sources"] == 10
    again = read_selection_manifest(out)
    assert again == m
    for info in m["sources"].values():
        assert len(info["selected_images"]) == info["keep"]
        for fn in info["selected_images"]:
            assert (Path(info["images_dir"]) / fn).is_file()


def test_manifest_per_dataset_mode(tmp_path):
    roots = {f"L{i}": make_ds(tmp_path, f"L{i}", 800) for i in range(12)}
    out = tmp_path / "m.json"
    m = build_selection_manifest(roots, per_dataset=500, seed=42, out_path=str(out))
    assert m["target"] == 6000 and m["total_selected"] == 6000
