"""Tests for source-disjoint holdout and cross-validation splits.

The core invariant under test: no source ever appears on both sides of a
train/val boundary (the whole point of source-disjoint splitting).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_contrastive.data.downstream.splits import build_cv_splits, build_holdout_split


def make_manifest(tmp_path: Path, sources: dict[str, int], names=("pothole",)) -> dict:
    """Build real on-disk source folders + a selection-manifest dict referencing them."""
    man: dict = {"sources": {}}
    for src, n in sources.items():
        root = tmp_path / src
        idir, ldir = root / "train" / "images", root / "train" / "labels"
        idir.mkdir(parents=True)
        ldir.mkdir(parents=True)
        selected = []
        for i in range(n):
            fn = f"{src}_{i:04d}.jpg"
            (idir / fn).write_bytes(b"\xff\xd8\xff")
            (ldir / f"{src}_{i:04d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            selected.append(fn)
        (root / "data.yaml").write_text(yaml.safe_dump({"nc": len(names), "names": list(names)}))
        man["sources"][src] = {"images_dir": str(idir), "labels_dir": str(ldir),
                               "selected_images": selected}
    return man


def _read_txt(path: str) -> list[str]:
    return [ln for ln in Path(path).read_text().splitlines() if ln]


def _src_of(path: str) -> str:
    # .../<src>/train/images/<file>.jpg
    return Path(path).parent.parent.parent.name


# --------------------------------------------------------------------------- holdout
def test_holdout_is_source_disjoint(tmp_path):
    srcs = {f"ds_{i:02d}": 500 for i in range(10)}
    s = build_holdout_split(make_manifest(tmp_path, srcs), tmp_path / "out")
    out = tmp_path / "out" / "holdout"
    tr, va, te = (_read_txt(str(out / f"{x}.txt")) for x in ("train", "val", "test"))

    tr_src = {_src_of(p) for p in tr}
    va_src = {_src_of(p) for p in va}
    te_src = {_src_of(p) for p in te}
    # the defining property: train/val/test draw from disjoint sources
    assert tr_src.isdisjoint(va_src)
    assert tr_src.isdisjoint(te_src)
    assert va_src.isdisjoint(te_src)
    assert tr_src | va_src | te_src == set(srcs)          # every source placed
    assert len(tr) + len(va) + len(te) == 5000            # no image lost or duplicated
    # assignment is recorded
    assert set(s["assignment"]) == {"train", "val", "test"}


def test_holdout_majority_in_train(tmp_path):
    srcs = {f"ds_{i:02d}": 500 for i in range(10)}
    s = build_holdout_split(make_manifest(tmp_path, srcs), tmp_path / "out")
    assert s["counts"]["train"] > s["counts"]["val"] + s["counts"]["test"]


def test_holdout_two_way_no_test(tmp_path):
    srcs = {f"ds_{i:02d}": 500 for i in range(5)}
    s = build_holdout_split(make_manifest(tmp_path, srcs), tmp_path / "out",
                            ratios=(0.8, 0.2, 0.0))
    assert "test" not in s["counts"]
    data = yaml.safe_load((tmp_path / "out" / "holdout" / "data.yaml").read_text())
    assert "test" not in data


def test_holdout_too_few_sources_for_three_way(tmp_path):
    srcs = {"a": 100, "b": 100}  # need >= 3 for train/val/test
    with pytest.raises(ValueError):
        build_holdout_split(make_manifest(tmp_path, srcs), tmp_path / "o")


def test_holdout_bad_ratios(tmp_path):
    srcs = {"a": 10, "b": 10, "c": 10}
    with pytest.raises(ValueError):
        build_holdout_split(make_manifest(tmp_path, srcs), tmp_path / "o",
                            ratios=(0.7, 0.2, 0.2))


# --------------------------------------------------------------------------- group_kfold
def test_group_kfold_source_disjoint_and_partitions(tmp_path):
    srcs = {f"ds_{i:02d}": 500 for i in range(10)}
    s = build_cv_splits(make_manifest(tmp_path, srcs), tmp_path / "out",
                        scheme="group_kfold", k=5)
    assert s["n_folds"] == 5

    base = tmp_path / "out" / "cv" / "group_kfold"
    val_src_union: set[str] = set()
    for i in range(5):
        tr = _read_txt(str(base / f"fold_{i}" / "train.txt"))
        va = _read_txt(str(base / f"fold_{i}" / "val.txt"))
        tr_src = {_src_of(p) for p in tr}
        va_src = {_src_of(p) for p in va}
        assert tr_src.isdisjoint(va_src)            # no source in both train and val
        assert tr_src | va_src == set(srcs)
        assert val_src_union.isdisjoint(va_src)     # each source validates once
        val_src_union |= va_src
    assert val_src_union == set(srcs)


def test_group_kfold_k_exceeds_sources(tmp_path):
    srcs = {"a": 10, "b": 10, "c": 10}
    with pytest.raises(ValueError):
        build_cv_splits(make_manifest(tmp_path, srcs), tmp_path / "o",
                        scheme="group_kfold", k=5)


# --------------------------------------------------------------------------- logo
def test_logo_one_source_per_fold(tmp_path):
    srcs = {"a": 30, "b": 40, "c": 20}
    s = build_cv_splits(make_manifest(tmp_path, srcs), tmp_path / "out", scheme="logo")
    assert s["n_folds"] == 3
    assert {f["val_sources"][0] for f in s["folds"]} == {"a", "b", "c"}

    base = tmp_path / "out" / "cv" / "logo"
    for f in s["folds"]:
        va = _read_txt(str(base / f"fold_{f['fold']}" / "val.txt"))
        tr = _read_txt(str(base / f"fold_{f['fold']}" / "train.txt"))
        assert {_src_of(p) for p in va} == set(f["val_sources"])
        assert f["val_sources"][0] not in {_src_of(p) for p in tr}


# --------------------------------------------------------------------------- guards / misc
def test_image_level_kfold_rejected(tmp_path):
    srcs = {"a": 10, "b": 10}
    with pytest.raises(ValueError, match="source-disjoint"):
        build_cv_splits(make_manifest(tmp_path, srcs), tmp_path / "o", scheme="kfold")


def test_unknown_scheme(tmp_path):
    srcs = {"a": 10, "b": 10}
    with pytest.raises(ValueError):
        build_cv_splits(make_manifest(tmp_path, srcs), tmp_path / "o", scheme="bootstrap")


def test_determinism(tmp_path):
    srcs = {f"ds_{i:02d}": 500 for i in range(10)}
    man = make_manifest(tmp_path, srcs)
    a = build_holdout_split(man, tmp_path / "o1", seed=42)
    b = build_holdout_split(man, tmp_path / "o2", seed=42)
    assert a["assignment"] == b["assignment"]


def test_names_override(tmp_path):
    srcs = {"a": 10, "b": 10, "c": 10}
    s = build_cv_splits(make_manifest(tmp_path, srcs), tmp_path / "o",
                        scheme="group_kfold", k=2, names=["crack"])
    assert s["names"] == ["crack"]
