"""Tests for source-spec loading — the N-scalable entry point."""

from __future__ import annotations

import pytest

from yolo_contrastive.data.downstream.sources import load_sources


def test_dict_passthrough():
    out = load_sources({"a": "https://x/1", "b": "https://x/2"})
    assert out == {"a": "https://x/1", "b": "https://x/2"}


def test_list_auto_names():
    out = load_sources(["https://x/1", "https://x/2", "https://x/3"])
    assert list(out) == ["ds_01", "ds_02", "ds_03"]
    assert out["ds_02"] == "https://x/2"


def test_list_padding_widens_past_99():
    out = load_sources([f"https://x/{i}" for i in range(100)])
    assert "ds_001" in out and "ds_100" in out


def test_txt_unnamed(tmp_path):
    f = tmp_path / "sources.txt"
    f.write_text("# comment\nhttps://x/1\n\nhttps://x/2\n")
    out = load_sources(f)
    assert list(out) == ["ds_01", "ds_02"]


def test_txt_named(tmp_path):
    f = tmp_path / "sources.txt"
    f.write_text("pothole_a https://x/1\npothole_b\thttps://x/2\n")
    out = load_sources(f)
    assert out == {"pothole_a": "https://x/1", "pothole_b": "https://x/2"}


def test_txt_mixed_named_unnamed_raises(tmp_path):
    f = tmp_path / "sources.txt"
    f.write_text("named https://x/1\nhttps://x/2\n")
    with pytest.raises(ValueError):
        load_sources(f)


def test_yaml_mapping(tmp_path):
    f = tmp_path / "sources.yaml"
    f.write_text("a: https://x/1\nb: https://x/2\n")
    assert load_sources(f) == {"a": "https://x/1", "b": "https://x/2"}


def test_yaml_list(tmp_path):
    f = tmp_path / "sources.yml"
    f.write_text("- https://x/1\n- https://x/2\n")
    assert list(load_sources(f)) == ["ds_01", "ds_02"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sources(tmp_path / "nope.txt")


def test_empty_spec_raises():
    with pytest.raises(ValueError):
        load_sources([])
