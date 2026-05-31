"""Tests for cross-validation eval orchestration (offline, mock detection runner)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from yolo_contrastive.eval.cross_val import (
    aggregate_cv_results,
    build_cv_matrix,
    load_backbones,
    run_cv_eval,
)
from yolo_contrastive.eval.run_matrix import CSV_COLUMNS


def make_fold_dir(tmp_path: Path, n_folds: int = 3) -> Path:
    base = tmp_path / "cv" / "logo"
    folds = []
    for i in range(n_folds):
        d = base / f"fold_{i}"
        d.mkdir(parents=True)
        (d / "data.yaml").write_text("train: t.txt\nval: v.txt\nnc: 1\nnames: [pothole]\n")
        folds.append({"fold": i, "val_sources": [f"ds_{i}"], "data_yaml": str(d / "data.yaml")})
    (base / "summary.json").write_text(json.dumps({"folds": folds}))
    return base


_BASE = {"bbA": 0.40, "bbB": 0.30, "bbC": 0.20}


def mock_detect(cell, hp):
    """Deterministic fake detection result keyed by backbone + fold index."""
    v = _BASE.get(cell["method"]["name"], 0.15) + 0.01 * int(cell["dataset"]["name"].split("_")[1])
    return {"metric": "mAP50-95", "metric_value": v, "mAP50": round(v + 0.05, 4),
            "precision": 0.5, "recall": 0.5}


# --------------------------------------------------------------------------- load_backbones
def test_load_backbones_dict():
    assert load_backbones({"a": "/a.pt"}) == [{"name": "a", "backbone_ckpt": "/a.pt"}]


def test_load_backbones_list_paths_autonamed():
    bb = load_backbones(["/a.pt", "/b.pt"])
    assert [b["name"] for b in bb] == ["bb_01", "bb_02"]


def test_load_backbones_txt(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("# comment\ngasp_v6 /g.pt\nmocov3\t/m.pt\n")
    assert load_backbones(f) == [
        {"name": "gasp_v6", "backbone_ckpt": "/g.pt"},
        {"name": "mocov3", "backbone_ckpt": "/m.pt"},
    ]


def test_load_backbones_yaml(tmp_path):
    f = tmp_path / "b.yaml"
    f.write_text("a: /a.pt\nb: /b.pt\n")
    assert [b["name"] for b in load_backbones(f)] == ["a", "b"]


def test_load_backbones_duplicate_names():
    with pytest.raises(ValueError):
        load_backbones([{"name": "x", "backbone_ckpt": "/1"},
                        {"name": "x", "backbone_ckpt": "/2"}])


# --------------------------------------------------------------------------- build_cv_matrix
def test_build_cv_matrix(tmp_path):
    folds = make_fold_dir(tmp_path, 3)
    cfg = build_cv_matrix({"bbA": "/a.pt", "bbB": "/b.pt"}, folds, seed=0, hp={"epochs": 10})
    assert cfg["task"] == "detection"
    assert [m["name"] for m in cfg["methods"]] == ["bbA", "bbB"]
    assert [d["name"] for d in cfg["datasets"]] == ["fold_0", "fold_1", "fold_2"]
    assert cfg["fractions"] == [1.0] and cfg["seeds"] == [0]
    assert cfg["hp"]["epochs"] == 10 and cfg["hp"]["imgsz"] == 320  # override + default


def test_build_cv_matrix_no_folds(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_cv_matrix({"a": "/a.pt"}, tmp_path / "empty")


# --------------------------------------------------------------------------- baselines
def test_build_cv_matrix_with_baselines(tmp_path):
    folds = make_fold_dir(tmp_path, 3)
    cfg = build_cv_matrix({"bbA": "/a.pt"}, folds, baselines=("coco", "scratch"))
    by_name = {m["name"]: m for m in cfg["methods"]}
    assert set(by_name) == {"bbA", "coco_baseline", "scratch"}
    assert by_name["bbA"]["backbone_ckpt"] == "/a.pt"
    # baselines carry a base_model and NO backbone_ckpt
    assert by_name["coco_baseline"]["base_model"] == "yolov8n.pt"
    assert "backbone_ckpt" not in by_name["coco_baseline"]
    assert by_name["scratch"]["base_model"] == "yolov8n.yaml"
    assert "backbone_ckpt" not in by_name["scratch"]


def test_baseline_unknown_raises(tmp_path):
    folds = make_fold_dir(tmp_path, 3)
    with pytest.raises(ValueError, match="unknown baseline"):
        build_cv_matrix({"bbA": "/a.pt"}, folds, baselines=("imagenet",))


def test_baseline_name_collision_raises(tmp_path):
    folds = make_fold_dir(tmp_path, 3)
    with pytest.raises(ValueError, match="duplicate method names"):
        build_cv_matrix({"coco_baseline": "/a.pt"}, folds, baselines=("coco",))


def test_run_cv_eval_with_baselines(tmp_path):
    folds = make_fold_dir(tmp_path, 3)
    csv_p = tmp_path / "res.csv"
    s = run_cv_eval({"bbA": "/x/a.pt"}, folds, str(csv_p), baselines=("coco", "scratch"),
                    runners={"detection": mock_detect})
    names = {b["name"] for b in s["backbones"]}
    assert names == {"bbA", "coco_baseline", "scratch"}          # baselines aggregated too
    assert all(b["n_folds"] == 3 for b in s["backbones"])         # each ran on all folds


# --------------------------------------------------------------------------- aggregate
def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def test_aggregate_mean_std_and_incomplete(tmp_path):
    csv_p = tmp_path / "res.csv"
    _write_csv(csv_p, [
        {"method": "bbA", "dataset": "fold_0", "status": "ok", "mAP50": 0.5},
        {"method": "bbA", "dataset": "fold_1", "status": "ok", "mAP50": 0.7},
        {"method": "bbB", "dataset": "fold_0", "status": "ok", "mAP50": 0.3},
        {"method": "bbB", "dataset": "fold_1", "status": "failed", "error": "boom"},
    ])
    s = aggregate_cv_results(csv_p, metric="mAP50")
    assert [b["name"] for b in s["backbones"]] == ["bbA", "bbB"]  # sorted by mean desc
    bbA = s["backbones"][0]
    assert bbA["mean"] == 0.6 and bbA["n_folds"] == 2
    assert abs(bbA["std"] - 0.1414) < 1e-3
    bbB = s["backbones"][1]
    assert bbB["n_folds"] == 1 and bbB["missing_folds"] == ["fold_1"] and bbB["n_failed"] == 1


# --------------------------------------------------------------------------- end-to-end (mock)
def test_run_cv_eval_end_to_end(tmp_path):
    folds = make_fold_dir(tmp_path, 3)
    backbones = {"bbA": "/x/a.pt", "bbB": "/x/b.pt", "bbC": "/x/c.pt"}
    csv_p = tmp_path / "res.csv"
    s = run_cv_eval(backbones, folds, str(csv_p), runners={"detection": mock_detect})

    ok = [r for r in csv.DictReader(open(csv_p)) if r["status"] == "ok"]
    assert len(ok) == 9                                   # 3 backbones x 3 folds
    assert [b["name"] for b in s["backbones"]] == ["bbA", "bbB", "bbC"]
    assert s["metric"] == "mAP50"
    # bbA mAP50 over folds = 0.45, 0.46, 0.47 -> mean 0.46
    assert s["backbones"][0]["mean"] == 0.46
    assert all(b["n_folds"] == 3 and not b["n_failed"] for b in s["backbones"])


def test_run_cv_eval_resume(tmp_path):
    folds = make_fold_dir(tmp_path, 3)
    backbones = {"bbA": "/x/a.pt", "bbB": "/x/b.pt"}
    csv_p = tmp_path / "res.csv"
    run_cv_eval(backbones, folds, str(csv_p), runners={"detection": mock_detect})
    n1 = sum(r["status"] == "ok" for r in csv.DictReader(open(csv_p)))
    run_cv_eval(backbones, folds, str(csv_p), runners={"detection": mock_detect})  # rerun
    n2 = sum(r["status"] == "ok" for r in csv.DictReader(open(csv_p)))
    assert n1 == 6 and n2 == 6      # resume: cells not re-run, no duplicate rows added
