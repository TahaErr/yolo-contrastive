"""Tests for RunMatrix orchestrator."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from typing import Any, Dict, List

import pytest

from yolo_contrastive.eval import RunMatrix, CSV_COLUMNS


# ── helpers ──────────────────────────────────────────────────────────────


def _basic_config() -> Dict[str, Any]:
    """Minimal valid config for testing."""
    return {
        "task": "linear_probe",
        "methods": [
            {"name": "method_a", "backbone_ckpt": "/fake/a.pt"},
            {"name": "method_b", "backbone_ckpt": "/fake/b.pt"},
        ],
        "datasets": [
            {"name": "ds_x", "data_yaml": "x.yaml", "num_classes": 5},
            {"name": "ds_y", "data_yaml": "y.yaml", "num_classes": 3},
        ],
        "fractions": [0.1, 0.5, 1.0],
        "seeds": [42, 43],
    }


def _make_matrix(config=None, runner=None, csv_dir=None) -> tuple:
    """Build a RunMatrix with optional mock runner.

    Returns (matrix, csv_path, tmp_dir).
    """
    if config is None:
        config = _basic_config()
    tmp = csv_dir or tempfile.mkdtemp(prefix="ycl_rm_test_")
    csv_path = os.path.join(tmp, "results.csv")

    runners = None
    if runner is not None:
        runners = {config["task"]: runner}

    rm = RunMatrix(config=config, output_csv=csv_path, runners=runners)
    return rm, csv_path, tmp


def _ok_runner(cell, hp) -> Dict[str, Any]:
    """Mock runner that always returns mAP=0.5."""
    return {"metric": "mAP", "metric_value": 0.5}


def _fail_runner(cell, hp) -> Dict[str, Any]:
    raise RuntimeError("simulated failure")


def _selective_fail_runner(cell, hp) -> Dict[str, Any]:
    """Fail only for fraction=0.1 cells."""
    if abs(cell["fraction"] - 0.1) < 1e-9:
        raise RuntimeError("fail at 0.1")
    return {"metric": "mAP", "metric_value": 0.5}


# ── construction ─────────────────────────────────────────────────────────


class TestConstruction:
    def test_with_config_dict(self):
        rm, _, tmp = _make_matrix()
        try:
            assert rm.config["task"] == "linear_probe"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_with_yaml_file(self):
        import yaml
        cfg = _basic_config()
        tmp = tempfile.mkdtemp(prefix="ycl_rm_yaml_")
        try:
            yaml_path = os.path.join(tmp, "cfg.yaml")
            with open(yaml_path, "w") as f:
                yaml.safe_dump(cfg, f)
            csv_path = os.path.join(tmp, "out.csv")
            rm = RunMatrix(
                config_path=yaml_path, output_csv=csv_path,
                runners={"linear_probe": _ok_runner},
            )
            assert rm.config["task"] == "linear_probe"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_config_raises(self):
        with pytest.raises(ValueError, match="config"):
            RunMatrix(output_csv="x.csv")

    def test_missing_required_keys(self):
        for missing in ("task", "methods", "datasets", "fractions", "seeds"):
            cfg = _basic_config()
            del cfg[missing]
            tmp = tempfile.mkdtemp()
            try:
                with pytest.raises(ValueError, match=missing):
                    RunMatrix(config=cfg,
                                output_csv=os.path.join(tmp, "x.csv"))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    def test_invalid_task(self):
        cfg = _basic_config()
        cfg["task"] = "bogus_task"
        tmp = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError, match="bogus_task"):
                RunMatrix(config=cfg, output_csv=os.path.join(tmp, "x.csv"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_invalid_fraction(self):
        cfg = _basic_config()
        cfg["fractions"] = [0.5, 1.5]
        tmp = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError, match="fraction"):
                RunMatrix(config=cfg, output_csv=os.path.join(tmp, "x.csv"),
                            runners={"linear_probe": _ok_runner})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_method_missing_name(self):
        cfg = _basic_config()
        cfg["methods"][0] = {"backbone_ckpt": "/fake.pt"}  # no name
        tmp = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError, match="name"):
                RunMatrix(config=cfg, output_csv=os.path.join(tmp, "x.csv"),
                            runners={"linear_probe": _ok_runner})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── expansion ────────────────────────────────────────────────────────────


class TestExpansion:
    def test_cartesian_product_size(self):
        rm, _, tmp = _make_matrix(runner=_ok_runner)
        try:
            cells = rm.expand()
            # 2 methods × 2 datasets × 3 fractions × 2 seeds = 24
            assert len(cells) == 24
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cells_have_required_keys(self):
        rm, _, tmp = _make_matrix(runner=_ok_runner)
        try:
            cells = rm.expand()
            for c in cells:
                for k in ("method", "dataset", "fraction", "seed", "task", "hp"):
                    assert k in c
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cells_unique(self):
        rm, _, tmp = _make_matrix(runner=_ok_runner)
        try:
            cells = rm.expand()
            keys = {rm._cell_key(c) for c in cells}
            assert len(keys) == len(cells)  # no duplicates
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── exclude filter ──────────────────────────────────────────────────────


class TestExclude:
    def test_exclude_by_method(self):
        cfg = _basic_config()
        cfg["exclude"] = [{"method": "method_a"}]
        rm, _, tmp = _make_matrix(config=cfg, runner=_ok_runner)
        try:
            cells = rm.expand()
            # 24 - (1 method × 2 datasets × 3 fractions × 2 seeds = 12) = 12
            assert len(cells) == 12
            for c in cells:
                assert c["method"]["name"] != "method_a"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_exclude_by_method_and_fraction(self):
        cfg = _basic_config()
        # Exclude only the (method_a, fraction=0.1) combo
        cfg["exclude"] = [{"method": "method_a", "fraction": 0.1}]
        rm, _, tmp = _make_matrix(config=cfg, runner=_ok_runner)
        try:
            cells = rm.expand()
            # 24 - (1 method × 2 datasets × 1 fraction × 2 seeds = 4) = 20
            assert len(cells) == 20
            for c in cells:
                assert not (
                    c["method"]["name"] == "method_a"
                    and abs(c["fraction"] - 0.1) < 1e-9
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_exclude_keeps_all(self):
        rm, _, tmp = _make_matrix(runner=_ok_runner)
        try:
            cells = rm.expand()
            assert len(cells) == 24
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── CSV writing ─────────────────────────────────────────────────────────


class TestCSV:
    def test_header_created(self):
        rm, csv_path, tmp = _make_matrix(runner=_ok_runner)
        try:
            cfg = _basic_config()
            cfg["fractions"] = [0.5]; cfg["seeds"] = [42]
            cfg["methods"] = cfg["methods"][:1]; cfg["datasets"] = cfg["datasets"][:1]
            rm = RunMatrix(config=cfg, output_csv=csv_path,
                            runners={"linear_probe": _ok_runner})
            rm.run(verbose=False)

            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
            assert header == CSV_COLUMNS
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rows_written(self):
        rm, csv_path, tmp = _make_matrix(runner=_ok_runner)
        try:
            results = rm.run(verbose=False)
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 24
            for r in rows:
                assert r["status"] == "ok"
                assert r["metric"] == "mAP"
                assert float(r["metric_value"]) == 0.5
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_failed_row_has_error(self):
        rm, csv_path, tmp = _make_matrix(runner=_fail_runner)
        try:
            rm.run(on_error="continue", verbose=False)
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for r in rows:
                assert r["status"] == "failed"
                assert "RuntimeError" in r["error"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── error handling ──────────────────────────────────────────────────────


class TestErrorHandling:
    def test_continue_keeps_running(self):
        """on_error='continue': single failure shouldn't halt."""
        rm, csv_path, tmp = _make_matrix(runner=_selective_fail_runner)
        try:
            results = rm.run(on_error="continue", verbose=False)
            assert len(results) == 24  # all attempted
            statuses = [r["status"] for r in results]
            n_fail = statuses.count("failed")
            n_ok = statuses.count("ok")
            # 0.1 cells: 2 methods × 2 datasets × 2 seeds = 8 failures
            # 0.5 + 1.0 cells: 16 successes
            assert n_fail == 8
            assert n_ok == 16
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_raise_propagates(self):
        rm, csv_path, tmp = _make_matrix(runner=_fail_runner)
        try:
            with pytest.raises(RuntimeError, match="simulated"):
                rm.run(on_error="raise", verbose=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_invalid_on_error_value(self):
        rm, _, tmp = _make_matrix(runner=_ok_runner)
        try:
            with pytest.raises(ValueError, match="on_error"):
                rm.run(on_error="bogus", verbose=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── resume ──────────────────────────────────────────────────────────────


class TestResume:
    def test_resume_skips_completed(self):
        """If CSV has status=ok rows, resume mode skips them."""
        cfg = _basic_config()
        cfg["fractions"] = [0.5]
        cfg["seeds"] = [42]
        # 2 methods × 2 datasets × 1 fraction × 1 seed = 4 cells
        rm, csv_path, tmp = _make_matrix(config=cfg, runner=_ok_runner)
        try:
            # First run — completes all 4
            r1 = rm.run(verbose=False)
            assert all(r["status"] == "ok" for r in r1)

            # Second run with resume=True — all should skip
            r2 = rm.run(resume=True, verbose=False)
            assert all(r["status"] == "skipped" for r in r2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resume_reruns_failed(self):
        """Resume should NOT skip cells that previously failed."""
        cfg = _basic_config()
        cfg["fractions"] = [0.5]
        cfg["seeds"] = [42]
        cfg["methods"] = cfg["methods"][:1]
        cfg["datasets"] = cfg["datasets"][:1]  # single cell

        # Run 1: failing runner
        rm1, csv_path, tmp = _make_matrix(config=cfg, runner=_fail_runner)
        try:
            rm1.run(on_error="continue", verbose=False)

            # Run 2: now successful runner — same CSV path, should rerun
            rm2 = RunMatrix(config=cfg, output_csv=csv_path,
                              runners={"linear_probe": _ok_runner})
            r2 = rm2.run(resume=True, verbose=False)
            assert len(r2) == 1
            assert r2[0]["status"] == "ok"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resume_disabled(self):
        cfg = _basic_config()
        cfg["fractions"] = [0.5]
        cfg["seeds"] = [42]
        cfg["methods"] = cfg["methods"][:1]
        cfg["datasets"] = cfg["datasets"][:1]
        rm, csv_path, tmp = _make_matrix(config=cfg, runner=_ok_runner)
        try:
            rm.run(verbose=False)
            r2 = rm.run(resume=False, verbose=False)
            # All cells re-run, none skipped
            assert all(r["status"] == "ok" for r in r2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── mock-runner end-to-end ──────────────────────────────────────────────


class TestEndToEnd:
    def test_full_run_writes_summary(self):
        rm, csv_path, tmp = _make_matrix(runner=_ok_runner)
        try:
            results = rm.run(verbose=False)
            assert len(results) == 24
            assert os.path.exists(csv_path)
            # CSV size should match
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 24
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_repr(self):
        rm, _, tmp = _make_matrix(runner=_ok_runner)
        try:
            r = repr(rm)
            assert "RunMatrix" in r
            assert "task='linear_probe'" in r
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── config defaults / hp passing ────────────────────────────────────────


class TestHyperparameters:
    def test_hp_passed_to_runner(self):
        captured = {}

        def capture_runner(cell, hp):
            captured.update(hp)
            return {"metric": "mAP", "metric_value": 0.5}

        cfg = _basic_config()
        cfg["fractions"] = [0.5]
        cfg["seeds"] = [42]
        cfg["methods"] = cfg["methods"][:1]
        cfg["datasets"] = cfg["datasets"][:1]
        cfg["hp"] = {"epochs": 5, "lr": 0.001, "weight_decay": 0.01}

        rm, _, tmp = _make_matrix(config=cfg, runner=capture_runner)
        try:
            rm.run(verbose=False)
            assert captured == {"epochs": 5, "lr": 0.001, "weight_decay": 0.01}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_hp_section_works(self):
        cfg = _basic_config()
        cfg["fractions"] = [0.5]
        cfg["seeds"] = [42]
        cfg["methods"] = cfg["methods"][:1]
        cfg["datasets"] = cfg["datasets"][:1]
        # No 'hp' key
        rm, _, tmp = _make_matrix(config=cfg, runner=_ok_runner)
        try:
            results = rm.run(verbose=False)
            assert results[0]["status"] == "ok"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
