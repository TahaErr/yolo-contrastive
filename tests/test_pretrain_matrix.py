"""Tests for :class:`PretrainMatrix`.

All tests use a mock runner — we never instantiate ``DenseSSLPretrainer``
in unit tests. End-to-end correctness of the trainer itself is covered by
``tests/test_dense_ssl_pretrainer.py``; here we only verify orchestration
logic (expansion, exclude DSL, resume, CSV schema, error handling).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from typing import Any, Dict

import pytest
import yaml

from yolo_contrastive.pretrain.run_matrix import (
    CSV_COLUMNS,
    PretrainMatrix,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────


def _basic_config() -> Dict[str, Any]:
    """Minimum valid config covering all required keys."""
    return {
        "base": {
            "images_dir": "/data/ssl_pool",
            "model": "yolov8n.pt",
            "epochs": 10,
            "batch_size": 16,
        },
        "grid": {
            "saps_mode": ["none", "within", "both"],
            "saps_t_scale": [0.5, 1.0],
        },
        "seeds": [42, 43],
    }


def _ok_runner(cell, base):
    return {
        "metric": "final_loss",
        "metric_value": 0.5,
        "backbone_path": f"/fake/{cell['cell_id']}.pt",
    }


def _fail_runner(cell, base):
    raise RuntimeError(f"simulated failure for {cell['cell_id']}")


def _mode_specific_runner(cell, base):
    """Fails iff saps_mode == 'within' — useful for testing selective failures."""
    if cell["axes"].get("saps_mode") == "within":
        raise RuntimeError("mode=within disabled")
    return {"metric": "final_loss", "metric_value": 0.1, "backbone_path": ""}


def _make_matrix(*, config=None, runner=None):
    """Helper: instantiate PretrainMatrix with a tempdir CSV + mock runner."""
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "results.csv")
    cfg = config if config is not None else _basic_config()
    runners = {"pretrain": runner if runner is not None else _ok_runner}
    pm = PretrainMatrix(config=cfg, output_csv=csv_path, runners=runners)
    return pm, csv_path, tmp


# ─────────────────────────────────────────────────────────────────────────
# Config validation
# ─────────────────────────────────────────────────────────────────────────


class TestConfigValidation:
    def test_minimum_valid_config(self):
        # Should not raise
        _make_matrix()

    def test_missing_base_raises(self):
        cfg = _basic_config()
        del cfg["base"]
        with pytest.raises(ValueError, match="base"):
            PretrainMatrix(config=cfg, output_csv="/tmp/x.csv")

    def test_missing_grid_raises(self):
        cfg = _basic_config()
        del cfg["grid"]
        with pytest.raises(ValueError, match="grid"):
            PretrainMatrix(config=cfg, output_csv="/tmp/x.csv")

    def test_missing_seeds_raises(self):
        cfg = _basic_config()
        del cfg["seeds"]
        with pytest.raises(ValueError, match="seeds"):
            PretrainMatrix(config=cfg, output_csv="/tmp/x.csv")

    def test_empty_grid_raises(self):
        cfg = _basic_config()
        cfg["grid"] = {}
        with pytest.raises(ValueError, match="grid"):
            PretrainMatrix(config=cfg, output_csv="/tmp/x.csv")

    def test_scalar_axis_value_rejected(self):
        cfg = _basic_config()
        cfg["grid"]["saps_mode"] = "within"  # scalar, not list
        with pytest.raises(ValueError, match="non-empty list"):
            PretrainMatrix(config=cfg, output_csv="/tmp/x.csv")

    def test_empty_axis_value_list_rejected(self):
        cfg = _basic_config()
        cfg["grid"]["saps_mode"] = []
        with pytest.raises(ValueError, match="non-empty list"):
            PretrainMatrix(config=cfg, output_csv="/tmp/x.csv")

    def test_non_list_seeds_rejected(self):
        cfg = _basic_config()
        cfg["seeds"] = 42
        with pytest.raises(ValueError, match="seeds"):
            PretrainMatrix(config=cfg, output_csv="/tmp/x.csv")

    def test_unknown_task_rejected(self):
        cfg = _basic_config()
        with pytest.raises(ValueError, match="task"):
            PretrainMatrix(
                config={**cfg, "task": "bogus"},
                output_csv="/tmp/x.csv",
                runners={"pretrain": _ok_runner},
            )

    def test_exclude_not_mapping_rejected(self):
        cfg = _basic_config()
        cfg["exclude"] = ["not a dict"]
        with pytest.raises(ValueError, match="exclude"):
            PretrainMatrix(config=cfg, output_csv="/tmp/x.csv")

    def test_output_csv_from_config(self):
        cfg = _basic_config()
        cfg["output_csv"] = "/tmp/from_config.csv"
        pm = PretrainMatrix(config=cfg)
        assert pm.output_csv == "/tmp/from_config.csv"

    def test_missing_output_csv_raises(self):
        cfg = _basic_config()
        with pytest.raises(ValueError, match="output_csv"):
            PretrainMatrix(config=cfg)


# ─────────────────────────────────────────────────────────────────────────
# Expansion (cartesian product)
# ─────────────────────────────────────────────────────────────────────────


class TestExpansion:
    def test_cartesian_size(self):
        pm, _, tmp = _make_matrix()
        try:
            cells = pm.expand()
            # 3 modes × 2 scales × 2 seeds = 12
            assert len(cells) == 12
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_each_cell_has_all_keys(self):
        pm, _, tmp = _make_matrix()
        try:
            cells = pm.expand()
            for c in cells:
                assert set(c.keys()) == {"axes", "seed", "base", "cell_id"}
                assert set(c["axes"].keys()) == {"saps_mode", "saps_t_scale"}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cell_ids_unique_across_grid(self):
        pm, _, tmp = _make_matrix()
        try:
            ids = [c["cell_id"] for c in pm.expand()]
            assert len(ids) == len(set(ids))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cell_id_deterministic_across_runs(self):
        # Same config → same cell_ids in the same order
        cfg = _basic_config()
        pm1, _, t1 = _make_matrix(config=cfg)
        pm2, _, t2 = _make_matrix(config=cfg)
        try:
            ids1 = [c["cell_id"] for c in pm1.expand()]
            ids2 = [c["cell_id"] for c in pm2.expand()]
            assert ids1 == ids2
        finally:
            shutil.rmtree(t1, ignore_errors=True)
            shutil.rmtree(t2, ignore_errors=True)

    def test_seeds_multiply_grid(self):
        cfg = _basic_config()
        cfg["seeds"] = [1, 2, 3]
        cfg["grid"] = {"x": ["a", "b"]}
        pm, _, tmp = _make_matrix(config=cfg)
        try:
            cells = pm.expand()
            assert len(cells) == 6  # 2 × 3
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# Exclude DSL — the main academic contribution
# ─────────────────────────────────────────────────────────────────────────


class TestExcludeDSL:
    def test_scalar_exclude_matches_one_value(self):
        """Original eval/run_matrix semantics — scalar equality."""
        cfg = _basic_config()
        cfg["exclude"] = [{"saps_mode": "within"}]
        pm, _, tmp = _make_matrix(config=cfg)
        try:
            cells = pm.expand()
            # Drops 1 mode × 2 scales × 2 seeds = 4 cells. 12 - 4 = 8.
            assert len(cells) == 8
            assert all(c["axes"]["saps_mode"] != "within" for c in cells)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_list_exclude_matches_multiple_values(self):
        """New DSL: list as 'in' filter."""
        cfg = _basic_config()
        cfg["exclude"] = [{"saps_mode": ["none", "within"]}]
        pm, _, tmp = _make_matrix(config=cfg)
        try:
            cells = pm.expand()
            # Drops 2 modes × 2 scales × 2 seeds = 8. 12 - 8 = 4.
            assert len(cells) == 4
            assert all(c["axes"]["saps_mode"] == "both" for c in cells)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_combined_fields_use_AND(self):
        """Multiple fields in one exclude entry combine with AND."""
        cfg = _basic_config()
        cfg["exclude"] = [{"saps_mode": "within", "saps_t_scale": 1.0}]
        pm, _, tmp = _make_matrix(config=cfg)
        try:
            cells = pm.expand()
            # Drops only (within, 1.0) × 2 seeds = 2 cells. 12 - 2 = 10.
            assert len(cells) == 10
            for c in cells:
                assert not (
                    c["axes"]["saps_mode"] == "within"
                    and c["axes"]["saps_t_scale"] == 1.0
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_list_combined_with_scalar(self):
        """Mix scalar and list within one exclude — natural for the canonical use case."""
        cfg = _basic_config()
        # "λ-irrelevant" pattern: when mode != both, only keep one t_scale.
        cfg["exclude"] = [
            {"saps_mode": ["none", "within"], "saps_t_scale": [1.0]}
        ]
        pm, _, tmp = _make_matrix(config=cfg)
        try:
            cells = pm.expand()
            # Drops 2 modes × 1 scale × 2 seeds = 4. 12 - 4 = 8.
            assert len(cells) == 8
            for c in cells:
                if c["axes"]["saps_mode"] in ("none", "within"):
                    assert c["axes"]["saps_t_scale"] != 1.0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_multiple_exclude_entries_use_OR(self):
        """Each exclude entry is independent — match any → exclude."""
        cfg = _basic_config()
        cfg["exclude"] = [
            {"saps_mode": "none"},
            {"saps_mode": "within", "saps_t_scale": 0.5},
        ]
        pm, _, tmp = _make_matrix(config=cfg)
        try:
            cells = pm.expand()
            # Drop 'none' entirely (4 cells) + (within, 0.5) × 2 seeds (2 cells)
            # = 6. 12 - 6 = 6.
            assert len(cells) == 6
            for c in cells:
                assert c["axes"]["saps_mode"] != "none"
                assert not (
                    c["axes"]["saps_mode"] == "within"
                    and c["axes"]["saps_t_scale"] == 0.5
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_seed_in_exclude(self):
        cfg = _basic_config()
        cfg["exclude"] = [{"seed": 42}]
        pm, _, tmp = _make_matrix(config=cfg)
        try:
            cells = pm.expand()
            assert all(c["seed"] != 42 for c in cells)
            assert len(cells) == 6  # 3 × 2 × (2-1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unknown_axis_does_not_exclude(self):
        """Defensive: typo in exclude key should not silently drop cells."""
        cfg = _basic_config()
        cfg["exclude"] = [{"saps_typo": "within"}]
        pm, _, tmp = _make_matrix(config=cfg)
        try:
            cells = pm.expand()
            assert len(cells) == 12  # nothing dropped
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_exclude_keeps_all(self):
        pm, _, tmp = _make_matrix()
        try:
            cells = pm.expand()
            assert len(cells) == 12
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# Run / CSV / resume
# ─────────────────────────────────────────────────────────────────────────


class TestRun:
    def test_run_writes_csv_with_correct_schema(self):
        pm, csv_path, tmp = _make_matrix()
        try:
            results = pm.run(verbose=False)
            assert len(results) == 12
            assert os.path.exists(csv_path)

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames == CSV_COLUMNS
                rows = list(reader)
            assert len(rows) == 12

            # axes_json round-trips
            for row in rows:
                axes = json.loads(row["axes_json"])
                assert "saps_mode" in axes
                assert row["status"] == "ok"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_failure_recorded_continue(self):
        pm, csv_path, tmp = _make_matrix(runner=_mode_specific_runner)
        try:
            results = pm.run(on_error="continue", verbose=False)
            statuses = [r["status"] for r in results]
            # 1 of 3 modes fails: (1 × 2 × 2) = 4 failures, 8 ok
            assert statuses.count("failed") == 4
            assert statuses.count("ok") == 8
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_failure_raises_when_configured(self):
        pm, _, tmp = _make_matrix(runner=_fail_runner)
        try:
            with pytest.raises(RuntimeError, match="simulated"):
                pm.run(on_error="raise", verbose=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_invalid_on_error_value(self):
        pm, _, tmp = _make_matrix()
        try:
            with pytest.raises(ValueError, match="on_error"):
                pm.run(on_error="bogus", verbose=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestResume:
    def test_resume_skips_completed_cells(self):
        pm, csv_path, tmp = _make_matrix()
        try:
            r1 = pm.run(verbose=False)
            assert all(r["status"] == "ok" for r in r1)

            # Second run — every cell is already "ok", so all should skip
            r2 = pm.run(resume=True, verbose=False)
            assert all(r["status"] == "skipped" for r in r2)
            assert len(r2) == 12
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resume_reruns_failed(self):
        cfg = _basic_config()
        cfg["grid"] = {"saps_mode": ["within"]}  # single failing axis
        cfg["seeds"] = [42]

        # First run: failing runner
        pm1, csv_path, tmp = _make_matrix(config=cfg, runner=_fail_runner)
        try:
            pm1.run(on_error="continue", verbose=False)

            # Second run: same CSV, succeeding runner
            pm2 = PretrainMatrix(
                config=cfg,
                output_csv=csv_path,
                runners={"pretrain": _ok_runner},
            )
            r2 = pm2.run(resume=True, verbose=False)
            assert len(r2) == 1
            assert r2[0]["status"] == "ok"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resume_disabled_reruns_everything(self):
        pm, _, tmp = _make_matrix()
        try:
            pm.run(verbose=False)
            r2 = pm.run(resume=False, verbose=False)
            assert all(r["status"] == "ok" for r in r2)  # not skipped
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# YAML round-trip — file-based loading should equal in-memory loading
# ─────────────────────────────────────────────────────────────────────────


class TestYamlRoundtrip:
    def test_load_from_yaml_file(self):
        cfg = _basic_config()
        tmp = tempfile.mkdtemp()
        try:
            yaml_path = os.path.join(tmp, "cfg.yaml")
            with open(yaml_path, "w") as f:
                yaml.safe_dump(cfg, f)

            csv_path = os.path.join(tmp, "results.csv")
            pm = PretrainMatrix(
                config_path=yaml_path,
                output_csv=csv_path,
                runners={"pretrain": _ok_runner},
            )
            cells = pm.expand()
            assert len(cells) == 12
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_constructor_requires_config(self):
        with pytest.raises(ValueError, match="config_path or config"):
            PretrainMatrix(output_csv="/tmp/x.csv")


# ─────────────────────────────────────────────────────────────────────────
# _run_pretrain interface — the REAL runner's kwarg routing.
#
# Every other test here uses a mock runner; the orchestration logic is
# covered, but the real _run_pretrain ↔ DenseSSLPretrainer interface —
# how merged YAML keys split into __init__ vs train() vs meta — was never
# exercised. That gap let the output_dir bug ship (plan §10.33): a base
# key that is neither a train kwarg nor an __init__ param was routed into
# init_kwargs, crashing the constructor. These tests stub DenseSSLPretrainer
# so the split is verified without running a real trainer.
# ─────────────────────────────────────────────────────────────────────────


class TestRunPretrainInterface:
    @staticmethod
    def _stub_trainer_module(monkeypatch):
        """Replace DenseSSLPretrainer with a stub that records the kwargs it
        was constructed / trained with. Returns the record dict."""
        rec = {"init_kwargs": None, "train_kwargs": None}

        class _StubPretrainer:
            # __init__ signature MUST mirror the real DenseSSLPretrainer so
            # _run_pretrain's signature-based filtering sees the same params.
            def __init__(self, model=None, weights=None, saps_mode="none",
                         out_dim=128, queue_size=4096, queue_update_strategy="pooled",
                         queue_subsample_n=16, saps_both_lambda=1.0,
                         saps_t_scale=1.0, saps_strict_negatives=False,
                         n_query=128, pos_radius=0.07, match_mode="threshold",
                         momentum=0.99, temperature=0.2, imgsz=640,
                         aug_kwargs=None, device=None, logger=None):
                rec["init_kwargs"] = dict(
                    model=model, saps_mode=saps_mode, out_dim=out_dim,
                    queue_size=queue_size,
                    queue_update_strategy=queue_update_strategy,
                    imgsz=imgsz, device=device,
                )

            def train(self, images_dir=None, epochs=100, batch_size=32,
                      lr=1e-3, warmup_epochs=5, weight_decay=0.0,
                      num_workers=4, output="bb.pt", save_every=25,
                      print_every=10):
                rec["train_kwargs"] = dict(
                    images_dir=images_dir, epochs=epochs,
                    batch_size=batch_size, output=output,
                )
                return output

            def cleanup(self):
                pass

        import yolo_contrastive.pretrain.dense_trainer as dt_mod
        monkeypatch.setattr(dt_mod, "DenseSSLPretrainer", _StubPretrainer)
        return rec

    def test_output_dir_not_routed_to_init(self, monkeypatch):
        """output_dir is an orchestrator meta key — it must NOT reach the
        trainer constructor. This is the exact bug from the Faz 5.1 smoke."""
        from yolo_contrastive.pretrain.run_matrix import _run_pretrain

        rec = self._stub_trainer_module(monkeypatch)
        cell = {
            "cell_id": "abc123",
            "axes": {"saps_mode": "both"},
            "base": {},
        }
        base = {
            "model": "yolov8n.pt", "imgsz": 320, "out_dim": 128,
            "queue_size": 4096, "epochs": 30, "batch_size": 32,
            "images_dir": "/data/pool",
            "output_dir": "/runs/stage1",      # the meta key — must be dropped
        }
        # Pre-fix: this raised TypeError(unexpected kwarg 'output_dir').
        out = _run_pretrain(cell, base)

        assert "output_dir" not in rec["init_kwargs"]
        assert out["backbone_path"] == "/runs/stage1/abc123.pt"

    def test_init_vs_train_kwarg_split(self, monkeypatch):
        """Constructor params land in __init__, train params land in train()."""
        from yolo_contrastive.pretrain.run_matrix import _run_pretrain

        rec = self._stub_trainer_module(monkeypatch)
        cell = {"cell_id": "c1", "axes": {"saps_mode": "within"}, "base": {}}
        base = {
            "model": "yolov8n.pt", "imgsz": 320, "out_dim": 256,
            "epochs": 30, "batch_size": 16, "images_dir": "/data/pool",
        }
        _run_pretrain(cell, base)

        # __init__ side
        assert rec["init_kwargs"]["saps_mode"] == "within"
        assert rec["init_kwargs"]["out_dim"] == 256
        assert rec["init_kwargs"]["imgsz"] == 320
        # train() side
        assert rec["train_kwargs"]["epochs"] == 30
        assert rec["train_kwargs"]["batch_size"] == 16
        assert rec["train_kwargs"]["images_dir"] == "/data/pool"

    def test_axes_override_base(self, monkeypatch):
        """cell['axes'] overrides a same-named base key in the merge."""
        from yolo_contrastive.pretrain.run_matrix import _run_pretrain

        rec = self._stub_trainer_module(monkeypatch)
        cell = {"cell_id": "c2",
                "axes": {"saps_mode": "both", "queue_update_strategy": "per_position"},
                "base": {}}
        base = {
            "model": "yolov8n.pt", "saps_mode": "none",   # axes должен override
            "queue_update_strategy": "pooled",
            "epochs": 10, "images_dir": "/d",
        }
        _run_pretrain(cell, base)
        assert rec["init_kwargs"]["saps_mode"] == "both"
        assert rec["init_kwargs"]["queue_update_strategy"] == "per_position"
