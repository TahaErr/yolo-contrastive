"""Tests for eval/leakage_check.py — cross-set leakage check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _write_image(path, color, size=64):
    """Write a solid-color PNG. Same color → same pHash."""
    import cv2
    img = np.full((size, size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _gradient_image(path, size=64, seed=0):
    """Write a deterministic textured image (distinct pHash per seed)."""
    import cv2
    rng = np.random.RandomState(seed)
    img = (rng.rand(size, size, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def _pool_parquet(tmp_path, phashes: dict):
    """Write a pool pHash parquet sidecar."""
    path = tmp_path / "pool_phash.parquet"
    pd.DataFrame(
        {"image_id": list(phashes.keys()), "phash": list(phashes.values())}
    ).to_parquet(path, index=False)
    return str(path)


# ═════════════════════════════════════════════════════════════════════════
# hash_image_dir
# ═════════════════════════════════════════════════════════════════════════


class TestHashImageDir:
    def test_hashes_all_images(self, tmp_path):
        from yolo_contrastive.eval.leakage_check import hash_image_dir

        d = tmp_path / "imgs"
        d.mkdir()
        for i in range(3):
            _gradient_image(d / f"img_{i}.png", seed=i)
        hashes = hash_image_dir(str(d))
        assert len(hashes) == 3
        # image_id is extension-free
        for iid in hashes:
            assert not iid.endswith(".png")

    def test_recursive_and_relative_id(self, tmp_path):
        from yolo_contrastive.eval.leakage_check import hash_image_dir

        d = tmp_path / "imgs"
        (d / "sub").mkdir(parents=True)
        _gradient_image(d / "a.png", seed=1)
        _gradient_image(d / "sub" / "b.png", seed=2)
        hashes = hash_image_dir(str(d))
        assert set(hashes.keys()) == {"a", "sub/b"}

    def test_missing_dir_raises(self):
        from yolo_contrastive.eval.leakage_check import hash_image_dir

        with pytest.raises(FileNotFoundError):
            hash_image_dir("/nonexistent/path/xyz")


# ═════════════════════════════════════════════════════════════════════════
# find_leakage
# ═════════════════════════════════════════════════════════════════════════


class TestFindLeakage:
    def test_exact_match(self):
        from yolo_contrastive.eval.leakage_check import find_leakage

        pool = {"pool/a": "abcd1234", "pool/b": "ffff0000"}
        downstream = {"eval/x": "abcd1234"}   # collides with pool/a
        pairs = find_leakage(pool, downstream, hamming_threshold=0)
        assert pairs == [("pool/a", "eval/x", 0)]

    def test_no_leakage(self):
        from yolo_contrastive.eval.leakage_check import find_leakage

        pool = {"pool/a": "1111"}
        downstream = {"eval/x": "2222"}
        assert find_leakage(pool, downstream, hamming_threshold=0) == []

    def test_near_duplicate_within_threshold(self):
        from yolo_contrastive.eval.leakage_check import find_leakage

        # 0x...0 vs 0x...1 — 1 bit apart
        pool = {"pool/a": "0000000000000000"}
        downstream = {"eval/x": "0000000000000001"}
        # exact → no match
        assert find_leakage(pool, downstream, hamming_threshold=0) == []
        # threshold 5 → match, dist 1
        pairs = find_leakage(pool, downstream, hamming_threshold=5)
        assert pairs == [("pool/a", "eval/x", 1)]

    def test_near_duplicate_beyond_threshold_excluded(self):
        from yolo_contrastive.eval.leakage_check import find_leakage

        # 8 bits apart
        pool = {"pool/a": "ff00000000000000"}
        downstream = {"eval/x": "0000000000000000"}
        assert find_leakage(pool, downstream, hamming_threshold=5) == []


# ═════════════════════════════════════════════════════════════════════════
# run_leakage_check
# ═════════════════════════════════════════════════════════════════════════


class TestRunLeakageCheck:
    def test_clean_no_leakage(self, tmp_path):
        from yolo_contrastive.eval.leakage_check import run_leakage_check, hash_image_dir

        # Pool images
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        for i in range(3):
            _gradient_image(pool_dir / f"p_{i}.png", seed=100 + i)
        pool_hashes = hash_image_dir(str(pool_dir))
        pool_parquet = _pool_parquet(tmp_path, pool_hashes)

        # Downstream — entirely different images
        ds_dir = tmp_path / "downstream"
        ds_dir.mkdir()
        for i in range(2):
            _gradient_image(ds_dir / f"d_{i}.png", seed=500 + i)

        report = run_leakage_check(pool_parquet, [str(ds_dir)])
        assert report["pool_size"] == 3
        assert report["total_leaking_pairs"] == 0
        assert report["leakage_rate"] == 0.0
        assert report["alert"] is False

    def test_detects_leakage(self, tmp_path):
        from yolo_contrastive.eval.leakage_check import run_leakage_check, hash_image_dir

        # Pool images
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        for i in range(4):
            _gradient_image(pool_dir / f"p_{i}.png", seed=i)
        pool_hashes = hash_image_dir(str(pool_dir))
        pool_parquet = _pool_parquet(tmp_path, pool_hashes)

        # Downstream — one image is a COPY of pool p_0 (same seed → same pHash)
        ds_dir = tmp_path / "downstream"
        ds_dir.mkdir()
        _gradient_image(ds_dir / "leaked.png", seed=0)     # == p_0
        _gradient_image(ds_dir / "clean.png", seed=999)

        report = run_leakage_check(pool_parquet, [str(ds_dir)])
        assert report["total_leaking_pairs"] >= 1
        assert len(report["leaking_pool_ids"]) >= 1
        assert report["leakage_rate"] > 0

    def test_leakage_rate_and_alert(self, tmp_path):
        from yolo_contrastive.eval.leakage_check import run_leakage_check, hash_image_dir

        # Small pool of 2 — one leaks → 50% rate → alert
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        _gradient_image(pool_dir / "p_0.png", seed=0)
        _gradient_image(pool_dir / "p_1.png", seed=1)
        pool_parquet = _pool_parquet(tmp_path, hash_image_dir(str(pool_dir)))

        ds_dir = tmp_path / "downstream"
        ds_dir.mkdir()
        _gradient_image(ds_dir / "leaked.png", seed=0)     # == p_0

        report = run_leakage_check(pool_parquet, [str(ds_dir)])
        assert report["leakage_rate"] == 0.5
        assert report["alert"] is True

    def test_output_file_written(self, tmp_path):
        from yolo_contrastive.eval.leakage_check import run_leakage_check, hash_image_dir

        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        _gradient_image(pool_dir / "p_0.png", seed=0)
        pool_parquet = _pool_parquet(tmp_path, hash_image_dir(str(pool_dir)))

        ds_dir = tmp_path / "downstream"
        ds_dir.mkdir()
        _gradient_image(ds_dir / "leaked.png", seed=0)

        out = tmp_path / "leaking.txt"
        run_leakage_check(pool_parquet, [str(ds_dir)], output=str(out))
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert len(lines) >= 1

    def test_multiple_downstream_dirs(self, tmp_path):
        from yolo_contrastive.eval.leakage_check import run_leakage_check, hash_image_dir

        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        for i in range(3):
            _gradient_image(pool_dir / f"p_{i}.png", seed=i)
        pool_parquet = _pool_parquet(tmp_path, hash_image_dir(str(pool_dir)))

        ds1 = tmp_path / "ds1"
        ds1.mkdir()
        _gradient_image(ds1 / "a.png", seed=900)
        ds2 = tmp_path / "ds2"
        ds2.mkdir()
        _gradient_image(ds2 / "b.png", seed=901)

        report = run_leakage_check(pool_parquet, [str(ds1), str(ds2)])
        assert set(report["downstream"].keys()) == {str(ds1), str(ds2)}
        assert all(d["n_images"] == 1 for d in report["downstream"].values())


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════


class TestCLI:
    def test_parser_builds(self):
        from yolo_contrastive.eval.leakage_check import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--pool-phash", "p.parquet",
            "--downstream", "d1", "d2",
            "--hamming-threshold", "5",
        ])
        assert args.pool_phash == "p.parquet"
        assert args.downstream == ["d1", "d2"]
        assert args.hamming_threshold == 5

    def test_main_runs(self, tmp_path, capsys):
        from yolo_contrastive.eval.leakage_check import main, hash_image_dir

        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        _gradient_image(pool_dir / "p_0.png", seed=0)
        pool_parquet = _pool_parquet(tmp_path, hash_image_dir(str(pool_dir)))

        ds_dir = tmp_path / "downstream"
        ds_dir.mkdir()
        _gradient_image(ds_dir / "clean.png", seed=777)

        rc = main([
            "--pool-phash", pool_parquet,
            "--downstream", str(ds_dir),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "leakage rate" in captured.out.lower()
