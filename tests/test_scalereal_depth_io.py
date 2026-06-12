"""Depth-cache I/O tests — uint16 PNG round-trip + the affine-ambiguity guard.

CPU-only, offline, synthetic numpy data. The soundness gate (log_depth_ratio
refusing relative-variant caches) is itself under test: the affine ambiguity
is a hard runtime error, not a docs footnote.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from yolo_contrastive.scalereal.depth_io import (
    DepthCache,
    central_sub_box,
    decode_inverse_depth,
    encode_inverse_depth,
    log_depth_ratio,
    normalize_sidecar,
    patch_depth_stats,
    require_metric_sidecar,
)

METRIC_SIDECAR = {
    "units": "1/m",
    "encoding": "inverse_depth",
    "variant": "dav2_metric_outdoor_small",
}
AFFINE_SIDECAR = {
    "units": "affine",
    "encoding": "inverse_depth",
    "variant": "dav2_rel_base",
}


# ── encode/decode + on-disk round trip ───────────────────────────────────────


class TestRoundTrip:
    def test_quantization_roundtrip_no_files(self):
        rng = np.random.default_rng(0)
        inv = rng.uniform(1.0 / 60.0, 0.5, size=(48, 64)).astype(np.float32)
        png, v_min, v_max = encode_inverse_depth(inv)
        back = decode_inverse_depth(png, v_min, v_max)
        rel = np.abs(back - inv) / inv
        assert float(rel.max()) < 1e-3

    def test_cache_roundtrip_with_sidecar(self, tmp_path):
        cache = DepthCache(tmp_path, variant="dav2_metric_outdoor_small")
        rng = np.random.default_rng(1)
        inv = rng.uniform(1.0 / 60.0, 0.5, size=(40, 56)).astype(np.float32)
        cache.save("city/img_000", inv)
        assert cache.has("city/img_000")
        assert "city/img_000" in cache

        back = cache.read("city/img_000")
        assert back.dtype == np.float32
        assert back.shape == inv.shape
        rel = np.abs(back - inv) / inv
        assert float(rel.max()) < 1e-3

        sidecar = cache.read_sidecar("city/img_000")
        assert sidecar["units"] == "1/m"
        assert sidecar["encoding"] == "inverse_depth"
        assert sidecar["variant"] == "dav2_metric_outdoor_small"
        assert "v_min" in sidecar and "v_max" in sidecar
        assert sidecar["cache_version"] == 1

    def test_slash_preserving_ids_and_len(self, tmp_path):
        cache = DepthCache(tmp_path)
        inv = np.full((8, 8), 0.1, dtype=np.float32)
        cache.save("mapillary/seq1/a", inv)
        cache.save("plain_id", inv)
        assert (cache.cache_dir / "mapillary" / "seq1" / "a.png").exists()
        assert len(cache) == 2

    def test_metric_save_clips_to_band(self, tmp_path):
        cache = DepthCache(tmp_path, variant="dav2_metric_outdoor_small")
        inv = np.array([[1.0 / 200.0, 5.0]], dtype=np.float32)  # Z=200m, Z=0.2m
        cache.save("clipme", inv)
        back = cache.read("clipme")
        assert back.min() >= 1.0 / 80.0 - 1e-6
        assert back.max() <= 1.0 / 0.5 + 1e-6

    def test_missing_raises(self, tmp_path):
        cache = DepthCache(tmp_path)
        with pytest.raises(FileNotFoundError):
            cache.read("nope")

    def test_unknown_variant_needs_explicit_units(self, tmp_path):
        cache = DepthCache(tmp_path, variant="custom_variant")
        inv = np.full((4, 4), 0.2, dtype=np.float32)
        with pytest.raises(ValueError, match="units"):
            cache.save("x", inv)
        cache.save("x", inv, units="1/m")  # explicit units OK
        assert cache.read_sidecar("x")["units"] == "1/m"


# ── the soundness gate ───────────────────────────────────────────────────────


class TestLogDepthRatio:
    def test_analytic_value(self):
        # Z_A = 5 m, Z_B = 20 m -> B appears 4x smaller, log_r = -1.386
        lr = log_depth_ratio(1.0 / 5.0, 1.0 / 20.0, METRIC_SIDECAR)
        assert lr == pytest.approx(math.log(5.0 / 20.0), abs=1e-9)
        assert lr == pytest.approx(-1.386, abs=1e-3)

    def test_antisymmetric(self):
        a, b = 1.0 / 3.0, 1.0 / 11.0
        assert log_depth_ratio(a, b, METRIC_SIDECAR) == pytest.approx(
            -log_depth_ratio(b, a, METRIC_SIDECAR), abs=1e-12
        )

    def test_refuses_relative_variant(self):
        """The affine-ambiguity guard IS the test: relative-cache ratios are
        mathematically invalid and must be a hard error."""
        with pytest.raises(ValueError, match="affine|metric|invalid"):
            log_depth_ratio(0.2, 0.05, AFFINE_SIDECAR)

    def test_refuses_missing_units(self):
        with pytest.raises(ValueError):
            log_depth_ratio(0.2, 0.05, {"encoding": "inverse_depth"})

    def test_refuses_bad_encoding(self):
        with pytest.raises(ValueError, match="encoding"):
            require_metric_sidecar({"units": "1/m", "encoding": "depth"})

    def test_refuses_nonpositive(self):
        with pytest.raises(ValueError, match="positive"):
            log_depth_ratio(0.0, 0.1, METRIC_SIDECAR)

    def test_cache_roundtrip_label_error_budget(self, tmp_path):
        """End-to-end: quantization error in the cache stays << the 1e-2
        label budget used by the geometry tests."""
        cache = DepthCache(tmp_path)
        inv = np.full((16, 16), 1.0 / 7.0, dtype=np.float32)
        inv[8:, :] = 1.0 / 21.0
        cache.save("img", inv)
        back = cache.read("img")
        sidecar = cache.read_sidecar("img")
        lr = log_depth_ratio(
            float(np.median(back[:8, :])), float(np.median(back[8:, :])), sidecar
        )
        # A: Z=7 m (top), B: Z=21 m (bottom) -> log_r = log(Z_A/Z_B) = -log 3
        assert lr == pytest.approx(math.log(7.0 / 21.0), abs=1e-3)


# ── geoteach-dialect compatibility ───────────────────────────────────────────


class TestGeoteachDialect:
    """The geoteach/depth_cache.py writer uses d_min/d_max + metric flags;
    every scalereal reader must accept that dialect and the metric guard
    must still fire correctly on its relative caches."""

    GEOTEACH_METRIC = {"d_min": 0.0125, "d_max": 2.0, "metric": True,
                       "depth_unit": "meters", "max_depth": 80.0}
    GEOTEACH_RELATIVE = {"d_min": 0.01, "d_max": 0.9}

    def test_normalize_maps_keys_and_units(self):
        n = normalize_sidecar(self.GEOTEACH_METRIC)
        assert n["v_min"] == 0.0125 and n["v_max"] == 2.0
        assert n["units"] == "1/m"
        assert n["encoding"] == "inverse_depth"
        n2 = normalize_sidecar(self.GEOTEACH_RELATIVE)
        assert n2["units"] == "affine"

    def test_log_depth_ratio_accepts_geoteach_metric(self):
        lr = log_depth_ratio(1.0 / 5.0, 1.0 / 20.0, self.GEOTEACH_METRIC)
        assert lr == pytest.approx(math.log(5.0 / 20.0), abs=1e-9)

    def test_log_depth_ratio_refuses_geoteach_relative(self):
        with pytest.raises(ValueError, match="metric"):
            log_depth_ratio(0.2, 0.05, self.GEOTEACH_RELATIVE)

    def test_read_geoteach_layout_cache(self, tmp_path):
        """Round-trip through geoteach's flat {root}/{tag}/ layout + dialect
        sidecar, read via DepthCache(subdir='')."""
        import json

        writer = DepthCache(tmp_path, variant="dav2_metric", subdir="",
                            model_tag="geoteach")
        rng = np.random.default_rng(5)
        inv = rng.uniform(1.0 / 60.0, 0.5, size=(24, 24)).astype(np.float32)
        writer.save("seq/img", inv, units="1/m")
        # rewrite the sidecar in the geoteach dialect
        sc_path = writer.cache_dir / "seq" / "img.json"
        spec = json.loads(sc_path.read_text(encoding="utf-8"))
        geoteach = {"d_min": spec["v_min"], "d_max": spec["v_max"],
                    "metric": True, "depth_unit": "meters",
                    "cache_h": 24, "cache_w": 24}
        sc_path.write_text(json.dumps(geoteach), encoding="utf-8")

        reader = DepthCache(tmp_path, variant="dav2_metric", subdir="")
        back = reader.read("seq/img")
        assert float((np.abs(back - inv) / inv).max()) < 1e-3
        sidecar = reader.read_sidecar("seq/img")
        require_metric_sidecar(sidecar)  # guard passes on the metric dialect


# ── patch statistics vs numpy reference ──────────────────────────────────────


class TestPatchStats:
    def test_median_and_iqr_match_numpy(self):
        rng = np.random.default_rng(2)
        inv = rng.uniform(0.05, 0.5, size=(32, 32)).astype(np.float32)
        box = np.array([0.25, 0.25, 0.75, 0.75])
        stats = patch_depth_stats(inv, box, central_fraction=1.0)
        region = inv[8:24, 8:24].reshape(-1)
        assert stats["median_inv"] == pytest.approx(float(np.median(region)), rel=1e-6)
        q25, q75 = np.percentile(region, [25, 75])
        assert stats["iqr_ratio"] == pytest.approx(
            float((q75 - q25) / np.median(region)), rel=1e-5
        )
        assert stats["z"] == pytest.approx(1.0 / float(np.median(region)), rel=1e-6)
        assert stats["n_px"] == 16 * 16

    def test_central_fraction_ignores_border(self):
        # border ring at 0.9, central core at 0.2: fraction 0.5 sees only core
        inv = np.full((32, 32), 0.9, dtype=np.float32)
        inv[12:20, 12:20] = 0.2
        box = np.array([8 / 32, 8 / 32, 24 / 32, 24 / 32])
        stats = patch_depth_stats(inv, box, central_fraction=0.5)
        assert stats["median_inv"] == pytest.approx(0.2, abs=1e-6)
        assert stats["iqr_ratio"] == pytest.approx(0.0, abs=1e-6)

    def test_central_sub_box_geometry(self):
        sub = central_sub_box(np.array([0.2, 0.4, 0.6, 0.8]), fraction=0.5)
        assert sub == pytest.approx([0.3, 0.5, 0.5, 0.7], abs=1e-9)
