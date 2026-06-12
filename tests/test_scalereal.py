"""End-to-end analytic GASP-Real test: files in, true log-ratios out.

Synthetic pinhole scenes (known-size squares at known metric depths) are
materialized to disk with their EXACT inverse-depth maps in the shared cache
format; the full mine_pool() driver runs over a real manifest + DepthCache +
parquet path with a stub embedder, and every recovered pair label must equal
log(Z_A / Z_B) within 1e-2. Resume idempotency and the mining-stats audit
artifact are covered on the same fixture. CPU-only, offline.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from yolo_contrastive.scalereal.config import ScaleRealConfig
from yolo_contrastive.scalereal.depth_io import DepthCache
from yolo_contrastive.scalereal.mine_pairs import mine_pool
from yolo_contrastive.scalereal.pair_manifest import (
    PairIndex,
    read_pairs,
    validate_pairs,
)
from yolo_contrastive.scalereal.synthetic import materialize_scene, two_class_scene


@pytest.fixture(scope="module")
def mined(tmp_path_factory):
    """Materialize 2 scene images + metric depth cache, run mine_pool once."""
    import pandas as pd

    tmp = tmp_path_factory.mktemp("scalereal_e2e")
    scene = two_class_scene()
    cache = DepthCache(tmp / "cache", variant="dav2_metric_outdoor_small")
    rows = []
    for image_id in ("img_000", "img_001"):  # both outside the probe holdout
        path = materialize_scene(scene, tmp / "images", image_id, depth_cache=cache)
        rows.append({"image_id": image_id, "dataset": "synthetic",
                     "materialized_path": path})
    manifest = pd.DataFrame(rows)
    cfg = ScaleRealConfig()
    out = tmp / "scalereal" / "pairs_v1.parquet"
    stats = mine_pool(manifest, cache, out, scene.make_stub_embedder(), cfg)
    return {"tmp": tmp, "scene": scene, "cache": cache, "manifest": manifest,
            "cfg": cfg, "out": out, "stats": stats}


class TestEndToEnd:
    def test_labels_match_pinhole_geometry(self, mined):
        """THE analytic requirement: mined log_r == log(Z_A/Z_B) +- 1e-2,
        through the full file pipeline (PNG image + uint16 depth cache +
        parquet round trip included in the error budget)."""
        scene = mined["scene"]
        pairs = read_pairs(mined["out"])
        assert len(pairs) >= 4  # both images yielded pairs
        validate_pairs(pairs)
        for _, row in pairs.iterrows():
            box_a = np.array([row["box_a_x1"], row["box_a_y1"],
                              row["box_a_x2"], row["box_a_y2"]])
            box_b = np.array([row["box_b_x1"], row["box_b_y1"],
                              row["box_b_x2"], row["box_b_y2"]])
            ia = scene.square_index_for_box(box_a)
            ib = scene.square_index_for_box(box_b)
            assert ia is not None and ib is not None
            truth = math.log(scene.squares[ia].z_m / scene.squares[ib].z_m)
            assert row["log_r"] == pytest.approx(truth, abs=1e-2), (ia, ib)

    def test_band_and_budget_respected(self, mined):
        cfg = mined["cfg"]
        pairs = read_pairs(mined["out"])
        abs_lr = pairs["log_r"].abs()
        assert (abs_lr >= cfg.log_ratio_min - 1e-9).all()
        assert (abs_lr <= cfg.log_ratio_max + 1e-9).all()
        assert (pairs.groupby("image_id").size() <= cfg.max_pairs_per_image).all()

    def test_mining_stats_written(self, mined):
        stats_path = mined["out"].parent / "mining_stats.json"
        assert stats_path.exists()
        d = json.loads(stats_path.read_text(encoding="utf-8"))
        assert d["counters"]["images_processed"] == 2
        assert d["counters"]["pairs_written"] == len(read_pairs(mined["out"]))
        assert d["image_yield"] == 1.0
        assert sum(d["log_r_hist"]) == d["counters"]["pairs_written"]
        assert d["per_source"]["synthetic"]["images"] == 2

    def test_resume_is_idempotent(self, mined):
        """Re-running mine_pool appends nothing (image_id set-difference)."""
        n_before = len(read_pairs(mined["out"]))
        stats2 = mine_pool(
            mined["manifest"], mined["cache"], mined["out"],
            mined["scene"].make_stub_embedder(), mined["cfg"],
        )
        assert stats2.counters["images_already_mined"] == 2
        assert stats2.counters["pairs_written"] == 0
        assert len(read_pairs(mined["out"])) == n_before

    def test_pair_index_serves_dataset_hook(self, mined):
        cfg = mined["cfg"]
        index = PairIndex.from_parquet(mined["out"])
        assert set(index.image_ids()) == {"img_000", "img_001"}
        t = index.prepare_targets("img_000")
        assert t is not None
        m = len(t["log_r"])
        assert 2 <= m <= cfg.max_pairs_per_image
        assert t["boxes_a"].shape == (m, 4)
        assert index.prepare_targets("missing") is None
        # both fixture ids are training-eligible (not probe holdout)
        assert set(index.eligible_image_ids(
            min_pairs=cfg.min_pairs_per_image,
            probe_fraction=cfg.probe_fraction,
        )) == {"img_000", "img_001"}

    def test_skip_images_without_depth(self, mined):
        """Images missing from the depth cache are counted and skipped."""
        import pandas as pd

        extra = pd.concat([
            mined["manifest"],
            pd.DataFrame([{"image_id": "img_nodepth", "dataset": "synthetic",
                           "materialized_path":
                               str(mined["tmp"] / "images" / "img_000.png")}]),
        ], ignore_index=True)
        stats = mine_pool(extra, mined["cache"], mined["out"],
                          mined["scene"].make_stub_embedder(), mined["cfg"])
        assert stats.counters["images_no_depth"] == 1
        assert stats.counters["pairs_written"] == 0

    def test_pano_and_source_exclusion(self, mined):
        import pandas as pd

        cfg = ScaleRealConfig(exclude_sources=("synthetic",))
        stats = mine_pool(mined["manifest"], mined["cache"],
                          mined["tmp"] / "excluded.parquet",
                          mined["scene"].make_stub_embedder(), cfg)
        assert stats.counters["images_skipped_source"] == 2

        pano = mined["manifest"].copy()
        pano["is_pano"] = True
        stats2 = mine_pool(pano, mined["cache"],
                           mined["tmp"] / "pano.parquet",
                           mined["scene"].make_stub_embedder(), mined["cfg"])
        assert stats2.counters["images_skipped_pano"] == 2
        assert isinstance(pano, pd.DataFrame)

    def test_plane_gate_skips_inconsistent_images(self, mined):
        """TERRA trust gate: gate_fn False -> image skipped (graceful hook)."""
        stats = mine_pool(
            mined["manifest"], mined["cache"], mined["tmp"] / "gated.parquet",
            mined["scene"].make_stub_embedder(), mined["cfg"],
            plane_gate_fn=lambda image_id: image_id != "img_000",
        )
        assert stats.counters["images_failed_plane_gate"] == 1
        pairs = read_pairs(mined["tmp"] / "gated.parquet")
        assert set(pairs["image_id"]) == {"img_001"}
