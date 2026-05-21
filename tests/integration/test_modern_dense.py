"""Hat A — Modern Dense Hat integration smoke tests.

Covers 17 scenarios from INVENTORY.md §2.1:
    A1-A4:    DenseSSLPretrainer construction + 4 saps modes + queue strategies + match modes
    A5-A11:   dense/ primitives (MultiScaleFeatureTap, FeatureQueue+combine, MomentumEncoder,
              SpatialTwoView, multi_scale_dense_loss, MultiScaleProjectionHead, saps_*)
    A12:      FinetuneDetectionTrainer end-to-end (Risk 16 v2 invariant, real YOLO)
    A13:      LinearProbeTrainer fit + early stopping (mock encoder)
    A14-A15:  eval/RunMatrix linear_probe + _run_detection runners
    A16-A17:  pretrain/PretrainMatrix orchestrator (2-cell + list-DSL exclude)

Integration scope:
    A1-A11 use a 23-layer mock YOLO encoder (same as tests/test_dense_*) for
    speed — dense/ primitives don't care whether the encoder is "real". This
    keeps these tests at ~1-2s each.

    A12, A15 are heavy — real YOLOv8n + 1-epoch finetune on a tiny dataset.
    These are the only integration tests that exercise the full
    SSL-backbone → finetune → mAP path. Risk 16 v2 fix invariants are
    validated here (head_norm > 0 after EMA sync).

    A16 uses mock encoder via PretrainMatrix's runner mechanism — we
    inject a tiny init_kwarg set so DenseSSLPretrainer build with a
    string model spec ('yolov8n.pt') is the heaviest part.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
# Mock 23-layer YOLO encoder (shared by A1-A11)
# ─────────────────────────────────────────────────────────────────────────


def _mock_yolo_encoder(channels: tuple = (32, 64, 128)) -> nn.Module:
    """23-layer Sequential mimicking YOLOv8 backbone+neck.

    Strides happen at indices 0/6/12/16/19. P3 at layer 15, P4 at 18, P5 at 21.
    Used wherever DenseSSL primitives need a callable but the actual encoder
    weights don't matter (most of Hat A).
    """
    p3, p4, p5 = channels
    layers = []
    layers.append(nn.Conv2d(3, p3, 3, stride=2, padding=1))  # 0: /2
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, 3, padding=1))       # 1-5
    layers.append(nn.Conv2d(p3, p3, 3, stride=2, padding=1)) # 6: /4
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, 3, padding=1))       # 7-11
    layers.append(nn.Conv2d(p3, p3, 3, stride=2, padding=1)) # 12: /8
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))           # 13
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))           # 14
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))           # 15: P3 out
    layers.append(nn.Conv2d(p3, p4, 3, stride=2, padding=1)) # 16: /16
    layers.append(nn.Conv2d(p4, p4, 3, padding=1))           # 17
    layers.append(nn.Conv2d(p4, p4, 3, padding=1))           # 18: P4 out
    layers.append(nn.Conv2d(p4, p5, 3, stride=2, padding=1)) # 19: /32
    layers.append(nn.Conv2d(p5, p5, 3, padding=1))           # 20
    layers.append(nn.Conv2d(p5, p5, 3, padding=1))           # 21: P5 out
    layers.append(nn.Conv2d(p5, p5, 1))                      # 22: dummy "Detect"
    return nn.Sequential(*layers)


def _make_dense_trainer(imgsz: int = 64, **overrides):
    """Quick factory mirroring tests/test_dense_ssl_pretrainer.py defaults."""
    from yolo_contrastive.pretrain import DenseSSLPretrainer

    kwargs = dict(
        model=_mock_yolo_encoder(),
        out_dim=16, queue_size=64, n_query=16,
        momentum=0.9, temperature=0.2, pos_radius=0.1,
        match_mode="threshold",
        imgsz=imgsz, device="cpu",
    )
    kwargs.update(overrides)
    return DenseSSLPretrainer(**kwargs)


# ═════════════════════════════════════════════════════════════════════════
# A1 — DenseSSLPretrainer construction (4 saps_mode)
# ═════════════════════════════════════════════════════════════════════════


class TestA1_DenseSSLConstruction:
    """All 4 saps modes construct successfully; state matches mode."""

    @pytest.mark.parametrize("mode", ["none", "within", "cross", "both"])
    def test_construction_per_mode(self, mode):
        tr = _make_dense_trainer(saps_mode=mode)
        try:
            assert tr.saps_mode == mode
            # within/cross/both → tagged queues
            if mode in ("cross", "both"):
                assert tr._needs_tagged_queues is True
                assert all(q.with_tags for q in tr.queues.values())
            else:  # none, within
                assert tr._needs_tagged_queues is False
            # Common state
            assert tr.out_dim == 16
            assert tr.queue_size == 64
            assert set(tr.queues.keys()) == {"P3", "P4", "P5"}
            assert tr.proj_online is not None
            assert tr.proj_momentum is not None
        finally:
            tr.cleanup()

    def test_invalid_saps_mode_raises(self):
        from yolo_contrastive.pretrain import DenseSSLPretrainer
        with pytest.raises(ValueError, match="saps_mode"):
            DenseSSLPretrainer(model=_mock_yolo_encoder(), saps_mode="bogus", device="cpu")


# ═════════════════════════════════════════════════════════════════════════
# A2 — DenseSSL train() 1 epoch for all 4 saps modes
# ═════════════════════════════════════════════════════════════════════════


class TestA2_DenseSSLTrainSingleEpoch:
    """train() completes a 1-epoch run in each saps mode and writes a .pt."""

    @pytest.mark.parametrize("mode", ["none", "within", "cross", "both"])
    def test_train_1epoch_per_mode(self, mode, dummy_images_dir, tmp_workspace):
        img_dir = dummy_images_dir(n=4, size=64, name=f"ssl_{mode}")
        out_path = tmp_workspace / f"backbone_{mode}.pt"

        tr = _make_dense_trainer(saps_mode=mode)
        try:
            result = tr.train(
                images_dir=str(img_dir),
                epochs=1, batch_size=2, lr=1e-3,
                warmup_epochs=0, num_workers=0,
                output=str(out_path),
                save_every=0, print_every=1,
            )
            assert result == str(out_path)
            assert out_path.exists()
            # Verify checkpoint loadable + schema matches save_backbone():
            #   {epoch, extra={loss, type='dense_ssl'}, model_state_dict, type='ssl_pretrained'}
            ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt
            assert ckpt.get("extra", {}).get("type") == "dense_ssl"
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# A3 — queue_update_strategy 3 modes (pooled/per_position/subsample)
# ═════════════════════════════════════════════════════════════════════════


class TestA3_QueueUpdateStrategy:
    """All 3 strategies enqueue the expected count per level per step."""

    def test_pooled_pushes_B_per_level(self):
        B = 2
        tr = _make_dense_trainer(queue_update_strategy="pooled")
        try:
            before = {lv: len(q) for lv, q in tr.queues.items()}
            _ = tr._step(torch.rand(B, 3, 64, 64))
            for lv, q in tr.queues.items():
                assert len(q) - before[lv] == B, (
                    f"pooled level {lv}: expected B={B}, got {len(q) - before[lv]}"
                )
        finally:
            tr.cleanup()

    def test_per_position_pushes_BHW(self):
        B = 2
        tr = _make_dense_trainer(
            queue_update_strategy="per_position", queue_size=100_000,
        )
        try:
            before = {lv: len(q) for lv, q in tr.queues.items()}
            _ = tr._step(torch.rand(B, 3, 64, 64))
            # Mock encoder at imgsz=64: P3=8x8, P4=4x4, P5=2x2 (strides 8/16/32)
            expected = {"P3": B * 8 * 8, "P4": B * 4 * 4, "P5": B * 2 * 2}
            for lv, q in tr.queues.items():
                assert len(q) - before[lv] == expected[lv]
        finally:
            tr.cleanup()

    def test_subsample_pushes_Bn(self):
        B = 2
        n = 4
        tr = _make_dense_trainer(
            queue_update_strategy="subsample",
            queue_subsample_n=n, queue_size=100_000,
        )
        try:
            before = {lv: len(q) for lv, q in tr.queues.items()}
            _ = tr._step(torch.rand(B, 3, 64, 64))
            for lv, q in tr.queues.items():
                # P5 HW=4 caps subsample at HW, but n=4 ≤ 4 so equals B*n
                added = len(q) - before[lv]
                assert added == B * n, (
                    f"subsample level {lv}: expected {B*n}, got {added}"
                )
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# A4 — match_mode 2 modes (threshold/nearest)
# ═════════════════════════════════════════════════════════════════════════


class TestA4_MatchMode:
    """Both match modes produce finite loss + valid gradient flow."""

    @pytest.mark.parametrize("mode", ["threshold", "nearest"])
    def test_match_mode_produces_finite_loss(self, mode):
        tr = _make_dense_trainer(match_mode=mode)
        try:
            out = tr._step(torch.rand(2, 3, 64, 64))
            assert torch.isfinite(out["loss"]).item()
            out["loss"].backward()
            # Online model must have gradients
            grads = [p.grad for p in tr.model.parameters() if p.grad is not None]
            assert any(g.abs().sum() > 0 for g in grads)
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# A5 — MultiScaleFeatureTap construction + forward hook
# ═════════════════════════════════════════════════════════════════════════


class TestA5_MultiScaleFeatureTap:
    """Tap installs at P3/P4/P5 layers, captures features via forward hook."""

    def test_tap_setup_and_capture(self):
        from yolo_contrastive.dense import MultiScaleFeatureTap

        enc = _mock_yolo_encoder()
        tap = MultiScaleFeatureTap(enc)
        tap.setup()
        try:
            _ = enc(torch.rand(2, 3, 64, 64))
            feats = tap.get_features()
            # All 3 levels populated
            for lv in ("P3", "P4", "P5"):
                assert feats[lv] is not None
                assert feats[lv].dim() == 4  # [B, C, H, W]
                assert feats[lv].shape[0] == 2  # batch dim
            # Channel counts from mock: P3=32, P4=64, P5=128
            assert feats["P3"].shape[1] == 32
            assert feats["P4"].shape[1] == 64
            assert feats["P5"].shape[1] == 128
        finally:
            tap.close()


# ═════════════════════════════════════════════════════════════════════════
# A6 — FeatureQueue + combine_queues
# ═════════════════════════════════════════════════════════════════════════


class TestA6_FeatureQueueCombine:
    """Untagged enqueue/get; tagged enqueue + combine_queues across levels."""

    def test_basic_enqueue_get(self):
        from yolo_contrastive.dense import FeatureQueue

        q = FeatureQueue(dim=8, K=16)
        keys = torch.randn(5, 8)
        q.enqueue(keys)
        assert len(q) == 5
        out = q.get()
        assert torch.allclose(out, keys)

    def test_combine_queues_with_tags(self):
        from yolo_contrastive.dense import FeatureQueue, combine_queues

        q3 = FeatureQueue(dim=8, K=16, with_tags=True)
        q4 = FeatureQueue(dim=8, K=16, with_tags=True)
        q5 = FeatureQueue(dim=8, K=16, with_tags=True)
        q3.enqueue(torch.full((3, 8), 3.0), tags=torch.zeros(3, dtype=torch.long))
        q4.enqueue(torch.full((4, 8), 4.0), tags=torch.ones(4, dtype=torch.long))
        q5.enqueue(torch.full((2, 8), 5.0), tags=torch.full((2,), 2, dtype=torch.long))

        keys, tags = combine_queues(
            {"P3": q3, "P4": q4, "P5": q5},
            level_to_id={"P3": 0, "P4": 1, "P5": 2},
        )
        assert keys.shape == (9, 8)
        assert tags.shape == (9,)
        assert set(tags.unique().tolist()) == {0, 1, 2}

    def test_ring_buffer_eviction(self):
        from yolo_contrastive.dense import FeatureQueue
        q = FeatureQueue(dim=4, K=4)
        q.enqueue(torch.full((6, 4), 1.0))  # B > K
        # Only last K entries remain
        assert q.is_full
        assert len(q) == 4


# ═════════════════════════════════════════════════════════════════════════
# A7 — MomentumEncoder EMA update
# ═════════════════════════════════════════════════════════════════════════


class TestA7_MomentumEncoderEMA:
    """EMA update follows θ_m ← m·θ_m + (1-m)·θ_online exactly."""

    def test_m_half_midpoint(self):
        from yolo_contrastive.dense import MomentumEncoder

        online = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        me = MomentumEncoder(online, m=0.5)
        m_init = [p.clone() for p in me.momentum.parameters()]

        # Set online params to a known value
        for p in online.parameters():
            p.data.fill_(10.0)

        me.update(online)
        for p_m, p_initial in zip(me.momentum.parameters(), m_init):
            expected = 0.5 * p_initial + 0.5 * 10.0
            assert torch.allclose(p_m, expected, atol=1e-6)

    def test_m_one_no_change(self):
        from yolo_contrastive.dense import MomentumEncoder

        online = nn.Sequential(nn.Linear(4, 4))
        me = MomentumEncoder(online, m=1.0)
        snapshot = [p.clone() for p in me.momentum.parameters()]
        for p in online.parameters():
            p.data.fill_(99.0)
        me.update(online)
        # m=1.0 → momentum stays unchanged
        for p_now, p_before in zip(me.momentum.parameters(), snapshot):
            assert torch.equal(p_now, p_before)


# ═════════════════════════════════════════════════════════════════════════
# A8 — SpatialTwoViewAugmentation coord tracking
# ═════════════════════════════════════════════════════════════════════════


class TestA8_SpatialTwoView:
    """Two views with independent coords, all in [0, 1]; named-tuple shape."""

    def test_two_view_shape_and_coords_in_unit_square(self):
        from yolo_contrastive.dense import SpatialTwoViewAugmentation, TwoView

        aug = SpatialTwoViewAugmentation(
            out_size=(32, 32),
            scale=(1.0, 1.0), ratio=(1.0, 1.0), hflip_prob=0.0,
        )
        imgs = torch.rand(2, 3, 64, 64)
        out = aug(imgs)

        assert isinstance(out, TwoView)
        assert out.view1.shape == (2, 3, 32, 32)
        assert out.view2.shape == (2, 3, 32, 32)
        assert out.coords1.shape == (2, 2, 32, 32)
        assert out.coords2.shape == (2, 2, 32, 32)

        # Identity-style aug: coords stay in [0, 1]
        for coords in (out.coords1, out.coords2):
            assert (coords >= 0.0).all()
            assert (coords <= 1.0).all()

    def test_hflip_inverts_x_coord(self):
        from yolo_contrastive.dense import SpatialTwoViewAugmentation

        aug = SpatialTwoViewAugmentation(
            out_size=(8, 8), scale=(1.0, 1.0), ratio=(1.0, 1.0), hflip_prob=1.0,
        )
        out = aug(torch.rand(2, 3, 64, 64))
        # Always flip → x at col 0 > x at col W-1
        x0 = out.coords1[:, 0, :, 0]
        x_last = out.coords1[:, 0, :, -1]
        assert (x0 > x_last).all()


# ═════════════════════════════════════════════════════════════════════════
# A9 — multi_scale_dense_loss numerical safety + reduction
# ═════════════════════════════════════════════════════════════════════════


class TestA9_MultiScaleDenseLoss:
    """Finite loss, weight normalization works, gradients flow."""

    def test_finite_loss_with_default_weights(self):
        from yolo_contrastive.dense import multi_scale_dense_loss

        torch.manual_seed(0)
        B, D = 2, 16
        # Use small sizes (matches FPN levels)
        q = {
            "P3": F.normalize(torch.randn(B, D, 16, 16), dim=1),
            "P4": F.normalize(torch.randn(B, D, 8, 8), dim=1),
            "P5": F.normalize(torch.randn(B, D, 4, 4), dim=1),
        }
        k = {
            "P3": F.normalize(torch.randn(B, D, 16, 16), dim=1),
            "P4": F.normalize(torch.randn(B, D, 8, 8), dim=1),
            "P5": F.normalize(torch.randn(B, D, 4, 4), dim=1),
        }
        qc = self._coord_grid(B, 64, 64)
        kc = self._coord_grid(B, 64, 64)

        loss, info = multi_scale_dense_loss(q, k, qc, kc, n_query=16)
        assert torch.isfinite(loss).item()
        assert "total" in info
        # All 3 levels active
        for lv in ("P3", "P4", "P5"):
            assert lv in info

    @staticmethod
    def _coord_grid(B: int, H: int, W: int) -> torch.Tensor:
        xs = (torch.arange(W).float() + 0.5) / W
        ys = (torch.arange(H).float() + 0.5) / H
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=0)
        return grid.unsqueeze(0).expand(B, -1, -1, -1).contiguous()


# ═════════════════════════════════════════════════════════════════════════
# A10 — MultiScaleProjectionHead shape + non-normalized output (caller's job)
# ═════════════════════════════════════════════════════════════════════════


class TestA10_MultiScaleProjectionHead:
    """Per-level MLP preserves spatial dims, projects channels to out_dim.
    Head does NOT L2-normalize — that's the caller's responsibility (see
    src/yolo_contrastive/dense/projection.py docstring + the unit test
    tests/test_projection.py::TestForwardShapes::test_output_not_normalized).
    Gradient flows through the head end-to-end."""

    def test_per_level_projection_shapes_and_grad(self):
        from yolo_contrastive.dense import MultiScaleProjectionHead

        in_channels = {"P3": 32, "P4": 64, "P5": 128}
        head = MultiScaleProjectionHead(
            in_channels=in_channels, out_dim=16, hidden_dim=32,
        )

        B = 2
        feats = {
            "P3": torch.randn(B, 32, 8, 8, requires_grad=True),
            "P4": torch.randn(B, 64, 4, 4, requires_grad=True),
            "P5": torch.randn(B, 128, 2, 2, requires_grad=True),
        }
        proj = head(feats)

        # 1) Shape preserved (spatial dims) + channel dim → out_dim
        assert proj["P3"].shape == (B, 16, 8, 8)
        assert proj["P4"].shape == (B, 16, 4, 4)
        assert proj["P5"].shape == (B, 16, 2, 2)

        # 2) NOT L2-normalized — head returns raw embeddings (caller normalizes).
        #    With large-magnitude inputs, per-pixel norms should differ from 1.
        for lv in ("P3", "P4", "P5"):
            norms = proj[lv].norm(dim=1)  # [B, H, W]
            # Some norms must be far from 1.0 — head is intentionally not norming.
            assert not torch.allclose(norms, torch.ones_like(norms), atol=0.1), (
                f"level {lv}: head should NOT L2-normalize (caller responsibility)"
            )

        # 3) Gradient flows from head output back to input features
        loss = sum(p.mean() for p in proj.values())
        loss.backward()
        for lv in ("P3", "P4", "P5"):
            assert feats[lv].grad is not None
            assert feats[lv].grad.abs().sum() > 0


# ═════════════════════════════════════════════════════════════════════════
# A11 — saps_within_loss + saps_cross_loss
# ═════════════════════════════════════════════════════════════════════════


class TestA11_SAPSLosses:
    """Both SAPS variants produce finite loss; cross_loss requires tagged queue."""

    @staticmethod
    def _coord_grid(B: int, H: int, W: int) -> torch.Tensor:
        xs = (torch.arange(W).float() + 0.5) / W
        ys = (torch.arange(H).float() + 0.5) / H
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=0)
        return grid.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    def test_saps_within_finite(self):
        from yolo_contrastive.dense import saps_within_loss

        torch.manual_seed(0)
        B, D = 2, 16
        q = {lv: F.normalize(torch.randn(B, D, s, s), dim=1)
             for lv, s in [("P3", 16), ("P4", 8), ("P5", 4)]}
        k = {lv: F.normalize(torch.randn(B, D, s, s), dim=1)
             for lv, s in [("P3", 16), ("P4", 8), ("P5", 4)]}
        qc = self._coord_grid(B, 64, 64)
        kc = self._coord_grid(B, 64, 64)

        loss, info = saps_within_loss(q, k, qc, kc, n_query=16)
        assert torch.isfinite(loss).item()
        # SAPS-within adds cross-scale in-image negatives
        for lv in ("P3", "P4", "P5"):
            assert "cross_scale_negs" in info[lv]
            assert info[lv]["cross_scale_negs"] > 0

    def test_saps_cross_with_tagged_queue(self):
        from yolo_contrastive.dense import (
            saps_cross_loss, FeatureQueue, combine_queues,
        )

        torch.manual_seed(0)
        B, D = 2, 16
        LEVEL_TO_ID = {"P3": 0, "P4": 1, "P5": 2}

        q = {lv: F.normalize(torch.randn(B, D, s, s), dim=1)
             for lv, s in [("P3", 16), ("P4", 8), ("P5", 4)]}
        k = {lv: F.normalize(torch.randn(B, D, s, s), dim=1)
             for lv, s in [("P3", 16), ("P4", 8), ("P5", 4)]}
        qc = self._coord_grid(B, 64, 64)
        kc = self._coord_grid(B, 64, 64)

        # Build tagged queue: 10 entries per level, tagged with level id
        queues = {lv: FeatureQueue(dim=D, K=32, with_tags=True) for lv in LEVEL_TO_ID}
        for lv, q_ in queues.items():
            keys = F.normalize(torch.randn(10, D), dim=1)
            tags = torch.full((10,), LEVEL_TO_ID[lv], dtype=torch.long)
            q_.enqueue(keys, tags=tags)
        queue_keys, queue_tags = combine_queues(queues, level_to_id=LEVEL_TO_ID)

        loss, info = saps_cross_loss(
            q, k, qc, kc,
            queue_keys=queue_keys, queue_tags=queue_tags,
            level_to_id=LEVEL_TO_ID,
            n_query=16, t_scale=1.0,
        )
        assert torch.isfinite(loss).item()
        # Info per level reports queue_neg_count
        for lv in ("P3", "P4", "P5"):
            assert info[lv]["queue_neg_count"] == 30  # 3 levels × 10 each


# ═════════════════════════════════════════════════════════════════════════
# A12 — FinetuneDetectionTrainer end-to-end (Risk 16 v2 invariant)
# ═════════════════════════════════════════════════════════════════════════


class TestA12_FinetuneEndToEnd:
    """Real YOLOv8n + tiny SSL backbone + 1-epoch finetune.
    Validates Risk 16 v2: head_norm > 0 after training (no EMA collapse).
    This is the only Hat A test that exercises the full SSL→FT path."""

    @pytest.mark.slow
    def test_finetune_1epoch_risk16_invariant(
        self, tiny_dense_backbone, dummy_yolo_dataset, env_isolation, tmp_workspace,
    ):
        from ultralytics import YOLO
        from yolo_contrastive.finetune import FinetuneDetectionTrainer

        # 1) Tiny SSL backbone (1 iter @ imgsz=64 for speed)
        backbone_pt = tiny_dense_backbone(epochs=1, n_images=4, imgsz=64)

        # 2) Tiny YOLO dataset @ imgsz=160 (multiple of 32, Ultralytics-safer)
        ds = dummy_yolo_dataset(n_train=6, n_val=2, num_classes=2, imgsz=160)

        # 3) Env vars (env_isolation will restore)
        os.environ["YCL_PRETRAINED"] = backbone_pt
        os.environ["YCL_FREEZE_BACKBONE"] = "10"
        os.environ["YCL_UNFREEZE_EPOCH"] = "0"  # never unfreeze in 1-epoch run
        os.environ["YCL_BACKBONE_LR_SCALE"] = "0.5"

        # 4) Real YOLO.train with FinetuneDetectionTrainer
        project = tmp_workspace / "yolo_runs"
        model = YOLO("yolov8n.pt")
        results = model.train(
            data=ds["data_yaml"],
            epochs=1, imgsz=160, batch=2,
            device="cpu",
            trainer=FinetuneDetectionTrainer,
            project=str(project), name="ft_smoke",
            exist_ok=True, verbose=False,
            workers=0,  # avoid multiprocessing in pytest
            plots=False,
        )

        # 5) Risk 16 v2 invariant: detection head must have non-zero norm
        # (v1 aliased EMA → head collapses to 0)
        head_norm = 0.0
        for name, p in model.model.named_parameters():
            # Heuristic: anything outside backbone layers 0-9 is "head"
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "model":
                try:
                    layer_idx = int(parts[1])
                    if layer_idx >= 10:  # head region
                        head_norm += p.detach().abs().sum().item()
                except ValueError:
                    pass

        assert head_norm > 0, (
            f"Head norm collapsed to 0 — Risk 16 v2 fix failed! head_norm={head_norm}"
        )


# ═════════════════════════════════════════════════════════════════════════
# A13 — LinearProbeTrainer construction + fit + early stopping
# ═════════════════════════════════════════════════════════════════════════


class TestA13_LinearProbe:
    """Mock encoder + synthetic dataset → fit completes; early stopping triggers."""

    def test_fit_returns_history(self):
        from yolo_contrastive.eval import LinearProbeTrainer
        from torch.utils.data import DataLoader, TensorDataset

        # Synthetic multi-label dataset
        torch.manual_seed(0)
        n, num_classes = 12, 3
        imgs = torch.rand(n, 3, 32, 32)
        # Random multi-hot targets
        targets = (torch.rand(n, num_classes) > 0.5).float()
        ds = TensorDataset(imgs, targets)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        probe = LinearProbeTrainer(
            backbone=_mock_yolo_encoder(),
            num_classes=num_classes,
            feat_level="P5", device="cpu",
        )
        try:
            result = probe.fit(loader, loader, epochs=2, verbose=False)
            assert "best_val_mAP" in result
            assert "history" in result
            assert len(result["history"]) == 2
            assert result["epochs_run"] == 2
            assert result["early_stopped"] is False
        finally:
            probe.cleanup()

    def test_early_stopping_triggers(self):
        from yolo_contrastive.eval import LinearProbeTrainer
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(0)
        n, num_classes = 8, 3
        imgs = torch.rand(n, 3, 32, 32)
        targets = (torch.rand(n, num_classes) > 0.5).float()
        ds = TensorDataset(imgs, targets)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        probe = LinearProbeTrainer(
            backbone=_mock_yolo_encoder(),
            num_classes=num_classes, feat_level="P5", device="cpu",
        )
        try:
            # Small dataset + patience=1 → likely plateaus quickly
            result = probe.fit(
                loader, loader, epochs=10, lr=1e-3, verbose=False,
                early_stopping_patience=1,
            )
            # Should stop before 10 epochs
            assert result["epochs_run"] < 10
            assert result["early_stopped"] is True
        finally:
            probe.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# A14 — eval/RunMatrix linear_probe runner (mock cell)
# ═════════════════════════════════════════════════════════════════════════


class TestA14_RunMatrixLinearProbe:
    """RunMatrix expands cells + runs linear_probe via custom mock runner.
    We don't need the real runner here — A13 covers LinearProbeTrainer
    end-to-end. This test exercises orchestrator wiring."""

    def test_run_with_mock_runner(self, tmp_workspace):
        from yolo_contrastive.eval import RunMatrix

        def _mock_runner(cell, hp):
            # Verify cell structure matches schema
            assert "method" in cell and "dataset" in cell
            assert "fraction" in cell and "seed" in cell
            return {"metric": "mAP", "metric_value": 0.42 + cell["fraction"] * 0.1}

        config = {
            "task": "linear_probe",
            "methods": [{"name": "ours", "backbone_ckpt": "/dev/null"}],
            "datasets": [{"name": "ds", "data_yaml": "/dev/null", "num_classes": 2}],
            "fractions": [0.1, 1.0],
            "seeds": [42],
            "hp": {"epochs": 1, "lr": 0.01},
        }

        csv_path = tmp_workspace / "results.csv"
        rm = RunMatrix(
            config=config, output_csv=str(csv_path),
            runners={"linear_probe": _mock_runner},
        )
        cells = rm.expand()
        assert len(cells) == 2  # 1 method × 1 dataset × 2 fractions × 1 seed

        rm.run(resume=False, on_error="raise")
        assert csv_path.exists()

        # Verify CSV has rows
        import csv as csv_mod
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        statuses = {r["status"] for r in rows}
        assert statuses == {"ok"}


# ═════════════════════════════════════════════════════════════════════════
# A15 — RunMatrix _run_detection real (mini)
# ═════════════════════════════════════════════════════════════════════════


class TestA15_RunMatrixDetection:
    """Run the actual _run_detection runner (Adım 2) end-to-end with tiny
    YOLO + tiny SSL backbone. Heaviest test in Hat A — exercises the
    detection runner contract."""

    @pytest.mark.slow
    def test_run_detection_e2e(
        self, tiny_dense_backbone, dummy_yolo_dataset, env_isolation, tmp_workspace,
    ):
        from yolo_contrastive.eval.run_matrix import _run_detection

        backbone_pt = tiny_dense_backbone(epochs=1, n_images=4, imgsz=64)
        ds = dummy_yolo_dataset(n_train=6, n_val=2, num_classes=2, imgsz=160)

        cell = {
            "method": {"name": "smoke_method", "backbone_ckpt": backbone_pt},
            "dataset": {"name": "smoke_ds", "data_yaml": ds["data_yaml"], "num_classes": 2},
            "seed": 42,
            "fraction": 1.0,
        }
        hp = {
            "epochs": 1, "imgsz": 160, "batch": 2,
            "freeze": 10, "unfreeze_epoch": 0, "backbone_lr_scale": 0.5,
            "device": "cpu",
            "project": str(tmp_workspace / "det_runs"),
        }

        result = _run_detection(cell, hp)
        # Contract: returns {"metric", "metric_value", ...}
        assert "metric" in result
        assert "metric_value" in result
        assert isinstance(result["metric_value"], float)
        # mAP50 in [0, 1]
        assert 0.0 <= result["metric_value"] <= 1.0


# ═════════════════════════════════════════════════════════════════════════
# A16 — PretrainMatrix 2-cell real mini grid
# ═════════════════════════════════════════════════════════════════════════


class TestA16_PretrainMatrixMini:
    """PretrainMatrix runs 2 cells with mock encoder via custom runner.
    The real DenseSSL pretrain is already exercised in A2 — here we test
    orchestration: expand + run + CSV persistence + per-cell axes_json."""

    def test_pretrain_matrix_2cell_with_mock_runner(self, tmp_workspace):
        from yolo_contrastive.pretrain.run_matrix import PretrainMatrix

        # Mock runner: pretends to train, returns canonical schema
        def _mock_pretrain_runner(cell, base):
            axes = cell["axes"]
            # Simulate a different "loss" per cell to confirm aggregation works
            fake_loss = 1.0 - 0.1 * axes.get("saps_t_scale", 1.0)
            return {
                "metric": "final_loss",
                "metric_value": fake_loss,
                "backbone_path": str(tmp_workspace / f"{cell['cell_id']}.pt"),
            }

        config = {
            "output_dir": str(tmp_workspace),
            "base": {"model": "yolov8n.pt", "epochs": 1},
            "grid": {
                "saps_mode": ["within", "cross"],
                "saps_t_scale": [1.0],
            },
            "seeds": [42],
        }

        csv_path = tmp_workspace / "pretrain.csv"
        pm = PretrainMatrix(
            config=config, output_csv=str(csv_path),
            runners={"pretrain": _mock_pretrain_runner},
        )

        cells = pm.expand()
        assert len(cells) == 2  # 2 saps_mode × 1 t_scale × 1 seed

        pm.run(resume=False, on_error="raise")
        assert csv_path.exists()

        import csv as csv_mod
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        # axes_json field captures the grid values
        import json
        for row in rows:
            axes = json.loads(row["axes_json"])
            assert "saps_mode" in axes
            assert "saps_t_scale" in axes


# ═════════════════════════════════════════════════════════════════════════
# A17 — PretrainMatrix list-DSL exclude
# ═════════════════════════════════════════════════════════════════════════


class TestA17_PretrainMatrixExclude:
    """The list-DSL exclude pattern reduces the grid as documented:
    'lambda redundant when saps_mode != both' — exclude saps_both_lambda > 0
    in modes other than 'both'."""

    def test_exclude_reduces_grid(self, tmp_workspace):
        from yolo_contrastive.pretrain.run_matrix import PretrainMatrix

        config = {
            "output_dir": str(tmp_workspace),
            "base": {"model": "yolov8n.pt"},
            "grid": {
                "saps_mode": ["none", "within", "cross", "both"],
                "saps_both_lambda": [0.0, 0.5, 1.0],
            },
            "seeds": [42],
            "exclude": [
                # When saps_mode is one of {none, within, cross}, exclude
                # saps_both_lambda values {0.5, 1.0} — only 0.0 remains
                {"saps_mode": ["none", "within", "cross"],
                 "saps_both_lambda": [0.5, 1.0]},
            ],
        }

        pm = PretrainMatrix(
            config=config,
            output_csv=str(tmp_workspace / "pm.csv"),
            runners={"pretrain": lambda cell, base: {
                "metric": "final_loss", "metric_value": 0.0,
                "backbone_path": "/dev/null",
            }},
        )

        cells = pm.expand()
        # Without exclude: 4 modes × 3 lambdas × 1 seed = 12
        # Exclude removes 3 modes × 2 lambdas × 1 seed = 6
        # Remaining = 12 - 6 = 6: (none/within/cross × λ=0) + (both × {0,0.5,1.0})
        assert len(cells) == 6

        # Verify shape: every cell with mode != "both" has lambda=0.0
        for cell in cells:
            ax = cell["axes"]
            if ax["saps_mode"] != "both":
                assert ax["saps_both_lambda"] == 0.0, (
                    f"exclude failed for cell {cell['cell_id']}: {ax}"
                )
