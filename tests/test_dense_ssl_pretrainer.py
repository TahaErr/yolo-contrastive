"""Structural tests for DenseSSLPretrainer.

Uses a mock 23-layer Sequential encoder so tests don't require ultralytics.
A separate test (test_real_yolo_smoke) is provided in test_dense_ssl_pretrainer_realyolo.py
and gets skipped if ultralytics isn't installed.

Note: DenseSSLPretrainer accepts either a string (passed to ultralytics YOLO)
or a pre-built nn.Module. We use the latter to avoid the YOLO dependency.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from yolo_contrastive.pretrain import DenseSSLPretrainer


# ── helpers ──────────────────────────────────────────────────────────────


def _mock_yolo_encoder(channels: tuple = (32, 64, 128)) -> nn.Sequential:
    """Build a 23-layer Sequential mimicking YOLOv8 backbone+neck topology.

    We need:
        - At least 22 layers (so layer indices 15/18/21 are valid)
        - Layer 15 output has channels[0] (P3)
        - Layer 18 output has channels[1] (P4)
        - Layer 21 output has channels[2] (P5)
        - Spatial size halves at appropriate points to mimic strides

    Strategy: simple chain that downsamples at layers 0, 6, 12, 18 and shifts
    channel counts at the right points. We don't care about realism beyond
    "tap captures something at indices 15/18/21".
    """
    p3, p4, p5 = channels
    layers = []
    # Layers 0-14: produce P3-resolution features ending in p3 channels
    layers.append(nn.Conv2d(3, p3, kernel_size=3, stride=2, padding=1))   # 0: /2
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, kernel_size=3, padding=1))        # 1-5
    layers.append(nn.Conv2d(p3, p3, kernel_size=3, stride=2, padding=1))  # 6: /4
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, kernel_size=3, padding=1))        # 7-11
    layers.append(nn.Conv2d(p3, p3, kernel_size=3, stride=2, padding=1))  # 12: /8
    layers.append(nn.Conv2d(p3, p3, kernel_size=3, padding=1))            # 13
    layers.append(nn.Conv2d(p3, p3, kernel_size=3, padding=1))            # 14
    layers.append(nn.Conv2d(p3, p3, kernel_size=3, padding=1))            # 15: P3 out
    # Layers 16-18: downsample to P4 resolution + p4 channels
    layers.append(nn.Conv2d(p3, p4, kernel_size=3, stride=2, padding=1))  # 16: /16, p4
    layers.append(nn.Conv2d(p4, p4, kernel_size=3, padding=1))            # 17
    layers.append(nn.Conv2d(p4, p4, kernel_size=3, padding=1))            # 18: P4 out
    # Layers 19-21: downsample to P5 resolution + p5 channels
    layers.append(nn.Conv2d(p4, p5, kernel_size=3, stride=2, padding=1))  # 19: /32, p5
    layers.append(nn.Conv2d(p5, p5, kernel_size=3, padding=1))            # 20
    layers.append(nn.Conv2d(p5, p5, kernel_size=3, padding=1))            # 21: P5 out
    layers.append(nn.Conv2d(p5, p5, kernel_size=1))                       # 22: dummy "Detect"
    return nn.Sequential(*layers)


def _make_trainer(imgsz: int = 64, **overrides) -> DenseSSLPretrainer:
    """Quick factory with sensible test defaults."""
    encoder = _mock_yolo_encoder()
    kwargs = dict(
        model=encoder,
        out_dim=16,            # tiny embedding
        queue_size=64,         # tiny queue
        momentum=0.9,          # fast updates so EMA test is meaningful
        temperature=0.2,
        n_query=16,
        pos_radius=0.1,
        match_mode="threshold",
        imgsz=imgsz,
        device="cpu",
    )
    kwargs.update(overrides)
    return DenseSSLPretrainer(**kwargs)


def _dummy_images_dir(n: int = 8, size: int = 64) -> str:
    """Create a tmp dir of small dummy images for full train() smoke."""
    import cv2
    tmp = tempfile.mkdtemp(prefix="ycl_dense_test_")
    for i in range(n):
        img = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(tmp, f"img_{i:03d}.png"), img)
    return tmp


# ── construction ─────────────────────────────────────────────────────────


class TestConstruction:
    def test_basic_init(self):
        tr = _make_trainer()
        try:
            assert tr.out_dim == 16
            assert tr.queue_size == 64
            assert tr.momentum_coef == 0.9
            assert set(tr.queues.keys()) == {"P3", "P4", "P5"}
            assert tr.proj_online is not None
            assert tr.proj_momentum is not None
            assert tr.online_tap._is_setup
            assert tr.momentum_tap._is_setup
        finally:
            tr.cleanup()

    def test_in_channels_inferred(self):
        tr = _make_trainer()
        try:
            ch = tr._in_channels
            assert ch == {"P3": 32, "P4": 64, "P5": 128}
        finally:
            tr.cleanup()

    def test_invalid_args(self):
        enc = _mock_yolo_encoder()
        with pytest.raises(ValueError, match="out_dim"):
            DenseSSLPretrainer(model=enc, out_dim=0, device="cpu")
        with pytest.raises(ValueError, match="queue_size"):
            DenseSSLPretrainer(model=enc, queue_size=0, device="cpu")
        with pytest.raises(ValueError, match="momentum"):
            DenseSSLPretrainer(model=enc, momentum=-0.1, device="cpu")
        with pytest.raises(ValueError, match="temperature"):
            DenseSSLPretrainer(model=enc, temperature=0, device="cpu")

    def test_repr(self):
        tr = _make_trainer()
        try:
            r = repr(tr)
            assert "DenseSSLPretrainer" in r
            assert "D=16" in r
            assert "K=64" in r
        finally:
            tr.cleanup()


# ── single step ──────────────────────────────────────────────────────────


class TestStep:
    def test_step_returns_finite_loss(self):
        tr = _make_trainer()
        try:
            imgs = torch.rand(2, 3, 64, 64)
            out = tr._step(imgs)
            assert "loss" in out and "info" in out
            assert torch.isfinite(out["loss"]).item()
            assert out["batch_size"] == 2
        finally:
            tr.cleanup()

    def test_step_loss_has_grad(self):
        tr = _make_trainer()
        try:
            imgs = torch.rand(2, 3, 64, 64)
            out = tr._step(imgs)
            out["loss"].backward()
            # Online model should have gradients
            grads = [p.grad for p in tr.model.parameters() if p.grad is not None]
            assert len(grads) > 0
            assert any(g.abs().sum() > 0 for g in grads)
            # Projection head should also have gradients
            head_grads = [p.grad for p in tr.proj_online.parameters() if p.grad is not None]
            assert len(head_grads) > 0
        finally:
            tr.cleanup()

    def test_step_no_grad_in_momentum(self):
        tr = _make_trainer()
        try:
            imgs = torch.rand(2, 3, 64, 64)
            tr._step(imgs)
            # Momentum encoder params should never have grad
            for p in tr.momentum.momentum.parameters():
                assert p.grad is None
            # Projection momentum head: plain nn.Module, no .momentum sub-attr
            for p in tr.proj_momentum.parameters():
                assert p.grad is None
        finally:
            tr.cleanup()

    def test_info_per_level_present(self):
        tr = _make_trainer()
        try:
            imgs = torch.rand(2, 3, 64, 64)
            out = tr._step(imgs)
            for lv in ("P3", "P4", "P5"):
                assert lv in out["info"]
            assert "total" in out["info"]
        finally:
            tr.cleanup()


# ── EMA update ───────────────────────────────────────────────────────────


class TestEMAUpdate:
    def test_ema_changes_momentum_after_optim_step(self):
        """After model params change + ema_update, momentum params should move."""
        tr = _make_trainer(momentum=0.5)  # fast EMA so change is visible
        try:
            # Snapshot momentum encoder weights
            mom_before = [p.detach().clone() for p in tr.momentum.momentum.parameters()]

            # Modify online model directly (simulate optimizer step)
            with torch.no_grad():
                for p in tr.model.parameters():
                    p.data.add_(torch.randn_like(p.data) * 0.5)

            tr._ema_update()

            mom_after = list(tr.momentum.momentum.parameters())
            # At least some params should differ
            differs = any(
                not torch.equal(b, a.detach())
                for b, a in zip(mom_before, mom_after)
            )
            assert differs, "momentum encoder didn't update after _ema_update"
        finally:
            tr.cleanup()

    def test_ema_updates_projection_head(self):
        tr = _make_trainer(momentum=0.5)
        try:
            proj_mom_before = [p.detach().clone()
                               for p in tr.proj_momentum.parameters()]
            with torch.no_grad():
                for p in tr.proj_online.parameters():
                    p.data.add_(torch.randn_like(p.data) * 0.5)
            tr._ema_update()
            proj_mom_after = list(tr.proj_momentum.parameters())
            differs = any(
                not torch.equal(b, a.detach())
                for b, a in zip(proj_mom_before, proj_mom_after)
            )
            assert differs
        finally:
            tr.cleanup()


# ── queue accumulation ──────────────────────────────────────────────────


class TestQueueAccumulation:
    def test_queue_grows_with_steps(self):
        tr = _make_trainer()
        try:
            assert all(len(q) == 0 for q in tr.queues.values())
            for _ in range(3):
                _ = tr._step(torch.rand(2, 3, 64, 64))
            # After 3 steps × 2 imgs = 6 entries per level
            for lv, q in tr.queues.items():
                assert len(q) == 6, f"queue {lv} has {len(q)} expected 6"
        finally:
            tr.cleanup()

    def test_queue_caps_at_K(self):
        tr = _make_trainer(queue_size=4)
        try:
            for _ in range(5):
                _ = tr._step(torch.rand(2, 3, 64, 64))
            for q in tr.queues.values():
                assert len(q) == 4  # K = 4
        finally:
            tr.cleanup()


# ── multi-step learning signal ──────────────────────────────────────────


class TestLearningSignal:
    def test_loss_decreases_over_optim_steps(self):
        """Smoke test: optimizer steps should reduce loss after queue warmup.

        Why warmup: queue starts empty and grows by B per step. Early-step
        loss is artificially low (few negatives), and "loss going up while
        queue fills" is mathematically expected (more negatives → more
        denominator mass), not a learning failure. To isolate the
        learning signal, we first fill the queue (warmup) WITHOUT optimizer
        steps, then measure whether optimizer steps reduce loss.

        Directional check on a noisy mock encoder: last quartile < first quartile.
        """
        tr = _make_trainer()
        try:
            torch.manual_seed(0)

            # ── Phase 1: warmup queue (no optimizer step, no EMA) ───────
            # Fill queue to capacity so subsequent loss values reflect
            # learning, not queue-fill artifacts.
            warmup_steps = (tr.queue_size // 4) + 4   # K // batch_size + buffer
            with torch.no_grad():
                for _ in range(warmup_steps):
                    imgs = torch.rand(4, 3, 64, 64)
                    _ = tr._step(imgs)
            # Queue should now be full (or near full)
            for q in tr.queues.values():
                assert len(q) > tr.queue_size // 2, "queue not warmed up enough"

            # ── Phase 2: measure with optimizer steps ───────────────────
            opt = torch.optim.AdamW(
                list(tr.model.parameters()) + list(tr.proj_online.parameters()),
                lr=5e-3,
            )
            losses = []
            n_steps = 16
            for _ in range(n_steps):
                imgs = torch.rand(4, 3, 64, 64)
                opt.zero_grad()
                out = tr._step(imgs)
                out["loss"].backward()
                opt.step()
                tr._ema_update()
                losses.append(out["loss"].item())

            q = n_steps // 4
            first = sum(losses[:q]) / q
            last = sum(losses[-q:]) / q
            assert last < first, (
                f"loss didn't decrease post-warmup: "
                f"first {q}-step avg={first:.4f}, last {q}-step avg={last:.4f}, "
                f"all losses={[f'{x:.3f}' for x in losses]}"
            )
        finally:
            tr.cleanup()


# ── cleanup ──────────────────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_closes_taps(self):
        tr = _make_trainer()
        assert tr.online_tap._is_setup
        assert tr.momentum_tap._is_setup
        tr.cleanup()
        assert not tr.online_tap._is_setup
        assert not tr.momentum_tap._is_setup

    def test_cleanup_idempotent(self):
        tr = _make_trainer()
        tr.cleanup()
        tr.cleanup()  # should not raise


# ── full train() smoke (uses dummy image dir) ───────────────────────────


class TestFullTrainSmoke:
    def test_train_2_epochs(self):
        """Run train() end-to-end on dummy images; verify checkpoint saved."""
        tmp_imgs = _dummy_images_dir(n=8, size=64)
        tmp_out_dir = tempfile.mkdtemp(prefix="ycl_dense_out_")
        try:
            output_path = os.path.join(tmp_out_dir, "backbone.pt")
            tr = _make_trainer()
            try:
                result = tr.train(
                    images_dir=tmp_imgs,
                    epochs=2,
                    batch_size=2,
                    lr=1e-3,
                    warmup_epochs=0,
                    num_workers=0,
                    output=output_path,
                    save_every=0,
                    print_every=1,
                )
            finally:
                tr.cleanup()
            assert result == output_path
            assert os.path.exists(output_path)
            # Verify checkpoint structure
            ckpt = torch.load(output_path, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt
            assert ckpt.get("epoch") == 2
            assert ckpt.get("extra", {}).get("type") == "dense_ssl"
        finally:
            shutil.rmtree(tmp_imgs, ignore_errors=True)
            shutil.rmtree(tmp_out_dir, ignore_errors=True)


# ── checkpoint roundtrip ────────────────────────────────────────────────


class TestCheckpointRoundtrip:
    def test_save_then_load_into_fresh_encoder(self):
        """Save backbone, build a fresh trainer, load weights — params match."""
        from yolo_contrastive.pretrain import save_backbone, load_backbone

        tr1 = _make_trainer()
        tmp_dir = tempfile.mkdtemp(prefix="ycl_dense_ckpt_")
        try:
            # Take a step so weights are non-trivial
            _ = tr1._step(torch.rand(2, 3, 64, 64))
            path = os.path.join(tmp_dir, "bb.pt")
            save_backbone(tr1.model, path, epoch=1, extra={"type": "dense_ssl"})
            # Snapshot tr1 weights before tearing down
            ref_state = {k: v.clone() for k, v in tr1.model.state_dict().items()}
        finally:
            tr1.cleanup()

        try:
            # Fresh encoder — different random init
            fresh = _mock_yolo_encoder()
            n = load_backbone(fresh, path, strict=False, verbose=False,
                              backbone_only=False)
            assert n > 0
            # Verify weights match
            for k, v_ref in ref_state.items():
                if k in fresh.state_dict():
                    assert torch.allclose(fresh.state_dict()[k], v_ref, atol=1e-6)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── logger integration ──────────────────────────────────────────────────


class _CapturingLogger:
    """Minimal BaseLogger-ish stub that records log_scalars calls."""
    def __init__(self):
        self.records = []
        self.config = None
        self.finished = False

    def log_scalars(self, metrics, step=None):
        self.records.append((step, dict(metrics)))

    def log_scalar(self, key, value, step=None):
        self.records.append((step, {key: value}))

    def log_config(self, config):
        self.config = dict(config)

    def finish(self):
        self.finished = True


class TestLoggerIntegration:
    def test_logger_receives_metrics(self):
        tmp_imgs = _dummy_images_dir(n=4, size=64)
        tmp_out_dir = tempfile.mkdtemp(prefix="ycl_dense_log_")
        logger = _CapturingLogger()
        try:
            tr = _make_trainer()
            tr.logger = logger
            try:
                tr.train(
                    images_dir=tmp_imgs,
                    epochs=1, batch_size=2, lr=1e-3, warmup_epochs=0,
                    num_workers=0,
                    output=os.path.join(tmp_out_dir, "out.pt"),
                    save_every=0, print_every=1,
                )
            finally:
                tr.cleanup()
            assert logger.config is not None
            assert "epochs" in logger.config
            assert len(logger.records) > 0
            # Each record should have loss
            keys_seen = set()
            for _, m in logger.records:
                keys_seen.update(m.keys())
            assert "loss" in keys_seen
            assert logger.finished
        finally:
            shutil.rmtree(tmp_imgs, ignore_errors=True)
            shutil.rmtree(tmp_out_dir, ignore_errors=True)
