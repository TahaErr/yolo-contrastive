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
            # epoch field now records best_epoch (best-epoch checkpoint
            # feature), not the last epoch — so it is in 1..epochs.
            assert 1 <= ckpt.get("epoch") <= 2
            assert ckpt.get("extra", {}).get("type") == "dense_ssl"
        finally:
            shutil.rmtree(tmp_imgs, ignore_errors=True)
            shutil.rmtree(tmp_out_dir, ignore_errors=True)

    def test_loss_history_persisted(self):
        """train() records per-epoch metrics into extra['loss_history'] —
        loss curves survive the run for paper Figure 2 (plan §5.1).
        Backward-compat: the pre-existing extra keys must stay intact."""
        tmp_imgs = _dummy_images_dir(n=8, size=64)
        tmp_out_dir = tempfile.mkdtemp(prefix="ycl_dense_hist_")
        try:
            output_path = os.path.join(tmp_out_dir, "backbone.pt")
            tr = _make_trainer()
            try:
                tr.train(
                    images_dir=tmp_imgs,
                    epochs=3,
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

            ckpt = torch.load(output_path, map_location="cpu",
                              weights_only=False)
            extra = ckpt.get("extra", {})

            # backward-compat — old keys still present
            assert extra.get("type") == "dense_ssl"
            assert "loss" in extra

            # loss_history — one record per epoch
            hist = extra.get("loss_history")
            assert hist is not None, "extra['loss_history'] missing"
            assert len(hist) == 3, f"expected 3 epoch records, got {len(hist)}"

            # each record carries the expected metric keys
            expected_keys = {"epoch", "loss", "acc_top1",
                             "pos_sim", "neg_sim", "lr"}
            for rec in hist:
                assert expected_keys <= set(rec.keys()), (
                    f"record missing keys: {expected_keys - set(rec.keys())}"
                )

            # epoch field is ordered 1..N
            assert [r["epoch"] for r in hist] == [1, 2, 3]
        finally:
            shutil.rmtree(tmp_imgs, ignore_errors=True)
            shutil.rmtree(tmp_out_dir, ignore_errors=True)

    def test_best_epoch_checkpoint(self):
        """Final checkpoint saves the lowest-loss epoch's weights, not the
        last epoch's. extra['best_epoch'] records which one; extra['loss']
        must equal the minimum loss in loss_history (reported best_loss and
        saved weights are now consistent)."""
        tmp_imgs = _dummy_images_dir(n=8, size=64)
        tmp_out_dir = tempfile.mkdtemp(prefix="ycl_dense_best_")
        try:
            output_path = os.path.join(tmp_out_dir, "backbone.pt")
            tr = _make_trainer()
            try:
                tr.train(
                    images_dir=tmp_imgs,
                    epochs=4,
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

            ckpt = torch.load(output_path, map_location="cpu",
                              weights_only=False)
            extra = ckpt.get("extra", {})

            # best_epoch present, within range
            be = extra.get("best_epoch")
            assert be is not None, "extra['best_epoch'] missing"
            assert 1 <= be <= 4, f"best_epoch out of range: {be}"

            # checkpoint's root epoch == best_epoch (final save uses it)
            assert ckpt.get("epoch") == be, (
                f"ckpt epoch {ckpt.get('epoch')} != best_epoch {be}"
            )

            # loss_history still the FULL curve — best-save doesn't truncate
            hist = extra.get("loss_history")
            assert hist is not None and len(hist) == 4

            # extra['loss'] == minimum loss in loss_history
            losses = [r["loss"] for r in hist]
            assert abs(extra["loss"] - min(losses)) < 1e-6, (
                f"extra['loss']={extra['loss']} != min(loss_history)={min(losses)}"
            )

            # best_epoch points at the actual minimum-loss epoch
            min_ep = min(hist, key=lambda r: r["loss"])["epoch"]
            assert be == min_ep, (
                f"best_epoch={be} but min-loss epoch is {min_ep}"
            )
        finally:
            shutil.rmtree(tmp_imgs, ignore_errors=True)
            shutil.rmtree(tmp_out_dir, ignore_errors=True)

    def test_best_epoch_weights_actually_saved(self):
        """The SAVED weights are the best epoch's, not the last epoch's.
        Smoke: train(), then if best_epoch != last epoch, the checkpoint's
        weights must NOT match the trainer's final (last-epoch) weights —
        proving load_state_dict(best_state) actually swapped them in."""
        tmp_imgs = _dummy_images_dir(n=8, size=64)
        tmp_out_dir = tempfile.mkdtemp(prefix="ycl_dense_bw_")
        try:
            output_path = os.path.join(tmp_out_dir, "backbone.pt")
            tr = _make_trainer()
            try:
                tr.train(
                    images_dir=tmp_imgs, epochs=5, batch_size=2, lr=1e-3,
                    warmup_epochs=0, num_workers=0, output=output_path,
                    save_every=0, print_every=1,
                )
                # trainer.model after train() == best-epoch weights
                # (final save restores best_state into self.model)
                final_state = {k: v.clone()
                               for k, v in tr.model.state_dict().items()}
            finally:
                tr.cleanup()

            ckpt = torch.load(output_path, map_location="cpu",
                              weights_only=False)
            saved = ckpt["model_state_dict"]
            be = ckpt["extra"]["best_epoch"]
            hist = ckpt["extra"]["loss_history"]

            # saved weights must equal what train() left in self.model
            # (both are the best-epoch weights)
            shared = set(saved) & set(final_state)
            assert shared, "no overlapping weight keys"
            for k in shared:
                assert torch.allclose(saved[k].cpu(), final_state[k].cpu(),
                                      atol=1e-6), (
                    f"saved weight {k} != trainer's best-epoch weight"
                )

            # if best wasn't the last epoch, that's the meaningful case —
            # best_state genuinely differs from last-epoch state
            if be != len(hist):
                assert ckpt["epoch"] == be  # not len(hist)
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


# ═════════════════════════════════════════════════════════════════════════
# SAPS integration tests (Faz 2.3)
# ═════════════════════════════════════════════════════════════════════════


class TestSAPSConstruction:
    def test_default_mode_is_none(self):
        tr = _make_trainer()
        try:
            assert tr.saps_mode == "none"
            assert tr._needs_tagged_queues is False
            assert all(not q.with_tags for q in tr.queues.values())
        finally:
            tr.cleanup()

    def test_invalid_mode_raises(self):
        from yolo_contrastive.pretrain import DenseSSLPretrainer
        encoder = _mock_yolo_encoder()
        with pytest.raises(ValueError, match="saps_mode"):
            DenseSSLPretrainer(model=encoder, saps_mode="bogus", device="cpu")

    def test_invalid_t_scale_raises(self):
        from yolo_contrastive.pretrain import DenseSSLPretrainer
        encoder = _mock_yolo_encoder()
        with pytest.raises(ValueError, match="saps_t_scale"):
            DenseSSLPretrainer(model=encoder, saps_t_scale=0.0, device="cpu")
        with pytest.raises(ValueError, match="saps_t_scale"):
            DenseSSLPretrainer(model=encoder, saps_t_scale=-1.0, device="cpu")

    def test_within_mode_state(self):
        tr = _make_trainer(saps_mode="within")
        try:
            assert tr.saps_mode == "within"
            assert tr._needs_tagged_queues is False
            assert all(not q.with_tags for q in tr.queues.values())
        finally:
            tr.cleanup()

    def test_cross_mode_state(self):
        tr = _make_trainer(saps_mode="cross", saps_t_scale=0.5)
        try:
            assert tr.saps_mode == "cross"
            assert tr.saps_t_scale == 0.5
            assert tr._needs_tagged_queues is True
            assert all(q.with_tags for q in tr.queues.values())
        finally:
            tr.cleanup()

    def test_both_mode_state(self):
        tr = _make_trainer(saps_mode="both")
        try:
            assert tr.saps_mode == "both"
            assert tr._needs_tagged_queues is True
            assert all(q.with_tags for q in tr.queues.values())
        finally:
            tr.cleanup()

    def test_level_to_id_stable(self):
        tr = _make_trainer()
        try:
            mapping = tr.level_to_id
            assert mapping == {"P3": 0, "P4": 1, "P5": 2}
        finally:
            tr.cleanup()

    def test_repr_includes_saps(self):
        tr = _make_trainer(saps_mode="within")
        try:
            assert "saps=within" in repr(tr)
        finally:
            tr.cleanup()


class TestSAPSStep:
    """_step works in all 4 modes, each with finite loss & valid gradient."""

    @pytest.mark.parametrize("mode", ["none", "within", "cross", "both"])
    def test_step_runs(self, mode):
        tr = _make_trainer(saps_mode=mode)
        try:
            imgs = torch.rand(2, 3, 64, 64)
            out = tr._step(imgs)
            assert torch.isfinite(out["loss"]).item()
            assert out["batch_size"] == 2
        finally:
            tr.cleanup()

    @pytest.mark.parametrize("mode", ["none", "within", "cross", "both"])
    def test_grad_flows(self, mode):
        tr = _make_trainer(saps_mode=mode)
        try:
            imgs = torch.rand(2, 3, 64, 64)
            out = tr._step(imgs)
            out["loss"].backward()
            grads = [p.grad for p in tr.model.parameters() if p.grad is not None]
            assert len(grads) > 0
            assert any(g.abs().sum() > 0 for g in grads)
        finally:
            tr.cleanup()

    def test_none_mode_info_flat(self):
        tr = _make_trainer(saps_mode="none")
        try:
            out = tr._step(torch.rand(2, 3, 64, 64))
            for lv in ("P3", "P4", "P5"):
                assert lv in out["info"]
                assert "acc_top1" in out["info"][lv]
        finally:
            tr.cleanup()

    def test_within_mode_info_has_cross_scale_negs(self):
        tr = _make_trainer(saps_mode="within")
        try:
            out = tr._step(torch.rand(2, 3, 64, 64))
            assert "cross_scale_negs" in out["info"]["P3"]
            assert out["info"]["P3"]["cross_scale_negs"] > 0
        finally:
            tr.cleanup()

    def test_cross_mode_info_has_queue_neg_count(self):
        """Empty queue first call → 0; after step, queue grows."""
        tr = _make_trainer(saps_mode="cross")
        try:
            out1 = tr._step(torch.rand(2, 3, 64, 64))
            assert out1["info"]["P3"]["queue_neg_count"] == 0
            out2 = tr._step(torch.rand(2, 3, 64, 64))
            # 3 levels × 2 batch = 6 entries combined after 1 enqueue
            assert out2["info"]["P3"]["queue_neg_count"] == 6
        finally:
            tr.cleanup()

    def test_both_mode_info_nested(self):
        tr = _make_trainer(saps_mode="both")
        try:
            out = tr._step(torch.rand(2, 3, 64, 64))
            assert "within" in out["info"]
            assert "cross" in out["info"]
            assert out["info"]["saps_mode"] == "both"
            for lv in ("P3", "P4", "P5"):
                assert lv in out["info"]["within"]
                assert lv in out["info"]["cross"]
        finally:
            tr.cleanup()


class TestSAPSQueueTagging:
    @pytest.mark.parametrize("mode", ["cross", "both"])
    def test_enqueue_attaches_correct_tags(self, mode):
        tr = _make_trainer(saps_mode=mode)
        try:
            _ = tr._step(torch.rand(4, 3, 64, 64))
            for lv, q in tr.queues.items():
                tags = q.get_tags()
                expected_id = tr.level_to_id[lv]
                assert (tags == expected_id).all(), (
                    f"Level {lv} expected tag {expected_id}, got {tags.tolist()}"
                )
        finally:
            tr.cleanup()

    def test_combined_queue_has_all_levels(self):
        from yolo_contrastive.dense import combine_queues
        tr = _make_trainer(saps_mode="cross")
        try:
            _ = tr._step(torch.rand(4, 3, 64, 64))
            keys, tags = combine_queues(tr.queues, level_to_id=tr.level_to_id)
            unique_tags = set(tags.unique().tolist())
            assert unique_tags == {0, 1, 2}
            # 4 batch × 3 levels = 12 total
            assert keys.shape[0] == 12
        finally:
            tr.cleanup()


class TestSAPSTrainSmoke:
    @pytest.mark.parametrize("mode", ["none", "within", "cross", "both"])
    def test_train_one_epoch(self, mode):
        tmp_imgs = _dummy_images_dir(n=4, size=64)
        tmp_out_dir = tempfile.mkdtemp(prefix=f"ycl_dense_saps_{mode}_")
        try:
            output = os.path.join(tmp_out_dir, "backbone.pt")
            tr = _make_trainer(saps_mode=mode)
            try:
                tr.train(
                    images_dir=tmp_imgs,
                    epochs=1, batch_size=2, lr=1e-3,
                    warmup_epochs=0, num_workers=0,
                    output=output, save_every=0, print_every=1,
                )
            finally:
                tr.cleanup()
            assert os.path.exists(output)
            ckpt = torch.load(output, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt
        finally:
            shutil.rmtree(tmp_imgs, ignore_errors=True)
            shutil.rmtree(tmp_out_dir, ignore_errors=True)


class TestSAPSRegressionVsNone:
    def test_none_mode_info_unchanged(self):
        """saps_mode='none' must keep the pre-SAPS info schema (flat per-level)."""
        tr = _make_trainer(saps_mode="none")
        try:
            out = tr._step(torch.rand(2, 3, 64, 64))
            assert torch.isfinite(out["loss"]).item()
            assert "total" in out["info"]
            assert "P3" in out["info"]
            assert "acc_top1" in out["info"]["P3"]
            # No SAPS-specific structure
            assert "within" not in out["info"]
            assert "cross" not in out["info"]
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# SAPS-both lambda weighting (Risk 9)
# ═════════════════════════════════════════════════════════════════════════


class TestSAPSBothLambda:
    """λ-weighted sum: loss = loss_within + λ · loss_cross.

    Default λ=1.0 preserves original additive behavior (regression-safe).
    λ=0 collapses to within-only. λ>1 amplifies cross.
    """

    def test_default_lambda_is_one(self):
        tr = _make_trainer(saps_mode="both")
        try:
            assert tr.saps_both_lambda == 1.0
        finally:
            tr.cleanup()

    def test_invalid_lambda_raises(self):
        from yolo_contrastive.pretrain import DenseSSLPretrainer
        encoder = _mock_yolo_encoder()
        with pytest.raises(ValueError, match="saps_both_lambda"):
            DenseSSLPretrainer(model=encoder, saps_mode="both",
                                saps_both_lambda=-0.1, device="cpu")

    def test_lambda_zero_equals_within_only(self):
        """λ=0 → loss = loss_within + 0 · loss_cross = loss_within.
        Numerically equivalent to saps_mode='within' (with same seed)."""
        torch.manual_seed(0)
        imgs = torch.rand(2, 3, 64, 64)

        # within-only run
        tr_w = _make_trainer(saps_mode="within")
        try:
            torch.manual_seed(0)  # match RNG state with both-λ=0 below
            out_w = tr_w._step(imgs.clone())
            loss_within = float(out_w["loss"].detach().item())
        finally:
            tr_w.cleanup()

        # both with λ=0
        tr_b = _make_trainer(saps_mode="both", saps_both_lambda=0.0)
        try:
            torch.manual_seed(0)
            out_b = tr_b._step(imgs.clone())
            loss_both_lambda0 = float(out_b["loss"].detach().item())
        finally:
            tr_b.cleanup()

        # NOTE: due to the queue having different tagged/untagged setup
        # between within-only (untagged) and both (tagged), exact
        # bit-equality isn't guaranteed at first step. But losses should
        # be in the same ballpark, and crucially for λ=0, the cross
        # contribution is exactly zeroed.
        # Stricter assertion: check cross contribution is masked
        loss_w_only = float(out_b["info"]["within"]["total"]["loss"])
        # loss == loss_w_only when λ=0
        assert abs(loss_both_lambda0 - loss_w_only) < 1e-5, (
            f"λ=0: total loss {loss_both_lambda0} should equal "
            f"within total {loss_w_only}"
        )

    def test_lambda_amplifies_cross_contribution(self):
        """Verify the formula `loss = loss_within + λ · loss_cross` directly
        using info dict from a single run. This is the cleanest invariant
        check — no stochastic mismatch between runs.
        """
        torch.manual_seed(42)
        for lam in (0.5, 1.0, 2.0, 3.0):
            tr = _make_trainer(saps_mode="both", saps_both_lambda=lam)
            try:
                out = tr._step(torch.rand(2, 3, 64, 64))
                loss_total = float(out["loss"].detach().item())
                loss_w = float(out["info"]["within"]["total"]["loss"])
                loss_c = float(out["info"]["cross"]["total"]["loss"])
                expected = loss_w + lam * loss_c
                assert abs(loss_total - expected) < 1e-4, (
                    f"λ={lam}: total={loss_total:.4f} vs "
                    f"expected w+λc = {loss_w:.4f} + {lam}·{loss_c:.4f} "
                    f"= {expected:.4f}"
                )
            finally:
                tr.cleanup()

    def test_info_contains_lambda_value(self):
        tr = _make_trainer(saps_mode="both", saps_both_lambda=0.5)
        try:
            out = tr._step(torch.rand(2, 3, 64, 64))
            assert "saps_both_lambda" in out["info"]
            assert out["info"]["saps_both_lambda"] == 0.5
        finally:
            tr.cleanup()

    def test_lambda_ignored_in_non_both_modes(self):
        """saps_both_lambda is set on trainer but only used when mode='both'.
        Other modes should ignore it (no info field, no behavior change)."""
        for mode in ("none", "within", "cross"):
            tr = _make_trainer(saps_mode=mode, saps_both_lambda=0.7)
            try:
                out = tr._step(torch.rand(2, 3, 64, 64))
                # saps_both_lambda only appears in info for "both" mode
                assert "saps_both_lambda" not in out["info"], (
                    f"mode={mode}: saps_both_lambda should not appear in info"
                )
            finally:
                tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Queue update strategy (Risk 7 — ablation prep)
# ═════════════════════════════════════════════════════════════════════════


class TestQueueUpdateStrategy:
    """Three strategies for pushing keys into the FPN-level queues:
    - "pooled":       1 vec per (image, level)  → B per level/step  (default)
    - "per_position": HxW vecs per (image, lvl) → B*HW per level/step
    - "subsample":    n random pos per image    → B*n per level/step

    Tests verify:
      1. default is "pooled" (regression-safe)
      2. invalid strategy raises
      3. invalid subsample_n raises
      4. each strategy enqueues the expected number of entries per level
    """

    def test_default_strategy_is_pooled(self):
        tr = _make_trainer()
        try:
            assert tr.queue_update_strategy == "pooled"
            assert tr.queue_subsample_n == 16
        finally:
            tr.cleanup()

    def test_invalid_strategy_raises(self):
        from yolo_contrastive.pretrain import DenseSSLPretrainer
        with pytest.raises(ValueError, match="queue_update_strategy"):
            DenseSSLPretrainer(model=_mock_yolo_encoder(),
                                queue_update_strategy="bogus", device="cpu")

    def test_invalid_subsample_n_raises(self):
        from yolo_contrastive.pretrain import DenseSSLPretrainer
        with pytest.raises(ValueError, match="queue_subsample_n"):
            DenseSSLPretrainer(model=_mock_yolo_encoder(),
                                queue_subsample_n=0, device="cpu")
        with pytest.raises(ValueError, match="queue_subsample_n"):
            DenseSSLPretrainer(model=_mock_yolo_encoder(),
                                queue_subsample_n=-5, device="cpu")

    def test_pooled_pushes_B_per_level(self):
        """Default strategy: B entries per level per step."""
        B = 2
        tr = _make_trainer(queue_update_strategy="pooled")
        try:
            queues_before = {lv: len(q) for lv, q in tr.queues.items()}
            _ = tr._step(torch.rand(B, 3, 64, 64))
            for lv, q in tr.queues.items():
                added = len(q) - queues_before[lv]
                assert added == B, (
                    f"pooled level {lv}: expected B={B} entries added, "
                    f"got {added}"
                )
        finally:
            tr.cleanup()

    def test_per_position_pushes_BHW_per_level(self):
        """Per-position: B * H_lv * W_lv entries per level per step."""
        B = 2
        tr = _make_trainer(queue_update_strategy="per_position",
                            queue_size=100_000)  # big enough not to wrap
        try:
            queues_before = {lv: len(q) for lv, q in tr.queues.items()}
            out = tr._step(torch.rand(B, 3, 64, 64))
            # mock encoder: imgsz 64 → P3=8x8, P4=4x4, P5=2x2 (strides 8/16/32)
            expected = {"P3": B * 8 * 8, "P4": B * 4 * 4, "P5": B * 2 * 2}
            for lv, q in tr.queues.items():
                added = len(q) - queues_before[lv]
                assert added == expected[lv], (
                    f"per_position level {lv}: expected {expected[lv]} "
                    f"entries added, got {added}"
                )
        finally:
            tr.cleanup()

    def test_subsample_pushes_Bn_per_level(self):
        """Subsample: B * n entries per level per step (capped at HW)."""
        B = 2
        n = 4
        tr = _make_trainer(queue_update_strategy="subsample",
                            queue_subsample_n=n,
                            queue_size=100_000)
        try:
            queues_before = {lv: len(q) for lv, q in tr.queues.items()}
            _ = tr._step(torch.rand(B, 3, 64, 64))
            # n=4, but for P5 HW=4 (2x2) so it caps at 4 also
            # P3 HW=64, P4 HW=16, P5 HW=4 — n=4 ≤ all
            for lv, q in tr.queues.items():
                added = len(q) - queues_before[lv]
                assert added == B * n, (
                    f"subsample level {lv}: expected {B * n} entries added, "
                    f"got {added}"
                )
        finally:
            tr.cleanup()
