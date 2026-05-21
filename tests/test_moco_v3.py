"""Tests for MoCoV3YOLOTrainer — MoCo-v3-YOLO baseline."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn


def _mock_yolo_encoder(channels=(32, 64, 128)) -> nn.Sequential:
    """23-layer Sequential mimicking YOLOv8 FPN — P3/P4/P5 at layers 15/18/21."""
    p3, p4, p5 = channels
    layers = []
    layers.append(nn.Conv2d(3, p3, 3, stride=2, padding=1))    # 0  /2
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, 3, padding=1))         # 1-5
    layers.append(nn.Conv2d(p3, p3, 3, stride=2, padding=1))   # 6  /4
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, 3, padding=1))         # 7-11
    layers.append(nn.Conv2d(p3, p3, 3, stride=2, padding=1))   # 12 /8
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))             # 13
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))             # 14
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))             # 15 P3
    layers.append(nn.Conv2d(p3, p4, 3, stride=2, padding=1))   # 16 /16
    layers.append(nn.Conv2d(p4, p4, 3, padding=1))             # 17
    layers.append(nn.Conv2d(p4, p4, 3, padding=1))             # 18 P4
    layers.append(nn.Conv2d(p4, p5, 3, stride=2, padding=1))   # 19 /32
    layers.append(nn.Conv2d(p5, p5, 3, padding=1))             # 20
    layers.append(nn.Conv2d(p5, p5, 3, padding=1))             # 21 P5
    layers.append(nn.Conv2d(p5, p5, 1))                        # 22
    return nn.Sequential(*layers)


def _make_trainer(**overrides):
    from yolo_contrastive.baselines.moco_v3 import MoCoV3YOLOTrainer

    kwargs = dict(
        model=_mock_yolo_encoder(), out_dim=32,
        proj_hidden=64, pred_hidden=64, momentum=0.9,
        temperature=0.2, imgsz=64, device="cpu",
    )
    kwargs.update(overrides)
    return MoCoV3YOLOTrainer(**kwargs)


def _dummy_image_dir(n=4, size=64):
    import cv2
    tmp = tempfile.mkdtemp(prefix="ycl_moco_")
    for i in range(n):
        img = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(tmp, f"img_{i:03d}.png"), img)
    return tmp


# ═════════════════════════════════════════════════════════════════════════
# Construction
# ═════════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_basic_build(self):
        tr = _make_trainer()
        try:
            assert tr.feat_level == "P5"
            assert tr.feat_dim == 128
            assert tr.momentum_coef == 0.9
        finally:
            tr.cleanup()

    def test_has_momentum_and_predictor(self):
        """MoCo-v3 has a momentum encoder and a query-side predictor."""
        tr = _make_trainer()
        try:
            assert tr.momentum is not None
            assert tr.predictor is not None
            assert tr.proj_momentum is not None
        finally:
            tr.cleanup()

    def test_no_queue(self):
        """MoCo-v3 dropped the memory queue — confirm no queue attr."""
        tr = _make_trainer()
        try:
            assert not hasattr(tr, "queues")
            assert not hasattr(tr, "queue")
        finally:
            tr.cleanup()

    def test_momentum_params_frozen(self):
        tr = _make_trainer()
        try:
            for p in tr.momentum.momentum.parameters():
                assert not p.requires_grad
            for p in tr.proj_momentum.parameters():
                assert not p.requires_grad
        finally:
            tr.cleanup()

    def test_bad_momentum_raises(self):
        from yolo_contrastive.baselines.moco_v3 import MoCoV3YOLOTrainer

        with pytest.raises(ValueError, match="momentum"):
            MoCoV3YOLOTrainer(model=_mock_yolo_encoder(), momentum=1.5,
                              device="cpu", imgsz=64)

    def test_bad_feat_level_raises(self):
        from yolo_contrastive.baselines.moco_v3 import MoCoV3YOLOTrainer

        with pytest.raises(ValueError, match="feat_level"):
            MoCoV3YOLOTrainer(model=_mock_yolo_encoder(), feat_level="P9",
                              device="cpu", imgsz=64)


# ═════════════════════════════════════════════════════════════════════════
# InfoNCE helper
# ═════════════════════════════════════════════════════════════════════════


class TestInfoNCE:
    def test_perfect_alignment_low_loss(self):
        """q == k → each query's positive dominates → low loss."""
        from yolo_contrastive.baselines.moco_v3 import _moco_v3_infonce

        torch.manual_seed(0)
        q = torch.randn(8, 16)
        k = q.clone()
        loss_aligned = _moco_v3_infonce(q, k, temperature=0.2)
        loss_random = _moco_v3_infonce(q, torch.randn(8, 16), temperature=0.2)
        assert loss_aligned < loss_random

    def test_finite_and_positive(self):
        from yolo_contrastive.baselines.moco_v3 import _moco_v3_infonce

        loss = _moco_v3_infonce(torch.randn(4, 16), torch.randn(4, 16), 0.2)
        assert torch.isfinite(loss).item()
        assert loss.item() > 0


# ═════════════════════════════════════════════════════════════════════════
# Embedding + step
# ═════════════════════════════════════════════════════════════════════════


class TestEmbedAndStep:
    def test_query_embed_shape(self):
        tr = _make_trainer(out_dim=32)
        try:
            q = tr._embed_query(torch.rand(2, 3, 64, 64))
            assert q.shape == (2, 32)
        finally:
            tr.cleanup()

    def test_key_embed_detached(self):
        tr = _make_trainer(out_dim=32)
        try:
            k = tr._embed_key(torch.rand(2, 3, 64, 64))
            assert k.shape == (2, 32)
            assert not k.requires_grad   # key branch is detached
        finally:
            tr.cleanup()

    def test_step_finite_loss(self):
        tr = _make_trainer()
        try:
            out = tr._step(torch.rand(4, 3, 64, 64))
            assert torch.isfinite(out["loss"]).item()
            assert out["loss"].item() > 0
        finally:
            tr.cleanup()

    def test_step_gradient_flow(self):
        tr = _make_trainer()
        try:
            out = tr._step(torch.rand(4, 3, 64, 64))
            out["loss"].backward()
            # online backbone, projector, predictor get gradients
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in tr.model.parameters())
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in tr.predictor.parameters())
            # momentum encoder must NOT get gradients
            for p in tr.momentum.momentum.parameters():
                assert p.grad is None
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# EMA update
# ═════════════════════════════════════════════════════════════════════════


class TestEMAUpdate:
    def test_ema_moves_momentum(self):
        tr = _make_trainer(momentum=0.5)   # fast update
        try:
            before = [p.detach().clone()
                      for p in tr.momentum.momentum.parameters()]
            with torch.no_grad():
                for p in tr.model.parameters():
                    p.data.add_(torch.randn_like(p) * 0.1)
            tr._ema_update()
            after = list(tr.momentum.momentum.parameters())
            assert any(not torch.equal(b, a) for b, a in zip(before, after))
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# train() smoke + checkpoint
# ═════════════════════════════════════════════════════════════════════════


class TestTrainSmoke:
    def test_train_one_epoch(self, tmp_path):
        import shutil

        img_dir = _dummy_image_dir(n=4, size=64)
        out = tmp_path / "moco.pt"
        tr = _make_trainer()
        try:
            result = tr.train(
                images_dir=img_dir, epochs=1, batch_size=2,
                lr=1e-3, warmup_epochs=0, num_workers=0,
                output=str(out), save_every=0, print_every=1,
            )
            assert result == str(out)
            assert out.exists()
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
            assert ckpt["extra"]["type"] == "moco_v3_yolo"
        finally:
            tr.cleanup()
            shutil.rmtree(img_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_train_real_yolo(self, tmp_path):
        import shutil
        from yolo_contrastive.baselines.moco_v3 import MoCoV3YOLOTrainer

        img_dir = _dummy_image_dir(n=4, size=64)
        out = tmp_path / "moco_real.pt"
        tr = MoCoV3YOLOTrainer(model="yolov8n.pt", out_dim=32,
                               imgsz=64, device="cpu")
        try:
            tr.train(
                images_dir=img_dir, epochs=1, batch_size=2,
                lr=1e-3, warmup_epochs=0, num_workers=0,
                output=str(out), save_every=0, print_every=1,
            )
            assert out.exists()
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
            assert ckpt["extra"]["type"] == "moco_v3_yolo"
        finally:
            tr.cleanup()
            shutil.rmtree(img_dir, ignore_errors=True)
