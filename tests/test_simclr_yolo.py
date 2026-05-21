"""Tests for SimCLRYOLOTrainer — SimCLR-YOLO baseline."""

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
    from yolo_contrastive.baselines.simclr_yolo import SimCLRYOLOTrainer

    kwargs = dict(
        model=_mock_yolo_encoder(), out_dim=32, proj_hidden=64,
        temperature=0.2, imgsz=64, device="cpu",
    )
    kwargs.update(overrides)
    return SimCLRYOLOTrainer(**kwargs)


def _dummy_image_dir(n=4, size=64):
    import cv2
    tmp = tempfile.mkdtemp(prefix="ycl_simclr_")
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
            assert tr.out_dim == 32
            assert tr.feat_dim == 128   # mock P5 channels
        finally:
            tr.cleanup()

    @pytest.mark.parametrize("level", ["P3", "P4", "P5"])
    def test_feat_level_choices(self, level):
        tr = _make_trainer(feat_level=level)
        try:
            assert tr.feat_level == level
            # feat_dim matches the mock channels for that level
            expected = {"P3": 32, "P4": 64, "P5": 128}[level]
            assert tr.feat_dim == expected
        finally:
            tr.cleanup()

    def test_bad_feat_level_raises(self):
        from yolo_contrastive.baselines.simclr_yolo import SimCLRYOLOTrainer

        with pytest.raises(ValueError, match="feat_level"):
            SimCLRYOLOTrainer(model=_mock_yolo_encoder(), feat_level="P9",
                              device="cpu", imgsz=64)

    def test_bad_out_dim_raises(self):
        from yolo_contrastive.baselines.simclr_yolo import SimCLRYOLOTrainer

        with pytest.raises(ValueError, match="out_dim"):
            SimCLRYOLOTrainer(model=_mock_yolo_encoder(), out_dim=0,
                              device="cpu", imgsz=64)

    def test_no_momentum_no_queue(self):
        """SimCLR has neither — confirm the trainer carries no such attrs."""
        tr = _make_trainer()
        try:
            assert not hasattr(tr, "momentum")
            assert not hasattr(tr, "queues")
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Embedding + step
# ═════════════════════════════════════════════════════════════════════════


class TestEmbedAndStep:
    def test_embed_shape(self):
        tr = _make_trainer(out_dim=32)
        try:
            view = torch.rand(2, 3, 64, 64)
            z = tr._embed(view)
            assert z.shape == (2, 32)
        finally:
            tr.cleanup()

    def test_step_finite_loss(self):
        tr = _make_trainer()
        try:
            out = tr._step(torch.rand(4, 3, 64, 64))
            assert torch.isfinite(out["loss"]).item()
            assert out["loss"].item() > 0
            assert out["batch_size"] == 4
        finally:
            tr.cleanup()

    def test_step_gradient_flow(self):
        tr = _make_trainer()
        try:
            out = tr._step(torch.rand(4, 3, 64, 64))
            out["loss"].backward()
            # backbone grads
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in tr.model.parameters())
            # projection head grads
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in tr.projection_head.parameters())
        finally:
            tr.cleanup()

    def test_two_views_differ(self):
        """Augmentation should produce two distinct views."""
        tr = _make_trainer()
        try:
            imgs = torch.rand(2, 3, 64, 64)
            v1 = tr.augmentation(imgs)
            v2 = tr.augmentation(imgs)
            # extremely unlikely to be identical given stochastic augmentation
            assert not torch.equal(v1, v2)
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# train() smoke + checkpoint
# ═════════════════════════════════════════════════════════════════════════


class TestTrainSmoke:
    def test_train_one_epoch(self, tmp_path):
        import shutil

        img_dir = _dummy_image_dir(n=4, size=64)
        out = tmp_path / "simclr.pt"
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
            assert ckpt["extra"]["type"] == "simclr_yolo"
            assert "model_state_dict" in ckpt
        finally:
            tr.cleanup()
            shutil.rmtree(img_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_train_real_yolo(self, tmp_path):
        import shutil
        from yolo_contrastive.baselines.simclr_yolo import SimCLRYOLOTrainer

        img_dir = _dummy_image_dir(n=4, size=64)
        out = tmp_path / "simclr_real.pt"
        tr = SimCLRYOLOTrainer(model="yolov8n.pt", out_dim=32,
                               imgsz=64, device="cpu")
        try:
            tr.train(
                images_dir=img_dir, epochs=1, batch_size=2,
                lr=1e-3, warmup_epochs=0, num_workers=0,
                output=str(out), save_every=0, print_every=1,
            )
            assert out.exists()
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
            assert ckpt["extra"]["type"] == "simclr_yolo"
        finally:
            tr.cleanup()
            shutil.rmtree(img_dir, ignore_errors=True)
