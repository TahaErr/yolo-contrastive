"""Tests for CoMADYOLOTrainer — CoMAD-YOLO baseline."""

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


def _make_trainer(n_teachers=3, **overrides):
    from yolo_contrastive.baselines.comad_yolo import CoMADYOLOTrainer

    kwargs = dict(
        model=_mock_yolo_encoder(),
        teachers=[_mock_yolo_encoder() for _ in range(n_teachers)],
        mask_ratio_teachers=tuple([0.1, 0.25, 0.4][:n_teachers]),
        patch_size=16, kl_temperature=4.0,
        imgsz=64, device="cpu",
    )
    kwargs.update(overrides)
    return CoMADYOLOTrainer(**kwargs)


def _dummy_image_dir(n=4, size=64):
    import cv2
    tmp = tempfile.mkdtemp(prefix="ycl_comad_")
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
            assert tr.n_teachers == 3
            assert tr.feat_level == "P5"
            assert len(tr.adapters) == 3
            assert len(tr.teacher_taps) == 3
        finally:
            tr.cleanup()

    def test_too_few_teachers_raises(self):
        from yolo_contrastive.baselines.comad_yolo import CoMADYOLOTrainer

        with pytest.raises(ValueError, match="2 teachers"):
            CoMADYOLOTrainer(
                model=_mock_yolo_encoder(),
                teachers=[_mock_yolo_encoder()],   # only 1
                mask_ratio_teachers=(0.1,),
                imgsz=64, device="cpu",
            )

    def test_mask_ratio_length_mismatch_raises(self):
        from yolo_contrastive.baselines.comad_yolo import CoMADYOLOTrainer

        with pytest.raises(ValueError, match="mask_ratio_teachers length"):
            CoMADYOLOTrainer(
                model=_mock_yolo_encoder(),
                teachers=[_mock_yolo_encoder() for _ in range(3)],
                mask_ratio_teachers=(0.1, 0.25),   # 2 vs 3 teachers
                imgsz=64, device="cpu",
            )

    def test_bad_feat_level_raises(self):
        from yolo_contrastive.baselines.comad_yolo import CoMADYOLOTrainer

        with pytest.raises(ValueError, match="feat_level"):
            CoMADYOLOTrainer(
                model=_mock_yolo_encoder(),
                teachers=[_mock_yolo_encoder() for _ in range(3)],
                mask_ratio_teachers=(0.1, 0.25, 0.4),
                feat_level="P9", imgsz=64, device="cpu",
            )

    def test_teacher_backbones_frozen(self):
        tr = _make_trainer()
        try:
            for teacher in tr.teachers:
                for p in teacher.parameters():
                    assert not p.requires_grad
        finally:
            tr.cleanup()

    def test_two_teachers_allowed(self):
        """CoMAD needs >= 2; verify 2 works (consensus still defined)."""
        tr = _make_trainer(n_teachers=2)
        try:
            assert tr.n_teachers == 2
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Asymmetric masking
# ═════════════════════════════════════════════════════════════════════════


class TestMasking:
    def test_mask_zeroes_expected_fraction(self):
        tr = _make_trainer()
        try:
            imgs = torch.ones(2, 3, 64, 64)
            # patch_size 16, 64/16 = 4x4 = 16 patches; ratio 0.5 → 8 masked
            masked = tr._apply_mask(imgs, ratio=0.5)
            # Count fully-zero 16x16 patches
            zero_patches = 0
            for r in range(4):
                for c in range(4):
                    patch = masked[0, :, r*16:(r+1)*16, c*16:(c+1)*16]
                    if patch.abs().sum() == 0:
                        zero_patches += 1
            assert zero_patches == 8
        finally:
            tr.cleanup()

    def test_mask_ratio_zero_unchanged(self):
        tr = _make_trainer()
        try:
            imgs = torch.rand(2, 3, 64, 64)
            assert torch.equal(tr._apply_mask(imgs, ratio=0.0), imgs)
        finally:
            tr.cleanup()

    def test_asymmetric_student_more_masked(self):
        """Student mask ratio must exceed every teacher's."""
        tr = _make_trainer()
        try:
            assert all(tr.mask_ratio_student > r
                       for r in tr.mask_ratio_teachers)
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Consensus gating
# ═════════════════════════════════════════════════════════════════════════


class TestConsensusGate:
    def test_fused_shape(self):
        tr = _make_trainer()
        try:
            student = torch.randn(2, 128, 4, 4)
            teachers = [torch.randn(2, 128, 4, 4) for _ in range(3)]
            fused = tr._consensus_gate(student, teachers)
            assert fused.shape == (2, 128, 4, 4)
        finally:
            tr.cleanup()

    def test_identical_teachers_fuse_to_same(self):
        """If all teachers are identical, fused == that feature."""
        tr = _make_trainer()
        try:
            student = torch.randn(2, 128, 4, 4)
            t = torch.randn(2, 128, 4, 4)
            fused = tr._consensus_gate(student, [t, t, t])
            assert torch.allclose(fused, t, atol=1e-4)
        finally:
            tr.cleanup()

    def test_cwd_kl_identical_is_zero(self):
        tr = _make_trainer()
        try:
            feat = torch.randn(2, 128, 4, 4)
            kl = tr._cwd_kl(feat, feat)
            assert abs(kl.item()) < 1e-4
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Step
# ═════════════════════════════════════════════════════════════════════════


class TestStep:
    def test_step_finite_loss(self):
        tr = _make_trainer()
        try:
            out = tr._step(torch.rand(4, 3, 64, 64))
            assert torch.isfinite(out["loss"]).item()
            assert out["batch_size"] == 4
        finally:
            tr.cleanup()

    def test_step_gradient_flow(self):
        tr = _make_trainer()
        try:
            out = tr._step(torch.rand(4, 3, 64, 64))
            out["loss"].backward()
            # student backbone gets gradients
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in tr.model.parameters())
            # teacher adapters get gradients
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in tr.adapters.parameters())
            # teacher backbones must NOT
            for teacher in tr.teachers:
                for p in teacher.parameters():
                    assert p.grad is None
        finally:
            tr.cleanup()

    def test_trainable_params_exclude_teachers(self):
        tr = _make_trainer()
        try:
            param_ids = {id(p) for p in tr._trainable_parameters()}
            for teacher in tr.teachers:
                for p in teacher.parameters():
                    assert id(p) not in param_ids
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# train() smoke + checkpoint
# ═════════════════════════════════════════════════════════════════════════


class TestTrainSmoke:
    def test_train_one_epoch(self, tmp_path):
        import shutil

        img_dir = _dummy_image_dir(n=4, size=64)
        out = tmp_path / "comad.pt"
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
            assert ckpt["extra"]["type"] == "comad_yolo"
            assert ckpt["extra"]["n_teachers"] == 3
        finally:
            tr.cleanup()
            shutil.rmtree(img_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_train_real_yolo(self, tmp_path):
        import shutil
        from yolo_contrastive.baselines.comad_yolo import CoMADYOLOTrainer

        img_dir = _dummy_image_dir(n=4, size=64)
        out = tmp_path / "comad_real.pt"
        # Real YOLOv8n student + 3 real YOLOv8n teachers (nn.Module form).
        from ultralytics import YOLO
        teachers = [YOLO("yolov8n.pt").model for _ in range(3)]
        tr = CoMADYOLOTrainer(
            model="yolov8n.pt", teachers=teachers,
            mask_ratio_teachers=(0.1, 0.25, 0.4),
            imgsz=64, device="cpu",
        )
        try:
            tr.train(
                images_dir=img_dir, epochs=1, batch_size=2,
                lr=1e-3, warmup_epochs=0, num_workers=0,
                output=str(out), save_every=0, print_every=1,
            )
            assert out.exists()
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
            assert ckpt["extra"]["type"] == "comad_yolo"
        finally:
            tr.cleanup()
            shutil.rmtree(img_dir, ignore_errors=True)
