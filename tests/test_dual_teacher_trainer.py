"""Tests for DualTeacherTrainer — DT-SAPS dual-teacher pretraining.

Mock-based: a synthetic 23-layer YOLOv8-like encoder stands in for the
student and both teachers. One slow test runs a real 1-epoch loop.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn


# Student channels (YOLOv8n-like) and COCO-teacher channels (wider).
_STUDENT_CH = (32, 64, 128)
_COCO_CH = (64, 128, 256)
_STUDENT_CH_DICT = {"P3": 32, "P4": 64, "P5": 128}


def _mock_yolo_encoder(channels=_STUDENT_CH) -> nn.Sequential:
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


def _ssl_kwargs():
    """Small DenseSSLPretrainer kwargs for fast tests."""
    return dict(out_dim=16, queue_size=64, n_query=16,
                momentum=0.9, temperature=0.2, pos_radius=0.1)


def _make_coco_teacher():
    from yolo_contrastive.dual_teacher import CocoTeacher
    return CocoTeacher(
        weights=_mock_yolo_encoder(_COCO_CH),
        student_channels=_STUDENT_CH_DICT, device="cpu",
    )


def _make_ssl_teacher():
    from yolo_contrastive.dual_teacher import CocoTeacher
    # SSL teacher = SAPS winner, student architecture → no adapter.
    return CocoTeacher(weights=_mock_yolo_encoder(_STUDENT_CH), device="cpu")


def _make_trainer(teacher_combo="both", **overrides):
    from yolo_contrastive.dual_teacher.dual_teacher_trainer import DualTeacherTrainer

    kwargs = dict(
        model=_mock_yolo_encoder(_STUDENT_CH),
        teacher_combo=teacher_combo,
        ssl_kwargs=_ssl_kwargs(),
        imgsz=64, device="cpu",
    )
    if teacher_combo in ("coco_only", "both"):
        kwargs["coco_teacher"] = _make_coco_teacher()
    if teacher_combo in ("ssl_only", "both"):
        kwargs["ssl_teacher"] = _make_ssl_teacher()
    kwargs.update(overrides)
    return DualTeacherTrainer(**kwargs)


def _dummy_image_dir(n=4, size=64):
    import cv2
    tmp = tempfile.mkdtemp(prefix="ycl_dt_")
    for i in range(n):
        img = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(tmp, f"img_{i:03d}.png"), img)
    return tmp


# ═════════════════════════════════════════════════════════════════════════
# Construction
# ═════════════════════════════════════════════════════════════════════════


class TestConstruction:
    @pytest.mark.parametrize("combo", ["none", "coco_only", "ssl_only", "both"])
    def test_all_combos_build(self, combo):
        tr = _make_trainer(teacher_combo=combo)
        try:
            assert tr.teacher_combo == combo
            # ssl_trainer always present (composition)
            assert tr.ssl_trainer is not None
        finally:
            tr.cleanup()

    def test_coco_only_without_coco_teacher_raises(self):
        from yolo_contrastive.dual_teacher.dual_teacher_trainer import DualTeacherTrainer

        with pytest.raises(ValueError, match="requires coco_teacher"):
            DualTeacherTrainer(
                model=_mock_yolo_encoder(), teacher_combo="coco_only",
                ssl_kwargs=_ssl_kwargs(), imgsz=64, device="cpu",
            )

    def test_ssl_only_without_ssl_teacher_raises(self):
        from yolo_contrastive.dual_teacher.dual_teacher_trainer import DualTeacherTrainer

        with pytest.raises(ValueError, match="requires ssl_teacher"):
            DualTeacherTrainer(
                model=_mock_yolo_encoder(), teacher_combo="ssl_only",
                ssl_kwargs=_ssl_kwargs(), imgsz=64, device="cpu",
            )

    def test_bad_combo_raises(self):
        from yolo_contrastive.dual_teacher.dual_teacher_trainer import DualTeacherTrainer

        with pytest.raises(ValueError, match="teacher_combo"):
            DualTeacherTrainer(
                model=_mock_yolo_encoder(), teacher_combo="bogus",
                ssl_kwargs=_ssl_kwargs(), imgsz=64, device="cpu",
            )

    def test_composition_not_inheritance(self):
        """DualTeacherTrainer holds a DenseSSLPretrainer, is not one."""
        from yolo_contrastive.dual_teacher.dual_teacher_trainer import DualTeacherTrainer
        from yolo_contrastive.pretrain import DenseSSLPretrainer

        tr = _make_trainer(teacher_combo="none")
        try:
            assert not isinstance(tr, DenseSSLPretrainer)
            assert isinstance(tr.ssl_trainer, DenseSSLPretrainer)
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Trainable parameters
# ═════════════════════════════════════════════════════════════════════════


class TestTrainableParameters:
    def test_includes_student_adapter_w_alpha(self):
        tr = _make_trainer(teacher_combo="both")
        try:
            params = tr._trainable_parameters()
            param_ids = {id(p) for p in params}
            # student backbone
            assert any(id(p) in param_ids
                       for p in tr.ssl_trainer.model.parameters())
            # COCO adapter
            assert any(id(p) in param_ids
                       for p in tr.coco_teacher.adapter.parameters())
            # ConsensusLoss fusion weight
            assert id(tr.consensus_loss.w_raw) in param_ids
            # learnable disagreement alpha
            assert id(tr.disagreement.alpha) in param_ids
        finally:
            tr.cleanup()

    def test_excludes_teacher_backbone(self):
        tr = _make_trainer(teacher_combo="both")
        try:
            param_ids = {id(p) for p in tr._trainable_parameters()}
            # COCO teacher backbone params must NOT be optimized
            for p in tr.coco_teacher.backbone.parameters():
                assert id(p) not in param_ids
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# _step
# ═════════════════════════════════════════════════════════════════════════


class TestStep:
    def test_step_both_finite_total(self):
        tr = _make_trainer(teacher_combo="both")
        try:
            out = tr._step(["a", "b"], torch.rand(2, 3, 64, 64))
            assert torch.isfinite(out["loss"]).item()
            assert out["info"]["saps"] > 0
            assert out["info"]["distill"] > 0
            # total = saps + distill_weight * distill
            expected = out["info"]["saps"] + 1.0 * out["info"]["distill"]
            assert abs(out["info"]["total"] - expected) < 1e-3
        finally:
            tr.cleanup()

    def test_step_none_is_saps_only(self):
        tr = _make_trainer(teacher_combo="none")
        try:
            out = tr._step(["a", "b"], torch.rand(2, 3, 64, 64))
            assert out["info"]["distill"] == 0.0
            assert abs(out["info"]["total"] - out["info"]["saps"]) < 1e-5
        finally:
            tr.cleanup()

    @pytest.mark.parametrize("combo", ["coco_only", "ssl_only"])
    def test_step_single_teacher(self, combo):
        tr = _make_trainer(teacher_combo=combo)
        try:
            out = tr._step(["a", "b"], torch.rand(2, 3, 64, 64))
            assert torch.isfinite(out["loss"]).item()
            assert out["info"]["distill"] > 0
        finally:
            tr.cleanup()

    def test_step_gradient_flow(self):
        tr = _make_trainer(teacher_combo="both")
        try:
            out = tr._step(["a", "b"], torch.rand(2, 3, 64, 64))
            out["loss"].backward()
            # student backbone grad
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in tr.ssl_trainer.model.parameters())
            # COCO adapter grad
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in tr.coco_teacher.adapter.parameters())
            # fusion weight grad
            assert tr.consensus_loss.w_raw.grad is not None
            # learnable alpha_d grad
            assert tr.disagreement.alpha.grad is not None
        finally:
            tr.cleanup()

    def test_distill_weight_scales(self):
        torch.manual_seed(0)
        tr = _make_trainer(teacher_combo="both", distill_weight=3.0)
        try:
            out = tr._step(["a", "b"], torch.rand(2, 3, 64, 64))
            expected = out["info"]["saps"] + 3.0 * out["info"]["distill"]
            assert abs(out["info"]["total"] - expected) < 1e-3
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Disagreement toggle
# ═════════════════════════════════════════════════════════════════════════


class TestDisagreementToggle:
    def test_disagreement_off(self):
        tr = _make_trainer(teacher_combo="both", use_disagreement=False)
        try:
            assert tr.disagreement is None
            out = tr._step(["a", "b"], torch.rand(2, 3, 64, 64))
            assert "alpha_d" not in out["info"]
        finally:
            tr.cleanup()

    def test_disagreement_on_reports_alpha(self):
        tr = _make_trainer(teacher_combo="both", use_disagreement=True)
        try:
            out = tr._step(["a", "b"], torch.rand(2, 3, 64, 64))
            assert "alpha_d" in out["info"]
            assert set(out["info"]["alpha_d"].keys()) == {"P3", "P4", "P5"}
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Teacher feature retrieval — live vs cache
# ═════════════════════════════════════════════════════════════════════════


class TestTeacherFeatureModes:
    def test_live_mode_coco(self):
        tr = _make_trainer(teacher_combo="coco_only")
        try:
            imgs = torch.rand(2, 3, 64, 64)
            feats = tr._teacher_features_from(
                tr.coco_teacher, None, ["a", "b"], imgs, apply_adapter=True,
            )
            # adapted → student channels
            assert feats["P3"].shape[1] == 32
            assert feats["P5"].shape[1] == 128
        finally:
            tr.cleanup()

    def test_cache_mode_coco(self, tmp_path):
        from yolo_contrastive.dual_teacher import TeacherCache

        tr = _make_trainer(teacher_combo="coco_only")
        try:
            # Build a small cache with RAW (teacher-channel) features.
            cache = TeacherCache(str(tmp_path), teacher_tag="coco_test")
            for iid in ("a", "b"):
                cache.save(iid, {
                    "P3": torch.randn(64, 8, 8),
                    "P4": torch.randn(128, 4, 4),
                    "P5": torch.randn(256, 2, 2),
                })
            imgs = torch.rand(2, 3, 64, 64)
            feats = tr._teacher_features_from(
                tr.coco_teacher, cache, ["a", "b"], imgs, apply_adapter=True,
            )
            # cache raw → adapter → student channels, batch stacked
            assert feats["P3"].shape == (2, 32, 8, 8)
            assert feats["P5"].shape == (2, 128, 2, 2)
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# train() smoke + checkpoint
# ═════════════════════════════════════════════════════════════════════════


class TestTrainSmoke:
    @pytest.mark.parametrize("combo", ["none", "both"])
    def test_train_one_epoch(self, combo, tmp_path):
        import shutil

        img_dir = _dummy_image_dir(n=4, size=64)
        out = tmp_path / f"dt_{combo}.pt"
        tr = _make_trainer(teacher_combo=combo)
        try:
            result = tr.train(
                images_dir=img_dir, epochs=1, batch_size=2,
                lr=1e-3, warmup_epochs=0, num_workers=0,
                output=str(out), save_every=0, print_every=1,
            )
            assert result == str(out)
            assert out.exists()
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
            assert ckpt["extra"]["type"] == "dt_saps"
            assert ckpt["extra"]["teacher_combo"] == combo
            assert "model_state_dict" in ckpt
        finally:
            tr.cleanup()
            shutil.rmtree(img_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_train_real_yolo_1epoch(self, tmp_path):
        """Real YOLOv8n student + real YOLOv8x COCO teacher, 1 epoch."""
        import shutil
        from yolo_contrastive.dual_teacher import CocoTeacher
        from yolo_contrastive.dual_teacher.dual_teacher_trainer import DualTeacherTrainer

        img_dir = _dummy_image_dir(n=4, size=64)
        out = tmp_path / "dt_real.pt"
        coco = CocoTeacher(
            weights="yolov8x.pt",
            student_channels={"P3": 64, "P4": 128, "P5": 256},
            device="cpu",
        )
        tr = DualTeacherTrainer(
            model="yolov8n.pt", teacher_combo="coco_only",
            coco_teacher=coco,
            ssl_kwargs=dict(out_dim=16, queue_size=16, n_query=4),
            imgsz=64, device="cpu",
        )
        try:
            result = tr.train(
                images_dir=img_dir, epochs=1, batch_size=2,
                lr=1e-3, warmup_epochs=0, num_workers=0,
                output=str(out), save_every=0, print_every=1,
            )
            assert out.exists()
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
            assert ckpt["extra"]["type"] == "dt_saps"
        finally:
            tr.cleanup()
            shutil.rmtree(img_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════
# Indexed dataset
# ═════════════════════════════════════════════════════════════════════════


class TestIndexedDataset:
    def test_yields_image_id_and_tensor(self):
        import shutil
        from yolo_contrastive.dual_teacher.dual_teacher_trainer import _IndexedImageDataset

        img_dir = _dummy_image_dir(n=3, size=64)
        try:
            ds = _IndexedImageDataset(img_dir, imgsz=64)
            assert len(ds) == 3
            image_id, img = ds[0]
            assert isinstance(image_id, str)
            assert img.shape == (3, 64, 64)
            # image_id is extension-free
            assert not image_id.endswith(".png")
        finally:
            shutil.rmtree(img_dir, ignore_errors=True)
