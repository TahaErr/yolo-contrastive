"""Integration smoke — DT-SAPS framework + external baselines.

Unit tests cover each module in isolation; this file exercises the
cross-module flows the paper pipeline actually runs:

  - DT-SAPS: CocoTeacher → (TeacherCache) → DualTeacherTrainer → checkpoint
             → load_backbone drop-in.
  - Baselines: SimCLR/MoCo-v3/CoMAD-YOLO trainer → checkpoint → load_backbone.
  - CoMAD-YOLO consuming other baselines' backbones as its three teachers
    (the real Faz 5.4 wiring).
  - data/dedup pHash → eval/leakage_check runner.
  - Trainer-convention consistency (train / cleanup) across the framework.

Most tests use a 23-layer mock encoder for speed; one slow test runs the
real YOLOv8n pretrain → finetune drop-in path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────
# Mock 23-layer YOLO encoder (P3/P4/P5 at layers 15/18/21)
# ─────────────────────────────────────────────────────────────────────────


def _mock_yolo_encoder(channels: tuple = (32, 64, 128)) -> nn.Sequential:
    p3, p4, p5 = channels
    layers = [nn.Conv2d(3, p3, 3, stride=2, padding=1)]
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, 3, padding=1))
    layers.append(nn.Conv2d(p3, p3, 3, stride=2, padding=1))
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, 3, padding=1))
    layers.append(nn.Conv2d(p3, p3, 3, stride=2, padding=1))
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))            # 15 P3
    layers.append(nn.Conv2d(p3, p4, 3, stride=2, padding=1))
    layers.append(nn.Conv2d(p4, p4, 3, padding=1))
    layers.append(nn.Conv2d(p4, p4, 3, padding=1))            # 18 P4
    layers.append(nn.Conv2d(p4, p5, 3, stride=2, padding=1))
    layers.append(nn.Conv2d(p5, p5, 3, padding=1))
    layers.append(nn.Conv2d(p5, p5, 3, padding=1))            # 21 P5
    layers.append(nn.Conv2d(p5, p5, 1))
    return nn.Sequential(*layers)


_STUDENT_CH = {"P3": 32, "P4": 64, "P5": 128}
_SSL_KWARGS = dict(out_dim=16, queue_size=64, n_query=16)


# ═════════════════════════════════════════════════════════════════════════
# DT-SAPS framework integration
# ═════════════════════════════════════════════════════════════════════════


class TestDTSAPSIntegration:
    """CocoTeacher + DualTeacherTrainer + TeacherCache wired together."""

    def test_dual_teacher_live_mode_step(self):
        """COCO + SSL teacher (live) → DualTeacherTrainer._step → finite loss."""
        from yolo_contrastive import CocoTeacher, DualTeacherTrainer

        coco = CocoTeacher(weights=_mock_yolo_encoder((64, 128, 256)),
                           student_channels=_STUDENT_CH, device="cpu")
        ssl_teacher = CocoTeacher(weights=_mock_yolo_encoder((32, 64, 128)),
                                  device="cpu")
        tr = DualTeacherTrainer(
            model=_mock_yolo_encoder(), teacher_combo="both",
            coco_teacher=coco, ssl_teacher=ssl_teacher,
            ssl_kwargs=_SSL_KWARGS, imgsz=64, device="cpu",
        )
        try:
            out = tr._step(["a", "b"], torch.rand(2, 3, 64, 64))
            assert torch.isfinite(out["loss"]).item()
            assert out["info"]["distill"] > 0
        finally:
            tr.cleanup()

    def test_dual_teacher_cache_mode_step(self, tmp_workspace):
        """TeacherCache (pre-built) → DualTeacherTrainer cache mode → finite loss."""
        from yolo_contrastive import CocoTeacher, TeacherCache, DualTeacherTrainer

        coco = CocoTeacher(weights=_mock_yolo_encoder((64, 128, 256)),
                           student_channels=_STUDENT_CH, device="cpu")
        ssl_teacher = CocoTeacher(weights=_mock_yolo_encoder((32, 64, 128)),
                                  device="cpu")
        # Pre-build small caches with RAW teacher features.
        coco_cache = TeacherCache(str(tmp_workspace / "cc"), teacher_tag="coco")
        ssl_cache = TeacherCache(str(tmp_workspace / "sc"), teacher_tag="ssl")
        for iid in ("a", "b"):
            coco_cache.save(iid, {"P3": torch.randn(64, 8, 8),
                                  "P4": torch.randn(128, 4, 4),
                                  "P5": torch.randn(256, 2, 2)})
            ssl_cache.save(iid, {"P3": torch.randn(32, 8, 8),
                                 "P4": torch.randn(64, 4, 4),
                                 "P5": torch.randn(128, 2, 2)})
        tr = DualTeacherTrainer(
            model=_mock_yolo_encoder(), teacher_combo="both",
            coco_teacher=coco, ssl_teacher=ssl_teacher,
            coco_cache=coco_cache, ssl_cache=ssl_cache,
            ssl_kwargs=_SSL_KWARGS, imgsz=64, device="cpu",
        )
        try:
            out = tr._step(["a", "b"], torch.rand(2, 3, 64, 64))
            assert torch.isfinite(out["loss"]).item()
        finally:
            tr.cleanup()

    def test_dt_saps_train_checkpoint_loadable(self, dummy_images_dir,
                                               tmp_workspace):
        """DualTeacherTrainer.train() → checkpoint → load_backbone drop-in."""
        from yolo_contrastive import CocoTeacher, DualTeacherTrainer, load_backbone

        img_dir = dummy_images_dir(n=4, size=64)
        out = tmp_workspace / "dt_saps.pt"
        coco = CocoTeacher(weights=_mock_yolo_encoder((64, 128, 256)),
                           student_channels=_STUDENT_CH, device="cpu")
        ssl_teacher = CocoTeacher(weights=_mock_yolo_encoder((32, 64, 128)),
                                  device="cpu")
        tr = DualTeacherTrainer(
            model=_mock_yolo_encoder(), teacher_combo="both",
            coco_teacher=coco, ssl_teacher=ssl_teacher,
            ssl_kwargs=_SSL_KWARGS, imgsz=64, device="cpu",
        )
        try:
            tr.train(images_dir=str(img_dir), epochs=1, batch_size=2,
                     warmup_epochs=0, num_workers=0, output=str(out),
                     save_every=0, print_every=1)
        finally:
            tr.cleanup()

        assert out.exists()
        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        assert ckpt["extra"]["type"] == "dt_saps"
        # Drop-in: load into a fresh encoder.
        fresh = _mock_yolo_encoder()
        n = load_backbone(fresh, str(out), strict=False, verbose=False)
        assert n > 0, "no params loaded from DT-SAPS checkpoint"


# ═════════════════════════════════════════════════════════════════════════
# Baseline integration
# ═════════════════════════════════════════════════════════════════════════


class TestBaselineIntegration:
    """Each baseline trainer: train() → checkpoint → load_backbone drop-in."""

    @pytest.mark.parametrize("which", ["simclr", "moco"])
    def test_baseline_train_checkpoint_loadable(self, which, dummy_images_dir,
                                                tmp_workspace):
        from yolo_contrastive import (
            SimCLRYOLOTrainer, MoCoV3YOLOTrainer, load_backbone,
        )
        img_dir = dummy_images_dir(n=4, size=64)
        out = tmp_workspace / f"{which}.pt"
        cls = SimCLRYOLOTrainer if which == "simclr" else MoCoV3YOLOTrainer
        tr = cls(model=_mock_yolo_encoder(), out_dim=32, imgsz=64, device="cpu")
        try:
            tr.train(images_dir=str(img_dir), epochs=1, batch_size=2,
                     warmup_epochs=0, num_workers=0, output=str(out),
                     save_every=0, print_every=1)
        finally:
            tr.cleanup()
        assert out.exists()
        fresh = _mock_yolo_encoder()
        assert load_backbone(fresh, str(out), strict=False, verbose=False) > 0

    def test_comad_consumes_baseline_backbones_as_teachers(self,
                                                           dummy_images_dir,
                                                           tmp_workspace):
        """CoMAD-YOLO with 3 teachers = the kind of backbones the other
        baselines produce — the real Faz 5.4 wiring (mock-encoder form)."""
        from yolo_contrastive import CoMADYOLOTrainer, load_backbone

        img_dir = dummy_images_dir(n=4, size=64)
        out = tmp_workspace / "comad.pt"
        # Three diverse SSL teachers (mock backbones, student architecture).
        teachers = [_mock_yolo_encoder() for _ in range(3)]
        tr = CoMADYOLOTrainer(
            model=_mock_yolo_encoder(), teachers=teachers,
            mask_ratio_teachers=(0.1, 0.25, 0.4),
            imgsz=64, device="cpu",
        )
        try:
            out_step = tr._step(torch.rand(2, 3, 64, 64))
            assert torch.isfinite(out_step["loss"]).item()
            tr.train(images_dir=str(img_dir), epochs=1, batch_size=2,
                     warmup_epochs=0, num_workers=0, output=str(out),
                     save_every=0, print_every=1)
        finally:
            tr.cleanup()
        assert out.exists()
        fresh = _mock_yolo_encoder()
        assert load_backbone(fresh, str(out), strict=False, verbose=False) > 0


# ═════════════════════════════════════════════════════════════════════════
# leakage_check ↔ data/dedup integration
# ═════════════════════════════════════════════════════════════════════════


class TestLeakageCheckIntegration:
    """eval/leakage_check on top of data/dedup pHash machinery."""

    def test_leakage_detected_across_modules(self, dummy_images_dir,
                                             tmp_workspace):
        """Pool hashed via leakage_check.hash_image_dir (which uses
        data/dedup.compute_phash) → run_leakage_check flags a planted
        duplicate downstream image."""
        import pandas as pd
        from yolo_contrastive.eval.leakage_check import (
            hash_image_dir, run_leakage_check,
        )

        # Pool of 4 images.
        pool_dir = dummy_images_dir(n=4, size=64, name="pool")
        pool_hashes = hash_image_dir(str(pool_dir))
        assert len(pool_hashes) == 4

        # Persist pool pHashes to the parquet sidecar run_leakage_check reads.
        pool_parquet = tmp_workspace / "pool_phash.parquet"
        pd.DataFrame({
            "image_id": list(pool_hashes.keys()),
            "phash": list(pool_hashes.values()),
        }).to_parquet(pool_parquet, index=False)

        # Downstream dir — copy one pool image so leakage is guaranteed.
        import shutil
        ds_dir = tmp_workspace / "downstream"
        ds_dir.mkdir()
        first_pool_img = sorted(Path(pool_dir).glob("*.jpg"))[0]
        shutil.copy(first_pool_img, ds_dir / "leaked.jpg")

        report = run_leakage_check(str(pool_parquet), [str(ds_dir)])
        assert report["pool_size"] == 4
        assert report["total_leaking_pairs"] >= 1
        assert len(report["leaking_pool_ids"]) >= 1


# ═════════════════════════════════════════════════════════════════════════
# Trainer-convention consistency
# ═════════════════════════════════════════════════════════════════════════


class TestTrainerConventions:
    """Every trainer in the framework shares the same train / cleanup shape."""

    def test_all_pretrainers_have_train_and_cleanup(self):
        from yolo_contrastive import (
            DenseSSLPretrainer, SimCLRYOLOTrainer, MoCoV3YOLOTrainer,
            CoMADYOLOTrainer, DualTeacherTrainer,
        )
        for cls in (DenseSSLPretrainer, SimCLRYOLOTrainer, MoCoV3YOLOTrainer,
                    CoMADYOLOTrainer, DualTeacherTrainer):
            assert callable(getattr(cls, "train", None)), f"{cls.__name__}.train"
            assert callable(getattr(cls, "cleanup", None)), f"{cls.__name__}.cleanup"

    def test_top_level_constructs_dual_teacher(self):
        """A trainer is constructible straight from the top-level export."""
        from yolo_contrastive import CocoTeacher, DualTeacherTrainer

        coco = CocoTeacher(weights=_mock_yolo_encoder((64, 128, 256)),
                           student_channels=_STUDENT_CH, device="cpu")
        ssl_teacher = CocoTeacher(weights=_mock_yolo_encoder((32, 64, 128)),
                                  device="cpu")
        tr = DualTeacherTrainer(
            model=_mock_yolo_encoder(), teacher_combo="both",
            coco_teacher=coco, ssl_teacher=ssl_teacher,
            ssl_kwargs=_SSL_KWARGS, imgsz=64, device="cpu",
        )
        try:
            assert tr.teacher_combo == "both"
        finally:
            tr.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Real-YOLO end-to-end (slow)
# ═════════════════════════════════════════════════════════════════════════


class TestRealYOLOEndToEnd:
    @pytest.mark.slow
    def test_dt_saps_real_yolo_pretrain_then_load(self, dummy_images_dir,
                                                  tmp_workspace):
        """Real YOLOv8n student + real YOLOv8x COCO teacher → DT-SAPS 1-epoch
        → checkpoint → load into a fresh real YOLO model."""
        from yolo_contrastive import CocoTeacher, DualTeacherTrainer, load_backbone
        from ultralytics import YOLO

        img_dir = dummy_images_dir(n=4, size=64)
        out = tmp_workspace / "dt_saps_real.pt"
        coco = CocoTeacher(
            weights="yolov8x.pt",
            student_channels={"P3": 64, "P4": 128, "P5": 256},
            device="cpu",
        )
        tr = DualTeacherTrainer(
            model="yolov8n.pt", teacher_combo="coco_only", coco_teacher=coco,
            ssl_kwargs=dict(out_dim=16, queue_size=16, n_query=4),
            imgsz=64, device="cpu",
        )
        try:
            tr.train(images_dir=str(img_dir), epochs=1, batch_size=2,
                     warmup_epochs=0, num_workers=0, output=str(out),
                     save_every=0, print_every=1)
        finally:
            tr.cleanup()
        assert out.exists()
        fresh = YOLO("yolov8n.pt").model
        n = load_backbone(fresh, str(out), strict=False, verbose=False,
                          backbone_only=True)
        assert n > 0, "DT-SAPS checkpoint not loadable into a real YOLO model"
