"""Real YOLOv8 smoke tests for DenseSSLPretrainer.

Skipped if ultralytics is not installed. Mock-encoder coverage is in
test_dense_ssl_pretrainer.py — these tests only verify that a *real*
YOLOv8 model integrates correctly (channel inference, single step,
single epoch, checkpoint).

Kept extremely lightweight: imgsz=64, batch=2, tiny config — pretrained
yolov8n.pt download is the only slow part and is cached after first run.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest
import torch


def _ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _ultralytics_available(),
    reason="ultralytics not installed",
)


def _dummy_images_dir(n: int = 8, size: int = 64) -> str:
    import cv2
    tmp = tempfile.mkdtemp(prefix="ycl_dense_realyolo_")
    for i in range(n):
        img = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(tmp, f"img_{i:03d}.png"), img)
    return tmp


# ── construction ─────────────────────────────────────────────────────────


class TestRealYOLOConstruction:
    def test_init_with_yolov8n(self):
        """DenseSSLPretrainer(model="yolov8n.pt") constructs without error."""
        from yolo_contrastive.pretrain import DenseSSLPretrainer

        tr = DenseSSLPretrainer(
            model="yolov8n.pt",
            out_dim=32,
            queue_size=64,
            n_query=8,
            imgsz=64,
            device="cpu",
        )
        try:
            # YOLOv8n FPN channel widths (verified from architecture)
            ch = tr._in_channels
            assert set(ch.keys()) == {"P3", "P4", "P5"}
            # Channels must be positive ints; exact values depend on YOLOv8
            # variant — we don't hardcode them, just sanity check.
            for lv, c in ch.items():
                assert c > 0, f"non-positive channel for {lv}: {c}"
        finally:
            tr.cleanup()

    def test_yolov8n_known_channels(self):
        """Document YOLOv8n's actual P3/P4/P5 channel widths.

        If this test fails after an ultralytics update, the channel widths
        in WORK_PLAN_v3 / projection head examples need updating too.
        """
        from yolo_contrastive.pretrain import DenseSSLPretrainer

        tr = DenseSSLPretrainer(
            model="yolov8n.pt", out_dim=16, queue_size=16,
            n_query=4, imgsz=64, device="cpu",
        )
        try:
            ch = tr._in_channels
            # YOLOv8n at width_multiple=0.25 produces these:
            assert ch == {"P3": 64, "P4": 128, "P5": 256}, (
                f"YOLOv8n channels changed: got {ch}. "
                f"Update docs and projection head examples."
            )
        finally:
            tr.cleanup()


# ── single step ──────────────────────────────────────────────────────────


class TestRealYOLOStep:
    def test_step_with_real_yolo(self):
        from yolo_contrastive.pretrain import DenseSSLPretrainer

        tr = DenseSSLPretrainer(
            model="yolov8n.pt",
            out_dim=32, queue_size=32, n_query=8,
            imgsz=64, device="cpu",
        )
        try:
            imgs = torch.rand(2, 3, 64, 64)
            out = tr._step(imgs)
            assert torch.isfinite(out["loss"]).item()
            assert out["batch_size"] == 2
            # Per-level info present
            for lv in ("P3", "P4", "P5"):
                assert lv in out["info"]
        finally:
            tr.cleanup()

    def test_grad_flows_through_real_yolo(self):
        from yolo_contrastive.pretrain import DenseSSLPretrainer

        tr = DenseSSLPretrainer(
            model="yolov8n.pt",
            out_dim=32, queue_size=32, n_query=8,
            imgsz=64, device="cpu",
        )
        try:
            imgs = torch.rand(2, 3, 64, 64)
            out = tr._step(imgs)
            out["loss"].backward()
            grads = [p.grad for p in tr.model.parameters()
                     if p.grad is not None]
            assert len(grads) > 0
            # At least some gradient magnitude
            total_grad = sum(g.abs().sum().item() for g in grads)
            assert total_grad > 0
        finally:
            tr.cleanup()


# ── EMA update ───────────────────────────────────────────────────────────


class TestRealYOLOEMA:
    def test_ema_updates_real_yolo(self):
        from yolo_contrastive.pretrain import DenseSSLPretrainer

        tr = DenseSSLPretrainer(
            model="yolov8n.pt",
            out_dim=32, queue_size=32, n_query=8,
            momentum=0.5,  # fast update
            imgsz=64, device="cpu",
        )
        try:
            mom_before = [p.detach().clone()
                          for p in tr.momentum.momentum.parameters()]
            with torch.no_grad():
                for p in tr.model.parameters():
                    p.data.add_(torch.randn_like(p.data) * 0.05)
            tr._ema_update()
            mom_after = list(tr.momentum.momentum.parameters())
            differs = any(
                not torch.equal(b, a.detach())
                for b, a in zip(mom_before, mom_after)
            )
            assert differs
        finally:
            tr.cleanup()


# ── full train() smoke ──────────────────────────────────────────────────


class TestRealYOLOTrain:
    def test_train_one_epoch(self):
        """Full train() loop with real YOLO + dummy images.

        Heavy-ish (~30s on CPU) but proves end-to-end integration.
        """
        from yolo_contrastive.pretrain import DenseSSLPretrainer

        tmp_imgs = _dummy_images_dir(n=4, size=64)
        tmp_out = tempfile.mkdtemp(prefix="ycl_dense_realyolo_out_")
        try:
            output = os.path.join(tmp_out, "backbone.pt")
            tr = DenseSSLPretrainer(
                model="yolov8n.pt",
                out_dim=32, queue_size=32, n_query=8,
                imgsz=64, device="cpu",
            )
            try:
                result = tr.train(
                    images_dir=tmp_imgs,
                    epochs=1, batch_size=2, lr=1e-3,
                    warmup_epochs=0, num_workers=0,
                    output=output, save_every=0, print_every=1,
                )
            finally:
                tr.cleanup()
            assert result == output
            assert os.path.exists(output)
            ckpt = torch.load(output, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt
            assert ckpt.get("extra", {}).get("type") == "dense_ssl"
        finally:
            shutil.rmtree(tmp_imgs, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)


# ── checkpoint cross-load (drop-in replacement check) ───────────────────


class TestRealYOLOCheckpointInterop:
    def test_save_then_load_into_finetune_path(self):
        """Save checkpoint with DenseSSLPretrainer, load via load_backbone
        the way FinetuneDetectionTrainer does it. This proves drop-in
        replacement compatibility.
        """
        from yolo_contrastive.pretrain import (
            DenseSSLPretrainer, save_backbone, load_backbone,
        )
        from ultralytics import YOLO

        tr = DenseSSLPretrainer(
            model="yolov8n.pt",
            out_dim=32, queue_size=32, n_query=8,
            imgsz=64, device="cpu",
        )
        tmp_dir = tempfile.mkdtemp(prefix="ycl_dense_interop_")
        try:
            # Step once so weights diverge from the pretrained baseline
            _ = tr._step(torch.rand(2, 3, 64, 64))
            path = os.path.join(tmp_dir, "dense_bb.pt")
            save_backbone(tr.model, path, epoch=1,
                          extra={"type": "dense_ssl"})
        finally:
            tr.cleanup()

        try:
            # Build a fresh YOLO model (the way finetune does)
            fresh = YOLO("yolov8n.pt").model
            n = load_backbone(fresh, path, strict=False, verbose=False,
                              backbone_only=True)
            assert n > 0, "no params loaded — incompatible checkpoint format"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
