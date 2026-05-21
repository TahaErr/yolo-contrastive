"""Hat B — Legacy Pretext + Adapters Hat integration smoke tests.

Covers 15 scenarios from INVENTORY.md §2.2 (revised against real API):
    B1-B8:   pretext/ — registry, heads, 6 tasks, CompositeTask
    B9-B12:  adapters/ — ConvLoRA, FreqGate, FreqGatedConvLoRA/TaskRouted, inject_lora
    B13-B15: pretrain/trainer.py — legacy SSLPretrainer (3 modes + adapter + train smoke)

Integration scope:
    pretext/ and adapters/ are FROZEN modules (WORK_PLAN_v9 §4: "❄ DONDURULDU").
    These smoke tests don't modify them — they pin the public API surface so
    the legacy comparison baseline (paper: `ours` vs legacy SSLPretrainer)
    can't silently rot. Unit-level invariants live in tests/test_pretext.py
    and tests/test_adapters.py; here we verify whole-path plumbing.

    B1-B11 use small synthetic tensors (pretext/adapter primitives don't
    need a real backbone) — ~1-2s each.
    B12 injects LoRA into a real YOLOv8n backbone.
    B15 runs a real 1-epoch SSLPretrainer.train() — @pytest.mark.slow.

Note on BatchNorm1d:
    PredictionHead / ProjectionHead contain BatchNorm1d, so every forward
    that hits a head needs batch >= 2. All head-touching tests use batch=4.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# Registry task names → (class_name, num_classes, difficulty)
# Verified against tests/test_pretext.py TASK_SPECS.
PRETEXT_SPECS = {
    "rotation":      ("RotationTask", 4, "trivial"),
    "solarization":  ("SolarizationTask", 4, "medium"),
    "color_perm":    ("ColorPermutationTask", 6, "hard"),
    "patch_shuffle": ("PatchShuffleTask", 24, "hard"),
    "blur":          ("BlurPredictionTask", 4, "medium"),
    "freq_band":     ("FrequencyBandPrediction", 7, "hard"),
}


# ═════════════════════════════════════════════════════════════════════════
# B1 — pretext registry: list_tasks / get_task / register_task
# ═════════════════════════════════════════════════════════════════════════


class TestB1_PretextRegistry:
    """Registry exposes exactly 6 tasks; get_task builds them; unknown raises."""

    def test_list_tasks_has_six(self):
        from yolo_contrastive.pretext import list_tasks
        tasks = list_tasks()
        assert len(tasks) == 6
        assert set(tasks) == set(PRETEXT_SPECS.keys())

    def test_get_task_builds_each(self):
        from yolo_contrastive.pretext import get_task
        for name, (cls_name, nc, diff) in PRETEXT_SPECS.items():
            task = get_task(name, feat_dim=64)
            assert type(task).__name__ == cls_name
            assert task.num_classes == nc
            assert task.difficulty == diff
            assert task.label_smoothing == 0.15

    def test_unknown_task_raises(self):
        from yolo_contrastive.pretext import get_task
        with pytest.raises(KeyError, match="nonexistent"):
            get_task("nonexistent", feat_dim=64)


# ═════════════════════════════════════════════════════════════════════════
# B2 — heads: PredictionHead + ProjectionHead forward shapes
# ═════════════════════════════════════════════════════════════════════════


class TestB2_Heads:
    """Both heads map [B, feat_dim] → expected output dims (batch>=2 for BN)."""

    def test_prediction_head_shape(self):
        from yolo_contrastive.pretext import PredictionHead
        head = PredictionHead(feat_dim=64, num_classes=7, hidden_dim=32)
        out = head(torch.randn(4, 64))
        assert out.shape == (4, 7)

    def test_projection_head_shape(self):
        from yolo_contrastive.pretext import ProjectionHead
        head = ProjectionHead(feat_dim=64, out_dim=128, hidden_dim=32)
        out = head(torch.randn(4, 64))
        assert out.shape == (4, 128)

    def test_prediction_head_grad_flow(self):
        from yolo_contrastive.pretext import PredictionHead
        head = PredictionHead(feat_dim=64, num_classes=4, hidden_dim=32)
        x = torch.randn(4, 64, requires_grad=True)
        head(x).sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


# ═════════════════════════════════════════════════════════════════════════
# B3 — RotationTask transform + forward
# ═════════════════════════════════════════════════════════════════════════


class TestB3_RotationTask:
    """RotationTask: transform produces 4-class labels, forward yields loss+acc."""

    def test_transform_and_forward(self):
        from yolo_contrastive.pretext import RotationTask

        task = RotationTask(feat_dim=64, hidden_dim=32)
        imgs = torch.rand(4, 3, 32, 32)
        rotated, labels = task.transform(imgs)

        assert rotated.shape == imgs.shape
        assert labels.shape == (4,)
        assert labels.dtype == torch.long
        assert labels.min() >= 0 and labels.max() < 4

        # forward: features [B, feat_dim] → (loss, accuracy)
        features = torch.randn(4, 64)
        loss, acc = task(features, labels)
        assert torch.isfinite(loss).item()
        assert 0.0 <= acc <= 1.0

    def test_rotate_batch_legacy_alias(self):
        from yolo_contrastive.pretext import RotationTask
        task = RotationTask(feat_dim=64, hidden_dim=32)
        imgs = torch.rand(4, 3, 32, 32)
        # rotate_batch is the documented legacy alias for transform
        out, labels = task.rotate_batch(imgs)
        assert out.shape == imgs.shape
        assert labels.shape == (4,)


# ═════════════════════════════════════════════════════════════════════════
# B4 — tasks.py: Solarization / ColorPermutation / PatchShuffle / Blur
# ═════════════════════════════════════════════════════════════════════════


class TestB4_TasksModule:
    """All 4 tasks.py pretext tasks: transform shape + forward loss/acc."""

    @pytest.mark.parametrize("name", ["solarization", "color_perm",
                                      "patch_shuffle", "blur"])
    def test_transform_forward(self, name):
        from yolo_contrastive.pretext import get_task

        nc = PRETEXT_SPECS[name][1]
        task = get_task(name, feat_dim=64)
        imgs = torch.rand(4, 3, 32, 32)

        out, labels = task.transform(imgs)
        assert out.shape == imgs.shape
        assert labels.shape == (4,)
        assert labels.dtype == torch.long
        assert labels.min() >= 0 and labels.max() < nc

        loss, acc = task(torch.randn(4, 64), labels)
        assert torch.isfinite(loss).item()
        assert 0.0 <= acc <= 1.0


# ═════════════════════════════════════════════════════════════════════════
# B5 — FrequencyBandPrediction 7-class + freq mask invariant
# ═════════════════════════════════════════════════════════════════════════


class TestB5_FrequencyBandPrediction:
    """freq_band: 7-class transform/forward + _apply_freq_mask band removal."""

    def test_transform_forward_7class(self):
        from yolo_contrastive.pretext import FrequencyBandPrediction

        task = FrequencyBandPrediction(feat_dim=64, hidden_dim=32)
        assert task.num_classes == 7

        imgs = torch.rand(4, 3, 64, 64)
        out, labels = task.transform(imgs)
        assert out.shape == imgs.shape
        assert labels.min() >= 0 and labels.max() < 7

        loss, acc = task(torch.randn(4, 64), labels)
        assert torch.isfinite(loss).item()

    def test_low_band_mask_reduces_center_energy(self):
        """band_id=1 (low removed) → FFT center magnitude drops."""
        from yolo_contrastive.pretext import FrequencyBandPrediction

        task = FrequencyBandPrediction(feat_dim=64, hidden_dim=32)
        img = torch.rand(3, 64, 64)
        masked = task._apply_freq_mask(img, band_id=1)
        assert masked.shape == img.shape

        orig_fft = torch.fft.fftshift(torch.fft.fft2(img))
        masked_fft = torch.fft.fftshift(torch.fft.fft2(masked))
        # Center 6x6 = low frequency region
        orig_center = orig_fft[:, 29:35, 29:35].abs().mean().item()
        masked_center = masked_fft[:, 29:35, 29:35].abs().mean().item()
        assert masked_center < orig_center * 0.5

    @pytest.mark.parametrize("band_id", [4, 5, 6])
    def test_dual_band_mask_valid(self, band_id):
        from yolo_contrastive.pretext import FrequencyBandPrediction
        task = FrequencyBandPrediction(feat_dim=64, hidden_dim=32)
        img = torch.rand(3, 64, 64)
        out = task._apply_freq_mask(img, band_id=band_id)
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0


# ═════════════════════════════════════════════════════════════════════════
# B6 — CompositeTask transform + forward (multi-task)
# ═════════════════════════════════════════════════════════════════════════


class TestB6_CompositeTask:
    """CompositeTask.from_names: cumulative transform → labels_dict;
    forward → (total_loss, avg_acc, details)."""

    def test_from_names_transform_forward(self):
        from yolo_contrastive.pretext import CompositeTask

        composite = CompositeTask.from_names(
            ["rotation", "solarization", "freq_band"],
            feat_dim=64, hidden_dim=32,
            weights=[1.0, 0.8, 0.5],
        )
        assert composite.num_heads == 3
        assert set(composite.task_names) == {"rotation", "solarization", "freq_band"}

        imgs = torch.rand(4, 3, 64, 64)
        augmented, labels_dict = composite.transform(imgs)
        assert augmented.shape == imgs.shape
        # labels_dict keyed by task_name
        assert set(labels_dict.keys()) == {"rotation", "solarization", "freq_band"}
        for lv in labels_dict.values():
            assert lv.shape == (4,)

        # forward returns 3-tuple
        features = torch.randn(4, 64)
        total_loss, avg_acc, details = composite(features, labels_dict)
        assert torch.isfinite(total_loss).item()
        assert 0.0 <= avg_acc <= 1.0
        assert set(details.keys()) == {"rotation", "solarization", "freq_band"}
        # Per-task detail carries loss/acc/weight
        for d in details.values():
            assert "loss" in d and "acc" in d and "weight" in d

    def test_weighted_sum_consistency(self):
        """total_loss equals Σ weight_i · task_loss_i."""
        from yolo_contrastive.pretext import CompositeTask

        composite = CompositeTask.from_names(
            ["rotation", "blur"], feat_dim=64, hidden_dim=32, weights=[1.0, 0.5],
        )
        imgs = torch.rand(4, 3, 32, 32)
        _, labels_dict = composite.transform(imgs)
        features = torch.randn(4, 64)
        total_loss, _, details = composite(features, labels_dict)

        manual = (1.0 * details["rotation"]["loss"]
                  + 0.5 * details["blur"]["loss"])
        assert torch.allclose(total_loss, manual, atol=1e-5)


# ═════════════════════════════════════════════════════════════════════════
# B7 — CompositeTask validation
# ═════════════════════════════════════════════════════════════════════════


class TestB7_CompositeTaskValidation:
    """Empty task list and weight/task length mismatch both raise ValueError."""

    def test_empty_tasks_raises(self):
        from yolo_contrastive.pretext import CompositeTask
        with pytest.raises(ValueError):
            CompositeTask.from_names([], feat_dim=64)

    def test_weight_mismatch_raises(self):
        from yolo_contrastive.pretext import CompositeTask, get_task
        with pytest.raises(ValueError):
            CompositeTask(
                [get_task("rotation", feat_dim=64)],
                weights=[1.0, 2.0],  # 2 weights, 1 task
            )


# ═════════════════════════════════════════════════════════════════════════
# B8 — full registry transform-shape invariant
# ═════════════════════════════════════════════════════════════════════════


class TestB8_AllTasksTransformInvariant:
    """Every registered task: transform preserves image shape, labels are
    long [B] within [0, num_classes)."""

    @pytest.mark.parametrize("name", list(PRETEXT_SPECS.keys()))
    def test_transform_shape_invariant(self, name):
        from yolo_contrastive.pretext import get_task

        nc = PRETEXT_SPECS[name][1]
        task = get_task(name, feat_dim=64)
        imgs = torch.rand(4, 3, 64, 64)
        out, labels = task.transform(imgs)

        assert out.shape == imgs.shape
        assert out.dtype == imgs.dtype
        assert labels.shape == (4,)
        assert labels.dtype == torch.long
        assert int(labels.min()) >= 0
        assert int(labels.max()) < nc


# ═════════════════════════════════════════════════════════════════════════
# B9 — ConvLoRA
# ═════════════════════════════════════════════════════════════════════════


class TestB9_ConvLoRA:
    """ConvLoRA: shape preservation, zero-init (output == frozen conv),
    frozen base conv, merge_weights roundtrip."""

    def test_forward_shape(self):
        from yolo_contrastive.adapters import ConvLoRA
        conv = nn.Conv2d(32, 64, 3, padding=1)
        lora = ConvLoRA(conv, rank=4)
        out = lora(torch.randn(4, 32, 16, 16))
        assert out.shape == (4, 64, 16, 16)

    def test_zero_init_matches_conv(self):
        """At init, LoRA path is zero → output equals frozen conv."""
        from yolo_contrastive.adapters import ConvLoRA
        conv = nn.Conv2d(32, 64, 3, padding=1)
        lora = ConvLoRA(conv, rank=4)
        x = torch.randn(2, 32, 8, 8)
        with torch.no_grad():
            diff = (lora(x) - conv(x)).abs().max().item()
        assert diff < 1e-5

    def test_base_conv_frozen(self):
        from yolo_contrastive.adapters import ConvLoRA
        conv = nn.Conv2d(32, 64, 3, padding=1)
        lora = ConvLoRA(conv, rank=4)
        assert not conv.weight.requires_grad
        assert lora.lora_down.weight.requires_grad

    def test_merge_weights_roundtrip(self):
        from yolo_contrastive.adapters import ConvLoRA
        conv = nn.Conv2d(32, 64, 3, padding=1)
        lora = ConvLoRA(conv, rank=4)
        merged = lora.merge_weights()
        assert isinstance(merged, nn.Conv2d)
        x = torch.randn(2, 32, 8, 8)
        with torch.no_grad():
            diff = (merged(x) - lora(x)).abs().max().item()
        assert diff < 1e-4


# ═════════════════════════════════════════════════════════════════════════
# B10 — FreqGate
# ═════════════════════════════════════════════════════════════════════════


class TestB10_FreqGate:
    """FreqGate: [B,C,H,W] → [B,1,1,1] gate in [0,1]; tiny param count; grad."""

    def test_gate_shape_and_range(self):
        from yolo_contrastive.adapters import FreqGate
        gate = FreqGate(hidden_dim=16)
        g = gate(torch.randn(4, 32, 32, 32))
        assert g.shape == (4, 1, 1, 1)
        assert g.min() >= 0.0 and g.max() <= 1.0

    def test_few_params(self):
        from yolo_contrastive.adapters import FreqGate
        gate = FreqGate(hidden_dim=16)
        assert gate.num_params < 200

    def test_gradient_flows(self):
        from yolo_contrastive.adapters import FreqGate
        gate = FreqGate(hidden_dim=16)
        gate(torch.randn(4, 32, 16, 16)).sum().backward()
        assert gate.mlp[0].weight.grad is not None


# ═════════════════════════════════════════════════════════════════════════
# B11 — FreqGatedConvLoRA + TaskRoutedConvLoRA
# ═════════════════════════════════════════════════════════════════════════


class TestB11_GatedAdapters:
    """Frequency-gated and task-routed LoRA: forward shape + param efficiency
    (trainable params < frozen params)."""

    def test_freq_gated_forward_and_efficiency(self):
        from yolo_contrastive.adapters import FreqGatedConvLoRA
        conv = nn.Conv2d(32, 64, 3, padding=1)
        fgl = FreqGatedConvLoRA(conv, rank=4)
        x = torch.randn(4, 32, 32, 32, requires_grad=True)
        out = fgl(x)
        assert out.shape == (4, 64, 32, 32)
        out.sum().backward()
        assert x.grad is not None
        # Parameter-efficient: LoRA + gate << frozen conv
        assert fgl.num_trainable_params < fgl.num_frozen_params

    def test_task_routed_forward(self):
        from yolo_contrastive.adapters import TaskRoutedConvLoRA
        conv = nn.Conv2d(32, 64, 3, padding=1)
        tr = TaskRoutedConvLoRA(conv, num_tasks=3, rank=4, use_gate=False)
        out = tr(torch.randn(4, 32, 16, 16))
        assert out.shape == (4, 64, 16, 16)
        assert tr.num_trainable_params < tr.num_frozen_params


# ═════════════════════════════════════════════════════════════════════════
# B12 — inject_lora into real YOLOv8n backbone
# ═════════════════════════════════════════════════════════════════════════


class TestB12_InjectLoRA:
    """inject_lora adapts a real YOLOv8n backbone; remove_lora reverses it."""

    def test_inject_and_remove(self, yolov8n_weights_path):
        from ultralytics import YOLO
        from yolo_contrastive.adapters import inject_lora, remove_lora

        model = YOLO(yolov8n_weights_path).model

        info = inject_lora(
            model, rank=4, adapter_type="freq_gated",
            backbone_layers=10, verbose=False,
        )
        # Contract: info dict reports injected count + param budgets
        assert info["injected"] > 0
        assert info["lora_params"] > 0
        assert info["total_trainable"] > 0

        # Forward still works after injection
        out = model(torch.randn(1, 3, 160, 160))
        assert out is not None

        # remove_lora reverses (returns count of removed adapters)
        removed = remove_lora(model, merge=True, verbose=False)
        assert removed == info["injected"]


# ═════════════════════════════════════════════════════════════════════════
# B13 — SSLPretrainer construction (3 modes)
# ═════════════════════════════════════════════════════════════════════════


class TestB13_SSLPretrainerModes:
    """Legacy SSLPretrainer: composite / legacy-rotation / cl-only init paths."""

    def test_composite_mode(self):
        from yolo_contrastive.pretrain import SSLPretrainer
        from yolo_contrastive.pretext import CompositeTask

        pt = SSLPretrainer(
            model="yolov8n.pt", aug_preset="simclr_v2", lambda_cl=1.0,
            pretext_tasks=["freq_band", "solarization"],
            pretext_weights=[1.0, 0.8], lambda_pretext=0.5, imgsz=64,
        )
        try:
            assert isinstance(pt.pretext_task, CompositeTask)
            assert pt._has_pretext
        finally:
            pt.cleanup()

    def test_legacy_rotation_mode(self):
        from yolo_contrastive.pretrain import SSLPretrainer
        pt = SSLPretrainer(
            model="yolov8n.pt", aug_preset="simclr_v2", lambda_cl=1.0,
            lambda_rot=0.3, imgsz=64,
        )
        try:
            assert pt.rot_task is not None
            assert pt.pretext_task is None
        finally:
            pt.cleanup()

    def test_cl_only_mode(self):
        from yolo_contrastive.pretrain import SSLPretrainer
        pt = SSLPretrainer(
            model="yolov8n.pt", aug_preset="simclr_v2", lambda_cl=1.0,
            lambda_rot=0.0, imgsz=64,
        )
        try:
            assert not pt._has_pretext
        finally:
            pt.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# B14 — SSLPretrainer + adapter injection
# ═════════════════════════════════════════════════════════════════════════


class TestB14_SSLPretrainerAdapter:
    """SSLPretrainer wires LoRA adapters during construction."""

    def test_freq_gated_adapter(self):
        from yolo_contrastive.pretrain import SSLPretrainer
        pt = SSLPretrainer(
            model="yolov8n.pt", aug_preset="simclr_v2", lambda_cl=1.0,
            pretext_tasks=["freq_band"], lambda_pretext=0.5,
            adapter="freq_gated", adapter_rank=4, imgsz=64,
        )
        try:
            assert pt._adapter_info is not None
            assert pt._adapter_info["injected"] > 0
        finally:
            pt.cleanup()

    def test_task_routed_adapter(self):
        from yolo_contrastive.pretrain import SSLPretrainer
        pt = SSLPretrainer(
            model="yolov8n.pt", aug_preset="simclr_v2", lambda_cl=1.0,
            pretext_tasks=["freq_band", "solarization", "patch_shuffle"],
            pretext_weights=[1.0, 0.8, 0.5], lambda_pretext=0.5,
            adapter="task_routed", adapter_rank=4, imgsz=64,
        )
        try:
            assert pt._adapter_info is not None
            assert pt._task_router is not None
            assert pt._task_router.num_tasks == 3
        finally:
            pt.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# B15 — SSLPretrainer train() 1-epoch smoke
# ═════════════════════════════════════════════════════════════════════════


class TestB15_SSLPretrainerTrain:
    """Legacy SSLPretrainer end-to-end: real 1-epoch composite-pretext run."""

    @pytest.mark.slow
    def test_train_1epoch_composite(self, dummy_images_dir, tmp_workspace):
        from yolo_contrastive.pretrain import SSLPretrainer

        img_dir = dummy_images_dir(n=6, size=64, name="ssl_legacy")
        out_path = tmp_workspace / "legacy_backbone.pt"

        pt = SSLPretrainer(
            model="yolov8n.pt", aug_preset="simclr_v2", lambda_cl=1.0,
            pretext_tasks=["solarization", "blur"],
            pretext_weights=[1.0, 0.5], lambda_pretext=0.3, imgsz=64,
        )
        try:
            result = pt.train(
                images_dir=str(img_dir), epochs=1, batch_size=4,
                lr=1e-3, warmup_epochs=0, num_workers=0,
                output=str(out_path), save_every=0, print_every=1,
            )
            assert result == str(out_path)
            assert out_path.exists()
        finally:
            pt.cleanup()
