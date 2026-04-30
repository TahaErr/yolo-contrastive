"""Tests for LinearProbeTrainer and multi-label mAP."""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from yolo_contrastive.eval import LinearProbeTrainer, LinearProbeHead
from yolo_contrastive.eval.linear_probe import multilabel_average_precision


# ── helpers ──────────────────────────────────────────────────────────────


def _mock_yolo_encoder(channels: tuple = (32, 64, 128)) -> nn.Sequential:
    """23-layer Sequential mimicking YOLOv8 backbone+neck topology
    (same convention as Faz 1 tests)."""
    p3, p4, p5 = channels
    layers = []
    layers.append(nn.Conv2d(3, p3, kernel_size=3, stride=2, padding=1))   # 0
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, kernel_size=3, padding=1))        # 1-5
    layers.append(nn.Conv2d(p3, p3, kernel_size=3, stride=2, padding=1))  # 6
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, kernel_size=3, padding=1))        # 7-11
    layers.append(nn.Conv2d(p3, p3, kernel_size=3, stride=2, padding=1))  # 12
    layers.append(nn.Conv2d(p3, p3, kernel_size=3, padding=1))            # 13
    layers.append(nn.Conv2d(p3, p3, kernel_size=3, padding=1))            # 14
    layers.append(nn.Conv2d(p3, p3, kernel_size=3, padding=1))            # 15: P3
    layers.append(nn.Conv2d(p3, p4, kernel_size=3, stride=2, padding=1))  # 16
    layers.append(nn.Conv2d(p4, p4, kernel_size=3, padding=1))            # 17
    layers.append(nn.Conv2d(p4, p4, kernel_size=3, padding=1))            # 18: P4
    layers.append(nn.Conv2d(p4, p5, kernel_size=3, stride=2, padding=1))  # 19
    layers.append(nn.Conv2d(p5, p5, kernel_size=3, padding=1))            # 20
    layers.append(nn.Conv2d(p5, p5, kernel_size=3, padding=1))            # 21: P5
    layers.append(nn.Conv2d(p5, p5, kernel_size=1))                       # 22
    return nn.Sequential(*layers)


class _RandomMultiLabelDataset(Dataset):
    """Random images with random multi-hot labels."""

    def __init__(self, n: int, num_classes: int, imgsz: int = 64,
                 seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.imgs = torch.rand(n, 3, imgsz, imgsz, generator=gen)
        # Each image: each class with prob 0.3 → multi-hot
        probs = torch.rand(n, num_classes, generator=gen)
        self.labels = (probs < 0.3).float()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.labels[i]


class _LearnableMultiLabelDataset(Dataset):
    """Synthetic task: label depends linearly on a known feature signature.

    Each image has a "color signature" in its first 3 pixels — class C is
    "active" if pixel[C, 0, 0] > 0.5. Linear probe should be able to
    recover this perfectly (high mAP) given enough epochs, AS LONG AS
    the backbone preserves spatial information.

    But mock 23-layer encoder with random init may NOT preserve enough
    signal — so this test asserts only DIRECTIONAL improvement, not high
    absolute mAP.
    """

    def __init__(self, n: int, num_classes: int, imgsz: int = 64,
                 seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.imgs = torch.rand(n, 3, imgsz, imgsz, generator=gen)
        # Encode label in pixel (0, 0, 0/1/2)
        self.labels = (self.imgs[:, :num_classes, 0, 0] > 0.5).float()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.labels[i]


def _loader(ds: Dataset, batch_size: int = 4) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def _make_probe(num_classes: int = 5, feat_level: str = "P5") -> LinearProbeTrainer:
    return LinearProbeTrainer(
        backbone=_mock_yolo_encoder(),
        num_classes=num_classes,
        feat_level=feat_level,
        device="cpu",
    )


# ═════════════════════════════════════════════════════════════════════════
# Multi-label mAP unit tests
# ═════════════════════════════════════════════════════════════════════════


class TestMultilabelMAP:
    def test_perfect_predictions_give_one(self):
        """Logits that perfectly separate positive/negative → AP = 1 per class."""
        # 4 samples, 2 classes; perfect score = high logit iff positive
        targets = torch.tensor([
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
        ], dtype=torch.float32)
        # Scores are higher for positive entries
        logits = torch.tensor([
            [5.0, -5.0],
            [4.0, 5.0],
            [-3.0, -4.0],
            [-2.0, 4.0],
        ])
        result = multilabel_average_precision(logits, targets)
        assert result["mAP"] == 1.0
        assert (result["per_class_ap"] == 1.0).all()
        assert result["n_valid_classes"] == 2

    def test_class_with_no_positives_skipped(self):
        targets = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)  # all zero
        logits = torch.randn(2, 2)
        result = multilabel_average_precision(logits, targets)
        assert result["n_valid_classes"] == 0
        assert result["mAP"] == 0.0

    def test_random_predictions_around_chance(self):
        """Random predictions on a balanced dataset → mAP ≈ positive_rate."""
        torch.manual_seed(0)
        N, C = 200, 5
        targets = (torch.rand(N, C) < 0.3).float()  # 30% positive rate
        logits = torch.randn(N, C)
        result = multilabel_average_precision(logits, targets)
        # Random ranking → AP ≈ positive_rate ≈ 0.3 (loose bound)
        assert 0.15 < result["mAP"] < 0.5

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="logits"):
            multilabel_average_precision(
                torch.randn(2, 3), torch.zeros(2, 4),
            )

    def test_wrong_dim_raises(self):
        with pytest.raises(ValueError, match=r"\[N, C\]"):
            multilabel_average_precision(
                torch.randn(5), torch.zeros(5),
            )


# ═════════════════════════════════════════════════════════════════════════
# LinearProbeHead unit tests
# ═════════════════════════════════════════════════════════════════════════


class TestProbeHead:
    def test_basic_forward(self):
        head = LinearProbeHead(in_dim=32, num_classes=5)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 5)

    def test_normalize_changes_output(self):
        torch.manual_seed(0)
        h_norm = LinearProbeHead(in_dim=32, num_classes=5, normalize=True)
        h_raw = LinearProbeHead(in_dim=32, num_classes=5, normalize=False)
        # Use same weights for fair comparison
        h_norm.fc.weight.data.copy_(h_raw.fc.weight.data)
        h_norm.fc.bias.data.copy_(h_raw.fc.bias.data)
        x = torch.randn(4, 32) * 10  # large magnitude
        out_norm = h_norm(x)
        out_raw = h_raw(x)
        assert not torch.allclose(out_norm, out_raw)

    def test_invalid_init(self):
        with pytest.raises(ValueError, match="in_dim"):
            LinearProbeHead(in_dim=0, num_classes=5)
        with pytest.raises(ValueError, match="num_classes"):
            LinearProbeHead(in_dim=32, num_classes=0)

    def test_wrong_input_shape(self):
        head = LinearProbeHead(in_dim=32, num_classes=5)
        with pytest.raises(ValueError, match=r"\[B, D\]"):
            head(torch.randn(4, 32, 8))  # 3D


# ═════════════════════════════════════════════════════════════════════════
# LinearProbeTrainer construction
# ═════════════════════════════════════════════════════════════════════════


class TestProbeConstruction:
    def test_basic(self):
        probe = _make_probe()
        try:
            assert probe.num_classes == 5
            assert probe.feat_level == "P5"
            assert probe.head.in_dim == 128  # mock P5 channels
        finally:
            probe.cleanup()

    def test_feat_levels(self):
        for lv, expected_dim in [("P3", 32), ("P4", 64), ("P5", 128)]:
            probe = _make_probe(feat_level=lv)
            try:
                assert probe.feat_level == lv
                assert probe.head.in_dim == expected_dim
            finally:
                probe.cleanup()

    def test_invalid_feat_level(self):
        with pytest.raises(ValueError, match="feat_level"):
            LinearProbeTrainer(
                backbone=_mock_yolo_encoder(),
                num_classes=5,
                feat_level="P9",
                device="cpu",
            )

    def test_invalid_num_classes(self):
        with pytest.raises(ValueError, match="num_classes"):
            LinearProbeTrainer(
                backbone=_mock_yolo_encoder(),
                num_classes=0,
                device="cpu",
            )

    def test_repr(self):
        probe = _make_probe()
        try:
            r = repr(probe)
            assert "LinearProbeTrainer" in r
            assert "feat_level='P5'" in r
        finally:
            probe.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Backbone is hard-frozen (CRITICAL)
# ═════════════════════════════════════════════════════════════════════════


class TestBackboneFrozen:
    def test_backbone_params_have_no_grad(self):
        probe = _make_probe()
        try:
            for p in probe.backbone.parameters():
                assert not p.requires_grad, "backbone param has requires_grad=True"
        finally:
            probe.cleanup()

    def test_backbone_in_eval_mode(self):
        probe = _make_probe()
        try:
            assert not probe.backbone.training
        finally:
            probe.cleanup()

    def test_backbone_grads_zero_after_optim_step(self):
        """Run one fit() epoch; backbone params must have None or zero grads."""
        probe = _make_probe(num_classes=3)
        try:
            ds = _RandomMultiLabelDataset(n=8, num_classes=3, imgsz=32, seed=0)
            probe.fit(_loader(ds, batch_size=4),
                       _loader(ds, batch_size=4),
                       epochs=1, verbose=False)
            for p in probe.backbone.parameters():
                # Either no grad attribute or zero grad
                assert p.grad is None or p.grad.abs().sum().item() == 0.0, (
                    "backbone got non-zero gradient — freeze broke!"
                )
        finally:
            probe.cleanup()

    def test_only_head_params_change(self):
        """Snapshot backbone weights, train, verify they didn't move."""
        probe = _make_probe(num_classes=3)
        try:
            # Snapshot
            bb_snapshot = {
                name: p.detach().clone()
                for name, p in probe.backbone.named_parameters()
            }
            head_snapshot = {
                name: p.detach().clone()
                for name, p in probe.head.named_parameters()
            }

            ds = _RandomMultiLabelDataset(n=8, num_classes=3, imgsz=32, seed=0)
            probe.fit(_loader(ds, batch_size=4),
                       _loader(ds, batch_size=4),
                       epochs=2, verbose=False)

            # Backbone must NOT have moved
            for name, p in probe.backbone.named_parameters():
                assert torch.equal(p, bb_snapshot[name]), (
                    f"backbone param {name} changed!"
                )

            # Head SHOULD have moved
            head_moved = False
            for name, p in probe.head.named_parameters():
                if not torch.equal(p, head_snapshot[name]):
                    head_moved = True
                    break
            assert head_moved, "head parameters didn't move during training"
        finally:
            probe.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Training & evaluation
# ═════════════════════════════════════════════════════════════════════════


class TestTraining:
    def test_fit_returns_history(self):
        probe = _make_probe(num_classes=3)
        try:
            ds = _RandomMultiLabelDataset(n=12, num_classes=3, imgsz=32, seed=0)
            result = probe.fit(_loader(ds), _loader(ds),
                                epochs=3, verbose=False)
            assert "best_val_mAP" in result
            assert "best_epoch" in result
            assert "final_val_mAP" in result
            assert "history" in result
            assert len(result["history"]) == 3
            for h in result["history"]:
                for k in ("epoch", "train_loss", "val_mAP"):
                    assert k in h
        finally:
            probe.cleanup()

    def test_evaluate_runs(self):
        probe = _make_probe(num_classes=3)
        try:
            ds = _RandomMultiLabelDataset(n=8, num_classes=3, imgsz=32, seed=0)
            metrics = probe.evaluate(_loader(ds))
            assert "mAP" in metrics
            assert "per_class_ap" in metrics
            assert metrics["per_class_ap"].shape == (3,)
        finally:
            probe.cleanup()

    def test_evaluate_empty_loader(self):
        """Empty loader → metrics return safe zeros, no crash."""
        probe = _make_probe(num_classes=3)
        try:
            ds = _RandomMultiLabelDataset(n=0, num_classes=3, imgsz=32, seed=0)
            # shuffle=False because RandomSampler rejects empty datasets
            empty_loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
            metrics = probe.evaluate(empty_loader)
            assert metrics["mAP"] == 0.0
        finally:
            probe.cleanup()


class TestEarlyStopping:
    """Optional early stopping by val_mAP plateau."""

    def test_default_no_early_stop(self):
        """patience=None (default) → all epochs run, fields present."""
        probe = _make_probe(num_classes=3)
        try:
            ds = _RandomMultiLabelDataset(n=8, num_classes=3, imgsz=32, seed=0)
            result = probe.fit(_loader(ds), _loader(ds),
                                epochs=4, verbose=False)
            assert result["epochs_run"] == 4
            assert result["early_stopped"] is False
            assert len(result["history"]) == 4
        finally:
            probe.cleanup()

    def test_patience_triggers(self):
        """If val_mAP plateaus for `patience` epochs, training stops early.

        We use a deterministic-ish setup: small training dataset → mAP
        likely plateaus quickly. With patience=1, any non-improving epoch
        after the first triggers stop.
        """
        torch.manual_seed(0)
        probe = _make_probe(num_classes=3)
        try:
            ds = _RandomMultiLabelDataset(n=12, num_classes=3, imgsz=32, seed=0)
            # epochs=10 is the cap; we expect to stop well before
            result = probe.fit(_loader(ds), _loader(ds),
                                epochs=10, lr=1e-3, verbose=False,
                                early_stopping_patience=1)
            # Should not run all 10 epochs in plateau scenario
            assert result["epochs_run"] < 10
            # And early_stopped must be True
            assert result["early_stopped"] is True
            # best_epoch must be inside [1, epochs_run]
            assert 1 <= result["best_epoch"] <= result["epochs_run"]
        finally:
            probe.cleanup()

    def test_patience_huge_runs_full(self):
        """If patience is larger than total epochs, behavior matches no-stop."""
        probe = _make_probe(num_classes=3)
        try:
            ds = _RandomMultiLabelDataset(n=8, num_classes=3, imgsz=32, seed=0)
            result = probe.fit(_loader(ds), _loader(ds),
                                epochs=3, verbose=False,
                                early_stopping_patience=100)
            assert result["epochs_run"] == 3
            assert result["early_stopped"] is False
        finally:
            probe.cleanup()

    def test_invalid_patience_raises(self):
        probe = _make_probe(num_classes=3)
        try:
            ds = _RandomMultiLabelDataset(n=4, num_classes=3, imgsz=32, seed=0)
            with pytest.raises(ValueError, match="early_stopping_patience"):
                probe.fit(_loader(ds), _loader(ds),
                           epochs=2, verbose=False,
                           early_stopping_patience=0)
            with pytest.raises(ValueError, match="early_stopping_patience"):
                probe.fit(_loader(ds), _loader(ds),
                           epochs=2, verbose=False,
                           early_stopping_patience=-1)
        finally:
            probe.cleanup()


class TestLearningSignal:
    """The probe head should learn — given enough epochs on a learnable task,
    val_mAP should improve over training. We use a generous tolerance because
    mock encoder + random init is a noisy setup."""

    def test_loss_decreases(self):
        torch.manual_seed(0)
        probe = _make_probe(num_classes=3)
        try:
            ds_train = _LearnableMultiLabelDataset(n=64, num_classes=3,
                                                     imgsz=32, seed=0)
            ds_val = _LearnableMultiLabelDataset(n=32, num_classes=3,
                                                   imgsz=32, seed=1)
            result = probe.fit(
                _loader(ds_train, batch_size=8),
                _loader(ds_val, batch_size=8),
                epochs=8, lr=1e-2, verbose=False,
            )
            losses = [h["train_loss"] for h in result["history"]]
            # Directional: first epoch loss > last epoch loss
            assert losses[-1] < losses[0], (
                f"train loss didn't decrease: first={losses[0]:.4f}, "
                f"last={losses[-1]:.4f}, all={losses}"
            )
        finally:
            probe.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Cleanup
# ═════════════════════════════════════════════════════════════════════════


class TestCleanup:
    def test_cleanup_closes_tap(self):
        probe = _make_probe()
        assert probe.tap._is_setup
        probe.cleanup()
        assert not probe.tap._is_setup

    def test_cleanup_idempotent(self):
        probe = _make_probe()
        probe.cleanup()
        probe.cleanup()  # no error
