"""Test pretext tasks, composite, and freq_band."""

import pytest
import torch
from yolo_contrastive.pretext import (
    get_task, list_tasks, CompositeTask,
    RotationTask, SolarizationTask, ColorPermutationTask,
    PatchShuffleTask, BlurPredictionTask, FrequencyBandPrediction,
)

TASK_SPECS = {
    "rotation":     {"cls": RotationTask,          "nc": 4,  "diff": "trivial"},
    "solarization": {"cls": SolarizationTask,      "nc": 4,  "diff": "medium"},
    "color_perm":   {"cls": ColorPermutationTask,  "nc": 6,  "diff": "hard"},
    "patch_shuffle": {"cls": PatchShuffleTask,     "nc": 24, "diff": "hard"},
    "blur":         {"cls": BlurPredictionTask,    "nc": 4,  "diff": "medium"},
    "freq_band":    {"cls": FrequencyBandPrediction, "nc": 7, "diff": "hard"},
}


def test_registry():
    registry = list_tasks()
    assert len(registry) == 6
    assert set(registry) == set(TASK_SPECS.keys())


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        get_task("nonexistent", feat_dim=256)


@pytest.mark.parametrize("name,spec", TASK_SPECS.items())
def test_task_properties(name, spec):
    task = get_task(name, feat_dim=256)
    assert isinstance(task, spec["cls"])
    assert task.num_classes == spec["nc"]
    assert task.difficulty == spec["diff"]
    assert task.label_smoothing == 0.15


@pytest.mark.parametrize("name,spec", TASK_SPECS.items())
def test_task_transform(name, spec, batch_img):
    task = get_task(name, feat_dim=256).to(batch_img.device)
    out, labels = task.transform(batch_img)
    assert out.shape == batch_img.shape
    assert labels.shape == (batch_img.shape[0],)
    assert labels.dtype == torch.long
    assert labels.min() >= 0 and labels.max() < spec["nc"]


@pytest.mark.parametrize("name", TASK_SPECS.keys())
def test_task_forward_gradient(name, device):
    task = get_task(name, feat_dim=256).to(device)
    feat = torch.randn(4, 256, device=device, requires_grad=True)
    labels = torch.randint(0, task.num_classes, (4,), device=device)
    loss, acc = task(feat, labels)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    loss.backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()


class TestCompositeTask:
    def test_3task(self, batch_img):
        ct = CompositeTask.from_names(
            ["freq_band", "solarization", "patch_shuffle"],
            feat_dim=256, weights=[1.0, 0.8, 0.5],
        ).to(batch_img.device)
        assert ct.num_heads == 3
        assert ct.total_classes == 7 + 4 + 24

    def test_6task(self, batch_img):
        ct = CompositeTask.from_names(
            list(TASK_SPECS.keys()), feat_dim=256,
        ).to(batch_img.device)
        assert ct.num_heads == 6
        assert ct.total_classes == 4 + 4 + 6 + 24 + 4 + 7

    def test_transform_and_forward(self, batch_img):
        ct = CompositeTask.from_names(
            ["freq_band", "solarization"], feat_dim=256,
        ).to(batch_img.device)
        out, labels = ct.transform(batch_img)
        assert out.shape == batch_img.shape
        assert len(labels) == 2

        feat = torch.randn(4, 256, device=batch_img.device, requires_grad=True)
        loss, acc, det = ct(feat, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert feat.grad is not None

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            CompositeTask.from_names([], feat_dim=256)

    def test_weight_mismatch_raises(self):
        with pytest.raises(ValueError):
            CompositeTask(
                [get_task("rotation", feat_dim=256)],
                weights=[1.0, 2.0],
            )


class TestFreqBandVerification:
    def test_low_mask_reduces_center_energy(self, device):
        fb = get_task("freq_band", feat_dim=256).to(device)
        img = torch.rand(3, 64, 64, device=device)
        masked = fb._apply_freq_mask(img, band_id=1)
        orig_fft = torch.fft.fftshift(torch.fft.fft2(img))
        masked_fft = torch.fft.fftshift(torch.fft.fft2(masked))
        orig_e = orig_fft[:, 29:35, 29:35].abs().mean().item()
        masked_e = masked_fft[:, 29:35, 29:35].abs().mean().item()
        assert masked_e < orig_e * 0.5

    def test_high_mask_reduces_corner_energy(self, device):
        fb = get_task("freq_band", feat_dim=256).to(device)
        img = torch.rand(3, 64, 64, device=device)
        masked = fb._apply_freq_mask(img, band_id=3)
        orig_fft = torch.fft.fftshift(torch.fft.fft2(img))
        masked_fft = torch.fft.fftshift(torch.fft.fft2(masked))
        orig_e = orig_fft[:, :6, :6].abs().mean().item()
        masked_e = masked_fft[:, :6, :6].abs().mean().item()
        assert masked_e < orig_e * 0.5

    @pytest.mark.parametrize("band_id", [4, 5, 6])
    def test_dual_band(self, band_id, device):
        fb = get_task("freq_band", feat_dim=256).to(device)
        img = torch.rand(3, 64, 64, device=device)
        out = fb._apply_freq_mask(img, band_id=band_id)
        assert out.shape == img.shape
        assert out.min() >= 0
