"""Test adapter system."""

import torch
import torch.nn as nn
from yolo_contrastive.adapters import (
    ConvLoRA, FreqGate, FreqGatedConvLoRA,
    TaskRoutedConvLoRA, TaskRouter, inject_lora, remove_lora,
    compute_merge_alphas, merge_task_routed_model,
)


class TestConvLoRA:
    def test_output_shape(self, device):
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        lora = ConvLoRA(conv, rank=4).to(device)
        out = lora(torch.randn(4, 64, 32, 32, device=device))
        assert out.shape == (4, 128, 32, 32)

    def test_initial_zero(self, device):
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        lora = ConvLoRA(conv, rank=4).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        with torch.no_grad():
            diff = (lora(x) - conv(x)).abs().max().item()
        assert diff < 1e-5

    def test_frozen(self, device):
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        lora = ConvLoRA(conv, rank=4).to(device)
        assert not conv.weight.requires_grad
        assert lora.lora_down.weight.requires_grad

    def test_gradient(self, device):
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        lora = ConvLoRA(conv, rank=4).to(device)
        x = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
        lora(x).sum().backward()
        assert lora.lora_down.weight.grad is not None
        assert conv.weight.grad is None

    def test_merge(self, device):
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        lora = ConvLoRA(conv, rank=4).to(device)
        merged = lora.merge_weights()
        assert isinstance(merged, nn.Conv2d)
        x = torch.randn(2, 64, 16, 16, device=device)
        with torch.no_grad():
            diff = (merged(x) - lora(x)).abs().max().item()
        assert diff < 1e-4


class TestFreqGate:
    def test_shape(self, device):
        gate = FreqGate(hidden_dim=16).to(device)
        g = gate(torch.randn(4, 64, 32, 32, device=device))
        assert g.shape == (4, 1, 1, 1)
        assert g.min() >= 0 and g.max() <= 1

    def test_gradient(self, device):
        gate = FreqGate(hidden_dim=16).to(device)
        g = gate(torch.randn(4, 64, 32, 32, device=device))
        g.sum().backward()
        assert gate.mlp[0].weight.grad is not None

    def test_few_params(self):
        gate = FreqGate(hidden_dim=16)
        assert gate.num_params < 200


class TestFreqGatedConvLoRA:
    def test_forward(self, device):
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        fgl = FreqGatedConvLoRA(conv, rank=4).to(device)
        x = torch.randn(4, 64, 32, 32, device=device, requires_grad=True)
        out = fgl(x)
        assert out.shape == (4, 128, 32, 32)
        out.sum().backward()
        assert x.grad is not None

    def test_efficiency(self, device):
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        fgl = FreqGatedConvLoRA(conv, rank=4).to(device)
        assert fgl.num_trainable_params < fgl.num_frozen_params


class TestTaskRoutedConvLoRA:
    def test_routing(self, device):
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        tr = TaskRoutedConvLoRA(conv, num_tasks=3, rank=4, use_gate=False).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        for t in range(3):
            tr.set_active_task(t)
            out = tr(x)
            assert out.shape == (2, 128, 16, 16)

    def test_gradient_isolation(self, device):
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        tr = TaskRoutedConvLoRA(conv, num_tasks=3, rank=4, use_gate=False).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        tr.set_active_task(0)
        tr.zero_grad()
        tr(x).sum().backward()
        assert tr.branches[0].lora_down.weight.grad is not None
        assert tr.branches[1].lora_down.weight.grad is None

    def test_merge(self, device):
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        tr = TaskRoutedConvLoRA(conv, num_tasks=3, rank=4, use_gate=False).to(device)
        merged = tr.merge_weights(alphas=[0.5, 0.3, 0.2])
        assert isinstance(merged, nn.Conv2d)


class TestTaskRouter:
    def test_context_manager(self, device):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = TaskRoutedConvLoRA(
                    nn.Conv2d(3, 16, 3, padding=1), num_tasks=2, rank=2, use_gate=False)
            def forward(self, x):
                return self.c(x)

        model = M().to(device)
        router = TaskRouter(model)
        assert router.num_tasks == 2

        with router.task(1):
            assert model.c._active_task == 1
        assert model.c._active_task == 0


class TestMergeStrategies:
    def test_equal(self):
        a = compute_merge_alphas("equal", num_tasks=3)
        assert len(a) == 3
        assert abs(sum(a) - 1.0) < 1e-6

    def test_weighted(self):
        a = compute_merge_alphas("weighted", num_tasks=3,
                                  custom_alphas=[2.0, 1.0, 1.0])
        assert abs(sum(a) - 1.0) < 1e-6
        assert a[0] > a[1]

    def test_task_loss(self):
        a = compute_merge_alphas("task_loss", num_tasks=3,
                                  task_losses={"a": 2.0, "b": 1.0, "c": 0.5},
                                  task_names=["a", "b", "c"])
        assert abs(sum(a) - 1.0) < 1e-6
        assert a[2] > a[0]  # lowest loss → highest alpha


class TestInjectRemoveYOLO:
    def test_inject_forward_merge(self, yolo_model, device):
        info = inject_lora(yolo_model, rank=4, adapter_type="freq_gated", verbose=False)
        assert info["injected"] > 0

        dummy = torch.randn(1, 3, 640, 640, device=device)
        with torch.no_grad():
            out = yolo_model(dummy)
        assert out is not None

        n = remove_lora(yolo_model, merge=True, verbose=False)
        assert n == info["injected"]

        with torch.no_grad():
            out2 = yolo_model(dummy)
        assert out2 is not None

    def test_task_routed_inject(self, device):
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt").model.to(device)
        info = inject_lora(model, rank=4, adapter_type="task_routed",
                           num_tasks=3, verbose=False)
        assert info["injected"] > 0
        n = merge_task_routed_model(model, strategy="equal", verbose=False)
        assert n == info["injected"]
