"""TaskRoutedConvLoRA — Her pretext task için ayrı LoRA branch.

Konsept:
    SSL sırasında her task kendi branch'ini eğitir:
        task_0 aktif → branch_0 forward + gradient
        task_1 aktif → branch_1 forward + gradient
        ...

    Fine-tune'da tüm branch'ler merge edilir:
        ΔW = Σ(αᵢ × branch_i)

    TaskRouter tüm modüllerin aktif task'ını senkronize eder.

Akademik novelty:
    - Task-specific LoRA: NLP'de var (LoRA-MoE), vision SSL'de YOK
    - Multi-task routing + merge: ilk kez
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn

from .freq_gate import FreqGate


class LoRABranch(nn.Module):
    """Tek bir LoRA branch: down (C_in→rank) + up (rank→C_out)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride, padding, dilation, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.lora_down = nn.Conv2d(
            in_channels, rank, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=1, bias=False,
        )
        self.lora_up = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_up(self.lora_down(x))

    def compute_delta_w(self) -> torch.Tensor:
        """Effective weight delta: up @ down → [C_out, C_in, kH, kW]."""
        with torch.no_grad():
            down_w = self.lora_down.weight  # [rank, C_in, kH, kW]
            up_w = self.lora_up.weight.squeeze(-1).squeeze(-1)  # [C_out, rank]
            r, c_in, kh, kw = down_w.shape
            return (up_w @ down_w.view(r, -1)).view(-1, c_in, kh, kw)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TaskRoutedConvLoRA(nn.Module):
    """Multi-branch Conv-LoRA: her task kendi branch'ini kullanır.

    Args:
        conv: Orijinal Conv2d (frozen olacak)
        num_tasks: Branch sayısı (task sayısı)
        rank: Her branch'in rank'ı
        scale: LoRA scale
        dropout: Dropout
        use_gate: FreqGate kullan
        gate_hidden: Gate MLP hidden dim
    """

    def __init__(self, conv: nn.Conv2d, num_tasks: int, rank: int = 4,
                 scale: float = 1.0, dropout: float = 0.0,
                 use_gate: bool = True, gate_hidden: int = 16):
        super().__init__()

        self.num_tasks = num_tasks
        self.scale = scale

        # Frozen conv
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation

        # N branch
        self.branches = nn.ModuleList([
            LoRABranch(
                self.in_channels, self.out_channels,
                self.kernel_size, self.stride,
                self.padding, self.dilation, rank=rank,
            ) for _ in range(num_tasks)
        ])

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Shared FreqGate (opsiyonel)
        self.gate = FreqGate(hidden_dim=gate_hidden) if use_gate else None

        # Active task index
        self._active_task: int = 0

    def set_active_task(self, idx: int) -> None:
        if idx < 0 or idx >= self.num_tasks:
            raise ValueError(f"Task index {idx} out of range [0, {self.num_tasks})")
        self._active_task = idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.conv(x)
        branch = self.branches[self._active_task]
        lora_out = branch(self.lora_dropout(x))

        if self.gate is not None:
            g = self.gate(x)
            return frozen_out + g * self.scale * lora_out
        return frozen_out + self.scale * lora_out

    def forward_all_branches(self, x: torch.Tensor) -> torch.Tensor:
        """Tüm branch'lerin ortalaması (CL loss için)."""
        frozen_out = self.conv(x)
        dx = self.lora_dropout(x)
        lora_sum = sum(branch(dx) for branch in self.branches) / self.num_tasks

        if self.gate is not None:
            g = self.gate(x)
            return frozen_out + g * self.scale * lora_sum
        return frozen_out + self.scale * lora_sum

    def merge_weights(self, alphas: Optional[List[float]] = None) -> nn.Conv2d:
        """Branch'leri merge edip clean Conv2d döndür.

        Args:
            alphas: Branch ağırlıkları. None = equal weight.
        """
        if alphas is None:
            alphas = [1.0 / self.num_tasks] * self.num_tasks

        if len(alphas) != self.num_tasks:
            raise ValueError(f"alphas ({len(alphas)}) != num_tasks ({self.num_tasks})")

        with torch.no_grad():
            delta_w = sum(
                a * branch.compute_delta_w()
                for a, branch in zip(alphas, self.branches)
            )

            merged = nn.Conv2d(
                self.in_channels, self.out_channels,
                self.kernel_size, self.stride, self.padding,
                self.dilation, self.conv.groups,
                bias=self.conv.bias is not None,
            ).to(self.conv.weight.device)

            merged.weight.copy_(self.conv.weight + self.scale * delta_w)
            if self.conv.bias is not None:
                merged.bias.copy_(self.conv.bias)

        return merged

    @property
    def num_trainable_params(self) -> int:
        total = sum(b.num_params for b in self.branches)
        if self.gate is not None:
            total += self.gate.num_params
        return total

    @property
    def num_frozen_params(self) -> int:
        return sum(p.numel() for p in self.conv.parameters())

    @property
    def rank(self) -> int:
        return self.branches[0].rank if self.branches else 0

    def __repr__(self) -> str:
        return (f"TaskRoutedConvLoRA(in={self.in_channels}, out={self.out_channels}, "
                f"tasks={self.num_tasks}, rank={self.rank}, "
                f"trainable={self.num_trainable_params:,}, "
                f"frozen={self.num_frozen_params:,})")


class TaskRouter:
    """Tüm TaskRoutedConvLoRA modüllerinin aktif task'ını senkronize eder.

    Kullanım:
        router = TaskRouter(model)
        with router.task(0):
            out = model(x)  # branch 0 aktif

        # veya
        router.route(1)
        out = model(x)  # branch 1 aktif
    """

    def __init__(self, model: nn.Module):
        self.modules: List[TaskRoutedConvLoRA] = [
            m for m in model.modules() if isinstance(m, TaskRoutedConvLoRA)
        ]
        if not self.modules:
            raise ValueError("Model'de TaskRoutedConvLoRA bulunamadı")

    def route(self, task_idx: int) -> None:
        for m in self.modules:
            m.set_active_task(task_idx)

    @contextmanager
    def task(self, task_idx: int):
        prev = self.modules[0]._active_task if self.modules else 0
        self.route(task_idx)
        try:
            yield
        finally:
            self.route(prev)

    @property
    def num_modules(self) -> int:
        return len(self.modules)

    @property
    def num_tasks(self) -> int:
        return self.modules[0].num_tasks if self.modules else 0
