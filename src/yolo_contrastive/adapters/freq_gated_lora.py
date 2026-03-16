"""FreqGatedConvLoRA — Frequency-gated low-rank adapter."""

from __future__ import annotations
import torch
import torch.nn as nn

from .conv_lora import ConvLoRA
from .freq_gate import FreqGate


class FreqGatedConvLoRA(nn.Module):
    """output = frozen_conv(x) + gate(x) * scale * lora_path(x)"""

    def __init__(self, conv: nn.Conv2d, rank: int = 4, scale: float = 1.0,
                 dropout: float = 0.0, gate_hidden: int = 16,
                 gate_init_bias: float = 1.0, low_ratio: float = 0.1,
                 mid_ratio: float = 0.4):
        super().__init__()
        self.lora = ConvLoRA(conv, rank=rank, scale=scale, dropout=dropout)
        self.gate = FreqGate(low_ratio=low_ratio, mid_ratio=mid_ratio,
                             hidden_dim=gate_hidden, init_bias=gate_init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.lora.conv(x)
        lora_out = self.lora.lora_up(self.lora.lora_down(self.lora.lora_dropout(x)))
        g = self.gate(x)
        return frozen_out + g * self.lora.scale * lora_out

    def merge_weights(self) -> nn.Conv2d:
        return self.lora.merge_weights()

    @property
    def num_trainable_params(self) -> int:
        return self.lora.num_lora_params + self.gate.num_params

    @property
    def num_frozen_params(self) -> int:
        return self.lora.num_frozen_params

    @property
    def rank(self) -> int:
        return self.lora.rank

    def __repr__(self) -> str:
        return (f"FreqGatedConvLoRA(in={self.lora.in_channels}, "
                f"out={self.lora.out_channels}, rank={self.lora.rank}, "
                f"trainable={self.num_trainable_params:,}, "
                f"frozen={self.num_frozen_params:,})")
