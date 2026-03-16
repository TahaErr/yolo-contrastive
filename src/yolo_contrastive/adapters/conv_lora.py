"""ConvLoRA — Low-rank adapter for Conv2d layers."""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class ConvLoRA(nn.Module):
    """Low-rank parallel adapter for a frozen Conv2d.

    output = frozen_conv(x) + scale * lora_up(lora_down(x))
    """

    def __init__(self, conv: nn.Conv2d, rank: int = 4, scale: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.scale = scale

        # Orijinal conv — frozen
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        # LoRA path
        self.lora_down = nn.Conv2d(
            self.in_channels, rank,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, dilation=self.dilation,
            groups=1, bias=False,
        )
        self.lora_up = nn.Conv2d(rank, self.out_channels, kernel_size=1, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init: Kaiming down, zero up → başlangıçta ΔW = 0
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        lora_out = self.lora_up(self.lora_down(self.lora_dropout(x)))
        return out + self.scale * lora_out

    @property
    def num_lora_params(self) -> int:
        return sum(p.numel() for p in [self.lora_down.weight, self.lora_up.weight])

    @property
    def num_frozen_params(self) -> int:
        return sum(p.numel() for p in self.conv.parameters())

    def merge_weights(self) -> nn.Conv2d:
        """LoRA ağırlıklarını frozen conv'a merge et."""
        with torch.no_grad():
            down_w = self.lora_down.weight  # [rank, C_in, kH, kW]
            up_w = self.lora_up.weight.squeeze(-1).squeeze(-1)  # [C_out, rank]
            r, c_in, kh, kw = down_w.shape
            down_flat = down_w.view(r, -1)
            delta_w = (up_w @ down_flat).view(self.out_channels, c_in, kh, kw)

            merged = nn.Conv2d(
                self.in_channels, self.out_channels,
                self.kernel_size, self.stride, self.padding,
                self.dilation, self.groups,
                bias=self.conv.bias is not None,
            ).to(self.conv.weight.device)
            merged.weight.copy_(self.conv.weight + self.scale * delta_w)
            if self.conv.bias is not None:
                merged.bias.copy_(self.conv.bias)
        return merged

    def __repr__(self) -> str:
        return (f"ConvLoRA(in={self.in_channels}, out={self.out_channels}, "
                f"k={self.kernel_size}, rank={self.rank}, "
                f"lora={self.num_lora_params:,}, frozen={self.num_frozen_params:,})")
