"""FreqGate — Frequency-aware gating for Conv-LoRA."""

from __future__ import annotations
import torch
import torch.nn as nn


class FreqGate(nn.Module):
    """Input'un frekans profiline göre [0,1] gate üretir.

    Pipeline: x → FFT2D → band energy (low/mid/high) → MLP → sigmoid → gate
    """

    def __init__(self, low_ratio: float = 0.1, mid_ratio: float = 0.4,
                 hidden_dim: int = 16, init_bias: float = 1.0):
        super().__init__()
        self.low_ratio = low_ratio
        self.mid_ratio = mid_ratio

        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Init: gate ≈ sigmoid(1.0) ≈ 0.73
        nn.init.zeros_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, init_bias)

        self._grid_cache = {}

    def _get_band_energy(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_mean = x.mean(dim=1, keepdim=True)
        # AMP float16 ile cuFFT uyumsuz → float32 zorla
        freq = torch.fft.fft2(x_mean.float(), dim=(-2, -1))
        freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))
        power = freq_shifted.abs().squeeze(1)

        key = (H, W, str(x.device))
        if key not in self._grid_cache:
            fy = torch.arange(H, device=x.device, dtype=torch.float32) - H / 2.0
            fx = torch.arange(W, device=x.device, dtype=torch.float32) - W / 2.0
            gy, gx = torch.meshgrid(fy, fx, indexing="ij")
            max_r = (H**2 / 4.0 + W**2 / 4.0) ** 0.5
            dist = (gy**2 + gx**2).sqrt() / max_r
            if len(self._grid_cache) < 8:
                self._grid_cache[key] = dist
        else:
            dist = self._grid_cache[key]

        low_mask = (dist < self.low_ratio).float()
        mid_mask = ((dist >= self.low_ratio) & (dist < self.mid_ratio)).float()
        high_mask = (dist >= self.mid_ratio).float()

        total = power.sum(dim=(-2, -1)).clamp(min=1e-8)
        e_low = (power * low_mask).sum(dim=(-2, -1)) / total
        e_mid = (power * mid_mask).sum(dim=(-2, -1)) / total
        e_high = (power * high_mask).sum(dim=(-2, -1)) / total

        return torch.stack([e_low, e_mid, e_high], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = self._get_band_energy(x)
        gate = torch.sigmoid(self.mlp(energy))
        return gate.view(-1, 1, 1, 1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return f"FreqGate(low={self.low_ratio}, mid={self.mid_ratio}, params={self.num_params})"
