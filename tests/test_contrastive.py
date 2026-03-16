"""Test contrastive losses."""

import pytest
import torch
from yolo_contrastive import NTXentLoss, build_contrastive_loss, ContrastiveLossError


class TestNTXentLoss:
    def test_shape(self, device):
        loss_fn = NTXentLoss(temperature=0.2)
        z1 = torch.randn(4, 128, device=device)
        z2 = torch.randn(4, 128, device=device)
        loss = loss_fn(z1, z2)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_b1_returns_zero(self, device):
        loss_fn = NTXentLoss(temperature=0.2)
        z = torch.randn(1, 128, device=device)
        assert loss_fn(z, z).item() == 0.0

    def test_negative_temp_raises(self):
        with pytest.raises(ContrastiveLossError):
            NTXentLoss(temperature=-1.0)

    @pytest.mark.parametrize("name", ["ntxent", "infonce", "simclr"])
    def test_build(self, name, device):
        fn = build_contrastive_loss(name, temperature=0.5)
        z1 = torch.randn(4, 128, device=device)
        z2 = torch.randn(4, 128, device=device)
        assert torch.isfinite(fn(z1, z2))
