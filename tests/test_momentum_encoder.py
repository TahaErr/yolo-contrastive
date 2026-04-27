"""Tests for MomentumEncoder."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from yolo_contrastive.dense import MomentumEncoder, MultiScaleFeatureTap


# ── helpers ──────────────────────────────────────────────────────────────


def _simple_encoder() -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )


def _encoder_with_bn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.BatchNorm2d(8),
    )


# ── initialization ───────────────────────────────────────────────────────


class TestInit:
    def test_deep_copy_independent(self):
        online = _simple_encoder()
        me = MomentumEncoder(online)
        for p in online.parameters():
            p.data.fill_(99.0)
        for p in me.momentum.parameters():
            assert not torch.allclose(p, torch.full_like(p, 99.0))

    def test_momentum_params_no_grad(self):
        me = MomentumEncoder(_simple_encoder())
        for p in me.momentum.parameters():
            assert not p.requires_grad

    def test_momentum_initially_eval(self):
        me = MomentumEncoder(_encoder_with_bn())
        for module in me.momentum.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert not module.training

    def test_invalid_m_raises(self):
        online = _simple_encoder()
        with pytest.raises(ValueError, match="m must be in"):
            MomentumEncoder(online, m=-0.1)
        with pytest.raises(ValueError, match="m must be in"):
            MomentumEncoder(online, m=1.5)

    def test_m_boundary_values(self):
        online = _simple_encoder()
        MomentumEncoder(online, m=0.0)
        MomentumEncoder(online, m=1.0)


# ── EMA update ───────────────────────────────────────────────────────────


class TestUpdate:
    def test_m_one_no_change(self):
        online = _simple_encoder()
        me = MomentumEncoder(online, m=1.0)
        snapshot = [p.clone() for p in me.momentum.parameters()]
        for p in online.parameters():
            p.data.fill_(99.0)
        me.update(online)
        for p_now, p_before in zip(me.momentum.parameters(), snapshot):
            assert torch.equal(p_now, p_before)

    def test_m_zero_full_copy(self):
        online = _simple_encoder()
        me = MomentumEncoder(online, m=0.0)
        for p in online.parameters():
            p.data.fill_(7.0)
        me.update(online)
        for p_o, p_m in zip(online.parameters(), me.momentum.parameters()):
            assert torch.allclose(p_o, p_m)

    def test_m_half_midpoint(self):
        online = _simple_encoder()
        me = MomentumEncoder(online, m=0.5)
        m_init = [p.clone() for p in me.momentum.parameters()]
        for p in online.parameters():
            p.data.fill_(10.0)
        me.update(online)
        for p_m, p_init in zip(me.momentum.parameters(), m_init):
            expected = 0.5 * p_init + 0.5 * 10.0
            assert torch.allclose(p_m, expected, atol=1e-6)

    def test_compound_updates_converge(self):
        """After many updates, momentum approaches online value."""
        online = _simple_encoder()
        me = MomentumEncoder(online, m=0.9)
        for p in online.parameters():
            p.data.fill_(100.0)
        for _ in range(50):
            me.update(online)
        for p_m in me.momentum.parameters():
            mean = p_m.mean().item()
            assert abs(mean - 100.0) < 5.0  # 0.9^50 * |delta| ≈ small

    def test_buffers_ema_for_floats(self):
        """BN running_mean should be EMA'd, not copied."""
        online = _encoder_with_bn()
        x = torch.randn(8, 3, 16, 16)
        online.train()
        for _ in range(3):
            online(x)

        me = MomentumEncoder(online, m=0.5)
        rm_mom_init = me.momentum[1].running_mean.clone()

        # Push online's BN to very different stats
        x2 = torch.randn(8, 3, 16, 16) * 5.0
        for _ in range(3):
            online(x2)
        rm_online_new = online[1].running_mean.clone()

        me.update(online)
        rm_mom_after = me.momentum[1].running_mean
        expected = 0.5 * rm_mom_init + 0.5 * rm_online_new
        assert torch.allclose(rm_mom_after, expected, atol=1e-5)

    def test_int_buffer_copied_not_emaed(self):
        """num_batches_tracked is integer — copy verbatim."""
        online = _encoder_with_bn()
        online.train()
        for _ in range(7):
            online(torch.randn(2, 3, 8, 8))

        me = MomentumEncoder(online, m=0.99)
        for _ in range(5):
            online(torch.randn(2, 3, 8, 8))
        nbt_online_new = online[1].num_batches_tracked.clone()

        me.update(online)
        assert torch.equal(me.momentum[1].num_batches_tracked, nbt_online_new)


# ── forward ──────────────────────────────────────────────────────────────


class TestForward:
    def test_no_grad(self):
        online = _simple_encoder()
        me = MomentumEncoder(online)
        x = torch.randn(2, 8, requires_grad=True)
        out = me(x)
        assert not out.requires_grad

    def test_doesnt_update_online_during_forward(self):
        online = _simple_encoder()
        me = MomentumEncoder(online)
        snapshot = [p.clone() for p in online.parameters()]
        _ = me(torch.randn(2, 8))
        for p_now, p_before in zip(online.parameters(), snapshot):
            assert torch.equal(p_now, p_before)

    def test_eval_mode_bn_doesnt_update(self):
        """BN running stats must NOT change during momentum forward."""
        online = _encoder_with_bn()
        online.train()
        for _ in range(3):
            online(torch.randn(4, 3, 8, 8))
        me = MomentumEncoder(online)
        rm_before = me.momentum[1].running_mean.clone()
        rv_before = me.momentum[1].running_var.clone()

        for _ in range(10):
            _ = me(torch.randn(4, 3, 8, 8) * 100.0)

        assert torch.equal(rm_before, me.momentum[1].running_mean)
        assert torch.equal(rv_before, me.momentum[1].running_var)

    def test_force_fp32_disables_autocast(self):
        online = _simple_encoder()
        me = MomentumEncoder(online, force_fp32=True)
        x = torch.randn(2, 8)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = me(x)
        assert out.dtype == torch.float32

    def test_force_fp32_off_allows_autocast(self):
        online = _simple_encoder()
        me = MomentumEncoder(online, force_fp32=False)
        x = torch.randn(2, 8)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = me(x)
        # When autocast is active and force_fp32=False, output should be bf16
        assert out.dtype == torch.bfloat16


# ── train() override ─────────────────────────────────────────────────────


class TestTrainMode:
    def test_train_mode_keeps_momentum_eval(self):
        me = MomentumEncoder(_encoder_with_bn())
        me.train()
        for module in me.momentum.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert not module.training, "BN in momentum must stay eval"

    def test_eval_mode_works(self):
        me = MomentumEncoder(_encoder_with_bn())
        me.train()
        me.eval()
        for module in me.momentum.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert not module.training


# ── state dict ───────────────────────────────────────────────────────────


class TestStateDict:
    def test_state_dict_only_momentum(self):
        online = _simple_encoder()
        me = MomentumEncoder(online)
        sd = me.state_dict()
        assert all(k.startswith("momentum.") for k in sd.keys())

    def test_state_dict_roundtrip(self):
        online = _simple_encoder()
        me1 = MomentumEncoder(online, m=0.95)
        for p in online.parameters():
            p.data.fill_(3.0)
        me1.update(online)

        me2 = MomentumEncoder(online)
        me2.load_state_dict(me1.state_dict())

        for p1, p2 in zip(me1.momentum.parameters(), me2.momentum.parameters()):
            assert torch.allclose(p1, p2)


# ── hook stripping ───────────────────────────────────────────────────────


class TestHookStripping:
    def test_forward_hooks_stripped(self):
        """Hooks on online should NOT fire on momentum forward."""
        online = _simple_encoder()
        fired = []
        online.register_forward_hook(lambda m, i, o: fired.append("x"))

        me = MomentumEncoder(online)
        fired.clear()
        _ = me(torch.randn(2, 8))
        assert len(fired) == 0

    def test_pre_hooks_stripped(self):
        online = _simple_encoder()
        fired = []
        online.register_forward_pre_hook(lambda m, i: fired.append("x"))

        me = MomentumEncoder(online)
        fired.clear()
        _ = me(torch.randn(2, 8))
        assert len(fired) == 0

    def test_online_hooks_still_work_after_init(self):
        """MomentumEncoder init shouldn't damage the online encoder's hooks."""
        online = _simple_encoder()
        fired = []
        online.register_forward_hook(lambda m, i, o: fired.append("online"))

        _ = MomentumEncoder(online)  # shouldn't strip online's hooks

        _ = online(torch.randn(2, 8))
        assert "online" in fired


# ── integration with MultiScaleFeatureTap ───────────────────────────────


class TestIntegrationWithMSTap:
    def test_momentum_features_via_tap(self):
        """End-to-end: tap on momentum.momentum yields multi-scale features."""
        seq = nn.Sequential(*[nn.Conv2d(3, 3, 1) for _ in range(23)])
        me = MomentumEncoder(seq)

        tap = MultiScaleFeatureTap(me.momentum)
        tap.setup()

        _ = me(torch.randn(2, 3, 16, 16))
        feats = tap.get_features()

        for level in ("P3", "P4", "P5"):
            assert feats[level] is not None
            assert not feats[level].requires_grad
        tap.close()

    def test_momentum_tap_independent_from_online_tap(self):
        """Online tap and momentum tap should not interfere."""
        seq = nn.Sequential(*[nn.Conv2d(3, 3, 1) for _ in range(23)])
        me = MomentumEncoder(seq)

        online_tap = MultiScaleFeatureTap(seq)
        online_tap.setup()
        momentum_tap = MultiScaleFeatureTap(me.momentum)
        momentum_tap.setup()

        x_online = torch.randn(2, 3, 16, 16)
        x_momentum = torch.randn(2, 3, 16, 16) * 5.0  # very different

        _ = seq(x_online)
        _ = me(x_momentum)

        f_o = online_tap.get_features()
        f_m = momentum_tap.get_features()

        # They should differ since inputs differ AND outputs were captured separately
        assert not torch.allclose(f_o["P3"], f_m["P3"])
        online_tap.close()
        momentum_tap.close()


# ── repr ─────────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr(self):
        me = MomentumEncoder(_simple_encoder(), m=0.99)
        r = repr(me)
        assert "MomentumEncoder" in r
        assert "m=0.99" in r
        assert "params=" in r
