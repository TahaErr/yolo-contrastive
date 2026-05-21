"""Tests for DisagreementWeighter — per-position cosine disagreement weighting.

All fast — small synthetic feature maps, no model.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


_LEVELS = ("P3", "P4", "P5")


def _fake_feats(channels=8, batch=2):
    """Two-teacher-like feature maps for each level."""
    return {
        "P3": torch.randn(batch, channels, 8, 8),
        "P4": torch.randn(batch, channels, 4, 4),
        "P5": torch.randn(batch, channels, 2, 2),
    }


# ═════════════════════════════════════════════════════════════════════════
# cosine_disagreement
# ═════════════════════════════════════════════════════════════════════════


class TestCosineDisagreement:
    def test_identical_features_zero_disagreement(self):
        from yolo_contrastive.dual_teacher.disagreement import cosine_disagreement

        feat = torch.randn(2, 8, 4, 4)
        d = cosine_disagreement(feat, feat)
        assert d.shape == (2, 4, 4)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)

    def test_opposite_features_max_disagreement(self):
        from yolo_contrastive.dual_teacher.disagreement import cosine_disagreement

        feat = torch.randn(2, 8, 4, 4)
        d = cosine_disagreement(feat, -feat)
        # cos = -1 → d = 2
        assert torch.allclose(d, torch.full_like(d, 2.0), atol=1e-5)

    def test_orthogonal_features_unit_disagreement(self):
        from yolo_contrastive.dual_teacher.disagreement import cosine_disagreement

        # Construct orthogonal channel vectors at each position
        a = torch.zeros(1, 2, 3, 3)
        b = torch.zeros(1, 2, 3, 3)
        a[:, 0] = 1.0   # direction (1, 0)
        b[:, 1] = 1.0   # direction (0, 1)
        d = cosine_disagreement(a, b)
        assert torch.allclose(d, torch.ones_like(d), atol=1e-5)

    def test_shape_mismatch_raises(self):
        from yolo_contrastive.dual_teacher.disagreement import cosine_disagreement

        with pytest.raises(ValueError, match="same shape"):
            cosine_disagreement(torch.randn(2, 8, 4, 4), torch.randn(2, 8, 2, 2))


# ═════════════════════════════════════════════════════════════════════════
# Construction — 2x2 ablation matrix
# ═════════════════════════════════════════════════════════════════════════


class TestConstruction:
    @pytest.mark.parametrize("mode", ["fixed", "learnable"])
    @pytest.mark.parametrize("per_scale", [True, False])
    def test_all_four_variants_build(self, mode, per_scale):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(mode=mode, per_scale=per_scale)
        assert w.mode == mode
        assert w.per_scale == per_scale

    def test_fixed_mode_alpha_is_buffer(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(mode="fixed")
        # buffer → not in parameters()
        assert "alpha" not in dict(w.named_parameters())
        assert "alpha" in dict(w.named_buffers())

    def test_learnable_mode_alpha_is_parameter(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(mode="learnable")
        assert "alpha" in dict(w.named_parameters())
        assert w.alpha.requires_grad is True

    def test_per_scale_alpha_shape(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w_ps = DisagreementWeighter(per_scale=True, levels=_LEVELS)
        assert w_ps.alpha.shape == (3,)
        w_sh = DisagreementWeighter(per_scale=False, levels=_LEVELS)
        assert w_sh.alpha.shape == (1,)

    def test_bad_mode_raises(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        with pytest.raises(ValueError, match="mode"):
            DisagreementWeighter(mode="bogus")


# ═════════════════════════════════════════════════════════════════════════
# forward — weight semantics
# ═════════════════════════════════════════════════════════════════════════


class TestForwardSemantics:
    def test_output_shape(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter()
        out = w(_fake_feats(), _fake_feats())
        assert set(out.keys()) == set(_LEVELS)
        assert out["P3"].shape == (2, 8, 8)
        assert out["P5"].shape == (2, 2, 2)

    def test_alpha_zero_uniform_weight(self):
        """alpha_d = 0 → weight identically 1 (classic distillation)."""
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(mode="fixed", init_alpha=0.0)
        out = w(_fake_feats(), _fake_feats())
        for t in out.values():
            assert torch.allclose(t, torch.ones_like(t), atol=1e-5)

    def test_positive_alpha_amplifies_disagreement(self):
        """alpha_d > 0 → higher disagreement → higher weight."""
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(mode="fixed", init_alpha=1.0)
        # P3: identical (d=0 → w=1); construct an opposite pair separately
        same = {lv: torch.randn(1, 4, 4, 4) for lv in _LEVELS}
        out_same = w(same, same)            # d=0 → w=1
        out_opp = w(same, {lv: -v for lv, v in same.items()})  # d=2 → w=exp(2)
        for lv in _LEVELS:
            assert torch.all(out_opp[lv] > out_same[lv])

    def test_negative_alpha_amplifies_agreement(self):
        """alpha_d < 0 → higher disagreement → LOWER weight (consensus regime)."""
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(mode="fixed", init_alpha=-1.0)
        same = {lv: torch.randn(1, 4, 4, 4) for lv in _LEVELS}
        out_same = w(same, same)            # d=0 → w=1
        out_opp = w(same, {lv: -v for lv, v in same.items()})  # d=2 → w=exp(-2)<1
        for lv in _LEVELS:
            assert torch.all(out_opp[lv] < out_same[lv])

    def test_clamp_max_caps_weight(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        # Large alpha + max disagreement would explode; clamp caps it.
        w = DisagreementWeighter(mode="fixed", init_alpha=3.0, clamp_max=5.0)
        same = {lv: torch.randn(1, 4, 4, 4) for lv in _LEVELS}
        out = w(same, {lv: -v for lv, v in same.items()})   # d=2, exp(6)≈403
        for t in out.values():
            assert torch.all(t <= 5.0 + 1e-4)

    def test_missing_level_raises(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter()
        feats = _fake_feats()
        del feats["P5"]
        with pytest.raises(ValueError, match="missing levels"):
            w(feats, _fake_feats())


# ═════════════════════════════════════════════════════════════════════════
# alpha_clamp
# ═════════════════════════════════════════════════════════════════════════


class TestAlphaClamp:
    def test_init_alpha_beyond_clamp_is_bounded(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        # init_alpha 99 but alpha_clamp 3 → effective alpha is 3
        w = DisagreementWeighter(mode="fixed", init_alpha=99.0, alpha_clamp=3.0)
        alphas = w.get_alpha()
        for v in alphas.values():
            assert abs(v) <= 3.0 + 1e-5


# ═════════════════════════════════════════════════════════════════════════
# learnable alpha — gradient
# ═════════════════════════════════════════════════════════════════════════


class TestLearnableGradient:
    def test_gradient_reaches_alpha(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(mode="learnable", per_scale=True)
        a = _fake_feats()
        b = _fake_feats()
        out = w(a, b)
        loss = sum(t.mean() for t in out.values())
        loss.backward()
        assert w.alpha.grad is not None
        assert w.alpha.grad.abs().sum() > 0

    def test_fixed_alpha_no_grad(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(mode="fixed")
        # buffer has no requires_grad
        assert w.alpha.requires_grad is False


# ═════════════════════════════════════════════════════════════════════════
# get_alpha / per-scale independence
# ═════════════════════════════════════════════════════════════════════════


class TestGetAlpha:
    def test_get_alpha_keys_match_levels(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(levels=_LEVELS, init_alpha=0.7)
        alphas = w.get_alpha()
        assert set(alphas.keys()) == set(_LEVELS)
        for v in alphas.values():
            assert abs(v - 0.7) < 1e-5

    def test_per_scale_alphas_independent(self):
        """Per-scale: editing one level's alpha doesn't touch others."""
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(mode="learnable", per_scale=True, levels=_LEVELS)
        with torch.no_grad():
            w.alpha[0] = 2.0
            w.alpha[1] = -1.0
            w.alpha[2] = 0.5
        alphas = w.get_alpha()
        assert abs(alphas["P3"] - 2.0) < 1e-5
        assert abs(alphas["P4"] - (-1.0)) < 1e-5
        assert abs(alphas["P5"] - 0.5) < 1e-5


# ═════════════════════════════════════════════════════════════════════════
# subset levels
# ═════════════════════════════════════════════════════════════════════════


class TestSubsetLevels:
    def test_p5_only(self):
        from yolo_contrastive.dual_teacher.disagreement import DisagreementWeighter

        w = DisagreementWeighter(levels=("P5",), per_scale=True)
        assert w.alpha.shape == (1,)
        feats = {"P5": torch.randn(2, 8, 2, 2)}
        out = w(feats, feats)
        assert set(out.keys()) == {"P5"}
