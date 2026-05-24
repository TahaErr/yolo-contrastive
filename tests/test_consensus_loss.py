"""Tests for ConsensusLoss — Form B + Form C dual-teacher distillation.

All fast — small synthetic feature maps.
"""

from __future__ import annotations

import math

import pytest
import torch


_LEVELS = ("P3", "P4", "P5")


def _feats(channels=8, batch=2):
    """Student/teacher-like feature maps per level."""
    return {
        "P3": torch.randn(batch, channels, 8, 8),
        "P4": torch.randn(batch, channels, 4, 4),
        "P5": torch.randn(batch, channels, 2, 2),
    }


# ═════════════════════════════════════════════════════════════════════════
# Construction
# ═════════════════════════════════════════════════════════════════════════


class TestConstruction:
    @pytest.mark.parametrize("form", ["B", "C", "B+C"])
    def test_all_forms_build(self, form):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss = ConsensusLoss(distill_form=form)
        assert loss.distill_form == form

    def test_bad_form_raises(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        with pytest.raises(ValueError, match="distill_form"):
            ConsensusLoss(distill_form="X")

    def test_bad_w_init_raises(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        with pytest.raises(ValueError, match="w_init"):
            ConsensusLoss(w_init=1.5)

    def test_w_init_recovered(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        for wi in (0.3, 0.5, 0.7):
            loss = ConsensusLoss(w_init=wi)
            assert abs(loss.get_w() - wi) < 1e-5

    def test_w_is_learnable_parameter(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss = ConsensusLoss()
        assert "w_raw" in dict(loss.named_parameters())
        assert loss.w_raw.requires_grad is True


# ═════════════════════════════════════════════════════════════════════════
# Form B — learned weighted L2
# ═════════════════════════════════════════════════════════════════════════


class TestFormB:
    def test_student_equals_target_zero_loss(self):
        """If student == w*coco + (1-w)*ssl exactly, Form B loss is 0."""
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss(distill_form="B", w_init=0.5)
        w = loss_fn.get_w()
        coco = _feats()
        ssl = _feats()
        student = {lv: w * coco[lv] + (1 - w) * ssl[lv] for lv in _LEVELS}
        loss, info = loss_fn(student, coco, ssl)
        assert loss.item() < 1e-5
        assert info["form_B"] < 1e-5

    def test_form_b_positive_when_mismatched(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss(distill_form="B")
        loss, info = loss_fn(_feats(), _feats(), _feats())
        assert loss.item() > 0
        assert info["form_C"] == 0.0   # form C inactive


# ═════════════════════════════════════════════════════════════════════════
# Form C — channel-wise dual KL
# ═════════════════════════════════════════════════════════════════════════


class TestFormC:
    def test_student_equals_both_teachers_zero_kl(self):
        """student == coco == ssl → both KL terms ~0."""
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss(distill_form="C")
        feats = _feats()
        loss, info = loss_fn(feats, feats, feats)
        assert loss.item() < 1e-4
        assert info["form_B"] == 0.0   # form B inactive

    def test_form_c_positive_when_mismatched(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss(distill_form="C")
        loss, info = loss_fn(_feats(), _feats(), _feats())
        assert loss.item() > 0

    def test_cwd_kl_helper_identical_is_zero(self):
        from yolo_contrastive.dual_teacher.consensus_loss import _cwd_kl

        feat = torch.randn(2, 8, 4, 4)
        kl = _cwd_kl(feat, feat, T=4.0)
        # CWD: KL is summed over the spatial axis -> one scalar per channel.
        assert kl.shape == (2, 8)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_cwd_kl_non_negative_invariant(self):
        """KL divergence is >= 0 (Gibbs). _cwd_kl sums over the spatial axis,
        so every per-channel entry must be non-negative for ANY input — the
        regression guard for the Faz 5.3 negative-distill bug."""
        from yolo_contrastive.dual_teacher.consensus_loss import _cwd_kl

        for seed in range(30):
            torch.manual_seed(seed)
            student = torch.randn(2, 8, 4, 4)
            teacher = torch.randn(2, 8, 4, 4)
            kl = _cwd_kl(student, teacher, T=4.0)
            assert (kl >= -1e-6).all(), (
                f"seed {seed}: _cwd_kl produced a negative value "
                f"{kl.min().item()} — KL must be >= 0"
            )

    def test_form_c_non_negative_across_temperatures(self):
        """Form C loss stays >= 0 regardless of input or kl_temperature —
        guards the dual-KL aggregation in forward()."""
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        for T in (1.0, 2.0, 4.0, 8.0):
            for seed in range(8):
                torch.manual_seed(seed)
                loss_fn = ConsensusLoss(distill_form="C", kl_temperature=T)
                loss, info = loss_fn(_feats(), _feats(), _feats())
                assert loss.item() >= -1e-6, (
                    f"T={T} seed={seed}: Form C loss negative {loss.item()}"
                )
                assert info["form_C"] >= -1e-6


# ═════════════════════════════════════════════════════════════════════════
# B+C combination
# ═════════════════════════════════════════════════════════════════════════


class TestBPlusC:
    def test_bplusc_equals_weighted_sum(self):
        """B+C total == alpha*form_B + beta*form_C."""
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        torch.manual_seed(0)
        student, coco, ssl = _feats(), _feats(), _feats()

        loss_fn = ConsensusLoss(distill_form="B+C", alpha=2.0, beta=3.0)
        loss, info = loss_fn(student, coco, ssl)

        expected = 2.0 * info["form_B"] + 3.0 * info["form_C"]
        assert abs(loss.item() - expected) < 1e-4

    def test_alpha_beta_scaling(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        torch.manual_seed(1)
        student, coco, ssl = _feats(), _feats(), _feats()

        base = ConsensusLoss(distill_form="B+C", alpha=1.0, beta=1.0)
        scaled = ConsensusLoss(distill_form="B+C", alpha=2.0, beta=2.0)
        # Share the same w so the comparison is clean
        scaled.w_raw.data.copy_(base.w_raw.data)

        l_base, _ = base(student, coco, ssl)
        l_scaled, _ = scaled(student, coco, ssl)
        assert abs(l_scaled.item() - 2.0 * l_base.item()) < 1e-4


# ═════════════════════════════════════════════════════════════════════════
# disagreement weight modulation
# ═════════════════════════════════════════════════════════════════════════


class TestDisagreementWeight:
    def test_zero_weight_zeroes_loss(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss(distill_form="B+C")
        student, coco, ssl = _feats(), _feats(), _feats()
        dw = {
            "P3": torch.zeros(2, 8, 8),
            "P4": torch.zeros(2, 4, 4),
            "P5": torch.zeros(2, 2, 2),
        }
        loss, _ = loss_fn(student, coco, ssl, disagreement_weight=dw)
        assert loss.item() < 1e-5

    def test_weight_scales_loss(self):
        """Uniform weight 2.0 → loss exactly 2x the unweighted loss."""
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        torch.manual_seed(2)
        student, coco, ssl = _feats(), _feats(), _feats()
        loss_fn = ConsensusLoss(distill_form="B+C")

        l_plain, _ = loss_fn(student, coco, ssl)
        dw = {
            "P3": torch.full((2, 8, 8), 2.0),
            "P4": torch.full((2, 4, 4), 2.0),
            "P5": torch.full((2, 2, 2), 2.0),
        }
        l_weighted, _ = loss_fn(student, coco, ssl, disagreement_weight=dw)
        assert abs(l_weighted.item() - 2.0 * l_plain.item()) < 1e-4


# ═════════════════════════════════════════════════════════════════════════
# gradient flow
# ═════════════════════════════════════════════════════════════════════════


class TestGradientFlow:
    def test_gradient_reaches_student_and_w(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss(distill_form="B+C")
        student = {lv: t.clone().requires_grad_(True)
                   for lv, t in _feats().items()}
        coco, ssl = _feats(), _feats()

        loss, _ = loss_fn(student, coco, ssl)
        loss.backward()

        # student grads
        for lv in _LEVELS:
            assert student[lv].grad is not None
            assert student[lv].grad.abs().sum() > 0
        # fusion weight grad (Form B active → w used)
        assert loss_fn.w_raw.grad is not None

    def test_form_c_only_w_unused(self):
        """distill_form='C' → w_raw receives no gradient (Form B inactive)."""
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss(distill_form="C")
        student = {lv: t.clone().requires_grad_(True)
                   for lv, t in _feats().items()}
        loss, _ = loss_fn(student, _feats(), _feats())
        loss.backward()
        # w_raw not part of Form C graph
        assert loss_fn.w_raw.grad is None or loss_fn.w_raw.grad.abs().sum() == 0


# ═════════════════════════════════════════════════════════════════════════
# info dict
# ═════════════════════════════════════════════════════════════════════════


class TestInfoDict:
    def test_info_schema(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss(distill_form="B+C")
        _, info = loss_fn(_feats(), _feats(), _feats())
        assert set(info.keys()) >= {"total", "form_B", "form_C", "w_coco", "levels"}
        assert set(info["levels"].keys()) == set(_LEVELS)

    def test_w_coco_in_info_matches_get_w(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss(distill_form="B", w_init=0.6)
        _, info = loss_fn(_feats(), _feats(), _feats())
        assert abs(info["w_coco"] - loss_fn.get_w()) < 1e-6


# ═════════════════════════════════════════════════════════════════════════
# validation + subset
# ═════════════════════════════════════════════════════════════════════════


class TestValidationAndSubset:
    def test_missing_level_raises(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss()
        feats = _feats()
        del feats["P5"]
        with pytest.raises(ValueError, match="missing levels"):
            loss_fn(feats, _feats(), _feats())

    def test_p5_only_subset(self):
        from yolo_contrastive.dual_teacher.consensus_loss import ConsensusLoss

        loss_fn = ConsensusLoss(levels=("P5",), distill_form="B+C")
        one = {"P5": torch.randn(2, 8, 2, 2)}
        loss, info = loss_fn(one, {"P5": torch.randn(2, 8, 2, 2)},
                             {"P5": torch.randn(2, 8, 2, 2)})
        assert torch.isfinite(loss).item()
        assert set(info["levels"].keys()) == {"P5"}
