"""ConsensusLoss — dual-teacher distillation loss (Form B + Form C).

Faz 5.3 — DT-SAPS dual-teacher framework (WORK_PLAN_v9 §10.28).

Distils two teachers (COCO YOLOv8x + SSL momentum encoder) into the dense
student via two complementary mechanisms:

  Form B — learned weighted L2 (feature-space).
      target = w * f_coco + (1 - w) * f_ssl
      L_B    = || f_student - target ||^2   (per-position, channel-averaged)
      w is a LEARNABLE scalar (sigmoid-bounded to [0, 1]); its trajectory
      w_coco-vs-epoch is a paper figure. w_init biases the start (§13.8
      ablation: 0.3 / 0.5 / 0.7).

  Form C — channel-wise dual KL (logit-space).
      Per CWD (Channel-wise Knowledge Distillation, Shu et al. 2021): each
      channel's HxW activation map is softmax-normalized into a spatial
      distribution; KL divergence aligns student to teacher channel-wise.
      Softmax removes teacher/student magnitude-scale differences — the same
      reason cosine was chosen for the disagreement metric.
      Form C keeps the two teachers SEPARATE (no fusion):
          L_C = KL_cwd(student || coco) + KL_cwd(student || ssl)
      This keeps Form B (feature-space fusion) and Form C (logit-space dual)
      genuinely distinct mechanisms, so the B / C / B+C ablation is meaningful.

  B+C — alpha * L_B + beta * L_C  (multi-level transfer).

The per-position disagreement weight (disagreement.py) optionally modulates
both forms: where the two teachers disagree, the consensus loss can be
amplified or damped depending on alpha_d's sign.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cwd_kl(student: torch.Tensor, teacher: torch.Tensor, T: float) -> torch.Tensor:
    """Channel-wise KL divergence, per-position contribution.

    Each channel's HxW map is softmax-normalized (CWD, Shu et al. 2021).
    KL(teacher || student) is computed per channel; the per-position
    contribution is returned (channel-averaged) so a spatial weight map can
    modulate it.

    Args:
        student, teacher: ``[B, C, H, W]`` feature maps (same shape).
        T: softmax temperature.

    Returns:
        ``[B, H, W]`` per-position KL contribution, T^2-scaled (Hinton/CWD).
    """
    B, C, H, W = student.shape
    s = (student.float() / T).reshape(B, C, H * W)
    t = (teacher.float() / T).reshape(B, C, H * W)
    s_logp = F.log_softmax(s, dim=2)        # spatial softmax per channel
    t_logp = F.log_softmax(t, dim=2)
    t_p = t_logp.exp()
    kl = t_p * (t_logp - s_logp)            # [B, C, HW] per-position KL term
    kl = kl.reshape(B, C, H, W).mean(dim=1)  # channel-average → [B, H, W]
    return kl * (T * T)


class ConsensusLoss(nn.Module):
    """Dual-teacher distillation loss: Form B, Form C, or B+C.

    Args:
        levels: FPN levels distilled. Default ``("P3", "P4", "P5")``.
        distill_form: ``"B"`` | ``"C"`` | ``"B+C"`` (§10.28).
        alpha: Form B weight.
        beta: Form C weight.
        w_init: initial value of the learnable fusion weight w (in (0, 1)).
        kl_temperature: softmax temperature for Form C's channel-wise KL.
    """

    def __init__(
        self,
        levels: tuple = ("P3", "P4", "P5"),
        distill_form: str = "B+C",
        alpha: float = 1.0,
        beta: float = 1.0,
        w_init: float = 0.5,
        kl_temperature: float = 4.0,
    ):
        super().__init__()
        if distill_form not in ("B", "C", "B+C"):
            raise ValueError(
                f"distill_form must be 'B', 'C' or 'B+C', got {distill_form!r}"
            )
        if not 0.0 < w_init < 1.0:
            raise ValueError(f"w_init must be in (0, 1), got {w_init}")
        if kl_temperature <= 0:
            raise ValueError(f"kl_temperature must be positive, got {kl_temperature}")

        self.levels = tuple(levels)
        self.distill_form = distill_form
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kl_temperature = float(kl_temperature)

        # Learnable fusion weight — stored as logit, sigmoid-bounded to (0, 1).
        w_raw_init = math.log(w_init / (1.0 - w_init))
        self.w_raw = nn.Parameter(torch.tensor(w_raw_init, dtype=torch.float32))

    # ── fusion weight ────────────────────────────────────────────────────

    def get_w(self) -> float:
        """Current fusion weight w (= sigmoid(w_raw)) — for logging."""
        with torch.no_grad():
            return float(torch.sigmoid(self.w_raw))

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        student: Dict[str, torch.Tensor],
        coco: Dict[str, torch.Tensor],
        ssl: Dict[str, torch.Tensor],
        disagreement_weight: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute the dual-teacher distillation loss.

        Args:
            student: ``{level: [B, C, H, W]}`` student feature maps.
            coco: ``{level: [B, C, H, W]}`` COCO teacher features (adapted to
                student channels).
            ssl: ``{level: [B, C, H, W]}`` SSL teacher features.
            disagreement_weight: optional ``{level: [B, H, W]}`` per-position
                weights (from DisagreementWeighter). Modulates both forms.

        Returns:
            ``(loss, info)`` — scalar loss + breakdown dict.
        """
        for name, d in (("student", student), ("coco", coco), ("ssl", ssl)):
            missing = [lv for lv in self.levels if lv not in d]
            if missing:
                raise ValueError(f"{name} missing levels {missing}")

        use_b = self.distill_form in ("B", "B+C")
        use_c = self.distill_form in ("C", "B+C")
        w = torch.sigmoid(self.w_raw)

        device_type = "cuda" if student[self.levels[0]].is_cuda else "cpu"
        info: Dict = {"levels": {}}

        with torch.autocast(device_type=device_type, enabled=False):
            lb_terms = []
            lc_terms = []
            for lv in self.levels:
                s, c, sl = student[lv].float(), coco[lv].float(), ssl[lv].float()
                dw = None
                if disagreement_weight is not None:
                    dw = disagreement_weight[lv].float()  # [B, H, W]

                lv_info = {}

                if use_b:
                    target = w * c + (1.0 - w) * sl
                    lb_map = ((s - target) ** 2).mean(dim=1)   # [B, H, W]
                    if dw is not None:
                        lb_map = lb_map * dw
                    lb = lb_map.mean()
                    lb_terms.append(lb)
                    lv_info["form_B"] = float(lb.detach())

                if use_c:
                    lc_map = (
                        _cwd_kl(s, c, self.kl_temperature)
                        + _cwd_kl(s, sl, self.kl_temperature)
                    )  # [B, H, W]
                    if dw is not None:
                        lc_map = lc_map * dw
                    lc = lc_map.mean()
                    lc_terms.append(lc)
                    lv_info["form_C"] = float(lc.detach())

                info["levels"][lv] = lv_info

            # Average over levels, then weight.
            zero = student[self.levels[0]].new_zeros(())
            lb_total = (sum(lb_terms) / len(lb_terms)) if lb_terms else zero
            lc_total = (sum(lc_terms) / len(lc_terms)) if lc_terms else zero
            loss = self.alpha * lb_total + self.beta * lc_total

        info["form_B"] = float(lb_total.detach()) if use_b else 0.0
        info["form_C"] = float(lc_total.detach()) if use_c else 0.0
        info["w_coco"] = float(w.detach())
        info["total"] = float(loss.detach())
        return loss, info

    # ── repr ─────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"levels={self.levels}, distill_form={self.distill_form}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"kl_temperature={self.kl_temperature}"
        )
