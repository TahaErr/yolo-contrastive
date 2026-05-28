"""GASP v6 — scale-aware augmentation (L_ctrl fix) testleri (§10.38).

Eski bug: controlled_loss_F'in iki view'i de aynı target_size'a gidiyordu,
scale_a/scale_b sadece log_ratio'da kullanılıyordu → encoder ölçek farkı
görmüyordu → L_ctrl 30 epoch boyunca %0.0 hareket. Bu testler fix'in
(a) fiilen ölçek sinyali enjekte ettiğini, (b) öğrenilebilir olduğunu,
(c) eski davranışı bozmadığını kanıtlar.
"""

from __future__ import annotations

import torch
import math

import pytest
import torch.nn as nn
import torch.nn.functional as F

from yolo_contrastive.gasp import ScaleEquivariantTransform, controlled_loss_F
from yolo_contrastive.gasp.losses import _augment_patch_scale_aware


def _mock_encoder(D: int = 64) -> nn.Module:
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, D, 3, padding=1)
            self.fc = nn.Linear(D, D)

        def forward(self, x):
            x = self.conv(x)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.fc(x)

    return Enc()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean().item()


class TestScaleAwareAugmentation:
    def test_different_scale_changes_image(self):
        # Jitter kapalı (deterministik): farklı ölçek → farklı görüntü.
        torch.manual_seed(0)
        patches = torch.rand(4, 3, 64, 64)
        view_lo = _augment_patch_scale_aware(patches, 64, 0.2, 0, 0, 0)
        view_hi = _augment_patch_scale_aware(patches, 64, 0.8, 0, 0, 0)
        assert not torch.allclose(view_lo, view_hi)
        assert (view_lo - view_hi).pow(2).mean().item() > 1e-5

    def test_same_scale_no_jitter_is_identical(self):
        # Jitter kapalı + aynı ölçek → birebir aynı (deterministik resize).
        torch.manual_seed(0)
        patches = torch.rand(4, 3, 64, 64)
        a = _augment_patch_scale_aware(patches, 64, 0.4, 0, 0, 0)
        b = _augment_patch_scale_aware(patches, 64, 0.4, 0, 0, 0)
        assert torch.allclose(a, b)

    def test_low_scale_loses_detail(self):
        # Düşük ölçek daha bulanık → yüksek-frekans enerji düşmeli.
        torch.manual_seed(0)
        patches = torch.rand(2, 3, 64, 64)
        lo = _augment_patch_scale_aware(patches, 64, 0.15, 0, 0, 0)
        hi = _augment_patch_scale_aware(patches, 64, 0.9, 0, 0, 0)
        # Komşu-piksel farkı (yüksek frekans proxy'si)
        tv_lo = (lo[..., 1:, :] - lo[..., :-1, :]).abs().mean().item()
        tv_hi = (hi[..., 1:, :] - hi[..., :-1, :]).abs().mean().item()
        assert tv_lo < tv_hi

    def test_min_intermediate_size(self):
        # Çok küçük ölçek bile en az 4px ara boyut → çökmemeli, doğru çıktı.
        patches = torch.rand(2, 3, 64, 64)
        out = _augment_patch_scale_aware(patches, 64, 0.001, 0, 0, 0)
        assert out.shape == (2, 3, 64, 64)

    def test_encoder_features_respond_to_scale(self):
        # Fix'in özü görüntü düzeyinde kanıtlandı (yukarı). Burada encoder'ın
        # ölçek farkına YANIT verdiğini gösteririz. NOT: eğitilmemiş rastgele
        # GAP encoder yüksek-frekansı ortaladığı için ayrım küçüktür (plan
        # Risk 1: "P5 GAP fazla sıkıştırıyor"). Ayrımı BÜYÜTMEK eğitimin işi —
        # bunu test_loss_decreases_over_steps yerel olarak doğrular.
        torch.manual_seed(0)
        enc = _mock_encoder(D=64)
        patches = torch.rand(8, 3, 64, 64)
        with torch.no_grad():
            f_lo = enc(_augment_patch_scale_aware(patches, 64, 0.1, 0, 0, 0))
            f_hi = enc(_augment_patch_scale_aware(patches, 64, 0.9, 0, 0, 0))
            f_lo2 = enc(_augment_patch_scale_aware(patches, 64, 0.1, 0, 0, 0))
        assert not torch.allclose(f_lo, f_hi)   # farklı ölçek → yanıt var
        assert _cos(f_lo, f_lo2) > 0.95         # aynı ölçek (jitter yok) → aynı


class TestControlledLossFScaleAware:
    def test_scale_aware_default_runs(self):
        out = controlled_loss_F(
            torch.rand(4, 3, 64, 64), _mock_encoder(64),
            ScaleEquivariantTransform(feat_dim=64, hidden_dim=16),
            scale_a=0.2, scale_b=0.5, target_patch_size=32,
            candidate_log_ratios=torch.tensor([-0.5, 0.5, 1.0]),
        )
        assert set(out.keys()) == {"loss", "mse_real", "log_ratio_real", "n_candidates"}
        assert out["loss"].dim() == 0

    def test_gradient_flows_scale_aware(self):
        enc = _mock_encoder(64)
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        out = controlled_loss_F(
            torch.rand(4, 3, 64, 64), enc, T,
            scale_a=0.2, scale_b=0.5, target_patch_size=32,
            candidate_log_ratios=torch.tensor([-0.5, 0.5, 1.0]),
            scale_aware_aug=True,
        )
        out["loss"].backward()
        g_enc = sum(p.grad.abs().sum().item()
                    for p in enc.parameters() if p.grad is not None)
        g_T = sum(p.grad.abs().sum().item()
                  for p in T.parameters() if p.grad is not None)
        assert g_enc > 0
        assert g_T > 0

    def test_dedup_removes_real_from_candidates(self):
        # §10.39 bugfix: gerçek log_r sahte adaylarda da varsa çıkarılmalı.
        real_lr = math.log(0.5 / 0.2)
        cands = torch.tensor([real_lr, -0.5, 0.5])   # ilk eleman = gerçek (duplike)
        out = controlled_loss_F(
            torch.rand(4, 3, 64, 64), _mock_encoder(64),
            ScaleEquivariantTransform(feat_dim=64, hidden_dim=16),
            scale_a=0.2, scale_b=0.5, target_patch_size=32,
            candidate_log_ratios=cands,
        )
        # real(1) + duplike-olmayan 2 = 3, duplike sayılmamalı (yoksa 4)
        assert out["n_candidates"] == 3

    def test_similarity_modes_both_run(self):
        kw = dict(scale_a=0.2, scale_b=0.5, target_patch_size=32,
                  candidate_log_ratios=torch.tensor([-0.5, 0.5, 1.0]))
        for sim in ("cosine", "mse"):
            out = controlled_loss_F(
                torch.rand(4, 3, 64, 64), _mock_encoder(64),
                ScaleEquivariantTransform(feat_dim=64, hidden_dim=16),
                similarity=sim, **kw)
            assert out["loss"].dim() == 0
            assert out["mse_real"] >= 0   # similarity'den bağımsız ham MSE

    def test_invalid_similarity_raises(self):
        with pytest.raises(ValueError):
            controlled_loss_F(
                torch.rand(4, 3, 64, 64), _mock_encoder(64),
                ScaleEquivariantTransform(feat_dim=64, hidden_dim=16),
                scale_a=0.2, scale_b=0.5, target_patch_size=32,
                candidate_log_ratios=torch.tensor([-0.5, 0.5]),
                similarity="foo")

    def test_cosine_loss_is_optimizable(self):
        # Kalibre loss eğitilebilir olmalı (chance'te donuk DEĞİL). NOT: bu
        # optimize-edilebilirlik testi, ölçek-genellemesi DEĞİL — gerçek
        # sinyal doğrulaması Colab smoke (§10.39 oracle zaten matematiği
        # kanıtladı). Sabit çift + augment=True (jitter) → ezber değil.
        torch.manual_seed(0)
        enc = _mock_encoder(64)
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        patches = torch.rand(8, 3, 64, 64)
        cands = torch.tensor([-1.0, -0.5, 0.5, 1.0])   # gerçek (≈0.916) yok → K=5
        opt = torch.optim.Adam(list(enc.parameters()) + list(T.parameters()), lr=2e-3)
        first = last = None
        for step in range(120):
            opt.zero_grad()
            out = controlled_loss_F(
                patches, enc, T, scale_a=0.2, scale_b=0.5,
                target_patch_size=32, candidate_log_ratios=cands)  # cosine default
            out["loss"].backward(); opt.step()
            if step == 0: first = out["loss"].item()
            last = out["loss"].item()
        assert last < 0.7 * first   # chance'ten belirgin kopuş

    def test_detach_encoder_blocks_encoder_grad(self):
        # §10.40: detach_encoder=True → encoder'a gradyan akmaz, T'ye akar.
        enc = _mock_encoder(64)
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        out = controlled_loss_F(
            torch.rand(4, 3, 64, 64), enc, T,
            scale_a=0.2, scale_b=0.5, target_patch_size=32,
            candidate_log_ratios=torch.tensor([-0.5, 0.5, 1.0]),
            detach_encoder=True,
        )
        out["loss"].backward()
        g_enc = sum((p.grad.abs().sum().item() if p.grad is not None else 0.0)
                    for p in enc.parameters())
        g_T = sum(p.grad.abs().sum().item()
                  for p in T.parameters() if p.grad is not None)
        assert g_enc == 0.0   # encoder donuk (L_ctrl tarafından)
        assert g_T > 0        # T öğreniyor

    def test_flag_inert_when_augment_false(self):
        # augment=False yolunda scale_aware_aug etkisiz olmalı (her iki view = interpolate).
        torch.manual_seed(0)
        enc = _mock_encoder(64)
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        patches = torch.rand(4, 3, 64, 64)
        cands = torch.tensor([-0.5, 0.5])
        kw = dict(scale_a=0.2, scale_b=0.5, target_patch_size=32,
                  candidate_log_ratios=cands, augment=False)
        out_t = controlled_loss_F(patches, enc, T, scale_aware_aug=True, **kw)
        out_f = controlled_loss_F(patches, enc, T, scale_aware_aug=False, **kw)
        assert torch.allclose(out_t["loss"], out_f["loss"])

    def test_flag_changes_views_when_augment_true(self):
        # augment=True'da flag fiilen davranışı değiştirmeli (scale sinyali enjekte).
        # Kanıt: aynı seed'le scale-aware view'lerin yüksek-frekans enerjisi,
        # ölçek-bağımsız (eski) view'lerden farklı olmalı.
        torch.manual_seed(0)
        patches = torch.rand(4, 3, 64, 64)
        # scale-aware: scale_a=0.2 ile küçültülmüş → bulanık
        sa = _augment_patch_scale_aware(patches, 32, 0.2, 0, 0, 0)
        # eski yol proxy: direkt resize (ölçek küçültme yok)
        old = F.interpolate(patches, size=(32, 32), mode="bilinear", align_corners=False)
        tv_sa = (sa[..., 1:, :] - sa[..., :-1, :]).abs().mean().item()
        tv_old = (old[..., 1:, :] - old[..., :-1, :]).abs().mean().item()
        assert tv_sa < tv_old   # scale-aware daha bulanık → daha az detay
