"""GASP kayıp fonksiyonları — L_kontrollü ve L_doğal (§2.3).

L_kontrollü: bir yamaya iki farklı bilinen ölçek dönüşümü uygula
    (resize + hafif augmentation), iki feature al, T'nin onları log-oranla
    tutarlı biçimde bağladığını sına. "Kontrollü" çünkü ölçek oranı kesin
    bilinir (biz uyguluyoruz).

L_doğal (sonraki modülde): EMA teacher tabanlı eşleştirme ile doğal
    sahnenin perspektif-derinlik ölçek değişimini kullanır.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _augment_patch(
    patches: torch.Tensor,
    target_size: int,
    brightness: float = 0.2,
    contrast: float = 0.2,
    blur_sigma: float = 0.5,
) -> torch.Tensor:
    """Resize + hafif renk-jitter + (opsiyonel) Gaussian blur.

    Kontrollü-hile (§3 Korku 2) ön-savunması: model "basit
    bilinear-interpolasyon operatörünü tersine çevirme" hilesini
    öğrenmesin diye, hedef-ölçeğe resize ÖNCESİ renk perturbasyonu
    uygulanır. Renk farkları "ölçek dönüşümü" ile karışmaz (T sadece
    ölçeği temsil etmeli), ama "resize-tanıma hilesini" zorlaştırır.

    Args:
        patches: [N, 3, P, P]
        target_size: çıktı boyutu (resize hedefi)
        brightness, contrast: renk-jitter şiddeti (yüzde olarak ±)
        blur_sigma: σ=0 ise blur uygulanmaz

    Returns:
        [N, 3, target_size, target_size]
    """
    N = patches.shape[0]
    device = patches.device

    # Renk-jitter: per-patch rastgele brightness + contrast
    if brightness > 0 or contrast > 0:
        # brightness: x = x + b (b ~ U[-brightness, +brightness])
        # contrast:   x = (x - mean) * (1+c) + mean (c ~ U[-contrast, +contrast])
        b = (torch.rand(N, 1, 1, 1, device=device) * 2 - 1) * brightness
        c = (torch.rand(N, 1, 1, 1, device=device) * 2 - 1) * contrast
        mean = patches.mean(dim=(-2, -1), keepdim=True)
        patches = (patches - mean) * (1 + c) + mean + b

    # Hafif Gaussian blur (opsiyonel — sigma>0 ise)
    if blur_sigma > 0:
        k = max(3, int(2 * round(2 * blur_sigma) + 1))
        if k % 2 == 0:
            k += 1
        # 1D Gauss kernel
        x = torch.arange(k, device=device, dtype=patches.dtype) - (k - 1) / 2
        g = torch.exp(-(x ** 2) / (2 * blur_sigma ** 2))
        g = g / g.sum()
        kernel = g[:, None] * g[None, :]    # [k, k]
        kernel = kernel.expand(3, 1, k, k)
        patches = F.conv2d(patches, kernel, padding=k // 2, groups=3)

    # Hedef boyuta resize
    return F.interpolate(
        patches, size=(target_size, target_size),
        mode="bilinear", align_corners=False,
    )


def controlled_loss(
    patches: torch.Tensor,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    transform: nn.Module,
    scale_a: float,
    scale_b: float,
    target_patch_size: int,
    augment: bool = True,
) -> dict:
    """L_kontrollü — bir yamadan iki farklı bilinen ölçekte feature çıkar,
    T'nin onları log-oranla tutarlı biçimde bağladığını sına.

    Bir yamayı iki kez augment et (her seferinde farklı bir ölçek dönüşümüyle),
    encoder'dan iki feature al, T(f_a, log(s_b/s_a)) ≈ f_b ister.

    Args:
        patches: [N, 3, P, P] giriş yamaları. P, patch_size'tan farklı
            olabilir (örn. sampler'dan 64×64 gelir; biz augment içinde
            target_patch_size'a resize ederiz).
        encoder: yama → feature fonksiyonu, [N,3,Q,Q] → [N, D].
        transform: ScaleEquivariantTransform örneği.
        scale_a, scale_b: iki augment'ın ölçek faktörleri. Resize
            içinde dolaylı; mesele log(s_b/s_a) oranı.
        target_patch_size: encoder'a verilecek boyut (resize hedefi).
        augment: True ise renk-jitter + blur (kontrollü-hile savunması);
            ablation/debug için False yapılabilir.

    Returns:
        {"loss": skaler tensor, "mse": L2 ortalama, "log_ratio": skaler}
    """
    if patches.dim() != 4 or patches.shape[1] != 3:
        raise ValueError(f"patches [N,3,P,P] olmalı, alındı: {tuple(patches.shape)}")
    if scale_a <= 0 or scale_b <= 0:
        raise ValueError(f"scales > 0 olmalı, alındı: {scale_a}, {scale_b}")
    if target_patch_size <= 0:
        raise ValueError(f"target_patch_size > 0 olmalı, alındı: {target_patch_size}")

    N = patches.shape[0]
    device = patches.device

    # İki augment — aynı patch, iki ayrı renk/blur randomizasyonu
    if augment:
        view_a = _augment_patch(patches, target_patch_size)
        view_b = _augment_patch(patches, target_patch_size)
    else:
        view_a = F.interpolate(patches, size=(target_patch_size, target_patch_size),
                                mode="bilinear", align_corners=False)
        view_b = view_a

    # Encoder
    f_a = encoder(view_a)   # [N, D]
    f_b = encoder(view_b)   # [N, D]

    # log-oran: T(f_a, log(s_b/s_a)) ≈ f_b
    import math
    log_ratio_val = math.log(scale_b / scale_a)
    log_ratio = torch.full((N, 1), log_ratio_val, device=device, dtype=f_a.dtype)

    # T uygula
    f_a_transformed = transform(f_a, log_ratio)   # [N, D]

    # L2 tutarsızlık
    mse = F.mse_loss(f_a_transformed, f_b)

    return {
        "loss": mse,
        "mse": float(mse.detach()),
        "log_ratio": log_ratio_val,
    }



def natural_loss(
    online_features: torch.Tensor,
    ema_features: torch.Tensor,
    log_scales: torch.Tensor,
    image_ids: torch.Tensor,
    matcher,
    transform: nn.Module,
) -> dict:
    """L_doğal — EMA-eşleştirilmiş çiftlerle T tutarlılığı (GASP §2.3).

    Doğal sahnenin perspektif-derinlik ölçek değişimini kullan: bir
    görüntüden farklı ölçekli yamalar arasında EMA-teacher tabanlı
    karşılıklı en-yakın-komşu eşleştirmesi yap; eşleşen çiftler için
    T(f_a, log_r) ≈ f_b tutarlılığını sına (simetrik — her iki yönde).

    Eşleşme yoksa skaler 0.0 döner (Mod A semantiği — eşleşmeyen
    yamalar `L_doğal`'a katkı vermez).

    Args:
        online_features: [N, D] öğrenci encoder feature'ları (gradyan
            akar).
        ema_features: [N, D] EMA-teacher feature'ları (eşleştirme için;
            detached olmalı, çağıran taraf sorumlu).
        log_scales: [N, 1] her yamanın log-ölçeği.
        image_ids: [N] her yamanın hangi görüntüden.
        matcher: NaturalPairMatcher örneği.
        transform: ScaleEquivariantTransform örneği.

    Returns:
        {"loss": skaler tensor, "n_pairs": int, "mse": float}
    """
    if online_features.dim() != 2:
        raise ValueError(
            f"online_features [N, D] olmalı, alındı: "
            f"{tuple(online_features.shape)}"
        )
    if online_features.shape != ema_features.shape:
        raise ValueError(
            f"online_features {tuple(online_features.shape)} ve ema_features "
            f"{tuple(ema_features.shape)} aynı şekilde olmalı"
        )

    device = online_features.device
    dtype = online_features.dtype

    match_out = matcher.match(ema_features, log_scales, image_ids)
    if match_out is None:
        # Mod A: eşleşme yok → 0.0 katkı.
        # requires_grad=False, ama trainer α-ağırlıkla toplarken
        # 0.0 + L_kontrollü gradyanı bozmaz (sıfır + gradyanlı tensor = gradyanlı).
        zero = torch.zeros((), device=device, dtype=dtype)
        return {"loss": zero, "n_pairs": 0, "mse": 0.0}

    idx_a, idx_b, log_r = match_out
    log_r = log_r.unsqueeze(1)   # [M, 1] — T'nin beklediği şekil

    f_a = online_features[idx_a]   # [M, D]
    f_b = online_features[idx_b]   # [M, D]

    # Simetrik tutarlılık: T(f_a, +log_r) ≈ f_b VE T(f_b, -log_r) ≈ f_a
    # Eşdeğişirlik matematiksel olarak iki-yönlü; tek yön T'yi bias eder.
    f_a_to_b = transform(f_a, log_r)         # [M, D]
    f_b_to_a = transform(f_b, -log_r)        # [M, D]
    loss_fwd = F.mse_loss(f_a_to_b, f_b)
    loss_bwd = F.mse_loss(f_b_to_a, f_a)
    loss = 0.5 * (loss_fwd + loss_bwd)

    return {
        "loss": loss,
        "n_pairs": int(idx_a.numel()),
        "mse": float(loss.detach()),
    }



def feature_regularization_loss(
    features: torch.Tensor,
    variance_target: float = 1.0,
    eps: float = 1e-4,
) -> dict:
    """VICReg-style feature düzenleyici — kollaps engeli (GASP'ın bilgi-koruma şartı).

    GASP'ın iki kaybı (controlled, natural) T tabanlı tutarlılık öğretir,
    ama bilgi koruma şartı yoktu — kollaps doğrulandı (ep2'de varyans
    2.3e-5, cosine 0.9996). Bu kayıp eşdeğişirliğin VARLIK ŞARTI olarak
    eklenir: feature uzayı bilgi taşımıyorsa, eşdeğişirlik trivially
    sağlanır (her şey ~sabit vektör).

    İki terim (Bardes et al. 2022, VICReg):
        - Variance: her boyutun std'si γ eşiğinin altına düşmesin
            V(Z) = (1/D) Σ_d max(0, γ − √(Var(z_d) + ε))
        - Covariance: off-diagonal kovaryanslar küçük olsun
            C(Z) = (1/D) Σ_{i≠j} [Cov(Z)]_{i,j}²

    Args:
        features: [N, D] feature vektörleri (online encoder çıktısı).
        variance_target: γ — std için minimum hedef (default 1.0, VICReg
            varsayılanı).
        eps: numerik stabilite.

    Returns:
        {"variance": skaler, "covariance": skaler}
    """
    if features.dim() != 2:
        raise ValueError(f"features [N, D] olmalı, alındı: {tuple(features.shape)}")
    if features.shape[0] < 2:
        # tek örnek var — varyans tanımsız, sıfır kayıp
        zero = torch.zeros((), device=features.device, dtype=features.dtype)
        return {"variance": zero, "covariance": zero}

    N, D = features.shape

    # Center: variance ve covariance için ortalamayı çıkar
    centered = features - features.mean(dim=0, keepdim=True)   # [N, D]

    # ── Variance loss ──
    # Std per boyut; γ eşiğinin altındaysa cezalandır (hinge)
    std = torch.sqrt(centered.var(dim=0) + eps)                # [D]
    variance_loss = F.relu(variance_target - std).mean()

    # ── Covariance loss ──
    # Cov(Z) = (1/(N-1)) Z^T Z (centered için)
    cov = (centered.T @ centered) / (N - 1)                    # [D, D]
    # Off-diagonal: maske + kare ortalama
    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=features.device)
    covariance_loss = (cov[off_diag_mask] ** 2).sum() / D

    return {"variance": variance_loss, "covariance": covariance_loss}
