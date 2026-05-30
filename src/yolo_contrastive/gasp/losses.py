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


def _augment_patch_scale_aware(
    patches: torch.Tensor,
    target_size: int,
    view_scale: float,
    brightness: float = 0.2,
    contrast: float = 0.2,
    blur_sigma: float = 0.5,
) -> torch.Tensor:
    """Scale-aware augmentation (GASP §10.38 — L_ctrl fix).

    Yamayı önce `view_scale` oranına küçültür, sonra `target_size`'a geri
    büyütür. Düşük view_scale'de kaybolan detay GERİ GELMEZ — encoder iki
    view arasında GERÇEK ölçek farkını görür (eski `_augment_patch` ile her
    iki view aynı boyuta gidiyordu, ölçek sinyali sıfırdı: L_ctrl donuk bug).

    view_scale ∈ (0, 1] beklenir (örn. 0.2 → yama %20 boyutunda görünür,
    bulanık; 0.5 → %50, daha keskin). Mutlak ölçek semantiği: her view kendi
    ölçeğince bozulur, fark = log(scale_b/scale_a) ile tutarlı sinyal.

    Args:
        patches: [N, 3, P, P]
        target_size: encoder'a verilecek çıktı boyutu
        view_scale: görünür ölçek faktörü (0, 1]
        brightness, contrast, blur_sigma: _augment_patch ile aynı jitter

    Returns:
        [N, 3, target_size, target_size]
    """
    # Görünür-ölçek sinyali enjekte et: küçült → geri büyüt (detay kaybı kalıcı)
    intermediate = max(int(round(target_size * view_scale)), 4)
    small = F.interpolate(
        patches, size=(intermediate, intermediate),
        mode="bilinear", align_corners=False,
    )
    upscaled = F.interpolate(
        small, size=(target_size, target_size),
        mode="bilinear", align_corners=False,
    )
    # jitter + blur'ı _augment_patch'ten reuse et (son resize target→target no-op).
    return _augment_patch(
        upscaled, target_size,
        brightness=brightness, contrast=contrast, blur_sigma=blur_sigma,
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
        {"variance": skaler, "covariance": skaler, "isotropy": skaler}
        isotropy: boyut-varyanslarının normalize varyansı — rank-collapse
            engeli (kovaryansın kör olduğu varyans-eşitsizliğini cezalar).
    """
    if features.dim() != 2:
        raise ValueError(f"features [N, D] olmalı, alındı: {tuple(features.shape)}")
    if features.shape[0] < 2:
        # tek örnek var — varyans tanımsız, sıfır kayıp
        zero = torch.zeros((), device=features.device, dtype=features.dtype)
        return {"variance": zero, "covariance": zero, "isotropy": zero}

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

    # ── Isotropy loss (rank-collapse engeli) ──
    # Kovaryans terimi varyans-EŞİTSİZLİĞİNE kördür: boyutlar dik (cov≈0)
    # olsa bile varyans birkaç boyuta yığılırsa eff_rank düşer (boyutsal
    # collapse). Statik kanıt: çökmüş (eff_rank 20) ve sağlıklı (eff_rank
    # 205) feature için cov_loss aynı çıkar. Bu terim boyut-varyanslarının
    # normalize varyansını (varyasyon katsayısı²) cezalar: tüm varyanslar
    # eşitse ≈0, birkaç boyuta yığılıysa yüksek (çökmüşte ~24 vs sağlıklı
    # ~0.01). Ölçek-bağımsız (mean² ile normalize → γ kalibrasyonu yok).
    # Variance floor ile birlikte: floor global-sıfır-collapse'ı, isotropy
    # varyans-yığılmasını engeller.
    var_per_dim = centered.var(dim=0)                          # [D]
    isotropy_loss = var_per_dim.var() / (var_per_dim.mean() ** 2 + eps)

    return {
        "variance": variance_loss,
        "covariance": covariance_loss,
        "isotropy": isotropy_loss,
    }



def controlled_loss_F(
    patches: torch.Tensor,
    encoder,
    transform,
    scale_a: float,
    scale_b: float,
    target_patch_size: int,
    candidate_log_ratios: torch.Tensor = None,
    temperature: float = 0.07,
    augment: bool = True,
    scale_aware_aug: bool = True,
    similarity: str = "cosine",
    detach_encoder: bool = False,
) -> dict:
    """L_kontrollü_F — sahte-dönüşüm-aware controlled loss (GASP §10.37).

    log_r üzerinde InfoNCE: T uygulandığında f_b'ye en çok yaklaşan log_r
    GERÇEK log_r olmalı; sahte log_r adayları daha uzak vermeli. Bu, T'nin
    log_r'nin gerçek bir fonksiyonu olduğunu öğrenmesini zorlar (EquiMod'un
    T=identity'ye çökme tuzağının doğrudan adresi).

    Formülasyon:
        d_k = ||T(f_a, log_r_k) - f_b||²
        L = -log[ exp(-d_real/τ) / Σ_k exp(-d_k/τ) ]

    Args:
        patches: [N, 3, P, P] giriş yamaları.
        encoder: yama → feature ([N,3,Q,Q] → [N, D]).
        transform: ScaleEquivariantTransform.
        scale_a, scale_b: iki augment'ın ölçek faktörleri (gerçek log_r =
            log(scale_b / scale_a)).
        target_patch_size: encoder'a verilecek boyut.
        candidate_log_ratios: [K] sahte log_r adayları. None ise YALNIZ
            gerçek log_r kullanılır (L2 fallback, ablation için).
        temperature: InfoNCE sıcaklığı (default 0.07; cosine için kalibre,
            MoCo aralığında — §10.39 oracle sweep).
        augment: True ise renk-jitter + blur (kontrollü-hile savunması).
        scale_aware_aug: True (default) ise iki view scale_a/scale_b ile
            fiilen farklı görünür ölçeğe getirilir (L_ctrl fix, §10.38).
            False → eski davranış (her iki view aynı boyut; ablation/uyumluluk).
        similarity: "cosine" (default) → L2-normalize edilmiş cosine InfoNCE
            (kalibre, kontrast üretir). "mse" → eski ham kare-L2 (ablation;
            §10.39: oracle T'yle bile chance'in ~%2 altında — kullanma).
        detach_encoder: True ise f_a/f_b encoder grafiğinden koparılır →
            L_ctrl yalnız T'yi eğitir, encoder'a gradyan akmaz (§10.40).
            Gerekçe: L_ctrl sentetik augmentasyon; encoder'ı doğal L_nat
            şekillendirmeli. İzolasyon testi: encoder donukken T kusursuz
            öğreniyor; joint'te L_ctrl gradyanı encoder'ı bozuyordu.

    Returns:
        {"loss": skaler, "mse_real": float, "log_ratio_real": float,
         "n_candidates": int}
    """
    if patches.dim() != 4 or patches.shape[1] != 3:
        raise ValueError(f"patches [N,3,P,P] olmalı, alındı: {tuple(patches.shape)}")
    if scale_a <= 0 or scale_b <= 0:
        raise ValueError(f"scales > 0 olmalı, alındı: {scale_a}, {scale_b}")
    if temperature <= 0:
        raise ValueError(f"temperature > 0 olmalı, alındı: {temperature}")

    import math
    N = patches.shape[0]
    device = patches.device

    # İki augment
    if augment:
        if scale_aware_aug:
            # Scale-aware: iki view fiilen farklı görünür ölçek görür (§10.38 fix)
            view_a = _augment_patch_scale_aware(patches, target_patch_size, scale_a)
            view_b = _augment_patch_scale_aware(patches, target_patch_size, scale_b)
        else:
            # Eski davranış (ablation/uyumluluk): ölçek sinyali yok
            view_a = _augment_patch(patches, target_patch_size)
            view_b = _augment_patch(patches, target_patch_size)
    else:
        view_a = F.interpolate(patches, size=(target_patch_size, target_patch_size),
                                mode="bilinear", align_corners=False)
        view_b = view_a

    f_a = encoder(view_a)   # [N, D]
    f_b = encoder(view_b)   # [N, D]

    if detach_encoder:
        # L_ctrl yalnız T'yi eğitsin; encoder'a gradyan akmasın (§10.40).
        f_a = f_a.detach()
        f_b = f_b.detach()

    # Gerçek log_r
    log_ratio_real = math.log(scale_b / scale_a)

    # Aday log_r set: gerçek + sahteler (eğer verildiyse)
    if candidate_log_ratios is None or candidate_log_ratios.numel() == 0:
        # Fallback: yalnız gerçek → L2 MSE (sadece smoke için)
        log_r = torch.full((N, 1), log_ratio_real, device=device, dtype=f_a.dtype)
        f_a_T = transform(f_a, log_r)
        mse = F.mse_loss(f_a_T, f_b)
        return {
            "loss": mse,
            "mse_real": float(mse.detach()),
            "log_ratio_real": log_ratio_real,
            "n_candidates": 1,
        }

    # Aday set: real önce, sonra sahteler. İlk indeks = real.
    candidates = candidate_log_ratios.to(device=device, dtype=f_a.dtype)   # [K_other]
    # Duplike-temizleme (§10.39 bugfix): gerçek log_r sahte adaylar arasında
    # da varsa çıkar — yoksa pozitif aynı anda negatif olur, oracle T bile
    # p_real ≤ 0.5'e sıkışır.
    candidates = candidates[(candidates - log_ratio_real).abs() > 1e-6]
    real_t = torch.tensor([log_ratio_real], device=device, dtype=f_a.dtype)
    all_log_r = torch.cat([real_t, candidates], dim=0)   # [K_total], real ilk
    K = all_log_r.shape[0]

    # T'yi K aday için vektörize uygula: tile f_a → [K*N, D]
    f_a_tiled = f_a.unsqueeze(0).expand(K, N, -1).reshape(K * N, -1)   # [K*N, D]
    log_r_tiled = all_log_r.unsqueeze(1).expand(K, N).reshape(K * N, 1)
    f_a_T = transform(f_a_tiled, log_r_tiled)   # [K*N, D]
    f_a_T = f_a_T.reshape(K, N, -1)              # [K, N, D]
    f_b_exp = f_b.unsqueeze(0).expand(K, N, -1)   # [K, N, D]

    # InfoNCE logit'leri — gerçek (k=0) en yüksek skoru almalı.
    # (§10.39: ham MSE/τ kontrast üretmiyordu; cosine kalibre.)
    if similarity == "cosine":
        sim = F.cosine_similarity(f_a_T, f_b_exp, dim=-1)   # [K, N], yüksek=iyi
        logits = sim / temperature
    elif similarity == "mse":
        d_k = ((f_a_T - f_b_exp) ** 2).mean(dim=-1)         # [K, N], düşük=iyi
        logits = -d_k / temperature
    else:
        raise ValueError(f"similarity 'cosine' ya da 'mse' olmalı, alındı: {similarity!r}")

    log_probs = F.log_softmax(logits, dim=0)   # adaylar üzerinde softmax (per-sample)
    loss = -log_probs[0].mean()                # L = -log p(real) ortalama

    # mse_real: similarity'den bağımsız teşhis metriği (her zaman ham MSE)
    mse_real = float(((f_a_T[0] - f_b) ** 2).mean().detach())

    return {
        "loss": loss,
        "mse_real": mse_real,
        "log_ratio_real": log_ratio_real,
        "n_candidates": K,
    }



def natural_loss_F(
    online_features: torch.Tensor,
    ema_features: torch.Tensor,
    log_scales: torch.Tensor,
    image_ids: torch.Tensor,
    matcher,
    transform,
    candidate_log_ratios: torch.Tensor = None,
    temperature: float = 0.2,
    similarity: str = "cosine",
) -> dict:
    """L_doğal_F — sahte-dönüşüm-aware natural loss (GASP §10.37).

    EMA-eşleştirilmiş çiftler için log_r üzerinde InfoNCE. Her eşleşen
    çiftin (idx_a_i, idx_b_i, log_r_i) gerçek log_r_i'si var; sahte
    adaylar = scales'tan teorik üretilen TÜM OLASI log_r değerleri
    (eşleşmelerden değil — eşleşmelerde duplikat olur, InfoNCE softmax
    duplikatları ayırt edemez). Symmetric uygulama: hem T(f_a, +log_r) →
    f_b, hem T(f_b, -log_r) → f_a yönü eşit ağırlıkla InfoNCE.

    Args:
        online_features: [N, D] öğrenci feature (gradient akar).
        ema_features: [N, D] EMA-teacher feature (eşleştirme için).
        log_scales, image_ids: sampler'dan.
        matcher: NaturalPairMatcher.
        transform: ScaleEquivariantTransform.
        temperature: InfoNCE sıcaklığı (default 0.2).
        similarity: "cosine" (default, L2-normalize edilmiş cosine — genlik
            collapse'ına dayanıklı, controlled_loss_F ile tutarlı) veya
            "mse" (eski ham kare-L2, genliğe duyarlı).

    Returns:
        {"loss": skaler, "n_pairs": int, "mse_real": float}
    """
    if online_features.dim() != 2:
        raise ValueError(f"online_features [N, D] olmalı, alındı: {tuple(online_features.shape)}")
    if online_features.shape != ema_features.shape:
        raise ValueError(f"online {online_features.shape} ve ema {ema_features.shape} eşit değil")
    if temperature <= 0:
        raise ValueError(f"temperature > 0 olmalı, alındı: {temperature}")
    if similarity not in ("cosine", "mse"):
        raise ValueError(f"similarity 'cosine' veya 'mse' olmalı, alındı: {similarity}")

    device = online_features.device
    dtype = online_features.dtype

    match_out = matcher.match(ema_features, log_scales, image_ids)
    if match_out is None:
        zero = torch.zeros((), device=device, dtype=dtype)
        return {"loss": zero, "n_pairs": 0, "mse_real": 0.0}

    idx_a, idx_b, log_r = match_out
    M = idx_a.numel()

    # Fallback: tek eşleşme varsa InfoNCE anlamsız (K=1), L2 davran
    if M < 2:
        log_r_col = log_r.unsqueeze(1)
        f_a = online_features[idx_a]
        f_b = online_features[idx_b]
        f_a_T = transform(f_a, log_r_col)
        f_b_T = transform(f_b, -log_r_col)
        mse = 0.5 * (F.mse_loss(f_a_T, f_b) + F.mse_loss(f_b_T, f_a))
        return {"loss": mse, "n_pairs": M, "mse_real": float(mse.detach())}

    # Aday log_r seti: scales'tan üretilen TÜM olası log_r'ler
    # (candidate_log_ratios). Eğer verilmemişse, eşleşmelerden benzersiz
    # alarak fallback (eski mantık, duplikat riski var).
    f_a = online_features[idx_a]                # [M, D]
    f_b = online_features[idx_b]                # [M, D]

    if candidate_log_ratios is not None and candidate_log_ratios.numel() > 0:
        all_log_r = candidate_log_ratios.to(device=device, dtype=f_a.dtype)
    else:
        # Fallback: eşleşmelerden benzersiz log_r'ler
        all_log_r = torch.unique(log_r)
    K = all_log_r.shape[0]

    # Her çift için "doğru" log_r index'i: aday set'te real_log_r'ye en yakın
    # (genellikle birebir eşleşir; numerik tolerans için argmin)
    # all_log_r [K], log_r [M]: dist matrix [K, M]
    dist_to_candidates = (all_log_r.unsqueeze(1) - log_r.unsqueeze(0)).abs()
    target = dist_to_candidates.argmin(dim=0)   # [M] her çiftin aday-set indexi

    # ── Forward yön: T(f_a, log_r_k) ≈ f_b ──
    # K aday × M çift için T(f_a, log_r_k): tile f_a → [K*M, D]
    f_a_tiled = f_a.unsqueeze(0).expand(K, M, -1).reshape(K * M, -1)
    log_r_tiled = all_log_r.unsqueeze(1).expand(K, M).reshape(K * M, 1)
    f_a_T = transform(f_a_tiled, log_r_tiled).reshape(K, M, -1)
    f_b_exp = f_b.unsqueeze(0).expand(K, M, -1)
    d_fwd = ((f_a_T - f_b_exp) ** 2).mean(dim=-1)   # [K, M] — ham MSE (diagnostik)
    if similarity == "cosine":
        logits_fwd = F.cosine_similarity(f_a_T, f_b_exp, dim=-1) / temperature
    else:
        logits_fwd = -d_fwd / temperature
    log_probs_fwd = F.log_softmax(logits_fwd, dim=0)
    loss_fwd = -log_probs_fwd[target, torch.arange(M, device=device)].mean()

    # ── Backward yön: T(f_b, -log_r_k) ≈ f_a ──
    neg_log_r_tiled = -all_log_r.unsqueeze(1).expand(K, M).reshape(K * M, 1)
    f_b_tiled = f_b.unsqueeze(0).expand(K, M, -1).reshape(K * M, -1)
    f_b_T = transform(f_b_tiled, neg_log_r_tiled).reshape(K, M, -1)
    f_a_exp = f_a.unsqueeze(0).expand(K, M, -1)
    d_bwd = ((f_b_T - f_a_exp) ** 2).mean(dim=-1)
    if similarity == "cosine":
        logits_bwd = F.cosine_similarity(f_b_T, f_a_exp, dim=-1) / temperature
    else:
        logits_bwd = -d_bwd / temperature
    log_probs_bwd = F.log_softmax(logits_bwd, dim=0)
    # Backward yönde "doğru" -log_r için: -all_log_r[target] ≈ -log_r[m]
    # Yani target aynı kalır (negation symmetric)
    loss_bwd = -log_probs_bwd[target, torch.arange(M, device=device)].mean()

    loss = 0.5 * (loss_fwd + loss_bwd)
    mse_real = float(d_fwd[target, torch.arange(M, device=device)].mean().detach())

    return {"loss": loss, "n_pairs": M, "mse_real": mse_real}
