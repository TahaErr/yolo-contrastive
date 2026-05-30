"""Çapraz-ölçek rekonstrüksiyon (v8) — GASP'ın eksik İÇERİK eksenini ekler.

Kanıtlanmış collapse kök nedeni (v5–v7): GASP'ın objektifi tek-eksenliydi.
Yalnızca ÖLÇEK eksenini kurdu (FiLM-T eşdeğişirlik + resim-içi tutarlılık);
İÇERİK (kimlik) eksenini hiç zorlamadı. "Araba ARABA mıdır" sorusu hiç
sorulmadığı için encoder o bilgiyi feature'a koymak zorunda kalmadı — ölçek
tutarlılığını düşük-rank bir alt-uzayda trivially sağlayıp çöktü. Yüksek-rank'ı
zorlayan tek kuvvet "içerik ayrımı"dır ve o objektifte yoktu. Batch-istatistiği
regularizer'ları (var/cov/iso) bunu çözemedi: kanıtlandı ki variance terimi bir
TABAN, eşitleyici değil (std>γ olan boyuta sıfır gradyan), ve kovaryans rank'a
kör (matched-norm'da çökmüş vs sağlıklı için identik kayıp). Projektör (v7) de
tutmadı: collapse baskısı projektörden geçip backbone'a ulaştı (eff_rank 3.09).

Çözüm (v8): İÇERİK eksenini etiketsiz, YAPISAL bir kuvvetle ekle —
rekonstrüksiyon. Bir yamayı geri kurmak, içeriği feature'da tutmayı ZORUNLU
kılar (düşük-rank darboğazdan piksel üretilemez). Bu, batch-istatistiği proxy'si
DEĞİL; bilgi-koruma kısıtının kendisi. En yakın prior: Scale-MAE (Reed 2023).

ÇAPRAZ-ÖLÇEK çerçevesi (naif rekonstrüksiyon değil): model yamayı `scale_a`'da
görür, `scale_b`'de yeniden üretmek zorundadır. Bu, iki ekseni TEK objektifte
birleştirir: içerik (yamayı yeniden çiz → yüksek-rank) + ölçek (farklı ölçekte
yeniden çiz → feature ölçeği yapılı kodlamalı). "Uzaktaki arabayı gör, yakındaki
görünümünü kestir" — GASP'ın asıl yakalamak istediği ölçek farkı, artık
rekonstrüksiyon hedefine gömülü.

UZAMSAL DARBOĞAZ — resize-ezberleme hilesini (tasarım belgesi "Korku 2") yener:
decoder, view_a'nın PİKSELLERİNİ görmez — yalnız encode edilmiş P5 uzamsal
haritasını (M_a, 64px yamada ~2×2) görür. view_a'yı resize edemez çünkü elinde
yok; view_b'nin içeriğini üretmenin tek yolu M_a'nın o içeriği KODLAMASIDIR.
2×2 darboğaz, MAE-masking'in sağlayacağı bilgi-kısıtını ücretsiz verir (64px
yamada P5 zaten ~2×2, masking dejenere olurdu) ve hileyi yapısal olarak engeller.
Bu kuvvet GAP-vektörü değil P5 uzamsal haritası üzerinden gider — ama GAP o
haritanın havuzlanmışı olduğundan backbone'un GAP feature'ı (downstream/eff_rank
ölçtüğümüz) da zenginleşir.

Decoder, log_r ile FiLM-koşullu — eşdeğişirlik transform'u (T) ile AYNI mekanizma.
Yöntem böylece birleşik: ölçek her yerde FiLM ile ele alınır (T + decoder).
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import _augment_patch_scale_aware


class ScaleConditionedDecoder(nn.Module):
    """Ölçek-koşullu hafif konvolüsyonel decoder (v8).

    P5 uzamsal feature haritasından (M_a, [N, C, h, w]) hedef yamayı
    ([N, 3, P, P]) yeniden üretir. log_r ile FiLM-koşullu: hedef ölçek-oranı
    decode'u modüle eder (T ile aynı FiLM mekanizması → birleşik mimari).

    Mimari: FiLM(log_r) → M_a modülasyonu → ardışık 2× upsample blokları
    (interpolate + conv-BN-ReLU) → 3-kanal head → sigmoid ([0,1] piksel).

    HAFİF: backbone'dan (~3M) belirgin küçük (~0.4M). Bu kasıtlı — "yüksek-rank
    decoder kapasitesinden değil, objektifin backbone'u zorlamasından gelir"
    iddiasını korur (ablation'da savunması kolay). Girişin uzamsal boyutundan
    bağımsız: son adımda out_size'a interpolate eder (her patch_size'a dayanıklı).

    Args:
        in_channels: M_a kanal sayısı (= backbone feat_level kanalı = feat_dim).
        out_size: yeniden üretilen yama kenar boyutu (= target_patch_size).
        base: ilk upsample bloğunun kanal sayısı (sonra yarılanır).
        gap_bottleneck: True ise girdi haritası önce 1×1'e (GAP) havuzlanır;
            decoder yalnız global vektörü (downstream/eff_rank'in kullandığı GAP
            feature'ı) görür. İçerik baskısını tam o vektöre bindirir — 2×2
            haritadan (1024 sayı) rekonstrüksiyon kolaydı; 1×1'den (256 sayı)
            çok daha zor, backbone'u daha çok zorlar. Default False (v8 2×2).
    """

    def __init__(self, in_channels: int, out_size: int = 64, base: int = 128,
                 gap_bottleneck: bool = False):
        super().__init__()
        self.out_size = out_size
        self.gap_bottleneck = gap_bottleneck
        # FiLM üreteci: log_r → (gamma, beta), her biri [N, in_channels].
        self.film = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2 * in_channels),
        )
        # Ardışık upsample blokları — kanallar yarılanır.
        chs = [in_channels, base, base // 2, base // 4, base // 8]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i], chs[i + 1], 3, padding=1),
                nn.BatchNorm2d(chs[i + 1]),
                nn.ReLU(inplace=True),
            )
            for i in range(len(chs) - 1)
        ])
        self.head = nn.Conv2d(chs[-1], 3, 3, padding=1)

    def forward(self, feat_map: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        """[N, C, h, w] + [N, 1] → [N, 3, out_size, out_size] (∈ [0,1])."""
        if self.gap_bottleneck:
            # Yalnız global vektörü gör (1×1) → içerik baskısı GAP-feature'ına biner.
            feat_map = F.adaptive_avg_pool2d(feat_map, 1)
        gamma, beta = self.film(log_ratio).chunk(2, dim=1)   # [N, C] her biri
        x = feat_map * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        for blk in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = blk(x)
        x = F.interpolate(
            x, size=(self.out_size, self.out_size),
            mode="bilinear", align_corners=False,
        )
        return torch.sigmoid(self.head(x))


def cross_scale_reconstruction_loss(
    patches: torch.Tensor,
    encoder_spatial: Callable[[torch.Tensor], torch.Tensor],
    decoder: ScaleConditionedDecoder,
    scale_a: float,
    scale_b: float,
    target_patch_size: int,
) -> Dict[str, object]:
    """Çapraz-ölçek rekonstrüksiyon kaybı (v8).

    Yamayı `scale_a`'da görür (view_a), `scale_b`'deki görünümünü (view_b)
    yeniden üretir. view_a'yı P5 uzamsal haritasına encode eder (gradyanlı,
    backbone'u şekillendirir), decoder'ı log_r = log(scale_b/scale_a) ile
    koşullandırır, view_b'ye karşı MSE alır.

    View'ler FOTOMETRİK JİTTER OLMADAN render edilir (brightness=contrast=
    blur=0) — tek fark ölçek olsun ki rekonstrüksiyon hedefi temiz/ulaşılabilir
    kalsın (rastgele parlaklık tahmin edilemez, kaybı kirletirdi). Ölçek-resampling
    `_augment_patch_scale_aware` ile yapılır (controlled_loss_F ile AYNI render
    — tek kaynak, tutarlı).

    Args:
        patches: [N, 3, P, P] ham yamalar (sampler'dan).
        encoder_spatial: yama → P5 uzamsal haritası [N, C, h, w] (GAP YOK).
            Gradyanlı online encoder olmalı (backbone'u zenginleştirir).
        decoder: ScaleConditionedDecoder.
        scale_a: girdi view ölçeği (model bunu görür).
        scale_b: hedef view ölçeği (model bunu yeniden üretir).
        target_patch_size: view kenar boyutu (decoder out_size ile eşleşmeli).

    Returns:
        {"loss": skaler MSE tensor, "log_ratio": float}
    """
    if scale_a <= 0 or scale_b <= 0:
        raise ValueError(f"scales > 0 olmalı, alındı: {scale_a}, {scale_b}")

    # Temiz ölçek-only view'ler (jitter yok → ulaşılabilir hedef).
    view_a = _augment_patch_scale_aware(
        patches, target_patch_size, scale_a,
        brightness=0.0, contrast=0.0, blur_sigma=0.0,
    )
    view_b = _augment_patch_scale_aware(
        patches, target_patch_size, scale_b,
        brightness=0.0, contrast=0.0, blur_sigma=0.0,
    )

    # view_a → P5 uzamsal harita (gradyanlı). Decoder yalnız bunu görür
    # (view_a piksellerini DEĞİL) → resize-ezberleme hilesi yapısal olarak imkansız.
    feat_map = encoder_spatial(view_a)            # [N, C, h, w]

    log_r = math.log(scale_b / scale_a)
    log_ratio = torch.full(
        (patches.size(0), 1), log_r,
        device=patches.device, dtype=feat_map.dtype,
    )
    recon = decoder(feat_map, log_ratio)          # [N, 3, P, P] ∈ [0,1]

    loss = F.mse_loss(recon, view_b)
    return {"loss": loss, "log_ratio": log_r}
