"""NaturalPairMatcher — EMA-teacher tabanlı açık eşleştirme (GASP §4 Karar 1+2).

Bir görüntüden çıkarılan farklı-ölçekli yamalar arasında, aynı nesnenin
iki ölçek versiyonu olan çiftleri belirler. Eşleştirme kararı, modelin
EMA-yumuşatılmış kopyasının (teacher) feature'larında karşılıklı
en-yakın-komşu + cosine eşik kuralıyla verilir.

Mod A (açık eşleştirme): eşleşen çift bulunursa onlar `L_doğal`'a girer,
bulunmazsa o yama doğal kayba katkı vermez. Yumuşak ağırlıklı varyant
(Mod B) yerine bu seçildi çünkü paper anlatısı net olur ("aynı nesnenin
iki ölçeği"), qualitative figure üretilebilir.

Tasarım:
    - Aday havuzu: AYNI görüntü içinde + FARKLI ölçek. Per-image
      gruplama §4 Karar 3'ün gereği. Kendi ölçeğinde eşleşme
      aranmaz — T'yi log_ratio≈0'a iter, ölçek-farklılığı sinyali ölür.
    - Karşılıklı en-yakın-komşu: tek-yönlü en-yakın gürültülü;
      karşılıklı eşleşme kalite filtresi (yamalar birbirini SEÇER).
    - Cosine eşik (τ): karşılıklı olsa bile düşük benzerlikte eşleşmeyi
      at. Gürültü filtresi; eşik aşılmazsa o yama eşleşmez.
    - EMA encoder dışarıda: matcher EMA tutmaz, çağrıda ema_features
      alır. EMA güncellemesi trainer sorumluluğunda (MoCo-v3 paterni).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class NaturalPairMatcher:
    """Açık çift eşleştirme — karşılıklı en-yakın-komşu + cosine eşik.

    Args:
        similarity_threshold: cosine benzerlik eşiği (τ). Karşılıklı
            en-yakın olsa bile benzerlik < τ ise eşleşme reddedilir.
            Default 0.7 — tipik SSL similarity-bootstrapping aralığı.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        if not (-1.0 <= similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold ∈ [-1, 1] olmalı, alındı: "
                f"{similarity_threshold}"
            )
        self.similarity_threshold = similarity_threshold

    def match(
        self,
        ema_features: torch.Tensor,
        log_scales: torch.Tensor,
        image_ids: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Çift eşleştir.

        Args:
            ema_features: [N, D] EMA-teacher feature'ları (detached
                önerilir; matcher gradient'i durdurmaz, çağıran taraf
                sorumlu).
            log_scales: [N, 1] her yamanın log-ölçeği.
            image_ids: [N] her yamanın hangi görüntüden geldiği.

        Returns:
            None — hiç eşleşme yoksa.
            (idx_a, idx_b, log_ratio): eşleşen çiftlerin indeksleri ve
                log(scale_b / scale_a) oranları. Her şekil [M], M =
                eşleşme sayısı.
        """
        if ema_features.dim() != 2:
            raise ValueError(
                f"ema_features [N, D] olmalı, alındı: "
                f"{tuple(ema_features.shape)}"
            )
        N = ema_features.shape[0]
        if log_scales.shape != (N, 1):
            raise ValueError(
                f"log_scales [{N}, 1] olmalı, alındı: "
                f"{tuple(log_scales.shape)}"
            )
        if image_ids.shape != (N,):
            raise ValueError(
                f"image_ids [{N}] olmalı, alındı: {tuple(image_ids.shape)}"
            )

        if N < 2:
            return None

        # L2-normalize → cosine = inner product
        f_norm = F.normalize(ema_features, dim=1)
        # [N, N] benzerlik matrisi (tüm çiftler)
        sim = f_norm @ f_norm.t()

        # Aday maskesi: aynı görüntü + farklı ölçek + i != j
        same_image = image_ids.unsqueeze(0) == image_ids.unsqueeze(1)   # [N, N]
        same_scale = (
            log_scales.squeeze(1).unsqueeze(0)
            == log_scales.squeeze(1).unsqueeze(1)
        )                                                                # [N, N]
        diff_scale = ~same_scale
        diagonal = torch.eye(N, dtype=torch.bool, device=ema_features.device)
        valid = same_image & diff_scale & ~diagonal                      # [N, N]

        # Geçersiz adayları -inf yap (argmax filtrelensin)
        masked_sim = sim.masked_fill(~valid, float("-inf"))

        # Her satır için en yakın komşu (argmax)
        best_for_each = masked_sim.argmax(dim=1)   # [N]
        # Skor: en yakın komşu benzerliği
        best_score = masked_sim.gather(1, best_for_each.unsqueeze(1)).squeeze(1)

        # i'nin en yakını j, j'nin en yakını da i mi (karşılıklı)?
        # AYRICA: i'nin geçerli komşusu var mı (best_score > -inf)?
        has_neighbor = best_score > float("-inf")
        # j'lerin en yakınını al
        best_of_best = best_for_each[best_for_each]   # [N], dolaylı erişim
        mutual = (best_of_best == torch.arange(N, device=ema_features.device))

        # Çift maskesi: karşılıklı + komşusu var + eşik üstü + ikilik tekilleştirme
        eligible = mutual & has_neighbor & (best_score >= self.similarity_threshold)
        # (i, j) ve (j, i) çiftlerini tekilleştir: yalnız i < j olanları al
        idx_a_all = torch.arange(N, device=ema_features.device)
        idx_b_all = best_for_each
        keep = eligible & (idx_a_all < idx_b_all)

        if not keep.any():
            return None

        idx_a = idx_a_all[keep]
        idx_b = idx_b_all[keep]

        # log_ratio = log(scale_b / scale_a) = log_scale_b - log_scale_a
        log_r = (log_scales[idx_b] - log_scales[idx_a]).squeeze(1)

        return idx_a, idx_b, log_r
