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

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        stratify_by_scale_pair: bool = False,
    ):
        if not (-1.0 <= similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold ∈ [-1, 1] olmalı, alındı: "
                f"{similarity_threshold}"
            )
        self.similarity_threshold = similarity_threshold
        self.stratify_by_scale_pair = stratify_by_scale_pair

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
        log_scale_col = log_scales.squeeze(1)
        same_scale = log_scale_col.unsqueeze(0) == log_scale_col.unsqueeze(1)
        diff_scale = ~same_scale
        diagonal = torch.eye(N, dtype=torch.bool, device=ema_features.device)
        base_valid = same_image & diff_scale & ~diagonal                 # [N, N]

        if self.stratify_by_scale_pair:
            return self._match_stratified(
                sim, base_valid, log_scale_col, log_scales, ema_features
            )

        valid = base_valid                                               # [N, N]

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

    def _match_stratified(
        self,
        sim: torch.Tensor,
        base_valid: torch.Tensor,
        log_scale_col: torch.Tensor,
        log_scales: torch.Tensor,
        ema_features: torch.Tensor,
    ):
        """Scale-pair-stratified matching: her benzersiz (s_a, s_b) çifti için
        ayrı karşılıklı en-yakın-komşu eşleştirmesi yap; sonuçları birleştir.

        Sebebi: karşılıklı en-yakın doğal olarak "kolay" (yakın-ölçek)
        çiftleri tercih ediyor — log_r dağılımı bozuk oluyor (örn. 14 çiftin
        6'sı tek bir log_r'de). Stratified mod log_r dağılımını uniform yapar:
        her olası ölçek çiftinden bağımsız eşleşmeler toplanır.
        """
        N = ema_features.shape[0]
        device = ema_features.device

        unique_scales = torch.unique(log_scale_col)
        all_idx_a = []
        all_idx_b = []

        # Tüm benzersiz (s_a, s_b) çiftleri, s_a < s_b (her çift bir kez)
        for i in range(len(unique_scales)):
            for j in range(i + 1, len(unique_scales)):
                s_a, s_b = unique_scales[i], unique_scales[j]
                # Bu özel ölçek çifti için maske
                is_s_a = log_scale_col == s_a   # [N]
                is_s_b = log_scale_col == s_b   # [N]
                # row in s_a, col in s_b — bu yön için karşılıklı en-yakın
                pair_mask = is_s_a.unsqueeze(1) & is_s_b.unsqueeze(0)   # [N, N]
                pair_mask = pair_mask & base_valid                       # [N, N]

                if not pair_mask.any():
                    continue

                # Bu submask altında karşılıklı en-yakın bul
                masked_sim = sim.masked_fill(~pair_mask, float("-inf"))
                # Her s_a satırı için en yakın s_b sütunu
                best_b_for_a = masked_sim.argmax(dim=1)     # [N]
                best_score_a = masked_sim.gather(1, best_b_for_a.unsqueeze(1)).squeeze(1)
                # Her s_b sütunu için en yakın s_a satırı (transpose ile)
                masked_sim_T = sim.t().masked_fill(~pair_mask.t(), float("-inf"))
                best_a_for_b = masked_sim_T.argmax(dim=1)   # [N]

                # Karşılıklılık: i'nin en yakını j, j'nin en yakını da i
                # AYRICA: benzerlik eşik üstü olmalı
                row_indices = torch.arange(N, device=device)
                mutual = (best_a_for_b[best_b_for_a] == row_indices)
                has_neighbor = best_score_a > float("-inf")
                eligible = mutual & has_neighbor & (best_score_a >= self.similarity_threshold)
                # Yalnız s_a olan satırlardan başla (is_s_a True ve mutual)
                eligible = eligible & is_s_a

                if eligible.any():
                    idx_a_pair = row_indices[eligible]
                    idx_b_pair = best_b_for_a[eligible]
                    all_idx_a.append(idx_a_pair)
                    all_idx_b.append(idx_b_pair)

        if not all_idx_a:
            return None

        idx_a = torch.cat(all_idx_a)
        idx_b = torch.cat(all_idx_b)
        log_r = (log_scales[idx_b] - log_scales[idx_a]).squeeze(1)

        return idx_a, idx_b, log_r
