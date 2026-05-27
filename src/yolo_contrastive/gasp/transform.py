"""ScaleEquivariantTransform — öğrenilen T(f, ratio) dönüşümü (GASP §2.1).

GASP'ın ölçek-eşdeğişirlik formülasyonu, bir feature vektörü f'in,
ölçek-oranı r altında öngörülebilir, tüm nesneler için TUTARLI bir
T altında dönüşmesini ister: f' ≈ T(f, ratio).

Parametrizasyon — affine (FiLM tarzı):
    T(f, r) = (1 + γ(log r)) ⊙ f + β(log r)

γ, β: log_ratio'dan iki küçük MLP ile üretilen kanal-başına ölçek ve
kayma vektörleri (Feature-wise Linear Modulation, Perez et al.).
Affine biçim seçildi çünkü ölçek değişimi feature uzayında hem kayma
(yeni detaylar farklı yerlere düşer) hem ölçek (bazı kanal-aktivasyonları
güç değiştirir) yaratır; sadece eklemeli kayma yetersiz.

Kritik kısıt — kapasite (GASP §2.1):
    T basit ve az parametreli tutulmalı. Serbest bir T, eşdeğişirliği
    GERÇEK öğrenmek yerine TAKLİT eder (T = identity'ye çökerek L'yi
    küçültür — MGD'nin "generator baypas" tuzağının akrabası).
    Hidden boyutu küçük tutulur (default 32); kanallar arası karıştırma
    YOK — γ, β kanal-başına skaler, vektör değil. Bu, T'nin yapabileceği
    şeyi kanal-başına affine ile sınırlar.

Identity çökmesi sınanır (smoke-test):
    Eğer eğitim sonunda ||γ|| ≈ 0 ve ||β|| ≈ 0 ise T identity öğrenmiş —
    eşdeğişirlik gerçekten öğrenilmemiş. Plan §3 Korku 3 felsefesi
    gereği bu önden mekanizma değil, smoke-test'te ölçülecek bir
    teşhis metriği.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScaleEquivariantTransform(nn.Module):
    """Affine, oran-koşullu, kanal-başına dönüşüm: T(f, r) = (1+γ(log r))⊙f + β(log r).

    Args:
        feat_dim: feature vektörünün boyutu (D).
        hidden_dim: γ, β'yı üreten MLP'nin gizli katman boyutu. Küçük
            tutulur (default 32) — T'nin kapasitesini sınırlamak için.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 32):
        super().__init__()
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {feat_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # log_ratio: [B, 1] → hidden → 2*feat_dim (γ ve β bitişik)
        self.gen = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * feat_dim),
        )
        # Başlangıçta T YAKLAŞIK identity: son katmanın bias'ını sıfırla,
        # weight'ı default (Kaiming-uniform) bırak. Niyet: anlamsız tahmin
        # yerine kimliğe yakın başla. NEDEN tam sıfır değil: son katmanı
        # tamamen sıfırlamak `gen[0]` katmanına gradyan akışını boğar
        # (zincir kuralında ∂γ/∂relu_output = gen[-1].weight = 0 olur);
        # ilk N batch boyunca yalnız son katman öğrenir, T etkin kapasitesi
        # daralır. Bias-yalnız sıfırlamak T'yi başlangıçta f'e çok yakın
        # tutar (γ, β küçük-rastgele) ama bütün ağa gradyan akar.
        nn.init.zeros_(self.gen[-1].bias)

    def forward(self, f: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        """Apply T(f, ratio).

        Args:
            f: [B, D] feature vektörleri.
            log_ratio: [B, 1] log oranlar — log(target_scale / source_scale).
                log kullanılır çünkü ölçek dönüşümü çarpımsal, log uzayında
                eklemeli ve daha pürüzsüz.

        Returns:
            [B, D] dönüştürülmüş feature.
        """
        if f.dim() != 2:
            raise ValueError(f"f must be [B, D], got shape {tuple(f.shape)}")
        if log_ratio.dim() != 2 or log_ratio.shape[1] != 1:
            raise ValueError(
                f"log_ratio must be [B, 1], got shape {tuple(log_ratio.shape)}"
            )
        if f.shape[0] != log_ratio.shape[0]:
            raise ValueError(
                f"batch size mismatch: f {f.shape[0]} vs log_ratio "
                f"{log_ratio.shape[0]}"
            )

        params = self.gen(log_ratio)            # [B, 2D]
        gamma, beta = params.chunk(2, dim=1)    # [B, D], [B, D]
        return (1.0 + gamma) * f + beta

    def identity_distance(self, log_ratio: torch.Tensor) -> torch.Tensor:
        """T identity'den ne kadar uzak — smoke-test teşhis metriği.

        Returns ||gamma||^2 + ||beta||^2 ortalaması. Eğitim sonunda
        ~0 ise T identity'ye çökmüş, eşdeğişirlik gerçekten öğrenilmemiş.

        Args:
            log_ratio: [B, 1] — bu oranlarda T'nin identity'den uzaklığı.

        Returns:
            skaler tensor (ortalama L2 mesafesi^2).
        """
        with torch.no_grad():
            params = self.gen(log_ratio)
            gamma, beta = params.chunk(2, dim=1)
            return (gamma.pow(2).mean() + beta.pow(2).mean())
