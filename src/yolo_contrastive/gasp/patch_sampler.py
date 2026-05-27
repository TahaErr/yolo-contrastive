"""MultiScalePatchSampler — bir görüntüden çoklu ölçekte konumsuz yamalar.

GASP §2.2 (konumsuz yamalar) + §4 Karar 3 (çoklu ölçek pencereleri,
kasıtlı ölçek-farklılığı) kod karşılığı.

Hedef: bir görüntüden iki grup yama üret (bir grup küçük, bir grup büyük) —
ki doğal eşleşme aramasında farklı ölçeklerde aynı-nesne adayları olsun.
Yamalar konumsuz dönülür: modele yalnızca yama tensoru + log-ratio
verilir, görüntüdeki (x, y) konumu DÖNÜLMEZ. Konum bilgisi y-koordinatı
kestirme tehlikesini açabilir (plan §3 Korku 1; literatür bölünmüş).

Tasarım:
    - Per-image gruplama: yamalar AYNI görüntüden — perspektif-derinlik
      tezinin gereği. Batch genelinde karıştırılmaz.
    - Deterministik dağılım: ``patches_per_scale`` her ölçek için sabit.
      Rastgele-per-yama ölçek seçimi yerine "tam N küçük + tam N büyük"
      → eşleşme adayları sayıca öngörülebilir, smoke-test ölçümü kolay.
    - Rastgele uniform konum: dış proposal/saliency YOK (Karar 3 elemesi,
      dış-bileşen riski). Boş yamalar (gökyüzü, düz yol) eşleştirme
      aşamasında doğal olarak elenir — önden filtreleme yok.
    - Yama boyutu (patch_size) ile yama kırpma alanı (scale × img_size)
      AYRI: küçük yama da büyük yama da patch_size'a resize edilir;
      "ölçek bilgisi" yamanın boyutunda değil, log_scale tensöründe durur.

API:
    sampler = MultiScalePatchSampler(scales=(0.2, 0.5), patches_per_scale=8)
    patches, log_scales, image_ids = sampler(images)
    # images: [B, 3, H, W]
    # patches: [B*N, 3, P, P]  (N = len(scales) * patches_per_scale)
    # log_scales: [B*N, 1]      (her yamanın log-ölçeği)
    # image_ids: [B*N]          (her yamanın hangi görüntüden — eşleştirme için)

Not — konum debug için opsiyonel:
    return_positions=True → yama merkez (x, y) konumları da döner.
    Bu YALNIZCA smoke-test/teşhis için (Korku 1 testleri); eğitimde False.
"""

from __future__ import annotations

from typing import Tuple, Union, Optional

import math
import torch
import torch.nn.functional as F


class MultiScalePatchSampler:
    """Bir görüntü batch'inden çoklu ölçekte rastgele konumlu, konumsuz yamalar.

    Args:
        scales: yama-alanının orijinal görüntüye oranları (tuple of floats,
            her biri 0 < s ≤ 1). Default (0.2, 0.5) — küçük + büyük.
        patches_per_scale: her görüntüden, her ölçek için kaç yama. Default 8.
            Toplam N = len(scales) * patches_per_scale yama/görüntü.
        patch_size: çıktı yamaların boyutu (piksel). Default 64.
            Tüm yamalar bu boyuta resize edilir; ölçek bilgisi log_scales'te.
        seed: rastgele konum üretici için opsiyonel seed (deterministik test
            için). None ise global torch RNG.
    """

    def __init__(
        self,
        scales: Tuple[float, ...] = (0.2, 0.5),
        patches_per_scale: int = 8,
        patch_size: int = 64,
        seed: Optional[int] = None,
    ):
        if not scales:
            raise ValueError("scales boş olamaz")
        for s in scales:
            if not (0.0 < s <= 1.0):
                raise ValueError(f"her scale 0 < s ≤ 1 olmalı, alındı: {s}")
        if patches_per_scale <= 0:
            raise ValueError(f"patches_per_scale > 0 olmalı, alındı: {patches_per_scale}")
        if patch_size <= 0:
            raise ValueError(f"patch_size > 0 olmalı, alındı: {patch_size}")
        self.scales = tuple(scales)
        self.patches_per_scale = patches_per_scale
        self.patch_size = patch_size
        self.seed = seed
        self._rng = None
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)

    @property
    def patches_per_image(self) -> int:
        return len(self.scales) * self.patches_per_scale

    def __call__(
        self,
        images: torch.Tensor,
        return_positions: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Görüntü batch'inden yamalar üret.

        Args:
            images: [B, 3, H, W] görüntü tensoru.
            return_positions: True ise yama merkez (x, y) konumları da
                döner — YALNIZCA smoke-test/teşhis için (default False;
                eğitimde konumsuz).

        Returns:
            patches: [B*N, 3, P, P]
            log_scales: [B*N, 1]
            image_ids: [B*N] — her yamanın hangi görüntüden
            (return_positions=True ise) positions: [B*N, 2] — (cx, cy)
                yama merkezleri, orijinal görüntü koordinatlarında.
        """
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                f"images [B, 3, H, W] olmalı, alındı: {tuple(images.shape)}"
            )
        B, _, H, W = images.shape
        N = self.patches_per_image
        device = images.device

        patches_list = []
        log_scales_list = []
        positions_list = []

        for b in range(B):
            img = images[b]   # [3, H, W]
            for s in self.scales:
                # Yama alanı boyutu (orijinal piksel cinsi)
                crop_h = int(round(s * H))
                crop_w = int(round(s * W))
                # En az 1×1 olsun
                crop_h = max(1, crop_h)
                crop_w = max(1, crop_w)
                # Maksimum başlangıç konumları (içine sığsın)
                max_y = max(1, H - crop_h + 1)
                max_x = max(1, W - crop_w + 1)
                # patches_per_scale tane rastgele başlangıç
                if self._rng is not None:
                    ys = torch.randint(0, max_y, (self.patches_per_scale,),
                                       generator=self._rng)
                    xs = torch.randint(0, max_x, (self.patches_per_scale,),
                                       generator=self._rng)
                else:
                    ys = torch.randint(0, max_y, (self.patches_per_scale,))
                    xs = torch.randint(0, max_x, (self.patches_per_scale,))
                for y, x in zip(ys.tolist(), xs.tolist()):
                    crop = img[:, y:y+crop_h, x:x+crop_w]   # [3, crop_h, crop_w]
                    # patch_size'a resize
                    crop = crop.unsqueeze(0)   # [1, 3, ch, cw]
                    crop = F.interpolate(
                        crop, size=(self.patch_size, self.patch_size),
                        mode="bilinear", align_corners=False,
                    )
                    patches_list.append(crop.squeeze(0))   # [3, P, P]
                    log_scales_list.append(math.log(s))
                    if return_positions:
                        cx = x + crop_w / 2.0
                        cy = y + crop_h / 2.0
                        positions_list.append((cx, cy))

        patches = torch.stack(patches_list, dim=0)            # [B*N, 3, P, P]
        log_scales = torch.tensor(
            log_scales_list, device=device, dtype=patches.dtype
        ).unsqueeze(1)                                          # [B*N, 1]
        image_ids = torch.arange(B, device=device).repeat_interleave(N)
        # ↑ [0,0,...,0, 1,1,...,1, ..., B-1,...] her id N kez ardışık

        if return_positions:
            positions = torch.tensor(positions_list, device=device,
                                     dtype=patches.dtype)
            return patches, log_scales, image_ids, positions
        return patches, log_scales, image_ids
