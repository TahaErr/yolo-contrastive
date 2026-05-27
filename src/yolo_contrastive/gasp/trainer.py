"""GASPTrainer — Geometry-Aware Scale Pretraining (GASP §2-§4 birleşimi).

Bütün GASP bileşenlerini birleştiren ana sınıf:
    - Online encoder (yolov8n veya başka)
    - EMA encoder (matching için, MoCo-v3 paterni)
    - ScaleEquivariantTransform (öğrenilen T)
    - MultiScalePatchSampler (konumsuz, çoklu ölçekli yamalar)
    - NaturalPairMatcher (Mod A, karşılıklı en-yakın-komşu)
    - controlled_loss + natural_loss (eşzamanlı, α sabit)

MoCo-v3 paterninden ödünç: __init__, _ema_update, train(epochs, resume,
scheduler, save_every), _save, cleanup. GASP'a özel: _step iki kayıp
hesaplar; sampler önce yamalar üretir, encoder + EMA encoder iki kez
çağrılır, T tutarlılığı her iki kayıpta sınanır.

Plan §3 Korku 3: α sabit (default 1.0). Adaptif α (GradNorm) ileride —
smoke-test gradyan-norm ölçümünden sonra.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .transform import ScaleEquivariantTransform
from .patch_sampler import MultiScalePatchSampler
from .natural_pair import NaturalPairMatcher
from .losses import controlled_loss, natural_loss, feature_regularization_loss, controlled_loss_F, natural_loss_F
from ..dense.multi_scale_tap import MultiScaleFeatureTap


class GASPTrainer:
    """GASP — geometry-aware scale pretraining (MoCo-v3 paterninde).

    Args:
        model: YOLO modeli ya da uyumlu mimari (P5 feature çıkarılır).
            "yolov8n.pt" gibi string ya da hazır nn.Module.
        feat_dim: encoder çıktı feature boyutu. Default 256 (yolov8n P5).
        scales: yama ölçekleri (sampler için). Default (0.2, 0.5).
        patches_per_scale: her görüntüden, her ölçek için kaç yama.
        patch_size: encoder'a verilecek yama boyutu. Default 64.
        target_patch_size: kontrollü kayıp augment hedefi. Default 64.
        alpha: L_kontrollü ağırlığı (L_doğal sabit 1.0). Default 1.0.
        momentum: EMA encoder momentum. Default 0.99.
        similarity_threshold: NaturalPairMatcher τ. Default 0.7.
        T_hidden_dim: ScaleEquivariantTransform hidden boyutu. Default 32.
        imgsz: input görüntü boyutu (sampler için). Default 320.
        device: "cuda" / "cpu".
    """

    def __init__(
        self,
        model,
        feat_dim: int = 256,
        feat_level: str = "P5",
        scales: Tuple[float, ...] = (0.2, 0.5),
        patches_per_scale: int = 8,
        patch_size: int = 64,
        target_patch_size: int = 64,
        alpha: float = 1.0,
        loss_variant: str = "F",
        temperature: float = 0.2,
        lambda_var: float = 0.0,
        lambda_cov: float = 0.0,
        variance_target: float = 1.0,
        momentum: float = 0.99,
        similarity_threshold: float = 0.7,
        T_hidden_dim: int = 32,
        imgsz: int = 320,
        device: str = "cuda",
    ):
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum ∈ [0, 1), got {momentum}")
        if alpha < 0:
            raise ValueError(f"alpha >= 0, got {alpha}")

        self.device = device
        self.feat_dim = feat_dim
        self.alpha = alpha
        if loss_variant not in ("F", "vanilla"):
            raise ValueError(f"loss_variant 'F' veya 'vanilla' olmalı, alındı: {loss_variant}")
        self.loss_variant = loss_variant
        self.temperature = temperature
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.variance_target = variance_target
        self.momentum = momentum
        self.imgsz = imgsz
        self.feat_level = feat_level
        self.patch_size = patch_size
        self.target_patch_size = target_patch_size
        self.scales = tuple(scales)

        # Encoder — yolov8 string ya da nn.Module
        if isinstance(model, str):
            from ultralytics import YOLO
            self.model = YOLO(model).model.to(device)
        else:
            self.model = model.to(device)
        self.model.train()
        # Ultralytics YOLO().model varsayilan olarak requires_grad=False ile gelir
        # (inference modu). Egitim icin parametreleri trainable yapmak ZORUNDAYIZ;
        # aksi halde optimizer.step() no-op olur (30K kosusunda 5 saatlik bug bu).
        for p in self.model.parameters():
            p.requires_grad = True

        # EMA encoder — deepcopy, eval mode, requires_grad=False
        self.ema_model = copy.deepcopy(self.model).to(device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

        # Feature tap'lar — MoCo-v3 paterni: forward çıktısı yerine hook ile
        # belirli FPN seviyesindeki feature'ı al. Ultralytics dict yapısına
        # bağımlı değiliz; tap level adıyla erişir.
        self.online_tap = MultiScaleFeatureTap(self.model, levels=(feat_level,))
        self.online_tap.setup()
        self.ema_tap = MultiScaleFeatureTap(self.ema_model, levels=(feat_level,))
        self.ema_tap.setup()

        # GASP bileşenleri
        self.transform = ScaleEquivariantTransform(
            feat_dim=feat_dim, hidden_dim=T_hidden_dim,
        ).to(device)
        self.sampler = MultiScalePatchSampler(
            scales=scales,
            patches_per_scale=patches_per_scale,
            patch_size=patch_size,
        )
        # F variant için scale-pair-stratified matching: log_r dağılımını
        # uniform yapar; karşılıklı en-yakın'ın "kolay çift" eğilimini düzeltir.
        # Vanilla variant: eski davranış (stratify=False).
        self.matcher = NaturalPairMatcher(
            similarity_threshold=similarity_threshold,
            stratify_by_scale_pair=(loss_variant == "F"),
        )

        # Sampler'dan gelen scales — her batch'te rastgele iki ölçek seçilir
        # (controlled_loss_F için). Tek scale verilirse self-self.
        if len(self.scales) < 2:
            self._scale_a = self.scales[0]
            self._scale_b = self.scales[0]
        # Çoklu scale durumunda dinamik seçim _step içinde yapılır.

        self.loss_history: list = []

    def _encode(self, patches: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        """Yama batch'ini feature vektörlerine çevir (MoCo-v3 paterni).

        Backbone forward çıktısı YOK SAYILIR (ultralytics dict döner: boxes/
        scores/feats); FPN feature MultiScaleFeatureTap hook'larından alınır.
        feat_level (default P5) ile seçilen seviye → adaptive_avg_pool2d →
        [N, D] vektörü.

        Args:
            patches: [N, 3, P, P]
            use_ema: True ise EMA encoder, no_grad.

        Returns:
            [N, D] feature vektörleri.
        """
        if use_ema:
            self.ema_tap.clear()
            with torch.no_grad():
                _ = self.ema_model(patches)
                feat = self.ema_tap.get_features()[self.feat_level]
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            return pooled.detach()
        else:
            self.online_tap.clear()
            _ = self.model(patches)
            feat = self.online_tap.get_features()[self.feat_level]
            return F.adaptive_avg_pool2d(feat, 1).flatten(1)

    @torch.no_grad()
    def _ema_update(self) -> None:
        """EMA encoder güncellemesi (MoCo-v3 paterni)."""
        for p, p_ema in zip(self.model.parameters(), self.ema_model.parameters()):
            p_ema.data.mul_(self.momentum).add_(p.data, alpha=1 - self.momentum)

    def _step(self, imgs: torch.Tensor) -> Dict:
        """Tek batch — iki kayıp hesapla."""
        imgs = imgs.to(self.device)

        # Sampler — yamalar, log_scales, image_ids
        patches, log_scales, image_ids = self.sampler(imgs)
        patches = patches.to(self.device)
        log_scales = log_scales.to(self.device)
        image_ids = image_ids.to(self.device)

        # Dinamik scale seçimi: her batch'te self.scales'tan rastgele iki tane
        # (controlled_loss_F için çeşitlilik kaynağı; tek-çift durumunda fallback)
        import random, math
        if len(self.scales) >= 2:
            scale_a, scale_b = random.sample(list(self.scales), 2)
        else:
            scale_a = scale_b = self.scales[0]

        # L_kontrollü_F için sahte log_r adayları: self.scales içindeki
        # tüm olası çiftlerin log_r değerleri (çeşitli, "gerçek olabilirdi" değerler)
        candidate_log_ratios_ctrl = None
        if self.loss_variant == "F" and len(self.scales) >= 3:
            all_pairs = [(sa, sb) for sa in self.scales for sb in self.scales if sa != sb]
            candidate_log_ratios_ctrl = torch.tensor(
                [math.log(sb / sa) for sa, sb in all_pairs],
                device=self.device,
            )

        if self.loss_variant == "F":
            ctrl_out = controlled_loss_F(
                patches=patches,
                encoder=lambda x: self._encode(x, use_ema=False),
                transform=self.transform,
                scale_a=scale_a,
                scale_b=scale_b,
                target_patch_size=self.target_patch_size,
                candidate_log_ratios=candidate_log_ratios_ctrl,
                temperature=self.temperature,
            )
        else:
            ctrl_out = controlled_loss(
                patches=patches,
                encoder=lambda x: self._encode(x, use_ema=False),
                transform=self.transform,
                scale_a=scale_a,
                scale_b=scale_b,
                target_patch_size=self.target_patch_size,
            )

        # L_doğal — F varyantı: InfoNCE on log_r; vanilla: symmetric L2
        online_feats = self._encode(patches, use_ema=False)
        ema_feats = self._encode(patches, use_ema=True)
        if self.loss_variant == "F":
            # Aday log_r set: tüm olası scale çiftleri (ctrl ile aynı set).
            # Hem +log_r hem -log_r olası yönlerini içersin (symmetric).
            nat_candidates = candidate_log_ratios_ctrl
            if nat_candidates is None and len(self.scales) >= 2:
                all_pairs = [(sa, sb) for sa in self.scales for sb in self.scales if sa != sb]
                nat_candidates = torch.tensor(
                    [math.log(sb / sa) for sa, sb in all_pairs],
                    device=self.device,
                )
            nat_out = natural_loss_F(
                online_features=online_feats,
                ema_features=ema_feats,
                log_scales=log_scales,
                image_ids=image_ids,
                matcher=self.matcher,
                transform=self.transform,
                candidate_log_ratios=nat_candidates,
                temperature=self.temperature,
            )
        else:
            nat_out = natural_loss(
                online_features=online_feats,
                ema_features=ema_feats,
                log_scales=log_scales,
                image_ids=image_ids,
                matcher=self.matcher,
                transform=self.transform,
            )

        # VICReg-style feature düzenleyici — kollaps engeli.
        # GASP'ın "bilgi koruma" şartı; T tabanlı tutarlılıkla birlikte
        # eşdeğişirliğin varlık şartını oluşturur.
        reg_out = feature_regularization_loss(
            online_feats,
            variance_target=self.variance_target,
        )
        L_var = reg_out["variance"]
        L_cov = reg_out["covariance"]

        total = (
            self.alpha * ctrl_out["loss"]
            + nat_out["loss"]
            + self.lambda_var * L_var
            + self.lambda_cov * L_cov
        )
        return {
            "loss": total,
            "L_ctrl": float(ctrl_out["loss"].detach()),
            "L_nat": float(nat_out["loss"].detach()) if nat_out["n_pairs"] > 0 else 0.0,
            "L_var": float(L_var.detach()),
            "L_cov": float(L_cov.detach()),
            "n_pairs": nat_out["n_pairs"],
        }

    def train(
        self,
        images_dir: str,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        num_workers: int = 4,
        output: str = "/content/gasp_yolov8n.pt",
        save_every: int = 5,
        print_every: int = 1,
        resume_from: Optional[str] = None,
        gradient_clip: Optional[float] = 1.0,
    ) -> str:
        """Pretraining ana döngü (MoCo-v3 paterninde)."""
        from ..pretrain.dataset import UnlabeledImageDataset
        ds = UnlabeledImageDataset(images_dir, imgsz=self.imgsz)
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True, pin_memory=True,
        )

        params = (
            list(self.model.parameters())
            + list(self.transform.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        steps_per_epoch = len(dl)
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * warmup_epochs

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            import math
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        start_epoch = 1
        if resume_from is not None and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.ema_model.load_state_dict(ckpt["ema_model"])
            self.transform.load_state_dict(ckpt["transform"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            self.loss_history = ckpt.get("loss_history", [])
            print(f"  ↳ resume edildi: ep{start_epoch-1}")

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            ep_loss = 0.0; ep_ctrl = 0.0; ep_nat = 0.0
            ep_var = 0.0; ep_cov = 0.0; ep_pairs = 0; nb = 0
            for imgs in dl:
                optimizer.zero_grad(set_to_none=True)
                out = self._step(imgs)
                out["loss"].backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, gradient_clip)
                optimizer.step()
                scheduler.step()
                self._ema_update()
                ep_loss += float(out["loss"].detach())
                ep_ctrl += out["L_ctrl"]
                ep_nat += out["L_nat"]
                ep_var += out["L_var"]
                ep_cov += out["L_cov"]
                ep_pairs += out["n_pairs"]
                nb += 1
            avg_loss = ep_loss / nb
            avg_ctrl = ep_ctrl / nb
            avg_nat = ep_nat / nb
            avg_var = ep_var / nb
            avg_cov = ep_cov / nb
            avg_pairs = ep_pairs / nb
            self.loss_history.append({
                "epoch": epoch, "loss": avg_loss,
                "L_ctrl": avg_ctrl, "L_nat": avg_nat,
                "L_var": avg_var, "L_cov": avg_cov,
                "n_pairs_avg": avg_pairs,
                "T_identity_dist": float(self.transform.identity_distance(
                    torch.tensor([[0.5]], device=self.device)
                ).item()),
            })
            if epoch % print_every == 0:
                print(f"[gasp] epoch {epoch}/{epochs} loss={avg_loss:.4f} "
                      f"L_ctrl={avg_ctrl:.4f} L_nat={avg_nat:.4f} "
                      f"L_var={avg_var:.4f} L_cov={avg_cov:.4f} "
                      f"pairs/batch={avg_pairs:.1f} ({time.time()-t0:.1f}s)")
            if epoch % save_every == 0 or epoch == epochs:
                self._save(output, epoch, optimizer, scheduler)

        return output

    def _save(self, output: str, epoch: int, optimizer, scheduler) -> None:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "transform": self.transform.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "loss_history": self.loss_history,
        }, output.replace(".pt", ".resume.pt"))
        torch.save({"model": self.model.state_dict()},
                   output.replace(".pt", f"_ep{epoch}.pt"))
        with open(output.replace(".pt", "_loss_history.json"), "w") as f:
            json.dump({"loss_history": self.loss_history}, f, indent=2)

    def cleanup(self) -> None:
        """Belleği temizle (MoCo-v3 paterni)."""
        if hasattr(self, "online_tap"):
            self.online_tap.close()
        if hasattr(self, "ema_tap"):
            self.ema_tap.close()
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "ema_model"):
            del self.ema_model
        if hasattr(self, "transform"):
            del self.transform
        torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (f"GASPTrainer(feat_dim={self.feat_dim}, scales={self.scales}, "
                f"α={self.alpha}, momentum={self.momentum})")
