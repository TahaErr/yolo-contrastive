"""SSLPretrainer — Backbone pretraining with unlabeled data."""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset import UnlabeledImageDataset
from .backbone_utils import save_backbone


class SSLPretrainer:
    def __init__(
        self,
        model: str = "yolov8n.pt",
        aug_preset: str = "simclr_v2",
        lambda_cl: float = 1.0,
        lambda_rot: float = 0.5,
        temperature: float = 0.2,
        proj_dim: int = 128,
        proj_hidden: int = 256,
        rot_hidden: int = 256,
        imgsz: int = 640,
        device: Optional[str] = None,
    ):
        self.imgsz = imgsz
        self.lambda_cl = lambda_cl
        self.lambda_rot = lambda_rot

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if isinstance(device, (int, float)) or (isinstance(device, str) and device.isdigit()):
                self.device = torch.device(f"cuda:{int(device)}")
            else:
                self.device = torch.device(device)

        from ultralytics import YOLO
        yolo = YOLO(model)
        self.model = yolo.model.to(self.device)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        print(f"[ycl] Model: {model} -> {self.device} ({trainable} trainable params)")

        from ..feature_tap import FeatureTap
        self.feature_tap = FeatureTap(self.model, min_channels=128, store_grad=True)
        self.feature_tap.setup(device=self.device, imgsz=imgsz)
        print(f"[ycl] FeatureTap: {self.feature_tap.layer_name}")

        self.feat_dim = self._detect_feat_dim()
        print(f"[ycl] Feature dim: {self.feat_dim}")

        from ..pretext.heads import ProjectionHead
        self.projection_head = ProjectionHead(
            feat_dim=self.feat_dim,
            out_dim=proj_dim,
            hidden_dim=proj_hidden,
        ).to(self.device)

        from ..contrastive.losses import build_contrastive_loss
        self.cl_loss_fn = build_contrastive_loss("ntxent", temperature=temperature)

        self.rot_task = None
        if lambda_rot > 0:
            from ..pretext.rotation import RotationTask
            self.rot_task = RotationTask(
                feat_dim=self.feat_dim,
                hidden_dim=rot_hidden,
            ).to(self.device)

        from ..augmentations.presets import build_pipeline
        self.augmentation = build_pipeline(aug_preset, imgsz=imgsz)
        print(f"[ycl] Augmentation: {aug_preset} ({len(self.augmentation)} ops)")

    def cleanup(self):
        if hasattr(self, "feature_tap"):
            self.feature_tap.close()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def _detect_feat_dim(self) -> int:
        # FIX: BN running stats'ları koruyarak dummy forward yap
        from ..trainer._helpers import preserve_bn_running_stats
        self.model.eval()
        with torch.no_grad(), preserve_bn_running_stats(self.model):
            dummy = torch.randn(1, 3, self.imgsz, self.imgsz, device=self.device)
            _ = self.model(dummy)
        self.model.train()
        emb = self.feature_tap.get_embedding()
        if emb is None:
            raise RuntimeError("[ycl] FeatureTap returned None")
        return emb.shape[1]

    def _get_embedding(self, imgs: torch.Tensor) -> torch.Tensor:
        _ = self.model(imgs)
        emb = self.feature_tap.get_embedding()
        if emb is None:
            raise RuntimeError("[ycl] FeatureTap returned None during training")
        return emb

    def train(
        self,
        images_dir: str,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        num_workers: int = 4,
        output: str = "pretrained_backbone.pt",
        save_every: int = 25,
        print_every: int = 10,
    ) -> str:
        dataset = UnlabeledImageDataset(images_dir, imgsz=self.imgsz)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        print(f"[ycl] Dataset: {len(dataset)} images, {len(dataloader)} batches/epoch")

        param_groups = [
            {"params": self.model.parameters(), "lr": lr},
            {"params": self.projection_head.parameters(), "lr": lr},
        ]
        if self.rot_task is not None:
            param_groups.append({"params": self.rot_task.parameters(), "lr": lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        total_steps = epochs * len(dataloader)
        warmup_steps = warmup_epochs * len(dataloader)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # FIX: LambdaLR.__init__ dahili step() çağırır → optimizer henüz step atmadığı
        # için PyTorch uyarı verir. Uyarıyı init sırasında bastır, loop sırası doğrudur.
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        use_amp = self.device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        print(f"\n[ycl] === SSL Pretraining Start ===")
        print(f"[ycl] epochs={epochs}, batch={batch_size}, lr={lr}")
        print(f"[ycl] lambda_cl={self.lambda_cl}, lambda_rot={self.lambda_rot}")
        print(f"[ycl] output={output}\n")

        global_step = 0
        best_loss = float("inf")
        t0_total = time.time()

        try:
            for epoch in range(1, epochs + 1):
                t0 = time.time()
                epoch_cl_loss = 0.0
                epoch_rot_loss = 0.0
                epoch_rot_acc = 0.0
                epoch_total = 0.0
                n_batches = 0

                self.model.train()
                self.projection_head.train()
                if self.rot_task:
                    self.rot_task.train()

                for imgs in dataloader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    optimizer.zero_grad()

                    # FIX: Augmentations OUTSIDE autocast and inside no_grad
                    with torch.no_grad():
                        view1 = self.augmentation(imgs)
                        view2 = self.augmentation(imgs)

                    rot_imgs = None
                    rot_labels = None
                    if self.rot_task is not None:
                        with torch.no_grad():
                            rot_imgs, rot_labels = self.rot_task.rotate_batch(imgs)

                    with torch.amp.autocast(self.device.type, enabled=use_amp):
                        z1 = self._get_embedding(view1)
                        p1 = self.projection_head(z1)

                        z2 = self._get_embedding(view2)
                        p2 = self.projection_head(z2)

                        cl_loss = self.cl_loss_fn(p1, p2)

                        rot_loss = torch.tensor(0.0, device=self.device)
                        rot_acc = 0.0
                        if self.rot_task is not None and rot_imgs is not None:
                            rot_features = self._get_embedding(rot_imgs)
                            rot_loss, rot_acc = self.rot_task(rot_features, rot_labels)

                        total = self.lambda_cl * cl_loss + self.lambda_rot * rot_loss

                    scaler.scale(total).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    with _warnings.catch_warnings():
                        _warnings.simplefilter("ignore")
                        scheduler.step()

                    global_step += 1
                    n_batches += 1
                    epoch_cl_loss += cl_loss.item()
                    epoch_rot_loss += rot_loss.item()
                    epoch_rot_acc += rot_acc
                    epoch_total += total.item()

                    if print_every > 0 and global_step % print_every == 0:
                        lr_now = optimizer.param_groups[0]["lr"]
                        print(
                            f"  step={global_step:>5d} "
                            f"cl={cl_loss.item():.3f} "
                            f"rot={rot_loss.item():.3f}(acc={rot_acc:.1%}) "
                            f"total={total.item():.3f} "
                            f"lr={lr_now:.2e}"
                        )

                n = max(1, n_batches)
                avg_cl = epoch_cl_loss / n
                avg_rot = epoch_rot_loss / n
                avg_acc = epoch_rot_acc / n
                avg_total = epoch_total / n
                elapsed = time.time() - t0

                print(
                    f"[ycl] Epoch {epoch:>3d}/{epochs} | "
                    f"cl={avg_cl:.4f} rot={avg_rot:.4f}(acc={avg_acc:.1%}) "
                    f"total={avg_total:.4f} | {elapsed:.1f}s"
                )

                if save_every > 0 and epoch % save_every == 0:
                    ckpt = output.replace(".pt", f"_ep{epoch}.pt")
                    save_backbone(self.model, ckpt, epoch=epoch, extra={
                        "cl_loss": avg_cl, "rot_loss": avg_rot, "rot_acc": avg_acc,
                    })
                    print(f"[ycl] Checkpoint saved: {ckpt}")

                if avg_total < best_loss:
                    best_loss = avg_total

            total_time = time.time() - t0_total
            save_backbone(self.model, output, epoch=epochs, extra={
                "total_time_sec": total_time,
                "best_loss": best_loss,
                "lambda_cl": self.lambda_cl,
                "lambda_rot": self.lambda_rot,
            })
            print(f"\n[ycl] === SSL Pretraining Complete ===")
            print(f"[ycl] {epochs} epochs in {total_time/60:.1f} min")
            print(f"[ycl] Best loss: {best_loss:.4f}")
            print(f"[ycl] Backbone saved: {output}")
            return output

        finally:
            self.feature_tap.close()
