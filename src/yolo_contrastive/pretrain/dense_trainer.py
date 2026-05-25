"""DenseSSLPretrainer — backbone pretraining with dense + multi-scale CL.

Faz 1.6b — Foundation closing module (WORK_PLAN_v3 §5).

Standalone trainer (NOT an Ultralytics subclass) that orchestrates the
Foundation building blocks:
    MultiScaleFeatureTap  — pulls P3/P4/P5 from YOLO backbone
    MomentumEncoder       — EMA copy with its own tap
    FeatureQueue (×3)     — per-level memory bank
    SpatialTwoViewAugmentation — coord-tracked two-view aug
    MultiScaleProjectionHead   — per-level projection
    multi_scale_dense_loss     — weighted-sum dense NT-Xent

Why standalone (not subclass of DetectionTrainer):
    Ultralytics' BaseTrainer assumes a detection task: dataloader emits
    labels, a detection loss is computed, NMS/eval at val time. SSL has
    none of these. Subclassing means fighting the framework on every
    step. A minimal standalone loop (~250 lines) is cleaner and lets us
    use the same logging/optimizer/scheduler patterns proven in the
    existing SSLPretrainer (which we copied / adapted from).

API parity:
    DenseSSLPretrainer.train() takes the same arguments as the legacy
    SSLPretrainer.train() so they can be swapped drop-in for
    head-to-head comparison.
"""

from __future__ import annotations

import math
import os
import time
import warnings as _warnings
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import UnlabeledImageDataset
from .backbone_utils import save_backbone

from ..dense import (
    MultiScaleFeatureTap,
    FeatureQueue,
    MomentumEncoder,
    SpatialTwoViewAugmentation,
    MultiScaleProjectionHead,
    multi_scale_dense_loss,
    infer_in_channels,
    combine_queues,
    saps_within_loss,
    saps_cross_loss,
)


class DenseSSLPretrainer:
    """Self-supervised pretrainer using dense + multi-scale contrastive learning.

    Args:
        model: Ultralytics model spec (e.g. ``"yolov8n.pt"``) or a pre-built
            ``DetectionModel``. If a string is given, ultralytics is imported
            and YOLO(model).model is extracted.
        out_dim: projection embedding dimension (D in the queue).
        queue_size: K, max negatives held per FPN level.
        momentum: m for EMA update, typically 0.99–0.999.
        temperature: NT-Xent temperature τ.
        n_query: positions sampled per image per level for the loss.
        pos_radius: positive-match coord threshold (normalized image coords).
        match_mode: ``"threshold"`` (multi-positive PixPro-style) or
            ``"nearest"`` (single positive DenseCL-style).
        weights: optional per-level loss weights, e.g. ``{"P3": 1, "P4": 1, "P5": 1}``.
            Auto-normalized. Default: equal.
        aug_kwargs: forwarded to ``SpatialTwoViewAugmentation`` (out_size auto-set
            from ``imgsz`` if missing).
        imgsz: input image size (square).
        device: ``"cuda"``, ``"cpu"``, or specific device. Auto-detected if None.
        logger: optional ``BaseLogger`` (e.g. ``MultiLogger``). If None, prints only.
    """

    def __init__(
        self,
        model: Any = "yolov8n.pt",
        out_dim: int = 256,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.2,
        n_query: int = 256,
        pos_radius: float = 0.07,
        match_mode: str = "threshold",
        weights: Optional[Dict[str, float]] = None,
        aug_kwargs: Optional[Dict[str, Any]] = None,
        imgsz: int = 640,
        device: Optional[str] = None,
        logger: Any = None,
        # ── SAPS (Faz 2.3) ──
        saps_mode: str = "none",
        saps_t_scale: float = 1.0,
        saps_strict_negatives: bool = False,
        saps_both_lambda: float = 1.0,
        # ── Queue update strategy (Risk 7) ──
        queue_update_strategy: str = "pooled",
        queue_subsample_n: int = 16,
    ) -> None:
        # ── basic validation ─────────────────────────────────────────────
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if queue_size <= 0:
            raise ValueError(f"queue_size must be positive, got {queue_size}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.out_dim = int(out_dim)
        self.queue_size = int(queue_size)
        self.momentum_coef = float(momentum)
        self.temperature = float(temperature)
        self.n_query = int(n_query)
        self.pos_radius = float(pos_radius)
        self.match_mode = match_mode
        self.weights = dict(weights) if weights is not None else None
        self.imgsz = int(imgsz)
        self.logger = logger

        # ── SAPS config ─────────────────────────────────────────────────
        if saps_mode not in ("none", "within", "cross", "both"):
            raise ValueError(
                f"saps_mode must be 'none', 'within', 'cross', or 'both', "
                f"got {saps_mode!r}"
            )
        if saps_t_scale <= 0:
            raise ValueError(f"saps_t_scale must be positive, got {saps_t_scale}")
        if saps_both_lambda < 0:
            raise ValueError(
                f"saps_both_lambda must be >= 0, got {saps_both_lambda}"
            )
        self.saps_mode = saps_mode
        self.saps_t_scale = float(saps_t_scale)
        self.saps_strict_negatives = bool(saps_strict_negatives)
        self.saps_both_lambda = float(saps_both_lambda)
        self._needs_tagged_queues = saps_mode in ("cross", "both")

        # ── Queue update strategy (Risk 7) ──
        if queue_update_strategy not in ("pooled", "per_position", "subsample"):
            raise ValueError(
                f"queue_update_strategy must be one of "
                f"{{'pooled', 'per_position', 'subsample'}}, "
                f"got {queue_update_strategy!r}"
            )
        if queue_subsample_n <= 0:
            raise ValueError(
                f"queue_subsample_n must be positive, got {queue_subsample_n}"
            )
        self.queue_update_strategy = queue_update_strategy
        self.queue_subsample_n = int(queue_subsample_n)

        # ── device ──────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, (int, float)) or (isinstance(device, str) and device.isdigit()):
            self.device = torch.device(f"cuda:{int(device)}")
        else:
            self.device = torch.device(device)

        # ── model ───────────────────────────────────────────────────────
        if isinstance(model, str):
            from ultralytics import YOLO
            yolo = YOLO(model)
            self.model = yolo.model.to(self.device)
        else:
            self.model = model.to(self.device)
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        # ── tap on online encoder ───────────────────────────────────────
        self.online_tap = MultiScaleFeatureTap(self.model)
        self.online_tap.setup()

        # ── probe channel widths ────────────────────────────────────────
        in_channels = infer_in_channels(
            self.model, self.online_tap,
            imgsz=min(self.imgsz, 64), device=self.device,
        )
        # tap captured features during probe — clear so first real forward starts fresh
        self.online_tap.clear()
        self._in_channels = in_channels

        # ── momentum encoder + its own tap ──────────────────────────────
        self.momentum = MomentumEncoder(
            self.model, m=self.momentum_coef, force_fp32=True,
        ).to(self.device)
        self.momentum_tap = MultiScaleFeatureTap(self.momentum.momentum)
        self.momentum_tap.setup()

        # ── projection heads (per-level) — online + EMA copy ────────────
        # MoCo-v3 / BYOL convention: the projection head is also momentum-
        # averaged. We can't reuse MomentumEncoder here because its forward()
        # expects a single tensor input, but our projection head consumes a
        # dict {level: tensor}. So we keep an online head and a manually-
        # EMA'd copy. Updates happen in _ema_update().
        import copy
        self.proj_online = MultiScaleProjectionHead(
            in_channels=in_channels, out_dim=out_dim,
        ).to(self.device)
        self.proj_momentum = copy.deepcopy(self.proj_online).to(self.device)
        for p in self.proj_momentum.parameters():
            p.requires_grad = False
        self.proj_momentum.eval()

        # ── per-level queues ────────────────────────────────────────────
        # When SAPS-cross or SAPS-both is active, we need scale tags on queue
        # entries so combine_queues() can produce a tagged pool for the
        # cross-image scale-aware reweighting.
        self.queues: Dict[str, FeatureQueue] = {
            lv: FeatureQueue(
                dim=out_dim, K=queue_size,
                with_tags=self._needs_tagged_queues,
            ).to(self.device)
            for lv in in_channels
        }
        # Stable level → integer id mapping (P3=0, P4=1, P5=2 in default order).
        # Used by SAPS-cross to broadcast scale-similarity weights.
        self.level_to_id: Dict[str, int] = {
            lv: i for i, lv in enumerate(in_channels.keys())
        }

        # ── augmentation ────────────────────────────────────────────────
        ak = dict(aug_kwargs or {})
        ak.setdefault("out_size", (self.imgsz, self.imgsz))
        self.aug = SpatialTwoViewAugmentation(**ak)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release tap hooks. Idempotent."""
        try:
            self.online_tap.close()
        except Exception:
            pass
        try:
            self.momentum_tap.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    # ── single training step (used by tests + train loop) ────────────────

    def _step(self, imgs: torch.Tensor) -> Dict[str, Any]:
        """One training step. Returns dict with loss tensor + info."""
        imgs = imgs.to(self.device, non_blocking=True)
        B = imgs.shape[0]

        # 1) Two views with coord tracking
        views = self.aug(imgs)
        # views.view{1,2}: [B, 3, H, W];  views.coords{1,2}: [B, 2, H, W]

        # 2) Online forward (view1) — features captured by online_tap
        self.online_tap.clear()
        _ = self.model(views.view1.to(self.device))
        q_raw = self.online_tap.get_features()
        q_proj = self.proj_online(q_raw)
        # L2-normalize (caller responsibility per dense_loss convention)
        q_norm = {lv: F.normalize(t, dim=1) for lv, t in q_proj.items()}

        # 3) Momentum forward (view2) — no_grad, separate tap
        self.momentum_tap.clear()
        with torch.no_grad():
            _ = self.momentum(views.view2.to(self.device))
            k_raw = self.momentum_tap.get_features()
            # Project via momentum projection head
            k_proj = self.proj_momentum(k_raw)
            k_norm = {lv: F.normalize(t, dim=1).detach() for lv, t in k_proj.items()}

        # 4) Per-level queue tensors (snapshot before this step's enqueue)
        queue_tensors: Dict[str, Optional[torch.Tensor]] = {}
        for lv, q in self.queues.items():
            t = q.get()
            queue_tensors[lv] = t if t.shape[0] > 0 else None

        # 5) Loss — branches on saps_mode
        coords1 = views.coords1.to(self.device)
        coords2 = views.coords2.to(self.device)

        common_kwargs = dict(
            q_features=q_norm,
            k_features=k_norm,
            q_coords=coords1,
            k_coords=coords2,
            weights=self.weights,
            temperature=self.temperature,
            n_query=self.n_query,
            pos_radius=self.pos_radius,
            match_mode=self.match_mode,
            return_info=True,
        )

        if self.saps_mode == "none":
            loss, info = multi_scale_dense_loss(
                queues=queue_tensors, **common_kwargs,
            )

        elif self.saps_mode == "within":
            loss, info = saps_within_loss(
                queues=queue_tensors,
                strict_negatives=self.saps_strict_negatives,
                **common_kwargs,
            )

        elif self.saps_mode == "cross":
            # Build a tagged combined queue from per-level tagged queues
            keys, tags = combine_queues(self.queues, level_to_id=self.level_to_id)
            loss, info = saps_cross_loss(
                queue_keys=keys,
                queue_tags=tags,
                level_to_id=self.level_to_id,
                t_scale=self.saps_t_scale,
                **common_kwargs,
            )

        else:  # "both"
            # Weighted sum: loss = loss_within + λ · loss_cross.
            # Default λ=1.0 preserves the previous (additive) behavior. λ=0
            # collapses to within-only (numerically equivalent to
            # saps_mode="within"), useful as an ablation control point.
            # See WORK_PLAN §7 + Risk 9.
            loss_w, info_w = saps_within_loss(
                queues=queue_tensors,
                strict_negatives=self.saps_strict_negatives,
                **common_kwargs,
            )
            keys, tags = combine_queues(self.queues, level_to_id=self.level_to_id)
            loss_c, info_c = saps_cross_loss(
                queue_keys=keys,
                queue_tags=tags,
                level_to_id=self.level_to_id,
                t_scale=self.saps_t_scale,
                **common_kwargs,
            )
            loss = loss_w + self.saps_both_lambda * loss_c
            info = {
                "within": info_w,
                "cross": info_c,
                "saps_mode": "both",
                "saps_both_lambda": self.saps_both_lambda,
            }

        # 6) Enqueue keys for next step (Risk 7 — strategy-switchable):
        #    - "pooled":       1 vec per (b, level)         → B per level
        #    - "per_position": HW vec per (b, level)         → B*HW per level
        #    - "subsample":    n random pos per (b, level)   → B*n per level
        # When tagged, attach level id so SAPS-cross can reweight by scale.
        with torch.no_grad():
            for lv, k_t in k_norm.items():
                # k_t: [B, D, H, W]
                if self.queue_update_strategy == "pooled":
                    entries = k_t.mean(dim=(2, 3))                   # [B, D]
                elif self.queue_update_strategy == "per_position":
                    B_, D_, H_, W_ = k_t.shape
                    entries = (
                        k_t.flatten(2)                                # [B, D, HW]
                           .permute(0, 2, 1)                          # [B, HW, D]
                           .reshape(-1, D_)                           # [B*HW, D]
                    )
                else:  # "subsample"
                    B_, D_, H_, W_ = k_t.shape
                    HW = H_ * W_
                    n = min(self.queue_subsample_n, HW)
                    idx = torch.randperm(HW, device=k_t.device)[:n]
                    flat = k_t.flatten(2)                             # [B, D, HW]
                    entries = (
                        flat[..., idx]                                # [B, D, n]
                            .permute(0, 2, 1)                         # [B, n, D]
                            .reshape(-1, D_)                          # [B*n, D]
                    )

                entries = F.normalize(entries, dim=1)
                if self._needs_tagged_queues:
                    tag_value = self.level_to_id[lv]
                    tags_b = torch.full(
                        (entries.shape[0],), tag_value,
                        dtype=torch.long, device=entries.device,
                    )
                    self.queues[lv].enqueue(entries, tags=tags_b)
                else:
                    self.queues[lv].enqueue(entries)

        return {"loss": loss, "info": info, "batch_size": B}

    @torch.no_grad()
    def _ema_update(self) -> None:
        """Update momentum encoder + projection head EMA."""
        # Encoder uses MomentumEncoder helper
        self.momentum.update(self.model)
        # Projection head: manual EMA (it's a plain nn.Module copy, not
        # wrapped in MomentumEncoder because that helper's forward() doesn't
        # handle dict inputs).
        m = self.momentum_coef
        for p_o, p_m in zip(
            self.proj_online.parameters(), self.proj_momentum.parameters()
        ):
            p_m.data.mul_(m).add_(p_o.data, alpha=1.0 - m)
        for b_o, b_m in zip(
            self.proj_online.buffers(), self.proj_momentum.buffers()
        ):
            if b_m.dtype.is_floating_point:
                b_m.data.mul_(m).add_(b_o.data.to(b_m.dtype), alpha=1.0 - m)
            else:
                b_m.data.copy_(b_o.data)

    # ── public training API ──────────────────────────────────────────────

    def train(
        self,
        images_dir: str,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        num_workers: int = 4,
        output: str = "dense_backbone.pt",
        save_every: int = 25,
        print_every: int = 10,
        resume_from: Optional[str] = None,
    ) -> str:
        """Run dense SSL pretraining.

        Args:
            resume_from: Path to a ``.resume.pt`` state file (written every
                ``save_every`` epochs). If given and the file exists, training
                resumes from the next epoch — model, optimizer, queues, EMA
                and loss_history are all restored. Resume is at epoch
                granularity (dataloader reshuffles); fine for SSL pretraining.

        Returns:
            Path to the saved backbone checkpoint.
        """
        # ── data ────────────────────────────────────────────────────────
        dataset = UnlabeledImageDataset(images_dir, imgsz=self.imgsz)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        steps_per_epoch = len(dataloader)
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        # ── optimizer ───────────────────────────────────────────────────
        param_groups = [
            {"params": self.model.parameters(), "lr": lr},
            {"params": self.proj_online.parameters(), "lr": lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        # ── scheduler: warmup + cosine ──────────────────────────────────
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ── AMP ─────────────────────────────────────────────────────────
        use_amp = self.device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # ── log header ──────────────────────────────────────────────────
        saps_extras = []
        if self.saps_mode != "none":
            saps_extras.append(f"t_scale={self.saps_t_scale}")
            saps_extras.append(f"strict_neg={self.saps_strict_negatives}")
            if self.saps_mode == "both":
                saps_extras.append(f"both_λ={self.saps_both_lambda}")
        saps_line = (
            f"[ycl-dense] saps={self.saps_mode}, " + ", ".join(saps_extras) + "\n"
            if self.saps_mode != "none" else ""
        )
        queue_line = (
            f"[ycl-dense] queue_strategy={self.queue_update_strategy}"
            + (f" (n={self.queue_subsample_n})"
               if self.queue_update_strategy == "subsample" else "")
            + "\n"
            if self.queue_update_strategy != "pooled" else ""
        )
        self._print(
            f"[ycl-dense] === Dense SSL Pretraining Start ===\n"
            f"[ycl-dense] epochs={epochs}, batch={batch_size}, lr={lr}\n"
            f"[ycl-dense] D={self.out_dim}, K={self.queue_size}, m={self.momentum_coef}\n"
            f"[ycl-dense] τ={self.temperature}, n_query={self.n_query}, "
            f"radius={self.pos_radius}, match={self.match_mode}\n"
            + saps_line
            + queue_line +
            f"[ycl-dense] dataset: {len(dataset)} imgs, {steps_per_epoch} batches/epoch\n"
            f"[ycl-dense] output: {output}\n"
        )
        if self.logger is not None:
            try:
                self.logger.log_config({
                    "epochs": epochs, "batch_size": batch_size, "lr": lr,
                    "out_dim": self.out_dim, "queue_size": self.queue_size,
                    "momentum": self.momentum_coef, "temperature": self.temperature,
                    "n_query": self.n_query, "pos_radius": self.pos_radius,
                    "match_mode": self.match_mode,
                    "saps_mode": self.saps_mode,
                    "saps_t_scale": self.saps_t_scale,
                    "saps_strict_negatives": self.saps_strict_negatives,
                    "saps_both_lambda": self.saps_both_lambda,
                    "queue_update_strategy": self.queue_update_strategy,
                    "queue_subsample_n": self.queue_subsample_n,
                })
            except Exception:
                pass

        # ── resume state (optional) ─────────────────────────────────────
        resume_path = output.replace(".pt", ".resume.pt")
        start_epoch = 1
        global_step = 0
        _resumed_loss_history: list = []
        _resumed_best = {"loss": float("inf"), "epoch": 0, "state": None}
        if resume_from is not None and os.path.exists(resume_from):
            rs = torch.load(resume_from, map_location=self.device,
                            weights_only=False)
            self.model.load_state_dict(rs["model"])
            self.proj_online.load_state_dict(rs["proj_online"])
            self.momentum.momentum.load_state_dict(rs["momentum_encoder"])
            self.proj_momentum.load_state_dict(rs["proj_momentum"])
            for lv, q in self.queues.items():
                if lv in rs["queues"]:
                    q.load_state_dict(rs["queues"][lv])
            optimizer.load_state_dict(rs["optimizer"])
            start_epoch = int(rs["epoch"]) + 1
            global_step = int(rs["global_step"])
            _resumed_loss_history = list(rs.get("loss_history", []))
            _resumed_best = rs.get("best", _resumed_best)
            # Fast-forward the scheduler to the resumed step count. Each
            # scheduler.step() here precedes any optimizer.step() in this
            # process, which PyTorch warns about — the warning is benign
            # (LR lands on the correct cosine value, verified) so it is
            # suppressed. last_epoch>=0 at construction was tried instead
            # but requires 'initial_lr' in param_groups — more fragile.
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                for _ in range(global_step):
                    scheduler.step()
            self._print(
                f"[ycl-dense] RESUMED from epoch {rs['epoch']} "
                f"→ continuing at epoch {start_epoch}/{epochs}"
            )

        # ── loop ────────────────────────────────────────────────────────
        best_loss = _resumed_best["loss"]
        # Track the weights of the lowest-loss epoch, not just the last.
        # train() runs a cosine LR to ~0; with some queue strategies the last
        # few epochs drift up in loss (Stage 1: pooled tail-rise). Saving the
        # best-epoch weights makes the checkpoint match the reported best_loss.
        import copy as _copy
        best_state = _resumed_best["state"]
        best_epoch = _resumed_best["epoch"]
        # Per-epoch metric history — persisted into the final checkpoint's
        # `extra` so loss curves survive the run (paper Figure 2, plan §5.1).
        # The console print is gated by print_every; this list is not.
        # On resume, prepend the epochs already completed.
        loss_history: list = list(_resumed_loss_history)
        t0_total = time.time()

        try:
            for epoch in range(start_epoch, epochs + 1):
                t0 = time.time()
                self.model.train()
                self.proj_online.train()

                ep_loss = 0.0
                ep_acc = 0.0
                ep_pos = 0.0
                ep_neg = 0.0
                n_batches = 0

                for imgs in dataloader:
                    optimizer.zero_grad(set_to_none=True)

                    # Forward + loss in autocast (loss internally moves to fp32)
                    with torch.amp.autocast(self.device.type, enabled=use_amp):
                        out = self._step(imgs)
                        loss = out["loss"]

                    # Backward + step
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
                    self._ema_update()

                    # Metrics
                    info = out["info"]
                    loss_v = float(loss.detach().item())
                    ep_loss += loss_v
                    n_batches += 1

                    # Per-level info access depends on saps_mode:
                    # - "none"/"within"/"cross" → info[level] = {acc_top1, ...}
                    # - "both"                  → info["within"][level], info["cross"][level]
                    # Use within's per-level stats as primary signal (it sees
                    # the broadest negative pool); fall back to top-level dict.
                    per_level_info = info.get("within", info)

                    accs, poss, negs = [], [], []
                    for lv in self._in_channels:
                        lv_info = per_level_info.get(lv, {})
                        if lv_info.get("skipped"):
                            continue
                        accs.append(lv_info.get("acc_top1", 0.0))
                        poss.append(lv_info.get("mean_pos_sim", 0.0))
                        negs.append(lv_info.get("mean_neg_sim", 0.0))
                    if accs:
                        ep_acc += sum(accs) / len(accs)
                        ep_pos += sum(poss) / len(poss)
                        ep_neg += sum(negs) / len(negs)

                    if self.logger is not None:
                        try:
                            self.logger.log_scalars({
                                "loss": loss_v,
                                "lr": optimizer.param_groups[0]["lr"],
                                "acc_top1_mean": (sum(accs) / len(accs)) if accs else 0.0,
                                "pos_sim_mean": (sum(poss) / len(poss)) if poss else 0.0,
                                "neg_sim_mean": (sum(negs) / len(negs)) if negs else 0.0,
                                "queue_filled_P3": len(self.queues.get("P3", FeatureQueue(1, 1))),
                            }, step=global_step)
                        except Exception:
                            pass

                    global_step += 1

                avg_loss = ep_loss / max(1, n_batches)
                avg_acc = ep_acc / max(1, n_batches)
                avg_pos = ep_pos / max(1, n_batches)
                avg_neg = ep_neg / max(1, n_batches)
                ep_time = time.time() - t0

                loss_history.append({
                    "epoch": epoch,
                    "loss": avg_loss,
                    "acc_top1": avg_acc,
                    "pos_sim": avg_pos,
                    "neg_sim": avg_neg,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                })

                if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
                    self._print(
                        f"[ycl-dense] epoch {epoch:3d}/{epochs} | "
                        f"loss={avg_loss:.4f} | acc@1={avg_acc:.3f} | "
                        f"pos={avg_pos:.3f} neg={avg_neg:.3f} | "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                        f"{ep_time:.1f}s"
                    )

                if save_every > 0 and epoch % save_every == 0:
                    chkpt = output.replace(".pt", f"_ep{epoch}.pt")
                    save_backbone(self.model, chkpt, epoch=epoch,
                                  extra={"loss": avg_loss, "type": "dense_ssl"})
                    self._print(f"[ycl-dense] checkpoint: {chkpt}")
                    # resume state — full restart point at this epoch boundary
                    torch.save({
                        "model": self.model.state_dict(),
                        "proj_online": self.proj_online.state_dict(),
                        "momentum_encoder": self.momentum.momentum.state_dict(),
                        "proj_momentum": self.proj_momentum.state_dict(),
                        "queues": {lv: q.state_dict()
                                   for lv, q in self.queues.items()},
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss_history": loss_history,
                        "best": {"loss": best_loss, "epoch": best_epoch,
                                 "state": best_state},
                    }, resume_path)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_epoch = epoch
                    best_state = _copy.deepcopy(self.model.state_dict())

            # Final save — restore the best-epoch weights, then save.
            # loss_history stays the full curve (paper Figure 2); only the
            # *weights* are the best epoch's. best_epoch records which one.
            if best_state is not None:
                self.model.load_state_dict(best_state)
            save_backbone(self.model, output, epoch=best_epoch or epochs,
                          extra={"loss": best_loss, "type": "dense_ssl",
                                 "loss_history": loss_history,
                                 "best_epoch": best_epoch or epochs})
            total_time = time.time() - t0_total
            # training completed cleanly — resume state no longer needed
            if os.path.exists(resume_path):
                os.remove(resume_path)
            self._print(
                f"[ycl-dense] === Done in {total_time:.1f}s | "
                f"best_loss={best_loss:.4f} @ epoch {best_epoch or epochs} "
                f"→ {output} ==="
            )

        finally:
            if self.logger is not None:
                try:
                    self.logger.finish()
                except Exception:
                    pass

        return output

    # ── helpers ──────────────────────────────────────────────────────────

    def _print(self, msg: str) -> None:
        try:
            from ultralytics.utils import LOGGER
            LOGGER.info(msg)
        except Exception:
            print(msg)

    def __repr__(self) -> str:
        return (
            f"DenseSSLPretrainer(D={self.out_dim}, K={self.queue_size}, "
            f"m={self.momentum_coef}, τ={self.temperature}, "
            f"levels={list(self._in_channels.keys())}, "
            f"saps={self.saps_mode})"
        )
