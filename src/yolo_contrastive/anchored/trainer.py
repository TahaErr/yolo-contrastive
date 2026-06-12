"""AnchoredJointTrainer — COCO-anchored joint training with pluggable aux channels.

The shared carrier for TERRA (geoteach), REVISIT (persistence) and GASP-Real
(scalereal): it reproduces the repo's ONLY historical win — COCO detection
loss + auxiliary loss in the SAME optimizer steps (0.6719 vs 0.6593) — and
bakes every measured failure's countermeasure into the loop:

    R3  COCO stays anchored: one replay detection batch (ultralytics
        v8DetectionLoss through the model's own COCO head) is backpropped in
        every optimizer step, gradient-accumulated with one batch per channel,
        then a single ``optimizer.step()``. Backbone LR 1e-4 (10x below the
        documented destructive runs), head-only warmup, EMA on.
    R5  The trainer never augments channel batches — loaders ship image and
        label maps already jointly transformed (see channel.py contract).
    R8  Export defaults to whole-detector transplant (anchored/export.py).
    R9  Sentinels (anchored/sentinels.py) run per epoch, followed by every
        channel's ``on_epoch_end`` sentinel hook; abort raises.
    E5  ultralytics models ship some params with requires_grad=False — all
        trainable parts are explicitly enabled (the fixed DFL integral conv is
        deliberately kept frozen: training it would corrupt box decoding);
        EMA is verified non-aliased at construction (Risk-16).

Loop skeleton modeled on pretrain/dense_trainer.py; the supervised+aux mixing
lineage is trainer/_core.py. Standalone (not an ultralytics BaseTrainer
subclass) for the same reason DenseSSLPretrainer is: the framework assumes a
single detection task per step, and we run 1 + len(channels) heterogeneous
batches per step.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from ..dense.multi_scale_tap import MultiScaleFeatureTap, _get_layer_sequence
from .channel import AuxChannel, probe_tap_channels
from .sentinels import SentinelLog, SentinelThresholds

#: YOLOv8 layout: layers 0-9 backbone, 10-(n-2) neck, last layer Detect head.
BACKBONE_LAYERS = 10


def _log(msg: str) -> None:
    try:
        from ultralytics.utils import LOGGER

        LOGGER.info(msg)
    except Exception:
        print(msg)


def _resolve_device(device) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return torch.device(f"cuda:{int(device)}")
    return torch.device(device)


def _dfl_param_ids(model: nn.Module) -> set:
    """Param ids of ultralytics DFL modules — a FIXED integral operator
    (conv weight = arange) that must never be trained or weight-decayed."""
    ids = set()
    for mod in model.modules():
        if type(mod).__name__ == "DFL":
            ids.update(id(p) for p in mod.parameters())
    return ids


class _Recycler:
    """Endless iterator over a re-iterable loader (re-iterates on exhaustion,
    unlike itertools.cycle which would cache an epoch of CUDA tensors)."""

    def __init__(self, loader: Iterable, name: str = "loader") -> None:
        if loader is None:
            raise ValueError(f"{name} is None")
        self.loader = loader
        self.name = name
        self._it = iter(loader)

    def __next__(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            try:
                return next(self._it)
            except StopIteration:
                raise RuntimeError(f"{self.name} is empty") from None

    def __iter__(self):
        return self


class AnchoredJointTrainer:
    """Joint COCO-replay + auxiliary-channel trainer (the anchored carrier).

    Args:
        model: ultralytics model spec — ``"yolov8n.pt"`` (COCO init; the
            production setting), ``"yolov8n.yaml"`` (offline random init;
            tests), or a pre-built ``DetectionModel``/nn.Module.
        replay_data: ultralytics data yaml for the COCO replay anchor
            (default ``"coco128.yaml"``; ultralytics may download it on first
            use — pass ``replay_loader`` to stay offline).
        channels: sequence of :class:`~.channel.AuxChannel` instances. May be
            empty — the trainer then degrades to the replay-only continuation
            control arm.
        lambda_aux: global weight multiplying the summed loss terms of every
            channel batch.
        epochs / imgsz / batch: training schedule defaults (used by
            :meth:`train` and the channel loader cfg).
        backbone_lr / neck_lr / head_lr: AdamW param-group LRs. Groups:
            backbone = ``model.model[0:backbone_layers]``;
            neck     = ``model.model[backbone_layers:]`` INCLUDING the COCO
                       Detect head (it keeps adapting under replay at the
                       moderate neck LR);
            heads    = all channel head params (fresh modules, highest LR).
        warmup_steps: head-only warmup — backbone+neck ``requires_grad=False``
            for the first N optimizer steps (COCO Detect head + channel heads
            stay trainable), then everything is explicitly re-enabled (E5).
        weight_decay: AdamW weight decay.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"`` | index | torch.device.
        amp: autocast + GradScaler when the device is CUDA (no-op on CPU).
        output_dir: run directory (sentinel CSV + exported checkpoints).
        replay_loader: optional pre-built replay loader (iterable of
            ultralytics-style detection batches: dict with ``img``,
            ``batch_idx``, ``cls``, ``bboxes``). Overrides ``replay_data``.
        workers: dataloader workers for built-in loaders / channel cfg.
        backbone_layers: backbone/neck split index (YOLOv8: 10).
        ema_decay / ema_tau: ultralytics ModelEMA schedule
            (d = decay * (1 - exp(-updates / tau))).
        grad_clip: global grad-norm clip (ultralytics convention: 10.0).
        sentinel_thresholds: warn/abort levels (see sentinels.py). Pass
            relaxed thresholds for tiny probes (tests).
        probe_batch: optional fixed sentinel probe images [B, 3, H, W]; if
            None, the first replay batch's images (up to 8) are captured.

    Typical use::

        trainer = AnchoredJointTrainer(
            model="yolov8n.pt", replay_data="coco128.yaml",
            channels=[TerraChannel(...)], lambda_aux=1.0,
            epochs=12, imgsz=512, batch=12,
        )
        ckpt = trainer.train()                  # -> output_dir/anchored_full.pt
        yolo = load_for_finetune(ckpt)          # -> ultralytics YOLO for FT
    """

    def __init__(
        self,
        model: Any = "yolov8n.pt",
        replay_data: str = "coco128.yaml",
        channels: Sequence[AuxChannel] = (),
        lambda_aux: float = 1.0,
        epochs: int = 12,
        imgsz: int = 512,
        batch: int = 12,
        backbone_lr: float = 1e-4,
        neck_lr: float = 2e-4,
        head_lr: float = 1e-3,
        warmup_steps: int = 300,
        weight_decay: float = 0.05,
        device: Any = "auto",
        amp: bool = True,
        output_dir: str = "runs/anchored",
        replay_loader: Optional[Iterable] = None,
        workers: int = 2,
        backbone_layers: int = BACKBONE_LAYERS,
        ema_decay: float = 0.9999,
        ema_tau: float = 2000,
        grad_clip: float = 10.0,
        sentinel_thresholds: Optional[SentinelThresholds] = None,
        probe_batch: Optional[torch.Tensor] = None,
    ) -> None:
        # ── validation ───────────────────────────────────────────────────
        if lambda_aux < 0:
            raise ValueError(f"lambda_aux must be >= 0, got {lambda_aux}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        for nm, lr in (("backbone_lr", backbone_lr), ("neck_lr", neck_lr), ("head_lr", head_lr)):
            if lr <= 0:
                raise ValueError(f"{nm} must be positive, got {lr}")
        names = [ch.name for ch in channels]
        if len(set(names)) != len(names):
            raise ValueError(f"channel names must be unique, got {names}")

        self.replay_data = replay_data
        self.channels: List[AuxChannel] = list(channels)
        self.lambda_aux = float(lambda_aux)
        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.backbone_lr = float(backbone_lr)
        self.neck_lr = float(neck_lr)
        self.head_lr = float(head_lr)
        self.warmup_steps = int(warmup_steps)
        self.weight_decay = float(weight_decay)
        self.output_dir = str(output_dir)
        self.workers = int(workers)
        self.backbone_layers = int(backbone_layers)
        self.grad_clip = float(grad_clip)
        self.sentinel_thresholds = sentinel_thresholds
        self._replay_loader = replay_loader
        self._probe_batch = probe_batch

        # ── device ───────────────────────────────────────────────────────
        self.device = _resolve_device(device)
        self.use_amp = bool(amp) and self.device.type == "cuda"

        # ── model ────────────────────────────────────────────────────────
        if isinstance(model, str):
            from ultralytics import YOLO  # lazy: optional [yolo] extra

            self.model = YOLO(model, task="detect").model
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            raise TypeError(f"model must be a str spec or nn.Module, got {type(model).__name__}")
        self.model.to(self.device).train()

        # v8DetectionLoss reads model.args.box/.cls/.dfl as ATTRIBUTES;
        # YOLO('*.yaml') ships args as a plain dict — normalize it.
        self._ensure_detection_args(self.model)

        # E5: ultralytics models can ship params with requires_grad=False.
        # Enable everything trainable; keep the fixed DFL conv frozen.
        self._dfl_ids = _dfl_param_ids(self.model)
        self._set_requires_grad_all(True)

        # ── EMA (BEFORE tap hooks — deepcopy must not capture hook closures)
        from ultralytics.utils.torch_utils import ModelEMA  # lazy

        self.ema = ModelEMA(self.model, decay=float(ema_decay), tau=float(ema_tau))
        self._assert_ema_independent()  # Risk-16 guard, hard error on aliasing

        # ── shared P3/P4/P5 taps ─────────────────────────────────────────
        self.taps = MultiScaleFeatureTap(self.model)
        self.taps.setup()
        self.tap_channels: Dict[str, int] = probe_tap_channels(
            self.model, self.taps, imgsz=min(self.imgsz, 64)
        )

        # ── attach channels ──────────────────────────────────────────────
        self.heads: Dict[str, nn.ModuleList] = {}
        for ch in self.channels:
            heads = ch.attach(self.model, self.taps)
            if not isinstance(heads, nn.ModuleList):
                raise TypeError(
                    f"channel {ch.name!r}.attach() must return nn.ModuleList, "
                    f"got {type(heads).__name__}"
                )
            heads.to(self.device).train()
            for p in heads.parameters():
                p.requires_grad_(True)  # E5
            self.heads[ch.name] = heads

        # ── optimizer: backbone / neck(+COCO head) / channel heads ───────
        self._seq = _get_layer_sequence(self.model)
        bb_end = min(self.backbone_layers, max(len(self._seq) - 1, 0))
        backbone_params = [
            p for m in list(self._seq)[:bb_end] for p in m.parameters()
            if id(p) not in self._dfl_ids
        ]
        neck_params = [
            p for m in list(self._seq)[bb_end:] for p in m.parameters()
            if id(p) not in self._dfl_ids
        ]
        head_params = [p for hl in self.heads.values() for p in hl.parameters()]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.backbone_lr,
                           "name": "backbone"})
        if neck_params:
            groups.append({"params": neck_params, "lr": self.neck_lr, "name": "neck"})
        if head_params:
            groups.append({"params": head_params, "lr": self.head_lr, "name": "heads"})
        if not groups:
            raise ValueError("no trainable parameters found")
        self.optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)

        # ── AMP scaler ───────────────────────────────────────────────────
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except TypeError:  # older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ── head-only warmup: freeze backbone+neck (NOT the Detect head) ─
        self._warmup_active = self.warmup_steps > 0
        if self._warmup_active:
            self._set_backbone_neck_frozen(True)

        # ── sentinels (lazy probe capture unless probe_batch given) ──────
        self.sentinels: Optional[SentinelLog] = None
        if probe_batch is not None:
            self._init_sentinels(probe_batch)

        # ── bookkeeping ──────────────────────────────────────────────────
        self.global_step = 0
        self.history: List[Dict[str, float]] = []
        self._epoch = 0
        self._criterion_warned = False

    # ── construction helpers ──────────────────────────────────────────────

    @staticmethod
    def _ensure_detection_args(model: nn.Module) -> None:
        args = getattr(model, "args", None)
        if args is not None and all(hasattr(args, k) for k in ("box", "cls", "dfl")):
            return
        try:
            from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
        except Exception:
            return  # non-ultralytics model; _replay_loss will raise clearly if used
        merged = dict(DEFAULT_CFG_DICT)
        if isinstance(args, dict):
            merged.update({k: v for k, v in args.items() if v is not None})
        elif args is not None:
            try:
                merged.update({k: v for k, v in vars(args).items() if v is not None})
            except TypeError:
                pass
        model.args = IterableSimpleNamespace(**merged)

    def _set_requires_grad_all(self, enabled: bool) -> None:
        for p in self.model.parameters():
            if id(p) in self._dfl_ids:
                p.requires_grad_(False)
            else:
                p.requires_grad_(enabled)

    def _set_backbone_neck_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze every layer EXCEPT the final Detect head.

        E5: unfreeze explicitly re-enables requires_grad (the documented
        ultralytics ships-frozen pitfall); the DFL conv stays frozen always.
        """
        for m in list(self._seq)[:-1]:
            for p in m.parameters():
                if id(p) in self._dfl_ids:
                    continue
                p.requires_grad_(not frozen)

    def _assert_ema_independent(self) -> None:
        """Risk-16: EMA update is in-place (v *= d) — any storage aliasing
        with the live model collapses weights within an epoch. Verify."""
        model_ptrs = {
            t.data_ptr()
            for t in list(self.model.parameters()) + list(self.model.buffers())
            if t.numel() > 0
        }
        ema_ptrs = {
            t.data_ptr()
            for t in list(self.ema.ema.parameters()) + list(self.ema.ema.buffers())
            if t.numel() > 0
        }
        shared = model_ptrs & ema_ptrs
        if shared:
            raise RuntimeError(
                f"Risk-16 violation: {len(shared)} EMA tensors share storage with the live "
                f"model. Never load EMA state with load_state_dict(assign=True); rebuild the "
                f"EMA from a deepcopy."
            )

    def _init_sentinels(self, probe_imgs: torch.Tensor) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.sentinels = SentinelLog(
            model=self.model,
            taps=self.taps,
            probe_batch=probe_imgs.detach()[:8].clone(),
            thresholds=self.sentinel_thresholds,
            csv_path=str(Path(self.output_dir) / "sentinels.csv"),
        )

    # ── batch plumbing ────────────────────────────────────────────────────

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
        img = out.get("img")
        if img is None:
            raise KeyError("batch is missing the required 'img' key")
        if img.dtype == torch.uint8:
            out["img"] = img.float() / 255.0
        return out

    def _autocast(self):
        return torch.amp.autocast(self.device.type, enabled=self.use_amp)

    def _replay_loss(self, batch: Dict[str, Any]):
        """Detection loss on the model's own COCO head (ultralytics
        v8DetectionLoss via DetectionModel.loss). Returns (scalar, items)."""
        if not hasattr(self.model, "loss"):
            raise TypeError(
                "replay anchor requires an ultralytics DetectionModel (model.loss(batch)); "
                f"got {type(self.model).__name__}"
            )
        out = self.model.loss(batch)
        loss, items = out if isinstance(out, tuple) else (out, None)
        if loss.dim() > 0:  # v8DetectionLoss returns the [box, cls, dfl] vector
            loss = loss.sum()
        return loss, items

    def loader_cfg(self) -> Dict[str, Any]:
        """The cfg dict handed to every channel's ``build_loader``."""
        return {
            "imgsz": self.imgsz,
            "batch": self.batch,
            "workers": self.workers,
            "device": str(self.device),
        }

    def _build_replay_loader(self):
        """Stock ultralytics detection train loader on ``replay_data``.

        NOTE: may download the dataset (e.g. coco128) on first use; tests
        pass ``replay_loader`` explicitly to stay offline.
        """
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_dataloader, build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset

        data = check_det_dataset(self.replay_data)
        head = list(self._seq)[-1]
        model_nc = getattr(head, "nc", None)
        data_nc = int(data.get("nc", 0) or 0)
        if model_nc is not None and data_nc and int(model_nc) != data_nc:
            raise ValueError(
                f"replay dataset nc={data_nc} != model nc={model_nc} — the replay anchor "
                f"must train the model's own COCO head (R3/R8), not a reshaped one."
            )
        cfg = get_cfg()
        cfg.imgsz = self.imgsz
        cfg.data = self.replay_data
        dataset = build_yolo_dataset(cfg, data["train"], self.batch, data, mode="train",
                                     rect=False)
        return build_dataloader(dataset, self.batch, self.workers, shuffle=True, rank=-1)

    # ── one optimizer step ────────────────────────────────────────────────

    def step(
        self,
        replay_batch: Dict[str, Any],
        channel_batches: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """One full optimizer step: replay detection loss + every channel's
        loss, gradient-accumulated, single ``optimizer.step()``, EMA update,
        warmup bookkeeping.

        Args:
            replay_batch: ultralytics-style detection batch
                (``img``, ``batch_idx``, ``cls``, ``bboxes``).
            channel_batches: ``{channel_name: batch}``; channels without an
                entry are skipped this step. None = replay-only step.

        Returns:
            Float metrics: ``replay/det_loss``, ``replay/cls_loss``,
            ``{name}/{term}`` and ``{name}/total`` per channel, ``total``.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics: Dict[str, float] = {}

        # 1) replay anchor (R3) — backward first, accumulate
        rb = self._to_device(replay_batch)
        if self.sentinels is None:
            self._init_sentinels(rb["img"])
        with self._autocast():
            det_loss, det_items = self._replay_loss(rb)
        self.scaler.scale(det_loss).backward()
        metrics["replay/det_loss"] = float(det_loss.detach())
        if det_items is not None and torch.is_tensor(det_items) and det_items.numel() >= 3:
            cls_v = float(det_items[1])  # [box, cls, dfl]
            metrics["replay/cls_loss"] = cls_v
            if self.sentinels is not None:
                self.sentinels.update_replay_cls(cls_v)

        # 2) channel batches — one forward + backward each (grad accumulation)
        total = metrics["replay/det_loss"]
        if channel_batches:
            for ch in self.channels:
                batch = channel_batches.get(ch.name)
                if batch is None:
                    continue
                cb = self._to_device(batch)
                self.taps.clear()
                with self._autocast():
                    _ = self.model(cb["img"])  # taps capture P3/P4/P5
                    terms = ch.loss(cb, self.taps)
                    if not terms:
                        continue
                    ch_loss = self.lambda_aux * sum(terms.values())
                self.scaler.scale(ch_loss).backward()
                for term, value in terms.items():
                    metrics[f"{ch.name}/{term}"] = float(value.detach())
                ch_total = float(ch_loss.detach())
                metrics[f"{ch.name}/total"] = ch_total
                total += ch_total

        # 3) clip + single optimizer step + EMA
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]],
            max_norm=self.grad_clip,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.update(self.model)

        # 4) warmup bookkeeping (E5: explicit re-enable after warmup)
        self.global_step += 1
        if self._warmup_active and self.global_step >= self.warmup_steps:
            self._set_backbone_neck_frozen(False)
            self._warmup_active = False
            _log(f"[ycl-anchored] warmup done at step {self.global_step}: "
                 f"backbone+neck unfrozen (DFL stays fixed)")

        metrics["total"] = total
        return metrics

    # ── full training run ─────────────────────────────────────────────────

    def train(
        self,
        epochs: Optional[int] = None,
        replay_loader: Optional[Iterable] = None,
        channel_loaders: Optional[Dict[str, Iterable]] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> str:
        """Run anchored joint training; returns the exported checkpoint path.

        Args:
            epochs: override ctor epochs.
            replay_loader: override ctor replay loader / replay_data build.
            channel_loaders: ``{channel_name: loader}``; if None, each
                channel's ``build_loader(self.loader_cfg())`` is called.
            steps_per_epoch: required if the replay loader has no ``len()``.

        Raises:
            SentinelAbort: if a sentinel crosses its abort threshold (R9).
        """
        n_epochs = int(epochs) if epochs is not None else self.epochs
        loader = replay_loader if replay_loader is not None else self._replay_loader
        if loader is None:
            loader = self._build_replay_loader()
        if channel_loaders is None:
            cfg = self.loader_cfg()
            channel_loaders = {ch.name: ch.build_loader(cfg) for ch in self.channels}
        missing = [ch.name for ch in self.channels if ch.name not in channel_loaders]
        if missing:
            raise ValueError(f"channel_loaders missing entries for: {missing}")
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(loader)  # type: ignore[arg-type]
            except TypeError:
                raise ValueError("steps_per_epoch is required for loaders without len()")
        if steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")

        replay_it = _Recycler(loader, "replay_loader")
        chan_its = {
            name: _Recycler(channel_loaders[name], f"channel_loader[{name}]")
            for name in channel_loaders
        }

        _log(
            f"[ycl-anchored] === Anchored joint training ===\n"
            f"[ycl-anchored] channels={[ch.name for ch in self.channels] or ['<replay-only>']}"
            f", lambda_aux={self.lambda_aux}\n"
            f"[ycl-anchored] epochs={n_epochs}, steps/epoch={steps_per_epoch}, "
            f"imgsz={self.imgsz}, batch={self.batch}, device={self.device}, "
            f"amp={self.use_amp}\n"
            f"[ycl-anchored] lr backbone={self.backbone_lr} neck={self.neck_lr} "
            f"heads={self.head_lr}, warmup_steps={self.warmup_steps}, EMA on"
        )

        t0_total = time.time()
        for epoch in range(self._epoch + 1, self._epoch + n_epochs + 1):
            t0 = time.time()
            sums: Dict[str, float] = {}
            n = 0
            for _ in range(steps_per_epoch):
                rb = next(replay_it)
                cbs = {name: next(it) for name, it in chan_its.items()}
                m = self.step(rb, cbs)
                for k, v in m.items():
                    if math.isfinite(v):
                        sums[k] = sums.get(k, 0.0) + v
                n += 1
            means = {k: v / max(1, n) for k, v in sums.items()}

            self._epoch = epoch
            sent = self.run_sentinels(epoch)  # raises SentinelAbort on hard failure
            chan_sent = self.run_channel_sentinels(epoch)  # R9: structural, per channel
            row = {"epoch": float(epoch), **means,
                   **{f"sentinel/{k}": v for k, v in sent.items() if k != "epoch"},
                   **chan_sent}
            self.history.append(row)
            _log(
                f"[ycl-anchored] epoch {epoch:3d} | total={means.get('total', float('nan')):.4f}"
                f" | det={means.get('replay/det_loss', float('nan')):.4f}"
                f" | eff_rank={sent.get('eff_rank', float('nan')):.1f}"
                f" | cka={sent.get('cka_prev_epoch', float('nan')):.3f}"
                f" | cls_drift={sent.get('replay_cls_drift', float('nan')):+.1%}"
                f" | {time.time() - t0:.1f}s"
            )

        path = self.export()
        _log(f"[ycl-anchored] === done in {time.time() - t0_total:.1f}s -> {path} ===")
        return path

    # ── sentinels / export / lifecycle ────────────────────────────────────

    def run_sentinels(self, epoch: int) -> Dict[str, float]:
        """Run the per-epoch sentinel suite (R9). Raises SentinelAbort on
        hard failure; returns the metric record otherwise."""
        if self.sentinels is None:
            return {}
        head_modules: Dict[str, nn.Module] = dict(self.heads)
        head_modules["coco_detect"] = list(self._seq)[-1]
        return self.sentinels.epoch_end(epoch, head_modules)

    def run_channel_sentinels(self, epoch: int) -> Dict[str, float]:
        """Invoke every channel's :meth:`AuxChannel.on_epoch_end` hook (R9 —
        channel sentinels run structurally, not by run-loop convention).
        Returns the merged dict keyed ``sentinel/{channel}/{metric}``."""
        out: Dict[str, float] = {}
        for ch in self.channels:
            hook = getattr(ch, "on_epoch_end", None)
            if not callable(hook):
                continue
            for k, v in (hook(epoch) or {}).items():
                if k == "epoch":
                    continue
                out[f"sentinel/{ch.name}/{k}"] = float(v)
        return out

    def export(self, path: Optional[str] = None, transplant: str = "full",
               use_ema: bool = True) -> str:
        """Save the (EMA) detector for downstream fine-tuning (R8: full
        transplant by default). Channel heads are NOT exported — inference
        cost stays exactly that of the base detector."""
        from .export import save_checkpoint

        src = self.ema.ema if (use_ema and self.ema is not None) else self.model
        if path is None:
            path = str(Path(self.output_dir) / f"anchored_{transplant}.pt")
        return save_checkpoint(
            src, path, transplant=transplant, epoch=self._epoch,
            extra={
                "channels": [ch.name for ch in self.channels],
                "lambda_aux": self.lambda_aux,
                "global_step": self.global_step,
                "ema": bool(use_ema),
            },
        )

    def cleanup(self) -> None:
        """Release tap hooks. Idempotent."""
        try:
            self.taps.close()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover - GC path
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AnchoredJointTrainer(channels={[ch.name for ch in self.channels]}, "
            f"lambda_aux={self.lambda_aux}, device={self.device}, "
            f"step={self.global_step}, warmup_active={self._warmup_active})"
        )
