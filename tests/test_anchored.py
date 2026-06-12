"""Tests for the anchored joint-training scaffold (anchored/).

CPU-only, offline: the detector is built from ``yolov8n.yaml`` (ships inside
the ultralytics package — no .pt download), replay batches are synthetic
ultralytics-style detection dicts, and the DummyChannel feeds random images
through a tiny parameter-dependent head.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from yolo_contrastive.anchored import (
    AnchoredJointTrainer,
    AuxChannel,
    SentinelAbort,
    SentinelLog,
    SentinelThresholds,
    effective_rank,
    linear_cka,
    load_for_finetune,
    probe_tap_channels,
    save_checkpoint,
)
from yolo_contrastive.anchored.export import _is_backbone_key
from yolo_contrastive.pretrain.backbone_utils import transplant_full

# Tiny 64px probes cannot reach the production eff_rank thresholds — relax
# every warn/abort level so trainer tests exercise the pipeline, not the bars.
RELAXED = SentinelThresholds(
    eff_rank_warn=0.0, eff_rank_abort=-1.0,
    cls_drift_warn=1e9, cls_drift_abort=1e9,
    cka_warn=-1.0, head_norm_growth_warn=1e9,
)


# ── shared helpers ────────────────────────────────────────────────────────────


class DummyChannel(AuxChannel):
    """Minimal channel: 1x1 conv head on P5, parameter-dependent MSE-to-zero."""

    name = "dummy"

    def __init__(self, n_batches: int = 4):
        self.n_batches = n_batches
        self.head: nn.Conv2d | None = None
        self.epoch_end_calls: list[int] = []

    def attach(self, model, taps):
        c5 = probe_tap_channels(model, taps)["P5"]
        self.head = nn.Conv2d(c5, 4, 1)
        return nn.ModuleList([self.head])

    def loss(self, batch, taps):
        f5 = taps.get_features()["P5"]
        return {"mse": self.head(f5).pow(2).mean()}

    def build_loader(self, cfg):
        b, s = cfg["batch"], cfg["imgsz"]
        return [{"img": torch.rand(b, 3, s, s)} for _ in range(self.n_batches)]

    def on_epoch_end(self, epoch):  # R9 channel-sentinel hook
        self.epoch_end_calls.append(epoch)
        return {"probe": 0.5, "epoch": float(epoch)}  # "epoch" must be dropped


def _replay_batch(b: int = 2, s: int = 64, seed: int = 0):
    """Synthetic ultralytics-style detection batch (2 boxes per image)."""
    g = torch.Generator().manual_seed(seed)
    n = 2 * b
    return {
        "img": torch.rand(b, 3, s, s, generator=g),
        "batch_idx": torch.arange(b).repeat_interleave(2).float(),
        "cls": torch.randint(0, 80, (n, 1), generator=g).float(),
        "bboxes": torch.cat(
            [0.3 + 0.4 * torch.rand(n, 2, generator=g),
             0.05 + 0.2 * torch.rand(n, 2, generator=g)],
            dim=1,
        ),
    }


def _make_trainer(output_dir, **kw):
    defaults = dict(
        model="yolov8n.yaml",       # offline: yaml ships inside ultralytics
        channels=[DummyChannel()],
        lambda_aux=1.0,
        epochs=1,
        imgsz=64,
        batch=2,
        warmup_steps=2,
        device="cpu",
        amp=False,
        output_dir=str(output_dir),
        sentinel_thresholds=RELAXED,
    )
    defaults.update(kw)
    return AnchoredJointTrainer(**defaults)


# ── import hygiene ────────────────────────────────────────────────────────────


def test_package_imports():
    import yolo_contrastive  # noqa: F401  (must stay lightweight, E2)
    import yolo_contrastive.anchored as anchored

    assert hasattr(anchored, "AnchoredJointTrainer")
    assert hasattr(anchored, "AuxChannel")


# ── sentinel math (analytic ground truth) ─────────────────────────────────────


class TestSentinelMath:
    def test_effective_rank_rank_one(self):
        x = torch.outer(torch.linspace(1.0, 2.0, 32), torch.linspace(1.0, 3.0, 16))
        assert abs(effective_rank(x, center=False) - 1.0) < 1e-4

    def test_effective_rank_equal_singular_values(self):
        # eye(16) stacked 4x: X^T X = 4I -> 16 equal singular values -> rank 16
        x = torch.eye(16).repeat(4, 1)
        assert abs(effective_rank(x, center=False) - 16.0) < 1e-3

    def test_effective_rank_4d_input(self):
        feats = torch.randn(2, 8, 4, 4)  # -> [32, 8] matrix
        er = effective_rank(feats)
        assert 1.0 <= er <= 8.0

    def test_cka_self_is_one(self):
        x = torch.randn(64, 16)
        assert abs(linear_cka(x, x) - 1.0) < 1e-5

    def test_cka_orthogonal_invariance(self):
        torch.manual_seed(0)
        x = torch.randn(64, 16)
        q, _ = torch.linalg.qr(torch.randn(16, 16))
        assert abs(linear_cka(x, x @ q) - 1.0) < 1e-4

    def test_cka_independent_is_low(self):
        torch.manual_seed(1)
        a, b = torch.randn(512, 16), torch.randn(512, 16)
        assert linear_cka(a, b) < 0.3

    def test_cka_sample_mismatch_raises(self):
        with pytest.raises(ValueError):
            linear_cka(torch.randn(8, 4), torch.randn(9, 4))


# ── SentinelLog on a stub model (no YOLO) ─────────────────────────────────────


class _TapStub:
    def __init__(self):
        self.f = None

    def clear(self):
        self.f = None

    def get_features(self):
        if self.f is None:
            raise RuntimeError("no forward captured")
        return {"P5": self.f}


class _StubModel(nn.Module):
    def __init__(self, stub):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=4, padding=1)
        self._stub = stub

    def forward(self, x):
        y = self.conv(x)
        self._stub.f = y
        return y


def _make_log(**th_kw):
    th = SentinelThresholds(
        eff_rank_warn=0.0, eff_rank_abort=-1.0,
        cls_drift_warn=1e9, cls_drift_abort=1e9,
        cka_warn=-1.0, head_norm_growth_warn=1e9,
    )
    for k, v in th_kw.items():
        setattr(th, k, v)
    stub = _TapStub()
    model = _StubModel(stub)
    return SentinelLog(
        model=model, taps=stub, probe_batch=torch.rand(4, 3, 32, 32),
        thresholds=th, cls_ema_momentum=0.0,
    )


class TestSentinelLog:
    def test_records_and_metrics(self, tmp_path):
        log = _make_log()
        log.csv_path = str(tmp_path / "s.csv")
        log.update_replay_cls(1.0)
        r1 = log.epoch_end(1, head_modules={"h": nn.Conv2d(2, 2, 1)})
        r2 = log.epoch_end(2)
        assert len(log.records) == 2
        assert r1["eff_rank"] > 1.0
        assert r1["cka_prev_epoch"] != r1["cka_prev_epoch"]  # nan at first epoch
        assert 0.99 < r2["cka_prev_epoch"] <= 1.0 + 1e-6     # identical weights
        assert r1["replay_cls_drift"] == 0.0                 # baseline epoch
        assert "head_norm/h" in r1
        assert (tmp_path / "s.csv").exists()

    def test_cls_drift_abort(self):
        log = _make_log(cls_drift_abort=0.30)
        log.update_replay_cls(1.0)
        log.epoch_end(1)  # sets the baseline
        log.update_replay_cls(2.0)  # momentum 0 -> EMA jumps to 2.0 (+100%)
        with pytest.raises(SentinelAbort, match="cls-loss"):
            log.epoch_end(2)
        assert len(log.records) == 2  # forensic record written before raising

    def test_cls_drift_warn(self):
        log = _make_log(cls_drift_warn=0.15, cls_drift_abort=10.0)
        log.update_replay_cls(1.0)
        log.epoch_end(1)
        log.update_replay_cls(1.2)
        with pytest.warns(RuntimeWarning, match="drift"):
            log.epoch_end(2)

    def test_eff_rank_abort(self):
        log = _make_log(eff_rank_abort=1e9)
        with pytest.raises(SentinelAbort, match="effective rank"):
            log.epoch_end(1)


# ── trainer: cheap validation (no model build) ────────────────────────────────


class TestTrainerValidation:
    def test_duplicate_channel_names_raise(self):
        with pytest.raises(ValueError, match="unique"):
            AnchoredJointTrainer(channels=[DummyChannel(), DummyChannel()])

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda_aux"):
            AnchoredJointTrainer(lambda_aux=-0.1)

    def test_bad_lr_raises(self):
        with pytest.raises(ValueError, match="backbone_lr"):
            AnchoredJointTrainer(backbone_lr=0.0)


# ── trainer: 3 optimizer steps end-to-end (one shared build) ─────────────────


@pytest.fixture(scope="module")
def stepped(tmp_path_factory):
    """Build one trainer (yolov8n.yaml random init), run 3 steps with
    warmup_steps=2, recording observations after each step."""
    tmp = tmp_path_factory.mktemp("anchored_run")
    t = _make_trainer(tmp, warmup_steps=2)

    bb = t._seq[0].conv.weight                # a backbone parameter
    head = t.heads["dummy"][0].weight         # the channel head parameter
    obs = SimpleNamespace(trainer=t, tmp=tmp)
    obs.bb_rg_at_ctor = bb.requires_grad
    obs.detect_rg_at_ctor = next(
        p.requires_grad for p in t._seq[-1].parameters() if id(p) not in t._dfl_ids
    )
    obs.bb0 = bb.detach().clone()
    obs.head0 = head.detach().clone()

    obs.metrics = []
    obs.bb_after = []
    obs.warmup_after = []
    for i in (1, 2, 3):
        m = t.step(_replay_batch(seed=i), {"dummy": {"img": torch.rand(2, 3, 64, 64)}})
        obs.metrics.append(m)
        obs.bb_after.append(bb.detach().clone())
        obs.warmup_after.append(t._warmup_active)
    obs.head_after = head.detach().clone()
    obs.sentinel_record = t.run_sentinels(1)
    yield obs
    t.cleanup()


class TestThreeSteps:
    def test_construction(self, stepped):
        t = stepped.trainer
        names = [g["name"] for g in t.optimizer.param_groups]
        lrs = [g["lr"] for g in t.optimizer.param_groups]
        assert names == ["backbone", "neck", "heads"]
        assert lrs == [1e-4, 2e-4, 1e-3]
        assert set(t.tap_channels) == {"P3", "P4", "P5"}
        assert isinstance(t.heads["dummy"], nn.ModuleList)

    def test_warmup_freezes_backbone_not_head(self, stepped):
        assert stepped.bb_rg_at_ctor is False       # backbone frozen at ctor
        assert stepped.detect_rg_at_ctor is True    # COCO Detect head trainable

    def test_losses_finite(self, stepped):
        for m in stepped.metrics:
            for k, v in m.items():
                assert torch.isfinite(torch.tensor(v)), f"{k} not finite: {v}"
            assert m["replay/det_loss"] > 0
            assert m["dummy/mse"] > 0
            assert m["total"] == pytest.approx(m["replay/det_loss"] + m["dummy/total"])

    def test_backbone_frozen_during_warmup_changes_after(self, stepped):
        assert torch.allclose(stepped.bb_after[0], stepped.bb0)   # step 1: frozen
        assert torch.allclose(stepped.bb_after[1], stepped.bb0)   # step 2: frozen
        assert stepped.warmup_after[1] is False                   # unfroze after step 2
        assert not torch.allclose(stepped.bb_after[2], stepped.bb0)  # step 3: trains

    def test_warmup_reenables_requires_grad(self, stepped):
        t = stepped.trainer
        for p in t.model.parameters():
            if id(p) in t._dfl_ids:
                assert not p.requires_grad  # the fixed DFL conv stays frozen
            else:
                assert p.requires_grad      # E5: explicit re-enable after warmup

    def test_channel_head_trains_from_step_one(self, stepped):
        assert not torch.allclose(stepped.head_after, stepped.head0)

    def test_ema_updated_and_not_aliased(self, stepped):
        t = stepped.trainer
        assert t.ema.updates == 3
        model_ptrs = {p.data_ptr() for p in t.model.parameters()}
        ema_ptrs = {p.data_ptr() for p in t.ema.ema.parameters()}
        assert not (model_ptrs & ema_ptrs)  # Risk-16: zero shared storage

    def test_sentinels_compute(self, stepped):
        r = stepped.sentinel_record
        assert r["eff_rank"] > 0
        assert "replay_cls_ema" in r and r["replay_cls_ema"] > 0
        assert "head_norm/dummy" in r
        assert "head_norm/coco_detect" in r

    # ── export + reload round-trip (R8) ───────────────────────────────────

    def test_export_full_roundtrip(self, stepped):
        t = stepped.trainer
        path = t.export(path=str(stepped.tmp / "full.pt"))  # use_ema=True default
        ckpt = torch.load(path, weights_only=True)           # weights_only-safe payload
        assert ckpt["transplant"] == "full"
        assert ckpt["type"] == "anchored_joint"

        yolo = load_for_finetune(path, base="yolov8n.yaml")
        src, dst = t.ema.ema.state_dict(), yolo.model.state_dict()
        assert set(src) == {k for k in dst if k in src}
        for key in ("model.0.conv.weight", f"model.{len(t._seq) - 1}.cv2.0.0.conv.weight"):
            assert torch.allclose(src[key], dst[key]), key
            assert src[key].data_ptr() != dst[key].data_ptr()  # value copy, no aliasing
        assert yolo.ckpt  # truthy -> .train() will carry the transplanted weights

    def test_export_backbone_only(self, stepped):
        t = stepped.trainer
        path = t.export(path=str(stepped.tmp / "bb.pt"), transplant="backbone",
                        use_ema=False)
        ckpt = torch.load(path, weights_only=True)
        keys = list(ckpt["model_state_dict"].keys())
        assert keys and all(_is_backbone_key(k) for k in keys)

        yolo = load_for_finetune(path, base="yolov8n.yaml")
        src, dst = t.model.state_dict(), yolo.model.state_dict()
        assert torch.allclose(src["model.0.conv.weight"], dst["model.0.conv.weight"])
        head_key = f"model.{len(t._seq) - 1}.cv2.0.0.conv.weight"
        assert not torch.allclose(src[head_key], dst[head_key])  # head NOT transplanted

    def test_save_checkpoint_rejects_bad_mode(self, stepped):
        with pytest.raises(ValueError, match="transplant"):
            save_checkpoint(stepped.trainer.model, str(stepped.tmp / "x.pt"),
                            transplant="neck")

    def test_transplant_full_counts(self, stepped):
        t = stepped.trainer
        n = transplant_full(t.model, t.model.state_dict(), verbose=False)
        assert n == len(t.model.state_dict())


# ── trainer.train(): one epoch through loaders + sentinels + export ──────────


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_train_one_epoch(tmp_path):
    t = _make_trainer(tmp_path / "run", warmup_steps=1)
    replay = [_replay_batch(seed=i) for i in (1, 2)]
    out = t.train(epochs=1, replay_loader=replay)  # channel loader via build_loader(cfg)
    try:
        assert Path(out).exists()
        assert len(t.history) == 1
        assert t.global_step == 2
        assert (tmp_path / "run" / "sentinels.csv").exists()
        ckpt = torch.load(out, weights_only=True)
        assert ckpt["transplant"] == "full"
        assert ckpt["extra"]["channels"] == ["dummy"]
        # R9 structural: the trainer invoked the channel sentinel hook and
        # logged its metrics (the hook's own "epoch" key is dropped)
        assert t.channels[0].epoch_end_calls == [1]
        assert t.history[0]["sentinel/dummy/probe"] == 0.5
        assert "sentinel/dummy/epoch" not in t.history[0]
    finally:
        t.cleanup()


def test_replay_only_control(tmp_path):
    """Empty channel list degrades to the replay-only continuation arm."""
    t = _make_trainer(tmp_path / "ro", channels=[], warmup_steps=0)
    try:
        names = [g["name"] for g in t.optimizer.param_groups]
        assert names == ["backbone", "neck"]
        m = t.step(_replay_batch(seed=5))
        assert torch.isfinite(torch.tensor(m["total"]))
        assert t.ema.updates == 1
    finally:
        t.cleanup()
