"""ScaleRealChannel contract tests — stub-backbone unit level + real YOLO tap.

CPU-only, offline. The stub path exercises the AuxChannel contract (attach /
loss / sentinels / guards) on a tiny stride-16 conv model; the integration
test builds the real AnchoredJointTrainer from yolov8n.yaml (ships inside
ultralytics, no download) and runs two optimizer steps with real P4 RoIAlign.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from yolo_contrastive.scalereal.channel import (
    ScaleRealChannel,
    ScaleRealPoolDataset,
    scalereal_collate,
)
from yolo_contrastive.scalereal.config import ScaleRealConfig


# ── stub model + taps (stride 16 like P4) ────────────────────────────────────


class _TapStub:
    def __init__(self):
        self.f = None

    def clear(self):
        self.f = None

    def get_features(self):
        if self.f is None:
            raise RuntimeError("no forward captured")
        return {"P4": self.f}


class _StubBackbone(nn.Module):
    """3 -> 16 channels at stride 16 (two stride-4 convs), feeding the tap."""

    def __init__(self, tap, stride2: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 4, stride=4)
        self.conv2 = nn.Conv2d(8, 16, stride2, stride=stride2)
        self._tap = tap

    def forward(self, x):
        y = self.conv2(torch.relu(self.conv1(x)))
        self._tap.f = y
        return y


def _make_batch(b=2, s=64, m=8, seed=0, theta=None):
    g = torch.Generator().manual_seed(seed)

    def _boxes():
        xy = 0.05 + 0.45 * torch.rand(m, 2, generator=g)
        wh = 0.30 + 0.15 * torch.rand(m, 2, generator=g)
        return torch.cat([xy, xy + wh], dim=1).clamp(0, 1)

    if theta is None:
        theta = torch.zeros(b, 2, 3)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
    sign = torch.where(torch.rand(m, generator=g) < 0.5, 1.0, -1.0)
    return {
        "img": torch.rand(b, 3, s, s, generator=g),
        "pair_batch_idx": torch.randint(0, b, (m,), generator=g),
        "boxes_a": _boxes(),
        "boxes_b": _boxes(),
        "log_r": sign * (0.5 + 1.2 * torch.rand(m, generator=g)),
        "aug_theta": theta,
        "image_id": [f"img_{i:03d}" for i in range(b)],
    }


def _attached_channel(**cfg_kw):
    cfg = ScaleRealConfig(**cfg_kw)
    tap = _TapStub()
    model = _StubBackbone(tap)
    ch = ScaleRealChannel(p4_channels=16, config=cfg)
    heads = ch.attach(model, tap)
    return ch, model, tap, heads


# ── contract: attach / loss / logs ───────────────────────────────────────────


class TestChannelContract:
    def test_attach_returns_modulelist_with_grads(self):
        ch, _, _, heads = _attached_channel()
        assert isinstance(heads, nn.ModuleList)
        assert len(heads) == 4  # descriptor, scale head, projector, predictor
        params = list(heads.parameters())
        assert params, "channel has no trainable parameters"
        assert all(p.requires_grad for p in params)  # E5

    def test_loss_terms_and_logs(self):
        ch, model, tap, _ = _attached_channel()
        batch = _make_batch()
        _ = model(batch["img"])
        terms = ch.loss(batch, tap)
        assert set(terms) == {"l_scale", "l_inv"}
        for v in terms.values():
            assert torch.isfinite(v)
            assert v.grad_fn is not None
        assert set(ch.last_logs) == {"n_pairs", "l_scale", "l_inv", "sign_acc", "pred_std"}
        assert ch.last_logs["n_pairs"] == 8

    def test_lambda_inv_zero_kills_inv_term(self):
        ch, model, tap, _ = _attached_channel(lambda_inv=0.0)
        batch = _make_batch()
        _ = model(batch["img"])
        terms = ch.loss(batch, tap)
        assert float(terms["l_inv"].detach()) == 0.0
        assert float(terms["l_scale"].detach()) > 0.0

    def test_zero_pair_batch_graph_connected(self):
        ch, model, tap, _ = _attached_channel()
        batch = _make_batch(m=8)
        for k in ("pair_batch_idx", "boxes_a", "boxes_b", "log_r"):
            batch[k] = batch[k][:0]
        _ = model(batch["img"])
        terms = ch.loss(batch, tap)
        total = sum(terms.values())
        assert float(total.detach()) == 0.0
        assert total.grad_fn is not None
        total.backward()  # must not raise
        assert ch.last_logs["n_pairs"] == 0

    def test_pair_cap_enforced(self):
        ch, model, tap, _ = _attached_channel(max_pairs_per_batch=4)
        batch = _make_batch(m=16)
        _ = model(batch["img"])
        _ = ch.loss(batch, tap)
        assert ch.last_logs["n_pairs"] == 4

    def test_gradients_flow_to_backbone(self):
        ch, model, tap, _ = _attached_channel()
        batch = _make_batch()
        _ = model(batch["img"])
        terms = ch.loss(batch, tap)
        sum(terms.values()).backward()
        assert model.conv1.weight.grad is not None
        assert float(model.conv1.weight.grad.abs().sum()) > 0


# ── runtime guards ───────────────────────────────────────────────────────────


class TestGuards:
    def test_aspect_distortion_assert(self):
        ch, model, tap, _ = _attached_channel()
        theta = torch.zeros(2, 2, 3)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 2.0  # anisotropy 2.0 > 1.2
        batch = _make_batch(theta=theta)
        _ = model(batch["img"])
        with pytest.raises(ValueError, match="aspect distortion"):
            ch.loss(batch, tap)

    def test_wrong_stride_assert(self):
        tap = _TapStub()
        model = _StubBackbone(tap, stride2=2)  # total stride 8, not 16
        ch = ScaleRealChannel(p4_channels=16)
        ch.attach(model, tap)
        batch = _make_batch()
        _ = model(batch["img"])
        with pytest.raises(ValueError, match="stride"):
            ch.loss(batch, tap)


# ── overfit sanity ───────────────────────────────────────────────────────────


class TestOverfit:
    def test_50_steps_halve_the_loss(self):
        torch.manual_seed(0)
        ch, model, tap, heads = _attached_channel()
        batch = _make_batch(seed=3)
        opt = torch.optim.Adam(
            list(heads.parameters()) + list(model.parameters()), lr=5e-3
        )
        first = None
        for _ in range(50):
            opt.zero_grad()
            tap.clear()
            _ = model(batch["img"])
            terms = ch.loss(batch, tap)
            loss = sum(terms.values())
            loss.backward()
            opt.step()
            if first is None:
                first = float(loss.detach())
        last = float(loss.detach())
        assert last < 0.5 * first, f"loss {first:.4f} -> {last:.4f} (<50% drop)"


# ── sentinels incl. the row-shortcut probe ───────────────────────────────────


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestSentinels:
    def test_epoch_sentinels_keys_and_records(self):
        ch, model, tap, _ = _attached_channel()
        ch.set_probe_batch(_make_batch(seed=5, m=32))
        rec = ch.epoch_sentinels(epoch=1)
        expected = {
            "epoch", "n_probe_pairs", "probe_smooth_l1", "sign_acc", "spearman",
            "pred_std", "r2_head", "r2_row", "r2_size", "row_probe_delta_r2",
            "flag_pred_collapse", "flag_row_shortcut",
        }
        assert expected == set(rec)
        assert rec["n_probe_pairs"] == 32
        assert math.isfinite(rec["row_probe_delta_r2"])
        assert len(ch.sentinel_records) == 1
        assert model.training  # train mode restored

    def test_row_shortcut_flag_after_deadline(self):
        """A constant head cannot beat the row regressor -> flagged by ep4."""
        ch, model, tap, _ = _attached_channel()
        # force a constant scale head: zero weights -> pred == 0 for all pairs
        for p in ch.scale_head.parameters():
            nn.init.zeros_(p)
        ch.set_probe_batch(_make_batch(seed=6, m=32))
        with pytest.warns(RuntimeWarning, match="row|collapsing"):
            rec = ch.epoch_sentinels(epoch=4)
        assert rec["flag_row_shortcut"] == 1.0
        assert rec["flag_pred_collapse"] == 1.0  # constant pred -> std 0

    def test_no_probe_batch_returns_empty(self):
        ch, _, _, _ = _attached_channel()
        assert ch.epoch_sentinels(epoch=1) == {}

    def test_on_epoch_end_trainer_hook(self):
        ch, model, tap, _ = _attached_channel()
        assert ch.on_epoch_end(1) == {}        # no probe, no loader -> silent no-op
        ch.set_probe_batch(_make_batch(seed=7, m=16))
        rec = ch.on_epoch_end(2)               # R9: called by the trainer
        assert rec["epoch"] == 2.0 and rec["n_probe_pairs"] == 16
        assert "r2_size" in rec and "r2_row" in rec


# ── loader: files -> batches (R5 inside the dataset) ────────────────────────


def _loader_fixture(tmp_path, n_images=3):
    """Synthetic scene images + a hand-written pair manifest."""
    from yolo_contrastive.scalereal.pair_manifest import (
        PairIndex,
        PairRecord,
        append_pairs,
        is_probe_image,
    )
    from yolo_contrastive.scalereal.synthetic import materialize_scene, two_class_scene

    import pandas as pd

    scene = two_class_scene(h=256, w=256)
    ids = [f"img_{i:03d}" for i in (0, 1, 2)][:n_images]
    assert not any(is_probe_image(i) for i in ids)  # all training-eligible
    rows, manifest_rows = [], []
    for image_id in ids:
        path = materialize_scene(scene, tmp_path / "images", image_id)
        manifest_rows.append({"image_id": image_id, "materialized_path": path})
        for k, (a, b) in enumerate([(0, 1), (2, 3)]):  # same-class square pairs
            ba, bb = scene.boxes_norm[a], scene.boxes_norm[b]
            za, zb = scene.squares[a].z_m, scene.squares[b].z_m
            rows.append(PairRecord(
                image_id=image_id, pair_id=f"{image_id}#p{k:03d}",
                box_a_x1=ba[0], box_a_y1=ba[1], box_a_x2=ba[2], box_a_y2=ba[3],
                box_b_x1=bb[0], box_b_y1=bb[1], box_b_x2=bb[2], box_b_y2=bb[3],
                log_r=math.log(za / zb), z_a=za, z_b=zb, sim=0.99,
                texture_a=0.2, texture_b=0.2, depth_iqr_a=0.0, depth_iqr_b=0.0,
            ))
    pairs_path = tmp_path / "pairs.parquet"
    append_pairs(pairs_path, rows)
    manifest = pd.DataFrame(manifest_rows)
    return pairs_path, manifest, PairIndex.from_parquet(pairs_path)


class TestLoader:
    def test_build_loader_yields_contract_batches(self, tmp_path):
        pairs_path, manifest, _ = _loader_fixture(tmp_path)
        cfg = ScaleRealConfig(min_patch_px=8.0)  # 256px scenes at 96px views
        ch = ScaleRealChannel(pairs_path=str(pairs_path), pool_manifest=manifest,
                              config=cfg, loader_seed=0)
        loader = ch.build_loader({"imgsz": 96, "batch": 2, "workers": 0,
                                  "device": "cpu"})
        batch = next(iter(loader))
        assert batch["img"].shape == (2, 3, 96, 96)
        assert batch["img"].min() >= 0 and batch["img"].max() <= 1
        assert batch["aug_theta"].shape == (2, 2, 3)
        m = batch["log_r"].shape[0]
        assert batch["boxes_a"].shape == (m, 4)
        assert batch["boxes_b"].shape == (m, 4)
        assert batch["pair_batch_idx"].shape == (m,)
        assert (batch["boxes_a"] >= 0).all() and (batch["boxes_a"] <= 1).all()
        assert len(batch["image_id"]) == 2
        # labels are aug-invariant: every survivor's log_r is one of the
        # two mined values (up to sign — both pair orders appear)
        if m:
            allowed = {round(abs(math.log(5.0 / 15.0)), 4),
                       round(abs(math.log(4.0 / 20.0)), 4)}
            got = {round(abs(float(v)), 4) for v in batch["log_r"]}
            assert got <= allowed

    def test_dataset_identity_mode_keeps_all_pairs(self, tmp_path):
        pairs_path, manifest, index = _loader_fixture(tmp_path, n_images=1)
        cfg = ScaleRealConfig(min_patch_px=8.0)
        records = [{"image_id": "img_000",
                    "path": str(tmp_path / "images" / "img_000.png")}]
        ds = ScaleRealPoolDataset(records, index, imgsz=96, cfg=cfg, augment=False)
        item = ds[0]
        assert item["log_r"].shape[0] == 2  # identity aug: nothing clipped
        # R5: identity theta -> boxes equal the manifest boxes exactly
        t = index.prepare_targets("img_000")
        assert torch.allclose(item["boxes_a"], torch.from_numpy(t["boxes_a"].copy()),
                              atol=1e-6)

    def test_collate_caps_pairs_uniformly(self):
        items = []
        for b in range(2):
            items.append({
                "img": torch.rand(3, 64, 64),
                "boxes_a": torch.rand(10, 4),
                "boxes_b": torch.rand(10, 4),
                "log_r": torch.randn(10),
                "theta": torch.eye(2, 3),
                "image_id": f"i{b}",
            })
        batch = scalereal_collate(items, max_pairs=6)
        assert batch["log_r"].shape[0] == 6
        assert batch["pair_batch_idx"].shape[0] == 6

    def test_missing_data_raises(self):
        ch = ScaleRealChannel()
        with pytest.raises(ValueError, match="pairs_path"):
            ch.build_loader({"imgsz": 64, "batch": 2, "workers": 0, "device": "cpu"})


# ── integration: real yolov8n.yaml trainer, two anchored steps ──────────────


def _replay_batch(b=2, s=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = 2 * b
    return {
        "img": torch.rand(b, 3, s, s, generator=g),
        "batch_idx": torch.arange(b).repeat_interleave(2).float(),
        "cls": torch.randint(0, 80, (n, 1), generator=g).float(),
        "bboxes": torch.cat(
            [0.3 + 0.4 * torch.rand(n, 2, generator=g),
             0.05 + 0.2 * torch.rand(n, 2, generator=g)], dim=1),
    }


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_anchored_trainer_integration(tmp_path):
    """Real yolov8n (yaml init, offline) + real P4 tap + RoIAlign, 2 steps."""
    from yolo_contrastive.anchored import AnchoredJointTrainer, SentinelThresholds

    relaxed = SentinelThresholds(
        eff_rank_warn=0.0, eff_rank_abort=-1.0,
        cls_drift_warn=1e9, cls_drift_abort=1e9,
        cka_warn=-1.0, head_norm_growth_warn=1e9,
    )
    ch = ScaleRealChannel()  # p4_channels inferred from the live tap
    trainer = AnchoredJointTrainer(
        model="yolov8n.yaml", channels=[ch], lambda_aux=0.3,
        epochs=1, imgsz=64, batch=2, warmup_steps=1, device="cpu", amp=False,
        output_dir=str(tmp_path), sentinel_thresholds=relaxed,
    )
    try:
        assert trainer.tap_channels["P4"] == ch.descriptor.in_channels == 128
        head_w = ch.scale_head.net[0].weight
        w0 = head_w.detach().clone()
        for i in (1, 2):
            m = trainer.step(_replay_batch(seed=i), {"scalereal": _make_batch(seed=i)})
            assert math.isfinite(m["scalereal/l_scale"])
            assert math.isfinite(m["scalereal/l_inv"])
            assert m["total"] == pytest.approx(
                m["replay/det_loss"] + m["scalereal/total"], rel=1e-5
            )
        assert not torch.allclose(head_w.detach(), w0)  # heads trained
        # channel sentinels on a fixed probe batch through the REAL model
        ch.set_probe_batch(_make_batch(seed=9, m=16))
        rec = ch.epoch_sentinels(epoch=1)
        assert rec["n_probe_pairs"] == 16
        # heads are NOT part of the exported detector (R8)
        path = trainer.export(path=str(tmp_path / "full.pt"))
        ckpt = torch.load(path, weights_only=True)
        assert all("descriptor" not in k and "scale_head" not in k
                   for k in ckpt["model_state_dict"])
    finally:
        trainer.cleanup()
