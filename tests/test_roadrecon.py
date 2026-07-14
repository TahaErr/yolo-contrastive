"""Tests for the roadrecon module (pure reconstruction-based pretraining: B2/M2/M3).

CPU-only, offline: the detector is built from ``yolov8n.yaml`` (ships inside the
ultralytics package — no .pt download), error maps / mining use synthetic analytic
arrays, and the channel round-trips through the real ``AnchoredJointTrainer`` on a
tiny 64px model. Follows the conventions of ``tests/test_geoteach.py`` /
``tests/test_anchored.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from yolo_contrastive.anchored import (
    AnchoredJointTrainer,
    SentinelThresholds,
    load_for_finetune,
)
from yolo_contrastive.roadrecon import (
    AnomalyMineConfig,
    ReconDecoder,
    RoadReconChannel,
    RoadReconNet,
    RoadReconstructor,
    box_iou_xywh,
    build_scratch_detector,
    detection_runners,
    full_transplant_detection_runner,
    mine_image_boxes,
    mining_fidelity,
)
from yolo_contrastive.roadrecon.eval_runner import _load_full_ncmatched, _read_nc

# Tiny 64px probes cannot reach production sentinel thresholds — relax them.
RELAXED = SentinelThresholds(
    eff_rank_warn=0.0, eff_rank_abort=-1.0,
    cls_drift_warn=1e9, cls_drift_abort=1e9,
    cka_warn=-1.0, head_norm_growth_warn=1e9,
)


def _replay_batch(b: int = 2, s: int = 64, seed: int = 0):
    """Synthetic ultralytics-style single-class detection batch."""
    g = torch.Generator().manual_seed(seed)
    n = 2 * b
    return {
        "img": torch.rand(b, 3, s, s, generator=g),
        "batch_idx": torch.arange(b).repeat_interleave(2).float(),
        "cls": torch.zeros(n, 1),  # single pothole class
        "bboxes": torch.cat(
            [0.3 + 0.4 * torch.rand(n, 2, generator=g),
             0.05 + 0.2 * torch.rand(n, 2, generator=g)], dim=1),
    }


def _synth_error(seed: int = 0):
    """A realistic low-noise error map with one small planted anomaly + its road mask + GT box."""
    rng = np.random.default_rng(seed)
    err = rng.normal(0.05, 0.01, (64, 64)).astype(np.float32).clip(0)
    err[42:50, 26:36] += 0.8                      # 8x10 strong anomaly (~2% of road)
    road = np.zeros((64, 64), dtype=bool)
    road[30:62, 6:58] = True
    gt = np.array([[(26 + 36) / 2 / 64, (42 + 50) / 2 / 64, 10 / 64, 8 / 64]], np.float32)
    return err, road, gt


# ── import hygiene ────────────────────────────────────────────────────────────


def test_package_import_stays_light():
    import sys
    import yolo_contrastive  # noqa: F401 — lazy top-level, must not pull heavy deps
    import yolo_contrastive.roadrecon as rr

    assert hasattr(rr, "RoadReconstructor")
    assert hasattr(rr, "RoadReconChannel")
    # importing roadrecon must not drag in ultralytics or cv2 (E2)
    assert "ultralytics" not in sys.modules
    assert "cv2" not in sys.modules
    # top-level lazy export resolves
    assert yolo_contrastive.RoadReconstructor is rr.RoadReconstructor


# ── decoder ────────────────────────────────────────────────────────────────────


class TestReconDecoder:
    def test_shape_and_range(self):
        dec = ReconDecoder(in_channels=32, out_size=64, base=64, up_steps=3)
        out = dec(torch.rand(2, 32, 8, 8)).detach()
        assert out.shape == (2, 3, 64, 64)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_validation(self):
        with pytest.raises(ValueError):
            ReconDecoder(in_channels=0, out_size=64)
        with pytest.raises(ValueError):
            ReconDecoder(in_channels=8, out_size=64, up_steps=0)


# ── recon net ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def net():
    n = RoadReconNet(model="yolov8n.yaml", imgsz=64, tap_level="P3", device="cpu")
    yield n
    n.cleanup()


class TestRoadReconNet:
    def test_forward_shape_and_range(self, net):
        out = net(torch.rand(2, 3, 64, 64)).detach()
        assert out.shape == (2, 3, 64, 64)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_error_map_shape(self, net):
        em = net.error_map(torch.rand(2, 3, 64, 64))
        assert em.shape == (2, 64, 64)
        assert float(em.min()) >= 0.0

    def test_encode_has_grad(self, net):
        feat = net.encode(torch.rand(1, 3, 64, 64))
        assert feat.requires_grad
        assert feat.shape[-2:] == (8, 8)  # P3 stride 8 at 64px

    def test_bad_tap_level(self):
        with pytest.raises(ValueError):
            RoadReconNet(model="yolov8n.yaml", tap_level="P9", device="cpu")


# ── reconstructor ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def reconstructor():
    r = RoadReconstructor(model="yolov8n.yaml", imgsz=64, tap_level="P3", device="cpu",
                          n_mask_patches=3, mask_patch_frac=0.2)
    yield r
    r.cleanup()


class TestRoadReconstructor:
    def test_step_finite_and_grad(self, reconstructor):
        out = reconstructor._step(torch.rand(2, 3, 64, 64))
        loss = out["loss"]
        assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
        loss.backward()
        gsum = sum(p.grad.abs().sum().item()
                   for p in reconstructor.net.encoder.parameters() if p.grad is not None)
        assert gsum > 0  # gradient reaches the encoder
        reconstructor.net.zero_grad(set_to_none=True)

    def test_corrupt_is_road_centered(self, reconstructor):
        imgs = torch.rand(1, 3, 64, 64)
        corrupted = reconstructor._corrupt(imgs)
        changed = (corrupted != imgs).any(dim=1)[0].cpu()    # [H, W]
        road = reconstructor._road_mask().cpu()
        assert bool(changed.sum() > 0)                       # something was masked
        # patches are centered on road pixels; more corruption lands inside the
        # road than outside (edge spill is allowed by design).
        assert int((changed & road).sum()) > int((changed & ~road).sum())

    def test_no_mask_patches_is_identity(self):
        r = RoadReconstructor(model="yolov8n.yaml", imgsz=64, device="cpu", n_mask_patches=0)
        imgs = torch.rand(1, 3, 64, 64)
        assert torch.equal(r._corrupt(imgs), imgs)
        r.cleanup()

    def test_pathlist_dataset(self, dummy_images):
        from pathlib import Path
        from yolo_contrastive.roadrecon.reconstructor import _PathListDataset
        paths = [str(p) for p in Path(dummy_images).glob("*")][:3]
        ds = _PathListDataset(paths, imgsz=64)
        assert len(ds) == 3
        x = ds[0]
        assert x.shape == (3, 64, 64)
        assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
        with pytest.raises(ValueError):
            _PathListDataset([], imgsz=64)

    def test_save_load_full_roundtrip(self, tmp_path):
        from yolo_contrastive.roadrecon import load_reconstructor
        r = RoadReconstructor(model="yolov8n.yaml", imgsz=64, device="cpu", n_mask_patches=1)
        x = torch.rand(1, 3, 64, 64)
        e0 = r.error_map(x)
        path = r.save(str(tmp_path / "full.pt"))
        r2 = load_reconstructor(path, device="cpu")
        e1 = r2.error_map(x)
        assert torch.allclose(e0, e1, atol=1e-5)   # encoder+decoder weights round-trip exactly
        r.cleanup()
        r2.cleanup()

    @pytest.mark.slow
    def test_train_saves_backbone(self, dummy_images, tmp_path):
        r = RoadReconstructor(model="yolov8n.yaml", imgsz=64, device="cpu", n_mask_patches=2)
        out = r.train(str(dummy_images), epochs=1, batch_size=2, num_workers=0,
                      output=str(tmp_path / "rr.pt"), save_every=0, print_every=1)
        ckpt = torch.load(out, weights_only=False)
        assert "model_state_dict" in ckpt
        assert ckpt.get("extra", {}).get("type") == "roadrecon"  # top-level type = "ssl_pretrained"
        r.cleanup()


# ── mining ─────────────────────────────────────────────────────────────────────


class TestMining:
    def test_recovers_planted_anomaly(self):
        err, road, gt = _synth_error()
        boxes = mine_image_boxes(err, road, AnomalyMineConfig(z_thresh=3.0, min_box_area_px=16))
        assert len(boxes) >= 1
        pred = np.array([[b.cx, b.cy, b.w, b.h] for b in boxes], np.float32)
        fid = mining_fidelity(pred, gt, iou_thr=0.2)
        assert fid["recall"] == 1.0 and fid["tp"] >= 1
        assert all(b.cls == 0 for b in boxes)

    def test_empty_on_flat(self):
        road = np.zeros((64, 64), bool)
        road[30:62, 6:58] = True
        err = np.full((64, 64), 0.05, np.float32)            # no anomaly
        assert mine_image_boxes(err, road) == []

    def test_flooded_road_gated(self):
        # a huge "anomaly" over the whole road → global-failure trust gate → empty
        road = np.zeros((64, 64), bool)
        road[30:62, 6:58] = True
        err = np.full((64, 64), 0.05, np.float32)
        err[road] += (np.arange(road.sum()) % 2) * 1.0       # half the road is hot
        out = mine_image_boxes(err, road, AnomalyMineConfig(max_anomaly_area_frac=0.10))
        assert out == []

    def test_box_iou_xywh(self):
        a = np.array([[0.5, 0.5, 0.2, 0.2]], np.float32)
        assert box_iou_xywh(a, a)[0, 0] == pytest.approx(1.0, abs=1e-5)
        b = np.array([[0.9, 0.9, 0.2, 0.2]], np.float32)
        assert box_iou_xywh(a, b)[0, 0] == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.slow
    def test_mine_anomaly_labels_writes_dataset(self, dummy_images, tmp_path):
        from pathlib import Path
        r = RoadReconstructor(model="yolov8n.yaml", imgsz=64, device="cpu", n_mask_patches=2)
        imgs = [(p.stem, str(p)) for p in Path(dummy_images).glob("*")]
        out_root = tmp_path / "mined"
        # z_thresh low so the untrained net's noise yields at least some boxes for the smoke
        from yolo_contrastive.roadrecon import mine_anomaly_labels
        stats = mine_anomaly_labels(r, imgs, str(out_root),
                                    cfg=AnomalyMineConfig(z_thresh=2.0, min_box_area_px=16),
                                    imgsz=64, log_every=0)
        assert stats["scanned"] == len(imgs)
        assert (out_root / "data.yaml").exists()
        r.cleanup()


# ── channel + anchored integration (fast, non-slow) ──────────────────────────────


class TestRoadReconChannel:
    def test_build_loader_yields_img_dict(self, dummy_images):
        ch = RoadReconChannel(images_dir=str(dummy_images), imgsz=64)
        loader = ch.build_loader({"imgsz": 64, "batch": 2, "workers": 0, "device": "cpu"})
        batch = next(iter(loader))
        assert "img" in batch and batch["img"].shape[1:] == (3, 64, 64)

    def test_bad_tap_level(self):
        with pytest.raises(ValueError):
            RoadReconChannel(images_dir="x", tap_level="P9")


@pytest.fixture(scope="module")
def stepped_channel(tmp_path_factory):
    """Run 3 anchored steps with a RoadReconChannel on a 64px yaml model."""
    ch = RoadReconChannel(images_dir="(unused-in-step)", tap_level="P3", imgsz=64)
    t = AnchoredJointTrainer(
        model="yolov8n.yaml", channels=[ch], lambda_aux=1.0, epochs=1, imgsz=64,
        batch=2, warmup_steps=2, device="cpu", amp=False,
        output_dir=str(tmp_path_factory.mktemp("rr_anchored")),
        sentinel_thresholds=RELAXED,
    )
    head0 = next(p.detach().clone() for p in t.heads["roadrecon"].parameters())
    metrics = [t.step(_replay_batch(seed=i), {"roadrecon": {"img": torch.rand(2, 3, 64, 64)}})
               for i in (1, 2, 3)]
    head1 = next(p.detach().clone() for p in t.heads["roadrecon"].parameters())
    from types import SimpleNamespace
    obs = SimpleNamespace(trainer=t, metrics=metrics, head0=head0, head1=head1,
                          tmp=tmp_path_factory.getbasetemp())
    yield obs
    t.cleanup()


class TestAnchoredIntegration:
    def test_losses_finite(self, stepped_channel):
        for m in stepped_channel.metrics:
            assert torch.isfinite(torch.tensor(m["total"]))
            assert m["roadrecon/recon"] > 0
            assert m["replay/det_loss"] > 0

    def test_decoder_head_trains(self, stepped_channel):
        assert not torch.allclose(stepped_channel.head0, stepped_channel.head1)

    def test_head_not_leaked_into_model(self, stepped_channel):
        t = stepped_channel.trainer
        model_ids = {id(p) for p in t.model.parameters()}
        assert all(id(p) not in model_ids for p in t.heads["roadrecon"].parameters())

    def test_export_full_roundtrip(self, stepped_channel):
        t = stepped_channel.trainer
        path = t.export(path=str(stepped_channel.tmp / "rr_full.pt"))
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["transplant"] == "full"
        yolo = load_for_finetune(path, base="yolov8n.yaml", verbose=False)
        # decoder head must NOT be in the exported detector
        assert not any("decoder" in k for k in yolo.model.state_dict())


# ── eval runner wiring (light) ───────────────────────────────────────────────────


class TestNcConsistency:
    def test_build_scratch_detector_nc(self):
        m = build_scratch_detector(nc=1, device="cpu")
        assert int(m.model[-1].nc) == 1

    def test_read_nc(self, tmp_path):
        y = tmp_path / "d.yaml"
        y.write_text("nc: 1\nnames: ['pothole']\n", encoding="utf-8")
        assert _read_nc(str(y)) == 1

    def test_nc1_anchored_export_transplants_head(self, tmp_path):
        # M3 end-to-end wiring: nc=1 scratch detector → anchored step → export →
        # nc-matched whole-detector transplant (the head must NOT be dropped).
        det = build_scratch_detector(nc=1, device="cpu")
        ch = RoadReconChannel(images_dir="x", tap_level="P3", imgsz=64)
        t = AnchoredJointTrainer(
            model=det, channels=[ch], lambda_aux=1.0, epochs=1, imgsz=64, batch=2,
            warmup_steps=1, device="cpu", amp=False, output_dir=str(tmp_path),
            sentinel_thresholds=RELAXED)
        t.step(_replay_batch(seed=1), {"roadrecon": {"img": torch.rand(2, 3, 64, 64)}})
        path = t.export(path=str(tmp_path / "nc1_full.pt"))
        yolo = _load_full_ncmatched(path, "yolov8n.yaml", nc=1, device="cpu")
        assert int(yolo.model.model[-1].nc) == 1
        src, dst = t.ema.ema.state_dict(), yolo.model.state_dict()
        # every EMA tensor (incl. the nc=1 head) transplants with matching shape
        assert all(k in dst and src[k].shape == dst[k].shape for k in src)
        t.cleanup()


class TestReviewFixes:
    def test_nc_mismatch_transplant_raises(self, tmp_path):
        # S1: an nc=1 checkpoint into an nc=4 arch would silently drop the head → refuse.
        det = build_scratch_detector(nc=1, device="cpu")
        p = tmp_path / "nc1.pt"
        torch.save({"model_state_dict": det.state_dict(), "transplant": "full"}, str(p))
        with pytest.raises(ValueError, match="nc mismatch"):
            _load_full_ncmatched(str(p), "yolov8n.yaml", nc=4, device="cpu")

    def test_mining_fidelity_matches_free_gt(self):
        # S2: a second prediction whose argmax GT is already claimed must match the
        # still-free GT above threshold (greedy argmax would under-count recall).
        gt = np.array([[0.30, 0.5, 0.30, 0.30], [0.55, 0.5, 0.30, 0.30]], np.float32)
        pred = np.array([[0.30, 0.5, 0.30, 0.30], [0.42, 0.5, 0.30, 0.30]], np.float32)
        fid = mining_fidelity(pred, gt, iou_thr=0.2)
        assert fid["tp"] == 2 and fid["recall"] == 1.0

    def test_runner_is_pure_and_passes_device_through(self, tmp_path, monkeypatch):
        # M1: never fall back to hp's COCO base; M2: device passed through, not str()'d.
        import yolo_contrastive.roadrecon.eval_runner as er
        y = tmp_path / "d.yaml"
        y.write_text("nc: 1\nnames: ['pothole']\n", encoding="utf-8")
        captured = {}

        class _Stop(Exception):
            pass

        def fake_load(ckpt, base, nc, device, verbose=False):
            captured["base"], captured["device"] = base, device
            raise _Stop()

        monkeypatch.setattr(er, "_load_full_ncmatched", fake_load)
        cell = {"method": {"name": "roadrecon_m3", "backbone_ckpt": str(tmp_path / "ck.pt")},
                "dataset": {"name": "fold0", "data_yaml": str(y)}}
        hp = {"base_model": "yolov8n.pt", "device": 0}   # DEFAULT_HP-style COCO + int device
        with pytest.raises(_Stop):
            full_transplant_detection_runner(cell, hp)
        assert captured["base"] == "yolov8n.yaml"   # M1: purity — not hp's COCO .pt
        assert captured["device"] == 0              # M2: int device, not "0"


class TestEvalRunner:
    def test_detection_runners_mapping(self):
        r = detection_runners()
        assert r["detection"] is full_transplant_detection_runner

    def test_baseline_without_ckpt_delegates(self, monkeypatch):
        # a method with no backbone_ckpt must fall through to the stock _run_detection
        called = {}

        def fake_run_detection(cell, hp):
            called["hit"] = True
            return {"metric": "mAP50-95", "metric_value": 0.0}

        import yolo_contrastive.eval.run_matrix as rm
        monkeypatch.setattr(rm, "_run_detection", fake_run_detection)
        out = full_transplant_detection_runner(
            {"method": {"name": "coco"}, "dataset": {"data_yaml": "x.yaml"}}, {})
        assert called.get("hit") and out["metric_value"] == 0.0
