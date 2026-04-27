"""Tests for MultiScaleProjectionHead and infer_in_channels."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_contrastive.dense import (
    MultiScaleProjectionHead,
    MultiScaleFeatureTap,
    infer_in_channels,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _yolov8n_channels():
    """Realistic YOLOv8n FPN channel counts."""
    return {"P3": 128, "P4": 256, "P5": 512}


def _make_features(channels: dict, B: int = 2,
                   sizes: dict = None) -> dict:
    if sizes is None:
        sizes = {"P3": 16, "P4": 8, "P5": 4}
    return {lv: torch.randn(B, channels[lv], sizes[lv], sizes[lv])
            for lv in channels}


# ── construction ─────────────────────────────────────────────────────────


class TestConstruction:
    def test_basic(self):
        head = MultiScaleProjectionHead(_yolov8n_channels(), out_dim=256)
        assert isinstance(head, nn.Module)
        assert set(head.towers.keys()) == {"P3", "P4", "P5"}

    def test_default_hidden(self):
        head = MultiScaleProjectionHead(_yolov8n_channels(), out_dim=256)
        assert head.hidden_dim == 512  # 2 * out_dim

    def test_explicit_hidden(self):
        head = MultiScaleProjectionHead(
            _yolov8n_channels(), out_dim=128, hidden_dim=384,
        )
        assert head.hidden_dim == 384

    def test_no_bn(self):
        head = MultiScaleProjectionHead(
            _yolov8n_channels(), out_dim=128, use_bn=False,
        )
        # No BatchNorm2d in any tower
        for lv, tower in head.towers.items():
            for module in tower:
                assert not isinstance(module, nn.BatchNorm2d)

    def test_with_bn(self):
        head = MultiScaleProjectionHead(_yolov8n_channels(), out_dim=128, use_bn=True)
        for lv, tower in head.towers.items():
            has_bn = any(isinstance(m, nn.BatchNorm2d) for m in tower)
            assert has_bn

    def test_invalid_in_channels(self):
        with pytest.raises(ValueError, match="empty"):
            MultiScaleProjectionHead({}, out_dim=128)
        with pytest.raises(ValueError, match="positive"):
            MultiScaleProjectionHead({"P3": 0}, out_dim=128)
        with pytest.raises(ValueError, match="positive"):
            MultiScaleProjectionHead({"P3": -1}, out_dim=128)

    def test_invalid_out_dim(self):
        with pytest.raises(ValueError, match="out_dim"):
            MultiScaleProjectionHead({"P3": 128}, out_dim=0)


# ── forward shapes ───────────────────────────────────────────────────────


class TestForwardShapes:
    def test_basic_forward(self):
        head = MultiScaleProjectionHead(_yolov8n_channels(), out_dim=64)
        feats = _make_features(_yolov8n_channels(), B=2)
        out = head(feats)
        assert set(out.keys()) == {"P3", "P4", "P5"}
        for lv, t in out.items():
            assert t.shape[0] == 2
            assert t.shape[1] == 64
            assert t.shape[2:] == feats[lv].shape[2:]  # spatial preserved

    def test_single_level(self):
        head = MultiScaleProjectionHead({"P3": 128}, out_dim=32)
        feats = {"P3": torch.randn(2, 128, 8, 8)}
        out = head(feats)
        assert out["P3"].shape == (2, 32, 8, 8)

    def test_different_batch_sizes(self):
        head = MultiScaleProjectionHead(_yolov8n_channels(), out_dim=64)
        for B in (1, 2, 8):
            feats = _make_features(_yolov8n_channels(), B=B)
            out = head(feats)
            for t in out.values():
                assert t.shape[0] == B

    def test_output_not_normalized(self):
        """Caller responsibility — head returns raw embeddings."""
        head = MultiScaleProjectionHead({"P3": 16}, out_dim=8)
        # Produce features with large magnitudes
        feats = {"P3": torch.randn(2, 16, 4, 4) * 10}
        out = head(feats)
        norms = out["P3"].flatten(2).norm(dim=1)  # [B, H*W]
        # Norms should NOT be ≈ 1 (we didn't normalize)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=0.1)


# ── input validation ─────────────────────────────────────────────────────


class TestInputValidation:
    def test_missing_level_raises(self):
        head = MultiScaleProjectionHead(_yolov8n_channels(), out_dim=64)
        feats = {"P3": torch.randn(2, 128, 8, 8)}  # missing P4, P5
        with pytest.raises(ValueError, match="Missing levels"):
            head(feats)

    def test_wrong_channel_count(self):
        head = MultiScaleProjectionHead({"P3": 128}, out_dim=64)
        feats = {"P3": torch.randn(2, 64, 8, 8)}  # wrong: 64 vs expected 128
        with pytest.raises(ValueError, match="channel mismatch"):
            head(feats)

    def test_wrong_feature_dim(self):
        head = MultiScaleProjectionHead({"P3": 128}, out_dim=64)
        feats = {"P3": torch.randn(2, 128, 8)}  # 3D not 4D
        with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
            head(feats)


# ── gradient flow ────────────────────────────────────────────────────────


class TestGradient:
    def test_grad_flows_through_head(self):
        head = MultiScaleProjectionHead(_yolov8n_channels(), out_dim=64)
        feats = _make_features(_yolov8n_channels(), B=2)
        # Make features require grad (simulate backbone output)
        feats = {lv: t.requires_grad_(True) for lv, t in feats.items()}
        out = head(feats)
        loss = sum(t.mean() for t in out.values())
        loss.backward()

        # Head params have grad
        for p in head.parameters():
            assert p.grad is not None

        # Input features have grad
        for lv, t in feats.items():
            assert t.grad is not None
            assert t.grad.abs().sum() > 0


# ── state dict ───────────────────────────────────────────────────────────


class TestStateDict:
    def test_roundtrip(self):
        h1 = MultiScaleProjectionHead(_yolov8n_channels(), out_dim=64)
        h2 = MultiScaleProjectionHead(_yolov8n_channels(), out_dim=64)

        # Verify initial state is different
        feats = _make_features(_yolov8n_channels(), B=1)
        out1_before = h1(feats)
        out2_before = h2(feats)
        assert not torch.allclose(out1_before["P3"], out2_before["P3"])

        # Roundtrip
        h2.load_state_dict(h1.state_dict())
        h1.eval(); h2.eval()  # disable BN running-stats updates for determinism
        out1_after = h1(feats)
        out2_after = h2(feats)
        assert torch.allclose(out1_after["P3"], out2_after["P3"], atol=1e-5)


# ── device transfer ──────────────────────────────────────────────────────


class TestDeviceTransfer:
    def test_to_cpu(self):
        head = MultiScaleProjectionHead({"P3": 16}, out_dim=8)
        head.to("cpu")
        feats = {"P3": torch.randn(1, 16, 4, 4)}
        out = head(feats)
        assert out["P3"].device.type == "cpu"


# ── repr ─────────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_config(self):
        head = MultiScaleProjectionHead(_yolov8n_channels(), out_dim=128)
        r = repr(head)
        assert "MultiScaleProjectionHead" in r
        assert "out_dim=128" in r


# ── infer_in_channels ────────────────────────────────────────────────────


class TestInferInChannels:
    def test_basic_inference(self):
        # Build a fake "YOLO-like" Sequential with known channels
        # 23 layers, layer 15→128 ch, layer 18→256 ch, layer 21→512 ch
        layers = []
        for i in range(23):
            if i == 14:
                layers.append(nn.Conv2d(3, 128, 1))   # next-layer input becomes 128
            elif i == 17:
                layers.append(nn.Conv2d(128, 256, 1))
            elif i == 20:
                layers.append(nn.Conv2d(256, 512, 1))
            else:
                # passthrough
                layers.append(nn.Identity())
        seq = nn.Sequential(*layers)

        tap = MultiScaleFeatureTap(seq)
        tap.setup()
        try:
            channels = infer_in_channels(seq, tap, imgsz=8)
        finally:
            tap.close()

        # Layer 15 = conv(3→128) output → P3=128
        # Layer 18 = conv(128→256) output → P4=256
        # Layer 21 = conv(256→512) output → P5=512
        # But our "passthrough" Identity doesn't change channels, and our
        # inserted Conv2d at layer 14 takes 3-ch input and produces 128 —
        # which propagates through Identity layer 15. Hooks fire AT layer
        # 15 which is Identity, so P3 = 128 (the channels coming through).
        assert channels == {"P3": 128, "P4": 256, "P5": 512}

    def test_works_after_no_grad_context(self):
        seq = nn.Sequential(*[nn.Conv2d(3, 3, 1) for _ in range(23)])
        tap = MultiScaleFeatureTap(seq)
        tap.setup()
        try:
            ch = infer_in_channels(seq, tap, imgsz=4)
            assert all(isinstance(v, int) for v in ch.values())
        finally:
            tap.close()


# ── end-to-end with real-ish pipeline ───────────────────────────────────


class TestEndToEnd:
    def test_with_dense_loss_pipeline(self):
        """Probe → tap → head → dense_loss runs end-to-end."""
        from yolo_contrastive.dense import (
            dense_ntxent_loss, coords_to_feature_map, SpatialTwoViewAugmentation,
        )

        torch.manual_seed(0)
        # Tiny "encoder": 23-layer Sequential preserving 3-ch
        encoder = nn.Sequential(*[
            nn.Conv2d(3, 3, 3, padding=1) for _ in range(23)
        ])

        tap = MultiScaleFeatureTap(encoder)
        tap.setup()

        try:
            in_ch = infer_in_channels(encoder, tap, imgsz=32)
            head = MultiScaleProjectionHead(in_ch, out_dim=16)

            aug = SpatialTwoViewAugmentation(out_size=(32, 32), seed=42)
            imgs = torch.rand(2, 3, 32, 32)
            views = aug(imgs)

            # Forward view1 through encoder → tap → head
            _ = encoder(views.view1)
            f1 = head(tap.get_features())
            tap.clear()

            _ = encoder(views.view2)
            f2 = head(tap.get_features())

            # Pick P3 for a single-level dense_loss check
            v1 = F.normalize(f1["P3"], dim=1)
            v2 = F.normalize(f2["P3"], dim=1).detach()
            qc = coords_to_feature_map(views.coords1, v1.shape[2], v1.shape[3])
            kc = coords_to_feature_map(views.coords2, v2.shape[2], v2.shape[3])

            loss, info = dense_ntxent_loss(v1, v2, qc, kc, n_query=16)
            assert torch.isfinite(loss).item()
            loss.backward()
        finally:
            tap.close()
