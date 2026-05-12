"""Risk 16 regression tests — PyTorch InferenceMode crash on EMA sync.

Documents and guards the fix described in WORK_PLAN §10.22 + §10.23.

Bug pattern (Yol 3 smoke, PT 2.x):
    Ultralytics' internal validator runs forward passes under
    torch.inference_mode(). After such a pass, parameters or buffers on
    the EMA copy can carry the InferenceMode flag. A subsequent plain
    self.ema.ema.load_state_dict(self.model.state_dict()) tries to
    in-place update those tensors and raises:

        RuntimeError: Inplace update to inference tensor outside
                      InferenceMode is not allowed.

Fix: load_state_dict(..., assign=True) — PyTorch 2.1+. Replaces target
tensors by reference instead of in-place mutation, so the InferenceMode
flag on the old tensors becomes irrelevant.

These three tests are deliberately split:
  1. baseline_crash_reproduces   — confirms the bug is real and the fix
                                   is solving something (paper evidence)
  2. assign_true_resolves_crash  — confirms the fix mechanism works on
                                   the same minimal reproducer
  3. safe_ema_sync_uses_assign_true — code-level sentinel: if someone
                                   removes assign=True from the helper,
                                   this test fails immediately, before
                                   the bug resurfaces in production

The first two test PyTorch behaviour (not our code) and would still pass
if our helper were deleted. The third closes that loop.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from yolo_contrastive.finetune.trainer import FinetuneDetectionTrainer


def _make_target_with_tainted_buffers() -> nn.Module:
    """Return a small Conv+BN module whose buffers carry the InferenceMode flag.

    We rebuild the buffers inside a torch.inference_mode() context so the
    fresh tensors inherit the flag — this mirrors what happens in
    Ultralytics when validation runs under inference_mode and the EMA
    copy ends up with tainted BN running stats.
    """
    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))
    with torch.inference_mode():
        for name, buf in list(m.named_buffers()):
            *path, leaf = name.split(".")
            parent = m
            for p in path:
                parent = getattr(parent, p)
            parent._buffers[leaf] = torch.zeros_like(buf)
    return m


class TestRisk16:
    """Three-step regression suite for the InferenceMode crash + fix."""

    def test_baseline_crash_reproduces(self):
        """Plain load_state_dict still crashes on tainted target buffers.

        If this ever stops raising, either PyTorch changed its
        InferenceMode semantics (good — then re-evaluate whether the fix
        is still needed) or our reproducer drifted (bad — fix the
        reproducer before trusting the fix).
        """
        target = _make_target_with_tainted_buffers()
        source = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))

        with pytest.raises(RuntimeError, match="inference"):
            target.load_state_dict(source.state_dict())

    def test_assign_true_resolves_crash(self):
        """assign=True on the same scenario succeeds and transfers state.

        Validates the fix mechanism in isolation, independent of our
        helper. Post-load invariants checked:
          - no RuntimeError raised
          - parameters now equal to source (real transfer, not no-op)
          - target buffers no longer carry the InferenceMode flag
          - forward pass still works (model not corrupted)
        """
        target = _make_target_with_tainted_buffers()
        source = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))
        # Give source distinctive values so the transfer is observable
        with torch.no_grad():
            for p in source.parameters():
                p.data.fill_(0.42)

        # Should not raise
        target.load_state_dict(source.state_dict(), assign=True)

        # Params transferred
        for (n1, p1), (n2, p2) in zip(
            target.named_parameters(), source.named_parameters()
        ):
            assert torch.equal(p1, p2), f"param {n1} not transferred"

        # No InferenceMode flag remains on target
        for n, t in target.state_dict().items():
            assert not t.is_inference(), f"InferenceMode flag still on {n}"

        # Forward still works and produces finite output
        target.eval()
        with torch.no_grad():
            out = target(torch.randn(1, 3, 16, 16))
        assert out.shape == (1, 8, 14, 14)
        assert torch.isfinite(out).all()

    def test_safe_ema_sync_uses_assign_true(self):
        """Code-level sentinel: the helper must contain assign=True.

        Source-level check so the test does not require an actual
        Ultralytics training context to run. If a refactor accidentally
        drops the fix, this fires before any Yol 3 / Faz 5 detection run
        wastes hours only to crash at final_eval.
        """
        helper_src = inspect.getsource(FinetuneDetectionTrainer._safe_ema_sync)
        assert "assign=True" in helper_src, (
            "Risk 16 fix removed from _safe_ema_sync — restore "
            "load_state_dict(..., assign=True) per WORK_PLAN §10.23."
        )

        setup_src = inspect.getsource(FinetuneDetectionTrainer._setup_train)
        save_src = inspect.getsource(FinetuneDetectionTrainer.save_model)
        assert "_safe_ema_sync" in setup_src, (
            "_setup_train no longer delegates to _safe_ema_sync"
        )
        assert "_safe_ema_sync" in save_src, (
            "save_model no longer delegates to _safe_ema_sync"
        )
