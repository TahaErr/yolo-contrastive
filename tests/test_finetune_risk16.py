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

    def test_safe_ema_sync_uses_v2_strategy(self):
        """Code-level sentinel for v2 fix (§10.25).

        Verifies the helper body (not docstring) avoids ``assign=True`` and
        uses the taint-cleanup-then-plain-load strategy. The previous v1
        sentinel only string-matched ``assign=True`` anywhere in the source,
        which silently passed when v2 mentioned ``assign=True`` in its
        docstring while no longer using it.

        Uses AST to strip the docstring before checking the executable body,
        so future docstring edits referencing v1 don't break this test.
        """
        import ast
        import textwrap

        raw = inspect.getsource(FinetuneDetectionTrainer._safe_ema_sync)
        src = textwrap.dedent(raw)
        funcdef = ast.parse(src).body[0]

        # Drop the docstring node if present
        body = funcdef.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            body = body[1:]
        code_only = ast.unparse(ast.Module(body=body, type_ignores=[]))

        # v2 invariants
        assert "assign=True" not in code_only, (
            "v2 fix violated — helper body uses assign=True again. "
            "This caused the EMA aliasing collapse documented in §10.25; "
            "use plain load_state_dict on a cleaned destination instead."
        )
        assert "is_inference" in code_only, (
            "v2 fix violated — taint-cleanup branch removed. "
            "Helper must check is_inference() on destination tensors."
        )
        assert "detach().clone" in code_only, (
            "v2 fix violated — taint cleanup must rebuild tainted tensors "
            "as detached clones."
        )

        # Delegation invariants (same as v1 — these are independent of strategy)
        setup_src = inspect.getsource(FinetuneDetectionTrainer._setup_train)
        save_src = inspect.getsource(FinetuneDetectionTrainer.save_model)
        assert "_safe_ema_sync" in setup_src, (
            "_setup_train no longer delegates to _safe_ema_sync"
        )
        assert "_safe_ema_sync" in save_src, (
            "save_model no longer delegates to _safe_ema_sync"
        )

    def test_v2_preserves_independent_storage(self):
        """v2 must NOT alias EMA tensors to model tensors (the v1 catastrophe).

        Reproduces the precise scenario where v1 silently broke: after a
        sync, do ``ema.weight`` and ``model.weight`` share storage? If yes,
        Ultralytics' in-place EMA update collapses both to zero within an
        epoch (§10.25 forensics).

        We invoke the helper in isolation via a minimal stand-in (a plain
        Module pair) so we don't need to instantiate the Ultralytics
        DetectionTrainer machinery. The helper's logic is what's under
        test, not how it's wired into Ultralytics.
        """
        # Build a stand-in trainer with just enough surface for the helper
        class _EMAStub:
            def __init__(self, mod):
                self.ema = mod

        class _Stub:
            pass

        stub = _Stub()
        stub.model = _make_target_with_tainted_buffers()  # tainted
        # Replace tainted model with a clean source — taint actually goes on EMA
        # in real Ultralytics, but the helper handles both directions.
        clean_source = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))
        stub.model = clean_source
        stub.ema = _EMAStub(_make_target_with_tainted_buffers())

        # Invoke the real helper method bound to our stub
        FinetuneDetectionTrainer._safe_ema_sync(stub)

        # Critical invariant: no aliasing
        aliased_params = [
            n for (n, p_e), (_, p_m) in zip(
                stub.ema.ema.named_parameters(),
                stub.model.named_parameters(),
            )
            if p_e.data_ptr() == p_m.data_ptr()
        ]
        assert not aliased_params, (
            f"v1 catastrophe reproduced — {len(aliased_params)} params alias model storage: "
            f"{aliased_params[:3]}. EMA update will collapse weights to zero. See §10.25."
        )

        # And no leftover InferenceMode taint
        leftover_taint = [
            n for n, t in stub.ema.ema.state_dict().items() if t.is_inference()
        ]
        assert not leftover_taint, (
            f"v2 cleanup incomplete — InferenceMode flag still on: {leftover_taint[:3]}"
        )

    def test_v2_survives_simulated_ema_updates(self):
        """v2 must keep weights stable under repeated EMA update steps.

        This is the production failure mode: v1 passed crash-prevention
        tests but collapsed weights by ~99.9% per EMA step (§10.25 sandbox
        forensics). v2 must show NO such collapse — weights stay at their
        initial magnitude after 10 simulated updates.

        We use a constant model (no optimizer.step) so the only forces on
        EMA weights are the update formula itself. If the formula collapses
        them, the bug is in the sync strategy.
        """
        class _EMAStub:
            def __init__(self, mod):
                self.ema = mod

        class _Stub:
            pass

        # Build a model where every weight starts at 1.0 (easy to inspect)
        def _init_ones(m):
            with torch.no_grad():
                for p in m.parameters():
                    p.data.fill_(1.0)
            return m

        stub = _Stub()
        stub.model = _init_ones(nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8)))
        stub.ema = _EMAStub(
            _make_target_with_tainted_buffers()  # destination has taint
        )

        FinetuneDetectionTrainer._safe_ema_sync(stub)

        # Simulate Ultralytics-style EMA update for 10 steps with early-training decay.
        # ModelEMA.decay(updates) = decay_max * (1 - exp(-updates / tau))
        decay_max, tau = 0.9999, 2000.0
        for step in range(10):
            d = decay_max * (1 - torch.exp(torch.tensor(-(step + 1) / tau))).item()
            msd = stub.model.state_dict()
            for k, v in stub.ema.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d)
                    v.add_(msd[k].detach(), alpha=1 - d)

        # All weights should still be ≈ 1.0 — no collapse, no aliasing exploit
        final = next(iter(stub.ema.ema.parameters()))[0, 0, 0, 0].item()
        assert abs(final - 1.0) < 1e-3, (
            f"EMA weight collapsed to {final:.6e} after 10 updates — "
            f"v1 aliasing bug pattern detected. See §10.25 for full forensics."
        )
