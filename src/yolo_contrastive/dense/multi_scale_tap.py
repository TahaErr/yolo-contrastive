"""Multi-scale feature tap — extracts P3/P4/P5 from YOLOv8 neck via forward hooks.

Faz 1.1 — Foundation for Dense + Multi-scale CL (WORK_PLAN_v3 §5).

Unlike the single-output FeatureTap (which produces one [B, D] embedding),
this module exposes the raw feature maps at three FPN levels so per-position
contrastive loss can be computed in dense_loss.py.

YOLOv8 architecture reference (yolov8.yaml head section):
    Layer 15: C2f → P3/8  output (fed to Detect head)
    Layer 18: C2f → P4/16 output
    Layer 21: C2f → P5/32 output
    Layer 22: Detect head [[15, 18, 21]]

Indices are identical for n/s/m/l/x — only channel widths differ.
v9/v10/v11 support is deferred to Faz 1.6 (probe-based detection).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..exceptions import FeatureTapError


# YOLOv8 P3/P4/P5 layer indices that feed into Detect head.
YOLOV8_FPN_LAYERS: Dict[str, int] = {
    "P3": 15,
    "P4": 18,
    "P5": 21,
}

# Expected stride per level (used by callers for shape verification).
YOLOV8_FPN_STRIDES: Dict[str, int] = {"P3": 8, "P4": 16, "P5": 32}


def _get_layer_sequence(model: nn.Module) -> nn.Sequential:
    """Locate the nn.Sequential of YOLO layers inside common wrapper structures.

    Handles:
        nn.Sequential                              → returned directly
        ultralytics.YOLO.model (DetectionModel)    → drills into .model
        DDP / nn.DataParallel                      → drills into .module(.model)
    """
    candidates: List[nn.Module] = [model]
    for chain in (("model",), ("model", "model"), ("module",), ("module", "model")):
        cur: nn.Module = model
        ok = True
        for attr in chain:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok:
            candidates.append(cur)

    for c in candidates:
        if isinstance(c, nn.Sequential):
            return c

    raise FeatureTapError(
        f"Could not locate nn.Sequential layer list in {type(model).__name__}. "
        f"Pass DetectionModel.model directly if needed."
    )


class MultiScaleFeatureTap:
    """Extracts feature maps from multiple FPN layers via forward hooks.

    Usage:
        tap = MultiScaleFeatureTap(detection_model)
        tap.setup()
        _ = detection_model(x)              # any forward pass triggers hooks
        feats = tap.get_features()          # {"P3": [B,C3,H/8,W/8], ...}
        tap.close()

    As context manager:
        with MultiScaleFeatureTap(model) as tap:
            tap.setup()
            ...
    """

    def __init__(
        self,
        model: nn.Module,
        levels: Tuple[str, ...] = ("P3", "P4", "P5"),
        layer_indices: Optional[Dict[str, int]] = None,
    ) -> None:
        self.model = model
        self.levels: Tuple[str, ...] = tuple(levels)

        if layer_indices is not None:
            self.layer_indices = dict(layer_indices)
        else:
            self.layer_indices = {
                k: YOLOV8_FPN_LAYERS[k] for k in self.levels if k in YOLOV8_FPN_LAYERS
            }

        for level in self.levels:
            if level not in self.layer_indices:
                raise FeatureTapError(
                    f"No layer index for level {level!r}. "
                    f"Pass layer_indices={{'{level}': <int>}} explicitly."
                )

        self._features: Dict[str, Optional[torch.Tensor]] = {k: None for k in self.levels}
        self._hooks: List = []
        self._is_setup: bool = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def __enter__(self) -> "MultiScaleFeatureTap":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def setup(self) -> None:
        """Install forward hooks on configured FPN layers. Idempotent."""
        if self._is_setup:
            return

        seq = _get_layer_sequence(self.model)
        try:
            for level in self.levels:
                idx = self.layer_indices[level]
                if idx < 0 or idx >= len(seq):
                    raise FeatureTapError(
                        f"Layer index {idx} for level {level!r} out of range "
                        f"(model has {len(seq)} layers)"
                    )
                hook = seq[idx].register_forward_hook(self._make_hook(level))
                self._hooks.append(hook)
        except Exception:
            # Roll back partial hook installation
            for h in self._hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            self._hooks.clear()
            raise

        self._is_setup = True

    def close(self) -> None:
        """Remove all hooks and reset state."""
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        self._features = {k: None for k in self.levels}
        self._is_setup = False

    # ── feature access ────────────────────────────────────────────────────

    def get_features(self) -> Dict[str, torch.Tensor]:
        """Return {level: feature_map}. Requires prior setup() and forward pass."""
        if not self._is_setup:
            raise FeatureTapError("Call setup() before get_features().")
        missing = [k for k, v in self._features.items() if v is None]
        if missing:
            raise FeatureTapError(
                f"Features for {missing} are None — was a forward pass executed?"
            )
        # Return shallow copy so external clear() on dict doesn't disturb us
        return dict(self._features)  # type: ignore[arg-type]

    def clear(self) -> None:
        """Reset captured features without removing hooks."""
        for k in self._features:
            self._features[k] = None

    # ── internals ─────────────────────────────────────────────────────────

    def _make_hook(self, level: str):
        def hook(_module, _input, output):
            # Some layers can output tuples (rare in YOLOv8 neck, but defensive)
            if isinstance(output, (tuple, list)):
                output = output[0]
            self._features[level] = output
        return hook

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "setup" if self._is_setup else "not setup"
        return (
            f"MultiScaleFeatureTap(levels={self.levels}, "
            f"indices={self.layer_indices}, status={status})"
        )
