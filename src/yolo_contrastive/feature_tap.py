"""Feature-tap module: auto-selects a backbone/neck layer and extracts [B, D] embeddings.

Fixes applied (audit report §1):
  1.1  [CRITICAL] Removed `c != nc` heuristic — only head-prefix blacklist is used now.
  1.2  [HIGH]     Blacklist approach: hooks ALL modules except head, filters by 4D output.
  1.3  [HIGH]     Hook cleanup wrapped in try/finally.
  1.4  [MEDIUM]   Probe input uses torch.randn (avoids div-by-zero in norms).
  1.5  [MEDIUM]   Removed L2 normalize here — kept only in losses.py (single source of truth).
  1.6  [MEDIUM]   `imgsz` accepts int or (H, W) tuple.
  1.7  [LOW]      `_unwrap_out` picks largest 4D tensor from tuple/list outputs.
  1.8  [LOW]      `_resolve_nc` handles string values gracefully.
  1.9  [LOW]      Added __enter__/__exit__ context-manager support + __del__ safety net.
  1.10 [MEDIUM]   MRO-based head detection (catches subclasses like DetectDFL).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch

from .exceptions import ConfigError, FeatureTapError
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_nc(model) -> Optional[int]:
    """Return the number of classes from the model, or None if unknown."""
    nc = getattr(model, "nc", None)
    if nc is not None:
        try:
            return int(nc)
        except (TypeError, ValueError):
            pass

    y = getattr(model, "yaml", None)
    if isinstance(y, dict):
        v = y.get("nc", None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return None


def _is_head_class(mod: torch.nn.Module, head_class_names: Iterable[str]) -> bool:
    """Check if *any* class in the module's MRO matches a head class name."""
    names = set(head_class_names)
    for cls in type(mod).__mro__:
        if cls.__name__ in names:
            return True
    return False


def _head_prefixes(model: torch.nn.Module, head_class_names: Iterable[str]) -> set[str]:
    """Return name-prefixes of all head modules (and their children)."""
    prefixes: set[str] = set()
    for name, mod in model.named_modules():
        if _is_head_class(mod, head_class_names):
            prefixes.add(name)
    return prefixes


def _in_prefix(name: str, prefixes: set[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _unwrap_out(out: Any) -> Any:
    """Extract the most likely feature tensor from a module output."""
    if isinstance(out, (tuple, list)) and len(out) > 0:
        best, best_n = None, -1
        for item in out:
            if torch.is_tensor(item) and item.ndim == 4:
                n = item.numel()
                if n > best_n:
                    best, best_n = item, n
        if best is not None:
            return best
        for item in out:
            if torch.is_tensor(item):
                return item
    return out


def _select_layer(
    acts: Dict[str, Any],
    head_prefixes: set[str],
    min_channels: int,
) -> Optional[str]:
    """Pick the last suitable 4-D feature map outside the head."""
    for name in reversed(list(acts.keys())):
        out = _unwrap_out(acts[name])
        if torch.is_tensor(out) and out.ndim == 4:
            _, c, _, _ = out.shape
            if c >= min_channels and not _in_prefix(name, head_prefixes):
                return name
    return None


def _parse_imgsz(imgsz: Union[int, Tuple[int, int], Sequence[int]]) -> Tuple[int, int]:
    """Normalise ``imgsz`` to an ``(H, W)`` tuple."""
    if isinstance(imgsz, int):
        return (imgsz, imgsz)
    if isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
        return (int(imgsz[0]), int(imgsz[1]))
    raise ConfigError(f"imgsz must be int or (H, W) tuple, got {imgsz!r}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureTap:
    """Auto-selects a backbone/neck layer and produces [B, D] embeddings via a forward hook.

    Usage::

        tap = FeatureTap(model, store_grad=True)
        tap.setup(device="cuda", imgsz=640)
        # ... run model forward ...
        z = tap.get_embedding()  # [B, D]
        tap.close()

    Or as a context manager::

        with FeatureTap(model, store_grad=True) as tap:
            tap.setup(device="cuda", imgsz=640)
            ...
    """

    def __init__(
        self,
        model: torch.nn.Module,
        min_channels: int = 128,
        store_grad: bool = False,
        head_class_names: Tuple[str, ...] = (
            "Detect", "Segment", "Pose", "OBB", "Classify",
        ),
    ):
        self.model = model
        self.min_channels = min_channels
        self.store_grad = store_grad
        self.head_class_names = head_class_names
        self.layer_name: Optional[str] = None
        self.last_embedding: Optional[torch.Tensor] = None
        self._fixed_hook: Any = None

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "FeatureTap":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -- public API ------------------------------------------------------

    def get_embedding(self) -> Optional[torch.Tensor]:
        """Explicit accessor for the last extracted embedding."""
        return self.last_embedding

    def setup(
        self,
        device: Union[str, torch.device] = "cpu",
        imgsz: Union[int, Tuple[int, int]] = 640,
    ) -> None:
        """Probe the model to auto-select a layer, then install a fixed hook."""
        h, w = _parse_imgsz(imgsz)
        device = torch.device(device) if isinstance(device, str) else device

        acts: Dict[str, Any] = {}
        hooks = []

        hp = _head_prefixes(self.model, self.head_class_names)

        def hook_factory(n: str):
            def fn(module, inp, out):
                acts[n] = out
            return fn

        for name, mod in self.model.named_modules():
            if name == "" or _in_prefix(name, hp):
                continue
            hooks.append(mod.register_forward_hook(hook_factory(name)))

        x = torch.randn(1, 3, h, w, device=device)
        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                _ = self.model(x)
        finally:
            for hook in hooks:
                hook.remove()

        if was_training:
            self.model.train()

        sel = _select_layer(acts, head_prefixes=hp, min_channels=self.min_channels)
        if sel is None:
            raise FeatureTapError(
                f"FeatureTap could not find a suitable feature layer "
                f"(4-D, C>={self.min_channels}, outside head). "
                f"Collected {len(acts)} activations from {len(hooks)} modules. "
                f"Head prefixes: {hp or '(none)'}."
            )

        self.layer_name = sel

        mods = dict(self.model.named_modules())
        target = mods[sel]

        def fixed_hook(module, inp, out):
            out_t = _unwrap_out(out)
            if torch.is_tensor(out_t) and out_t.ndim == 4:
                emb = F.adaptive_avg_pool2d(out_t, (1, 1)).flatten(1)
                self.last_embedding = emb if self.store_grad else emb.detach()

        self._fixed_hook = target.register_forward_hook(fixed_hook)

    def close(self) -> None:
        """Remove the permanent hook. Safe to call multiple times."""
        if self._fixed_hook is not None:
            try:
                self._fixed_hook.remove()
            except Exception:
                pass
            self._fixed_hook = None
