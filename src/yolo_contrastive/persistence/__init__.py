"""REVISIT — cross-traversal persistence pretraining for YOLOv8 backbones.

Repeated captures of the same street location (Mapillary, different sessions,
months apart) provide two label-like supervision signals on the shared
:class:`~yolo_contrastive.anchored.trainer.AnchoredJointTrainer`:

    Signal A  positive-only SimSiam consistency between P3 features at
              homography-corresponding points across traversals (real
              time/weather/camera variation as the "augmentation");
    Signal B  dense 3-class persistence labels (background / persistent /
              transient) from cross-traversal blob-proposal matching.

Offline pair factory (manifest-driven, resumable): mine -> download -> align
-> propose -> label, via ``python -m yolo_contrastive.persistence.cli``.

This package init is lazy (E2): importing ``yolo_contrastive.persistence``
needs nothing beyond the package itself; submodules pull torch / pandas /
cv2 / requests only when actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "PersistenceChannel",
    "PairDataset",
    "collate_pairs",
    "AlignConfig",
    "align_pair",
    "PersistenceLabelConfig",
    "match_proposals",
    "ProposalConfig",
    "cheap_proposals",
    "PairGateConfig",
    "mine_pairs",
    "download_images",
]

_LAZY = {
    "PersistenceChannel": ("channel", "PersistenceChannel"),
    "PairDataset": ("pair_dataset", "PairDataset"),
    "collate_pairs": ("pair_dataset", "collate_pairs"),
    "AlignConfig": ("align", "AlignConfig"),
    "align_pair": ("align", "align_pair"),
    "PersistenceLabelConfig": ("persistence_labels", "PersistenceLabelConfig"),
    "match_proposals": ("persistence_labels", "match_proposals"),
    "ProposalConfig": ("proposals", "ProposalConfig"),
    "cheap_proposals": ("proposals", "cheap_proposals"),
    "PairGateConfig": ("mapillary_pairs", "PairGateConfig"),
    "mine_pairs": ("mapillary_pairs", "mine_pairs"),
    "download_images": ("mapillary_pairs", "download_images"),
}

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from .align import AlignConfig, align_pair  # noqa: F401
    from .channel import PersistenceChannel  # noqa: F401
    from .mapillary_pairs import PairGateConfig, download_images, mine_pairs  # noqa: F401
    from .pair_dataset import PairDataset, collate_pairs  # noqa: F401
    from .persistence_labels import PersistenceLabelConfig, match_proposals  # noqa: F401
    from .proposals import ProposalConfig, cheap_proposals  # noqa: F401


def __getattr__(name: str):
    """Lazy attribute resolution (PEP 562) keeping the package import light."""
    try:
        mod_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(f".{mod_name}", __name__), attr)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
