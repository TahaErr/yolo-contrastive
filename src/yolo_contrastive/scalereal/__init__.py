"""GASP-Real — natural-scale supervision from metric monocular depth.

The successor to gasp/ (failed: synthetic-rescale blur shortcut, EMA-matcher
feedback, InfoNCE calibration minefield). GASP-Real keeps the original
insight — similar content at different distances carries scale information —
and makes the diagnosed shortcuts impossible by construction: no patch is
ever synthetically resized anywhere; pair labels are REAL apparent-scale
ratios ``log_r = log(Z_A / Z_B)`` from metric monocular depth; mining is an
offline pass with a frozen embedder; training runs as an
:class:`~yolo_contrastive.anchored.AuxChannel` on the COCO-anchored joint
trainer.

Public API (lazily resolved — ``import yolo_contrastive`` stays light, E2):

    ScaleRealConfig       — every threshold in one dataclass (config.py)
    ScaleRealChannel      — the AuxChannel implementation (channel.py)
    ScaleRealPoolDataset, scalereal_collate — the R5 joint-aug loader
    DepthCache, log_depth_ratio, patch_depth_stats — shared metric depth
                            cache I/O + the affine-ambiguity guard (depth_io)
    PairIndex, read_pairs, write_pairs, append_pairs, is_probe_image
                          — the pair manifest (pair_manifest.py)
    mine_image_pairs, mine_pool, grid_candidate_boxes, MiningStats,
    GridStatsEmbedder, DINOv2Embedder, render_audit — the offline miner
                            (mine_pairs.py; CLI: ``python -m
                            yolo_contrastive.scalereal.mine_pairs``)

Loss primitives (losses.py), heads (heads.py), the joint transform bridge
(pair_transform.py) and the synthetic pinhole generator (synthetic.py) are
importable as submodules.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_LAZY = {
    "ScaleRealConfig": ".config",
    "ScaleRealChannel": ".channel",
    "ScaleRealPoolDataset": ".channel",
    "scalereal_collate": ".channel",
    "DepthCache": ".depth_io",
    "log_depth_ratio": ".depth_io",
    "patch_depth_stats": ".depth_io",
    "PairIndex": ".pair_manifest",
    "read_pairs": ".pair_manifest",
    "write_pairs": ".pair_manifest",
    "append_pairs": ".pair_manifest",
    "is_probe_image": ".pair_manifest",
    "mine_image_pairs": ".mine_pairs",
    "mine_pool": ".mine_pairs",
    "grid_candidate_boxes": ".mine_pairs",
    "MiningStats": ".mine_pairs",
    "GridStatsEmbedder": ".mine_pairs",
    "DINOv2Embedder": ".mine_pairs",
    "render_audit": ".mine_pairs",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .channel import (  # noqa: F401
        ScaleRealChannel,
        ScaleRealPoolDataset,
        scalereal_collate,
    )
    from .config import ScaleRealConfig  # noqa: F401
    from .depth_io import (  # noqa: F401
        DepthCache,
        log_depth_ratio,
        patch_depth_stats,
    )
    from .mine_pairs import (  # noqa: F401
        DINOv2Embedder,
        GridStatsEmbedder,
        MiningStats,
        grid_candidate_boxes,
        mine_image_pairs,
        mine_pool,
        render_audit,
    )
    from .pair_manifest import (  # noqa: F401
        PairIndex,
        append_pairs,
        is_probe_image,
        read_pairs,
        write_pairs,
    )
