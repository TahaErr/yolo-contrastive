"""yolo-contrastive: Self-supervised pretraining + contrastive learning for YOLOv8+.

The top-level API is lazily loaded. ``import yolo_contrastive`` is lightweight
and pulls in **no** optional dependencies (ultralytics, opencv): a name is only
resolved — and its submodule imported — on first access. So the import works
even without ultralytics installed; classes that genuinely need it (e.g.
``FinetuneDetectionTrainer``) require it only when actually used.

Public API (all importable as ``from yolo_contrastive import X``):

  Pretraining
    DenseSSLPretrainer   — dense SAPS self-supervised pretraining
    SSLPretrainer        — legacy global contrastive pretraining
    save_backbone, load_backbone — backbone checkpoint I/O

  Dual-teacher framework (DT-SAPS)
    DualTeacherTrainer   — SAPS + dual-teacher distillation trainer
    CocoTeacher          — frozen YOLO feature teacher + adapter
    TeacherCache         — FP16 teacher-feature cache
    ConsensusLoss        — Form B + Form C distillation loss
    DisagreementWeighter — per-position cosine disagreement weighting

  External SSL baselines
    SimCLRYOLOTrainer, MoCoV3YOLOTrainer, CoMADYOLOTrainer

  Evaluation
    LinearProbeTrainer   — frozen-backbone linear probe
    RunMatrix            — YAML-driven ablation grid runner
    run_leakage_check    — pool/downstream cross-set leakage check

  Fine-tuning
    FinetuneDetectionTrainer — YOLO fine-tuning with a pretrained backbone

  Anchored joint training (nature's labels carrier)
    AnchoredJointTrainer — COCO-replay-anchored joint trainer (R3)
    AuxChannel           — channel plugin interface
    save_checkpoint, load_for_finetune — R8 whole-detector transplant I/O

  TERRA (geoteach — road-plane residuals from monocular depth)
    TerraChannel             — the AuxChannel
    run_depth_anything       — Stage-0 depth cache factory
    labels_from_inverse_depth — full per-image label pipeline

  REVISIT (persistence — cross-traversal supervision)
    PersistenceChannel   — the AuxChannel
    mine_pairs, download_images — Mapillary pair-pool factory

  GASP-Real (scalereal — natural scale from metric depth)
    ScaleRealChannel     — the AuxChannel
    ScaleRealConfig      — every threshold in one dataclass
    mine_pool            — offline pair miner

  High-level pipeline
    SSLFinetunePipeline, PipelineConfig, auto_train

  Core building blocks
    NTXentLoss, build_contrastive_loss, FeatureTap

  Dataset discovery
    discover, DatasetInfo, TrainMode

  Exceptions
    YoloContrastiveError, FeatureTapError, ContrastiveLossError,
    ConfigError, PatchError
"""

from __future__ import annotations

import importlib

__version__ = "0.2.0"

# name → submodule (relative to this package) that defines it.
_LAZY_EXPORTS = {
    # ── pretraining ───────────────────────────────────────────────────
    "DenseSSLPretrainer": "pretrain",
    "SSLPretrainer": "pretrain",
    "save_backbone": "pretrain",
    "load_backbone": "pretrain",
    # ── dual-teacher framework (DT-SAPS) ──────────────────────────────
    "DualTeacherTrainer": "dual_teacher",
    "CocoTeacher": "dual_teacher",
    "TeacherCache": "dual_teacher",
    "ConsensusLoss": "dual_teacher",
    "DisagreementWeighter": "dual_teacher",
    # ── external SSL baselines ────────────────────────────────────────
    "SimCLRYOLOTrainer": "baselines",
    "MoCoV3YOLOTrainer": "baselines",
    "CoMADYOLOTrainer": "baselines",
    # ── evaluation ────────────────────────────────────────────────────
    "LinearProbeTrainer": "eval",
    "RunMatrix": "eval",
    "run_cv_eval": "eval.cross_val",
    "aggregate_cv_results": "eval.cross_val",
    "run_leakage_check": "eval.leakage_check",
    # ── fine-tuning ───────────────────────────────────────────────────
    "FinetuneDetectionTrainer": "finetune",
    # ── anchored joint training (nature's labels carrier) ─────────────
    "AnchoredJointTrainer": "anchored",
    "AuxChannel": "anchored",
    "save_checkpoint": "anchored",
    "load_for_finetune": "anchored",
    # ── TERRA (geoteach) ──────────────────────────────────────────────
    "TerraChannel": "geoteach",
    "run_depth_anything": "geoteach",
    "labels_from_inverse_depth": "geoteach",
    # ── REVISIT (persistence) ─────────────────────────────────────────
    "PersistenceChannel": "persistence",
    "mine_pairs": "persistence",
    "download_images": "persistence",
    # ── GASP-Real (scalereal) ─────────────────────────────────────────
    "ScaleRealChannel": "scalereal",
    "ScaleRealConfig": "scalereal",
    "mine_pool": "scalereal",
    # ── high-level pipeline ───────────────────────────────────────────
    "SSLFinetunePipeline": "pipeline",
    "PipelineConfig": "pipeline",
    "auto_train": "pipeline",
    # ── core building blocks ──────────────────────────────────────────
    "NTXentLoss": "contrastive",
    "build_contrastive_loss": "contrastive",
    "FeatureTap": "feature_tap",
    # ── dataset discovery ─────────────────────────────────────────────
    "discover": "discovery",
    "DatasetInfo": "discovery",
    "TrainMode": "discovery",
    # ── exceptions ────────────────────────────────────────────────────
    "YoloContrastiveError": "exceptions",
    "FeatureTapError": "exceptions",
    "ContrastiveLossError": "exceptions",
    "ConfigError": "exceptions",
    "PatchError": "exceptions",
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str):
    """PEP 562 lazy attribute access — import the submodule on first use."""
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{submodule}", __name__)
    attr = getattr(module, name)
    globals()[name] = attr   # cache — subsequent access bypasses __getattr__
    return attr


def __dir__():
    """Expose lazy names to dir() / tab-completion."""
    return sorted([*globals().keys(), *_LAZY_EXPORTS])
