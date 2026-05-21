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
    "run_leakage_check": "eval.leakage_check",
    # ── fine-tuning ───────────────────────────────────────────────────
    "FinetuneDetectionTrainer": "finetune",
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
