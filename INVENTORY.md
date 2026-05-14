# YOLO-CONTRASTIVE — REPO INVENTORY

**Audit purpose:** UX iyileştirmesi öncesi kütüphanenin **her özelliğini** sistematik haritalama. Plan v9'da §4'te eksik veya yanlış sınıflandırılan modülleri açığa çıkarma.

**Generated:** 2026-05-14, commit `87d3e9d` (Adım 2 sonrası)

**Audit scope:** `src/yolo_contrastive/` + `tests/` + `configs/` + `README.md`. Repo'nun **tüm public ve hidden** yüzeyi.

---

## Executive Summary

### Repo gerçek mimari — 3 paralel API hattı (plan v9'da 2 yazmıştım)

Kütüphane **üç paralel API hattı** içerir. Plan v9 §4'te bunlardan ikisini yazdım, üçüncüsünü atladım. UX iyileştirmesi öncesi her üçünün de farkındalığı kritik.

| Hat | Çekirdek | Status | Plan v9'da | Test |
|---|---|---|---|---|
| **A — Modern Dense Hat** | `dense/`, `pretrain/dense_trainer.py`, `finetune/`, `eval/` | ✅ TAMAMLANDI (Faz 1-2) | §4.1 modern hat | 213 + 48 + 5 + 54 ≈ 320 |
| **B — Legacy Pretext Hat** | `contrastive/`, `pretext/`, `adapters/`, `pretrain/trainer.py`, `trainer/` | ✅ AKTİF (README dokümanlı) | §4.2 legacy hat (kısmen) — `pretext/`/`adapters/` "DONDURULDU" yazdım YANLIŞ | ~130 |
| **C — UX Façade Hat** | `pipeline.py`, `discovery.py`, `auto_train`, `_config.py` | ✅ AKTİF | §4.2'de pipeline var, `_config.py` yok | ~40 |
| **D — Data infrastructure** | `data/ssl_pool/`, `data/dedup/`, `data/label_fraction.py`, `data/unified_loader.py` | ✅ TAMAMLANDI | §4.1'de var (modern hat altında) | ~600 |

### Plan v9 §4 doğrulukları + yanlışları

**Doğru olanlar:**
- ✅ `dense/` 8 modül listesi doğru
- ✅ `pretrain/dense_trainer.py` (DenseSSLPretrainer) modern hat
- ✅ `pretrain/run_matrix.py` (PretrainMatrix) doğru
- ✅ `finetune/trainer.py` Risk 16 v2 ile production-validated
- ✅ `data/ssl_pool/` 4 adapter (BDD/A2D2/Cityscapes/Mapillary)
- ✅ `data/dedup/` pHash + leakage modülleri
- ✅ `data/label_fraction.py` + `data/unified_loader.py`
- ✅ `eval/run_matrix.py` detection runner STUB → implement (Adım 2 yapıldı)
- ✅ `pipeline.py` UX katmanı

**Yanlış veya eksik olanlar:**

| Plan v9'da | Gerçek | Aksiyon |
|---|---|---|
| `pretext/` "DONDURULDU" | ✅ AKTİF — 6 task + Composite, README "FrequencyBandPrediction novel" | Plan v9'a düzelt |
| `adapters/` "DONDURULDU" | ✅ AKTİF — LoRA sistemi (ConvLoRA + FreqGated + TaskRouted), inject_lora API | Plan v9'a düzelt |
| `_config.py` yok | ✅ AKTİF — `CLConfig.from_env()` 15+ env var | Plan v9'a ekle |
| `trainer/_helpers, _augmentation, _patching, _csv_logger` yok | ✅ AKTİF — `ContrastiveDetectionTrainer` mixin'leri | Plan v9'a ekle |
| `feature_tap.py` "legacy hat" | ✅ AKTİF — pretrain/trainer.py + trainer/_core.py kullanıyor | Plan v9'da legacy hatın bir parçası, doğru |

### En kritik tespit — paper'ı etkileyebilir

**README iddiası:** "FrequencyBandPrediction — novel frequency domain pretext task (**first in image SSL for detection**)"

Bu modül `pretext/freq_band.py`'de implement edilmiş, paper-grade novelty claim'i içeriyor, plan v9'da bahsedilmedi. Faz 5.3 DT-SAPS önerimle birlikte paper'ın "method" bölümünde **iki novelty** olabilir: (1) DT-SAPS dual-teacher framework, (2) FrequencyBandPrediction pretext task.

**Karar gerekiyor (sen):** FrequencyBandPrediction Faz 5'te ablation eksen olarak dahil mi edilecek, yoksa "legacy hat'tan ayrı bir paper" mı?

### Hidden state ve UX kırıcılar (Onay 3 — özel dikkat)

UX redesign'da en çok can yakacak şeyler:

| # | Hidden state | Etki | UX risk |
|---|---|---|---|
| 1 | **YCL_** env var sistemi (16+ var) | Tüm trainer'lar (`Finetune`, `Contrastive`) env'den yapılandırılır | Kullanıcı YAML yerine os.environ ile config yapmak zorunda. UX yüksek riskli. |
| 2 | **`_env_vars` context manager** (pipeline.py) | Geçici env değişimi, restore | Mevcut implementasyon doğru, sadece gizlilik soru işareti |
| 3 | **Global registry** (augmentations, pretext) | `register()` decorator + `_REGISTRY` dict | İmport sırası bağımlılığı, dynamic discovery |
| 4 | **CLConfig.from_env() gizli yükleme** | `_ensure_cl(batch)` ilk batch'te env'i okuyup config'i sabitler | Kullanıcı `os.environ` değişikliğini geç yaparsa cell'de çalışmaz |
| 5 | **pretrain/trainer.py legacy SSLPretrainer içinde cleanup zorunluluğu** | FeatureTap close, projection head del | Kullanıcı `with` yerine `try/finally` yazmazsa memory leak |
| 6 | **Forward hook patching** (PatchingMixin) | `_install_model_patches` Ultralytics' DetectionModel.forward'ı override eder | İki ContrastiveDetectionTrainer iç içe → double patching tehlikesi |
| 7 | **EMA aliasing risk** (Risk 16 v2 çözüldü) | InferenceMode tainted tensor + load_state_dict | Faz 5'te yeni trainer eklenince yine açığa çıkabilir |
| 8 | **Modeller torch.compile'la uyumsuz olabilir** | `_patch_loss_for_compile` mevcut ama untested | torch 2.5+ JIT modunda riskli |

### Eylem Listesi (Onay 4 — rapor formatı)

**Aşama A çıktısı (bu doküman):**
1. ✅ INVENTORY.md tam dokümante (bu)
2. Plan v9 patch — §4'e `pretext/`, `adapters/`, `_config.py`, `trainer/_*` ekle, "DONDURULDU" düzelt
3. Plan v9 patch — §14'e FrequencyBandPrediction novelty kararı sor

**Aşama B (sonraki — integration smoke suite):**
4. `tests/test_integration_smoke.py` — kütüphanenin tüm public API path'leri için end-to-end smoke
5. ~15-20 senaryo (12 başlangıç, envantere göre +5-8)
6. CPU-on tiny dummy data, 5-10 dakika max

**Aşama C (Adım 3 — UX rewire):**
7. Plan v9 §13.8 Seçenek Y implement (PipelineConfig'e ssl_method)
8. CLConfig env-var → PipelineConfig forwarding (UX iyileştirme)
9. Tüm legacy hat (pretext + adapters) UX'e bağlanmış (auto_train(use_pretext=...))

---

## 1. Public API Surface — `__init__.py` export'ları

`src/yolo_contrastive/__init__.py` (v0.2.0) en üst seviyede şunları export ediyor:

```python
__version__ = "0.2.0"

# ── Legacy contrastive hat ──
from .contrastive import NTXentLoss, build_contrastive_loss
from .feature_tap import FeatureTap

# ── UX façade hat ──
from .pipeline import SSLFinetunePipeline, PipelineConfig, auto_train
from .discovery import discover, DatasetInfo, TrainMode

# ── Exceptions ──
from .exceptions import (
    YoloContrastiveError, FeatureTapError,
    ContrastiveLossError, ConfigError, PatchError,
)

__all__ = [
    "__version__",
    "NTXentLoss", "build_contrastive_loss", "FeatureTap",
    "SSLFinetunePipeline", "PipelineConfig", "auto_train",
    "discover", "DatasetInfo", "TrainMode",
    "YoloContrastiveError", "FeatureTapError",
    "ContrastiveLossError", "ConfigError", "PatchError",
]
```

**Kritik gözlem:** Top-level `__init__.py` SADECE legacy hat + UX hat'ı export ediyor. **Modern hat hiç top-level export değil!** Kullanıcının `DenseSSLPretrainer`, `FinetuneDetectionTrainer`, SAPS, multi-scale loss'a ulaşabilmesi için **alt-paket import** gerek:

```python
from yolo_contrastive.pretrain import DenseSSLPretrainer  # alt-paket
from yolo_contrastive.finetune import FinetuneDetectionTrainer
from yolo_contrastive.dense import multi_scale_dense_loss, saps_within_loss
```

**UX implikasyonu:** Modern hat first-class değil. Plan v9 §13.8 Seçenek Y rewire'da bu düzeltilebilir — `auto_train` default `dense`'e bağlandığında modern hat **kullanıcının görmediği bir backend** olur, ki bu doğru tasarım.

### 1.1 Top-level public API (5 sınıf + 3 fonksiyon + 5 exception)

| Symbol | Type | Path | Görev |
|---|---|---|---|
| `NTXentLoss` | class | `contrastive/losses.py` | InfoNCE/SimCLR contrastive loss |
| `build_contrastive_loss` | function | `contrastive/losses.py` | Factory: `'ntxent'/'infonce'/'simclr'` → loss obj |
| `FeatureTap` | class | `feature_tap.py` | Single-output backbone embedding extractor [B, D] |
| `SSLFinetunePipeline` | class | `pipeline.py` | UX façade: discover → SSL → finetune |
| `PipelineConfig` | dataclass | `pipeline.py` | UX config (35+ alan) |
| `auto_train` | function | `pipeline.py` | One-call training entry point |
| `discover` | function | `discovery.py` | Auto dataset structure detection |
| `DatasetInfo` | dataclass | `discovery.py` | Discovery result |
| `TrainMode` | Enum | `discovery.py` | SSL_FINETUNE / DETECTION / SSL_ONLY |
| `YoloContrastiveError` | exception | `exceptions.py` | Base exception |
| `FeatureTapError` | exception | `exceptions.py` | FeatureTap failures |
| `ContrastiveLossError` | exception | `exceptions.py` | Loss construction failures |
| `ConfigError` | exception | `exceptions.py` | Config validation failures |
| `PatchError` | exception | `exceptions.py` | Model patching failures |

### 1.2 Alt-paket public exports

**`yolo_contrastive.contrastive`** (`contrastive/__init__.py`):
```python
__all__ = ["build_contrastive_loss", "NTXentLoss"]
```

**`yolo_contrastive.feature_tap`** (module, not subpackage): `FeatureTap` class only.

**`yolo_contrastive.augmentations`** (`augmentations/__init__.py`):
```python
__all__ = [
    # Registry / pipeline
    "BaseAugmentation", "PerImageAugmentation", "AugmentationPipeline",
    "register", "get_augmentation", "list_augmentations",
    "build_pipeline", "PRESETS",
    # Geometric (5)
    "RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation90",
    "RandomRotation", "RandomAffine",
    # Color (9)
    "RandomBrightness", "RandomContrast", "RandomSaturation", "RandomHue",
    "RandomColorJitter", "RandomGrayscale", "RandomSolarize", "RandomPosterize",
    "RandomEqualize",
    # Erasing (3)
    "RandomCutout", "RandomErasing", "GridMask",
    # Filtering (3)
    "RandomGaussianBlur", "GaussianNoise", "RandomSharpen",
]
```

**Toplam 20 augmentation primitif** + Pipeline + Registry. Built-in presets: `simclr_v1`, `simclr_v2`, `byol`, `aggressive`.

**`yolo_contrastive.pretext`** (`pretext/__init__.py`):
```python
__all__ = [
    "BasePretextTask", "register_task", "get_task", "list_tasks",
    "RotationTask",                              # legacy
    "SolarizationTask", "ColorPermutationTask",  # IE-Rot inspired
    "PatchShuffleTask", "BlurPredictionTask",    # IE-Rot inspired
    "FrequencyBandPrediction",                   # NOVEL CONTRIBUTION (README)
    "CompositeTask",
    "ProjectionHead", "PredictionHead",
]
```

**6 pretext task + Composite + 2 head sınıfı.** Hepsi `register_task` decorator ile global registry'ye kayıtlı.

**`yolo_contrastive.adapters`** (`adapters/__init__.py`):
```python
__all__ = [
    "ConvLoRA", "FreqGate", "FreqGatedConvLoRA",
    "TaskRoutedConvLoRA", "TaskRouter", "LoRABranch",
    "inject_lora", "remove_lora",
    "compute_merge_alphas", "merge_task_routed_model",
]
```

**LoRA adapter sistemi (10 symbol):** 3 adapter variant (plain/freq-gated/task-routed), inject/remove API, multi-task merging.

**`yolo_contrastive.pretrain`** (`pretrain/__init__.py`):
```python
__all__ = [
    "SSLPretrainer",           # legacy
    "DenseSSLPretrainer",      # modern (Faz 1-2)
    "UnlabeledImageDataset",
    "save_backbone", "load_backbone", "freeze_backbone", "unfreeze_all",
]
```

**`pretrain/run_matrix.py`** (`PretrainMatrix`) export'ta yok — alt-modülden import gerek.

**`yolo_contrastive.finetune`** (`finetune/__init__.py`):
```python
__all__ = ["FinetuneDetectionTrainer"]
```

**`yolo_contrastive.trainer`** (`trainer/__init__.py`):
```python
__all__ = ["ContrastiveDetectionTrainer"]
```

**`yolo_contrastive.dense`** (`dense/__init__.py`):
```python
__all__ = [
    "MultiScaleFeatureTap", "YOLOV8_FPN_LAYERS", "YOLOV8_FPN_STRIDES",
    "FeatureQueue", "combine_queues",
    "MomentumEncoder",
    "SpatialTwoViewAugmentation", "TwoView",
    "dense_ntxent_loss", "coords_to_feature_map",
    "multi_scale_dense_loss",
    "MultiScaleProjectionHead", "infer_in_channels",
    "saps_within_loss", "saps_cross_loss",
]
```

**Modern hat'ın 14 export'u.** Tüm modüller dokunulmaz (Faz 1-2 KAPATILDI).

**`yolo_contrastive.eval`** (`eval/__init__.py`):
```python
__all__ = ["LinearProbeTrainer", "LinearProbeHead", "RunMatrix", "CSV_COLUMNS"]
```

**`yolo_contrastive.data`** (`data/__init__.py`):
```python
__all__ = [
    "LabelFractionSplitter",
    "build_ssl_manifest",
    "MultiLabelImageDataset",
    "loaders_from_yolo_data_yaml",
]
```

**`yolo_contrastive.data.ssl_pool`** (`data/ssl_pool/__init__.py`):
```python
__all__ = [
    "DEFAULT_JPEG_QUALITY", "DEFAULT_LONG_SIDE",
    "MANIFEST_COLUMNS", "ManifestRow",
    "append_rows", "download_with_resume",
    "existing_image_ids", "is_readable_image",
    "read_manifest", "resize_and_save", "write_manifest",
]
```

**`yolo_contrastive.data.dedup`** (`data/dedup/__init__.py`):
```python
__all__ = [
    "DEFAULT_HASH_SIZE",
    "compute_phash", "compute_pool_phashes", "load_phashes",
    "hamming_distance",
    "find_exact_duplicates", "cross_set_leakage", "summarize_duplicates",
]
```

### 1.3 Hidden public API (export'tan dışlanmış ama kullanıcı erişebilir)

Bu modüller `__all__` listelerinde yok ama Python import edilebilir:

- `yolo_contrastive._config.CLConfig` — `from_env()` + 15+ env var alan
- `yolo_contrastive.pretrain.run_matrix.PretrainMatrix` + `CSV_COLUMNS` — ablation orchestrator
- `yolo_contrastive.pretrain.backbone_utils.{save_backbone, load_backbone, freeze_backbone, unfreeze_all}` — backbone IO
- `yolo_contrastive.pretrain.dataset.UnlabeledImageDataset` — pretrain dataset
- `yolo_contrastive.adapters.{ConvLoRA, FreqGate, ...}` — adapter primitives
- `yolo_contrastive.pretext.{BasePretextTask, ...}` — pretext base

**UX risk:** Bu modüller dokümante edilmemiş ama kullanılıyor. UX redesign'da hangi alanların `__all__`'a eklenip resmi olacağı kararı gerek.

---

## 2. Module-by-Module Breakdown

Repo'daki **her modülün** ne yaptığı + hangi durumda kullanılır + bağımlılıkları.

### 2.1 Hat A — Modern Dense Hat (Faz 1-2)

#### `src/yolo_contrastive/dense/` (8 modül)

| Modül | Görev | Bağımlılıklar | Faz |
|---|---|---|---|
| `multi_scale_tap.py` | YOLOv8 backbone'dan P3/P4/P5 forward hook ile çıkar | `torch` | 1.1 |
| `queue.py` | `FeatureQueue` (MoCo-style memory bank) + `combine_queues` (3-level merge with tags) | `torch` | 1.2 |
| `momentum_encoder.py` | EMA encoder (DenseSSLPretrainer'da kullanılır) | `torch`, `copy.deepcopy` | 1.3 |
| `spatial_aug.py` | `SpatialTwoViewAugmentation` — coord-tracked 2-view (random resized crop + hflip) | `torch.nn.functional` | 1.4a |
| `dense_loss.py` | `dense_ntxent_loss` — per-level dense NT-Xent + `coords_to_feature_map` helper | `torch` | 1.4b |
| `multi_scale_loss.py` | `multi_scale_dense_loss` — weighted sum across P3/P4/P5 | `dense_loss` | 1.5 |
| `projection.py` | `MultiScaleProjectionHead` — per-level 2-layer MLP, L2-norm, `infer_in_channels` | `torch.nn` | 1.6 |
| `saps.py` | `saps_within_loss` (cross-scale negatives) + `saps_cross_loss` (queue with scale weighting) | `dense_loss` | 2.1 + 2.2 |

**Toplam test:** 213 (23+32+26+26+29+19+21+37).

**Test dosyaları:** 
- `tests/test_dense_*.py` (multi_scale_tap, queue, momentum, spatial_aug, dense_loss, multi_scale_loss, projection)
- `tests/test_saps.py`

#### `src/yolo_contrastive/pretrain/`

| Modül | Görev | Bağımlılıklar |
|---|---|---|
| `__init__.py` | Re-exports: SSLPretrainer, DenseSSLPretrainer, UnlabeledImageDataset, backbone_utils | — |
| `dense_trainer.py` | **`DenseSSLPretrainer`** — modern hat trainer (Faz 1-2 entegrasyon) | `dense/`, `dataset`, `backbone_utils` |
| `trainer.py` | **`SSLPretrainer`** — legacy hat trainer (CL + pretext + adapter) | `contrastive`, `pretext`, `adapters`, `feature_tap`, `augmentations` |
| `dataset.py` | `UnlabeledImageDataset` — image dir loader, OpenCV decode, resize+/255 | `cv2`, `torch` |
| `backbone_utils.py` | `save_backbone`, `load_backbone`, `freeze_backbone`, `unfreeze_all` — DetectionModel state dict IO | `torch` |
| `run_matrix.py` | **`PretrainMatrix`** — YAML-driven ablation orchestrator (Faz 5.1) | `yaml`, `dense_trainer` (lazy) |

**Kritik dosya boyutları:**
- `dense_trainer.py`: ~600 satır (Foundation + SAPS state machine)
- `trainer.py` (SSLPretrainer): ~500 satır (CL+pretext+adapter integration)
- `run_matrix.py`: ~400 satır (orchestrator + list-DSL exclude)

**Test dosyaları:**
- `tests/test_dense_ssl_pretrainer.py` — 48 test, mock encoder
- `tests/test_dense_ssl_pretrainer_realyolo.py` — gerçek YOLOv8n smoke testleri
- `tests/test_ssl_pretrainer.py` — 5 test (init modes, slow tag ile train smoke)
- `tests/test_pretrain_matrix.py` — 34 test

#### `src/yolo_contrastive/finetune/`

| Modül | Görev | Bağımlılıklar |
|---|---|---|
| `__init__.py` | Re-exports FinetuneDetectionTrainer | — |
| `trainer.py` | **`FinetuneDetectionTrainer`** — Ultralytics DetectionTrainer subclass: pretrained backbone load + differential LR + freeze/unfreeze + Risk 16 v2 fix | `ultralytics.models.yolo.detect.train`, `pretrain.backbone_utils` |

**Tek dosya, ~200 satır.** Risk 16 v2 `_safe_ema_sync` helper bu dosyada (§10.25).

**Env var sözleşmesi:**
- `YCL_PRETRAINED` — SSL backbone .pt path
- `YCL_FREEZE_BACKBONE` — freeze edilecek layer sayısı (default 10)
- `YCL_UNFREEZE_EPOCH` — unfreeze epoch (default 0 = hiç)
- `YCL_BACKBONE_LR_SCALE` — backbone LR multiplier (default 0.5)

**Test:** `tests/test_finetune_risk16.py` — 5 regression test (v2 invariants).

#### `src/yolo_contrastive/eval/`

| Modül | Görev | Bağımlılıklar |
|---|---|---|
| `__init__.py` | Re-exports LinearProbeTrainer, LinearProbeHead, RunMatrix, CSV_COLUMNS | — |
| `linear_probe.py` | `LinearProbeTrainer` + `LinearProbeHead` — frozen backbone + linear head + multilabel AP + early stopping | `feature_tap`, `torch` |
| `run_matrix.py` | `RunMatrix` orchestrator + `_run_linear_probe` + `_run_detection` (Adım 2 ile implement edildi) | `yaml`, `ultralytics` (lazy), `finetune` (lazy) |

**`run_matrix.py` `_run_detection` (Adım 2 implement):**
- Ultralytics YOLO + FinetuneDetectionTrainer integration
- Env var pattern (YCL_PRETRAINED + freeze + unfreeze + lr_scale)
- Paper-grade defaults: epochs=30, imgsz=640, batch=16
- Returns `{metric, metric_value, mAP50, precision, recall}`

**Test:**
- `tests/test_linear_probe.py` — 28 test (early stopping ablation dahil)
- `tests/test_run_matrix.py` — 26 test (orchestrator core)
- `tests/test_run_matrix_detection.py` — 15 test (Adım 2 yeni — mock-based env lifecycle, return shape, integration)

### 2.2 Hat B — Legacy Pretext Hat

Bu hat plan v9'da **kısmen kapsanmıştı**, gerçekte zengin ve aktif.

#### `src/yolo_contrastive/contrastive/`

| Modül | Görev |
|---|---|
| `__init__.py` | Re-exports NTXentLoss, build_contrastive_loss |
| `losses.py` | `NTXentLoss` (InfoNCE/SimCLR) + `build_contrastive_loss` factory |

**Test:** `tests/test_contrastive.py`.

#### `src/yolo_contrastive/feature_tap.py`

**Tek modül**, `FeatureTap` sınıfı — single-output backbone embedding extractor.

- Auto-selects backbone/neck layer ≥ `min_channels`
- Forward hook ile [B, D] embedding üretir
- `store_grad=True` ile gradient akıtır
- Context manager (`__enter__/__exit__`) destekler
- `head_class_names` ile Detect/Segment/Pose/OBB/Classify head'lerini skip eder

**Test:** Implicit via `tests/test_dense_*` ve `tests/test_linear_probe.py`.

#### `src/yolo_contrastive/trainer/`

| Modül | Görev |
|---|---|
| `__init__.py` | Re-exports ContrastiveDetectionTrainer |
| `_core.py` | `ContrastiveDetectionTrainer` — Ultralytics DetectionTrainer + AugmentationMixin + PatchingMixin + CSVLoggerMixin |
| `_helpers.py` | `log`, `safe_scalar`, `extract_loss_from_out`, `replace_in_output`, `is_main_process`, `preserve_bn_running_stats` |
| `_augmentation.py` | `AugmentationMixin` — view2 generation (legacy gaussian blur, color jitter, gray, flip) |
| `_patching.py` | `PatchingMixin` — `_install_model_patches`, `_patch_forward_for_loss_return`, `_patch_loss_for_compile`, `_inject_all` (det + CL + pretext) |
| `_csv_logger.py` | `CSVLoggerMixin` — thread-safe per-step CSV writer |

**Tasarım:** Multiple inheritance — DetectionTrainer base + 3 mixin. **Kompleks**, plan v9'da bahsedilmedi.

**Test:** İndirekt (env var-driven, integration test'leri gerek).

#### `src/yolo_contrastive/pretext/` (NOVEL CONTRIBUTION içerir)

| Modül | Görev | Class count |
|---|---|---|
| `__init__.py` | Re-exports 12 symbol | — |
| `base.py` | `BasePretextTask` + `register_task` decorator + `get_task` + `list_tasks` + global registry | 4 |
| `heads.py` | `ProjectionHead`, `PredictionHead` | 2 |
| `rotation.py` | `RotationTask` — 4-class 0°/90°/180°/270° (legacy) | 1 |
| `tasks.py` | `SolarizationTask` (4), `ColorPermutationTask` (6), `PatchShuffleTask` (24), `BlurPredictionTask` (4) | 4 |
| `freq_band.py` | **`FrequencyBandPrediction`** (4-class, novel) — FFT2D → band mask → IFFT → predict band | 1 |
| `composite.py` | `CompositeTask` — multi-task combiner, `from_names` factory | 1 |

**Toplam 6 task + Composite + base infrastructure.**

**README iddiası:** `FrequencyBandPrediction` "first in image SSL for detection" — paper'a etki edebilir.

**Test:** `tests/test_pretext.py` — 6 task için per-instance parametric test, ~20 test.

#### `src/yolo_contrastive/adapters/` (LoRA sistemi)

| Modül | Görev |
|---|---|
| `__init__.py` | Re-exports 10 symbol |
| `conv_lora.py` | `ConvLoRA` — Conv2d için plain LoRA (rank-r decomposition) |
| `freq_gate.py` | `FreqGate` — frequency-domain gating module |
| `freq_gated_lora.py` | `FreqGatedConvLoRA` — ConvLoRA + FreqGate (frequency-aware adaptation) |
| `task_routed_lora.py` | `TaskRoutedConvLoRA` + `TaskRouter` + `LoRABranch` — multi-task LoRA routing |
| `inject.py` | `inject_lora` + `remove_lora` + `_freeze_backbone_non_lora` + `_get_backbone_convs` |
| `merge.py` | `compute_merge_alphas` + `merge_task_routed_model` — task-routed alphas computation |

**Toplam 10 export symbol** — production-ready LoRA framework for YOLOv8.

**SSLPretrainer ve ContrastiveDetectionTrainer içinden kullanılır** (`adapter="freq_gated"` veya `adapter="task_routed"` kwarg).

**Test:** `tests/test_adapters.py` — 20+ test (ConvLoRA shape/initial-zero/frozen/gradient/merge, FreqGate, FreqGatedConvLoRA, TaskRouted, inject/remove, merge).

#### `src/yolo_contrastive/augmentations/`

| Modül | Görev |
|---|---|
| `__init__.py` | 20 augmentation primitif + registry + presets re-export |
| `registry.py` | `BaseAugmentation`, `PerImageAugmentation`, `AugmentationPipeline`, `register` decorator, `_REGISTRY` dict, `get_augmentation`, `list_augmentations` |
| `presets.py` | `simclr_v1`, `simclr_v2`, `byol`, `aggressive` + `build_pipeline` + `PRESETS` dict |
| `geometric.py` | RandomHorizontalFlip, RandomVerticalFlip, RandomRotation90, RandomRotation, RandomAffine |
| `color.py` | RandomBrightness, RandomContrast, RandomSaturation, RandomHue, RandomColorJitter, RandomGrayscale, RandomSolarize, RandomPosterize, RandomEqualize |
| `erasing.py` | RandomCutout, RandomErasing, GridMask |
| `filtering.py` | RandomGaussianBlur, GaussianNoise, RandomSharpen |

**20 augmentation primitif + 4 preset.** Registry-based, decorator-driven.

**Test:** `tests/test_augmentations.py`.

### 2.3 Hat C — UX Façade Hat

#### `src/yolo_contrastive/pipeline.py`

**1 dosya, ~300 satır.** İçinde:

| Symbol | Type | Görev |
|---|---|---|
| `_env_vars` | context manager | YCL_* env var'larını geçici set + restore |
| `PipelineConfig` | dataclass | 35+ alan: model, imgsz, device, ssl_*, ft_*, cl_*, project, name |
| `SSLFinetunePipeline` | class | `discover_dataset()`, `run_ssl()`, `run_finetune()`, `run_detection()`, `run()` (auto), `summary()` |
| `auto_train` | function | One-call entry: `auto_train(data_yaml=..., unlabeled_dir=...)` |

**ÖNEMLİ:** `run_ssl()` şu an **LEGACY SSLPretrainer** kullanıyor, modern `DenseSSLPretrainer` değil. Plan v9 §13.8 Seçenek Y rewire bunu düzeltecek.

**PipelineConfig 35+ alan dökümü:**
```python
# Model
model: str = "yolov8n.pt"
imgsz: int = 640
device: Optional[str] = None

# SSL Pretraining (legacy SSLPretrainer için)
ssl_epochs: int = 100
ssl_batch: int = 32
ssl_lr: float = 1e-3
ssl_aug_preset: str = "simclr_v2"
ssl_lambda_cl: float = 1.0
ssl_lambda_rot: float = 0.5         # ← legacy, dense'de yok
ssl_temperature: float = 0.2
ssl_warmup_epochs: int = 5
ssl_num_workers: int = 4
ssl_save_every: int = 25
ssl_print_every: int = 10

# Fine-tuning / Detection
ft_epochs: int = 50
ft_batch: int = 16
ft_freeze_layers: int = 10
ft_unfreeze_epoch: int = 0
ft_backbone_lr_scale: float = 0.1

# Contrastive (detection modunda opsiyonel)
cl_lambda: float = 0.0
cl_temperature: float = 0.2
cl_two_view: bool = False
cl_aug_preset: str = ""
cl_lambda_rot: float = 0.0

# Output
backbone_path: str = "pretrained_backbone.pt"
project: str = "runs/pipeline"
name: str = "exp"
```

**Test:** Şu an pipeline-spesifik test yok (sadece import test'i). Aşama B integration smoke'unda kapsanacak.

#### `src/yolo_contrastive/discovery.py`

**1 dosya, ~200 satır.**

| Symbol | Type | Görev |
|---|---|---|
| `TrainMode` | Enum | `SSL_FINETUNE` / `DETECTION` / `SSL_ONLY` |
| `DatasetInfo` | dataclass | 10+ alan: mode, data_yaml, unlabeled_dir, train/val/test dirs, num_classes, class_names, n_train/val/test/unlabeled, `summary()` |
| `discover` | function | data.yaml + unlabeled_dir + dataset_dir input → DatasetInfo |
| `IMAGE_EXTS` | constant | `{".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}` |
| `_count_images` | helper | recursive image count |
| `_resolve_path` | helper | base_dir + rel_path → abs path |

**Auto-discovery algoritması:**
1. `data_yaml` bul (verilmişse kullan; yoksa `dataset_dir`'de ara)
2. data.yaml parse — train/val/test dirs + names + num_classes
3. `unlabeled_dir` bul (verilmişse; yoksa data.yaml'da "unlabeled" key veya `dataset_dir/unlabeled/`)
4. Mode belirle: labeled+unlabeled → SSL_FINETUNE; sadece labeled → DETECTION; sadece unlabeled → SSL_ONLY
5. Ne ikisi de yoksa `ConfigError`

#### `src/yolo_contrastive/exceptions.py`

**5 exception sınıfı:**
- `YoloContrastiveError` (base)
- `FeatureTapError`
- `ContrastiveLossError`
- `ConfigError`
- `PatchError`

#### `src/yolo_contrastive/_config.py`

**Hidden — `__all__`'a dahil değil ama aktif kullanılıyor.**

| Symbol | Type | Görev |
|---|---|---|
| `CLConfig` | dataclass | `lambda_cl`, `loss_name`, `temperature`, `two_view`, `pseudo_view`, `noise_std`, `aug_preset`, `pretext_tasks`, `pretext_weights`, `lambda_pretext`, `lambda_rot`, `rot_hidden_dim`, `adapter_type`, `adapter_rank`, `adapter_scale`, `flip_p`, `gray_p`, `blur_p`, `blur_k`, `blur_sigma`, `brightness_lo/hi`, `contrast_lo/hi` |
| `CLConfig.from_env()` | classmethod | 15+ env var okur, dataclass döner |
| `CLConfig.validate()` | method | Negatif değer/sıfır kontrolleri |
| `CLConfig.enabled` | property | `lambda_cl > 0` |
| `CLConfig.pretext_enabled` | property | pretext tasks + lambda |
| `CLConfig.adapter_enabled` | property | `bool(adapter_type)` |
| `CLConfig.rotation_enabled` | property | legacy: `lambda_rot > 0` (pretext yoksa) |

**env_int, env_float, env_str, env_bool, _parse_csv_floats, _parse_csv_strings** — env var parsing helpers.

**ContrastiveDetectionTrainer içinden `CLConfig.from_env()` ile yüklenir** — kullanıcı `os.environ`'a YCL_* set ederek config yapar.

### 2.4 Hat D — Data Infrastructure

#### `src/yolo_contrastive/data/`

| Modül | Görev |
|---|---|
| `__init__.py` | Re-exports LabelFractionSplitter, build_ssl_manifest, MultiLabelImageDataset, loaders_from_yolo_data_yaml |
| `label_fraction.py` | `LabelFractionSplitter` (stratify, nested, deterministic) + `class_distribution` + `verify_nested` + helpers |
| `unified_loader.py` | `build_ssl_manifest`, `MultiLabelImageDataset`, `loaders_from_yolo_data_yaml`, `_resolve_split` (Roboflow `..` fallback) |

**Test:**
- `tests/test_label_fraction.py` — 30 test (stratification, nested, output txt)
- `tests/test_unified_loader.py` — 32 test (SSL manifest, multi-label dataset, YOLO data.yaml loaders, Roboflow `..` resolution)

#### `src/yolo_contrastive/data/ssl_pool/`

| Modül | Görev |
|---|---|
| `__init__.py` | Re-exports ManifestRow, append_rows, read_manifest, existing_image_ids, write_manifest, resize_and_save, download_with_resume, is_readable_image |
| `common.py` | `resize_and_save`, `is_readable_image`, `download_with_resume`, `DEFAULT_JPEG_QUALITY`, `DEFAULT_LONG_SIDE` |
| `manifest.py` | `ManifestRow` dataclass, `MANIFEST_COLUMNS`, parquet I/O |
| `bdd100k.py` | BDD100K adapter — `ingest` zip → pool materialization, image IDs preserve nested paths |
| `a2d2.py` | A2D2 adapter — Audi dataset |
| `cityscapes.py` | Cityscapes adapter — coarse+fine, per-city subdirs |
| `mapillary.py` | Mapillary Vistas adapter |

**Manifest parquet schema:**
```
image_id        sha-prefixed unique id  (e.g. "bdd100k/train/abc-123")
dataset         dataset name            (e.g. "bdd100k", "a2d2")
original_split  source split            (e.g. "train", "val", "test", "train_extra")
materialized_path  pool-relative path
materialized_w, materialized_h  output dimensions
image_hash      sha256 (64-char hex)
```

**Test:**
- `tests/data/ssl_pool/test_bdd100k.py` — extensive (parsing, ingestion, flush, dedup, corrupt skip, nested subdirs)
- `tests/data/ssl_pool/test_*.py` — diğer 3 dataset için benzeri (kapsam belirsiz)

#### `src/yolo_contrastive/data/dedup/`

| Modül | Görev |
|---|---|
| `__init__.py` | Re-exports |
| `phash.py` | `compute_phash`, `compute_pool_phashes`, `load_phashes`, `hamming_distance`, parquet sidecar |
| `leakage.py` | `find_exact_duplicates`, `cross_set_leakage`, `summarize_duplicates` |

**Dependency:** `imagehash` external library — **Colab session her başında install gerek** (§13.5).

**Test:**
- `tests/data/dedup/test_phash.py` — compute, persist, hamming
- `tests/data/dedup/test_leakage.py` — exact dup, cross-set, summarize

**Toplam ~510 test (commit 557e151).**

---

## 3. Entry Points — Kullanıcının kütüphaneye ulaşabileceği yollar

Kullanıcı kütüphaneye **6 farklı seviyede** ulaşabilir. UX iyileştirmesi her seviye için ayrı karar gerektirir.

### 3.1 Seviye 1 — `auto_train` (UX top-level)

**En basit kullanım, tek satır:**
```python
from yolo_contrastive import auto_train
auto_train(data_yaml="data.yaml", unlabeled_dir="/path/to/pool", ssl_epochs=100, ft_epochs=50)
```

**Davranış:** `discover` → `SSLFinetunePipeline.run` → mode'a göre SSL+Finetune veya sadece Detection.

**Şu anki sınırlama:** SSL legacy hat (`SSLPretrainer`) kullanıyor. Plan v9 §13.8 ile düzeltilecek.

### 3.2 Seviye 2 — `SSLFinetunePipeline` (orta UX)

**Aşamalı kontrol:**
```python
from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

cfg = PipelineConfig(model="yolov8n.pt", ssl_epochs=100, ft_epochs=50, ft_freeze_layers=10)
pipeline = SSLFinetunePipeline(config=cfg)
pipeline.discover_dataset(data_yaml="data.yaml", unlabeled_dir="/path/to/pool")
backbone = pipeline.run_ssl()
results = pipeline.run_finetune()
print(pipeline.summary())
```

**3 ana metod:** `run_ssl()`, `run_finetune()`, `run_detection()` (sadece detection için CL eklentili).

### 3.3 Seviye 3 — Modern hat'a doğrudan erişim

**Modern hat (paper'ın kalbi):**
```python
from yolo_contrastive.pretrain import DenseSSLPretrainer
from yolo_contrastive.finetune import FinetuneDetectionTrainer

# SSL pretrain (modern dense SAPS)
trainer = DenseSSLPretrainer(
    model="yolov8n.pt",
    out_dim=128, queue_size=4096, momentum=0.99, temperature=0.2,
    n_query=128, pos_radius=0.07, match_mode="threshold",
    saps_mode="both", saps_both_lambda=1.0, saps_t_scale=1.0,
    queue_update_strategy="pooled",
    imgsz=640, device="cuda",
)
backbone = trainer.train(
    images_dir="/path/to/pool",
    epochs=100, batch_size=64, lr=1e-3,
    output="dense_backbone.pt",
)

# Finetune (Risk 16 v2 fix kullanılır)
import os
os.environ["YCL_PRETRAINED"] = "dense_backbone.pt"
os.environ["YCL_FREEZE_BACKBONE"] = "10"
os.environ["YCL_UNFREEZE_EPOCH"] = "5"
os.environ["YCL_BACKBONE_LR_SCALE"] = "0.5"

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.train(
    data="data.yaml", epochs=50, imgsz=640, batch=16,
    trainer=FinetuneDetectionTrainer,
)
```

**UX risk:** Env var-driven config alışılmamış pattern.

### 3.4 Seviye 4 — Legacy hat'a doğrudan erişim

**Legacy SSL + pretext (README'de dokümante):**
```python
from yolo_contrastive.pretrain import SSLPretrainer

# Composite pretext (recommended, README dolgu)
pretrainer = SSLPretrainer(
    model="yolov8n.pt",
    aug_preset="simclr_v2",
    lambda_cl=1.0,
    pretext_tasks=["freq_band", "solarization", "patch_shuffle"],
    pretext_weights=[1.0, 0.8, 0.5],
    lambda_pretext=0.5,
    adapter="freq_gated",
    adapter_rank=4,
)
pretrainer.train(images_dir="/path/to/pool", epochs=100, output="legacy_backbone.pt")
```

**Veya contrastive detection (label'lı eğitim):**
```python
import os
os.environ["YCL_LAMBDA"] = "0.1"
os.environ["YCL_PRETEXT_TASKS"] = "freq_band,solarization,patch_shuffle"
os.environ["YCL_PRETEXT_WEIGHTS"] = "1.0,0.8,0.5"
os.environ["YCL_LAMBDA_PRETEXT"] = "0.3"

from ultralytics import YOLO
from yolo_contrastive.trainer import ContrastiveDetectionTrainer

model = YOLO("yolov8n.pt")
model.train(data="coco128.yaml", epochs=10, trainer=ContrastiveDetectionTrainer)
```

**Çok fazla env var.** UX redesign'ın asıl hedef alanı.

### 3.5 Seviye 5 — Ablation orchestration (PretrainMatrix + RunMatrix)

**Faz 5 ablation grid'leri:**
```python
from yolo_contrastive.pretrain.run_matrix import PretrainMatrix
from yolo_contrastive.eval import RunMatrix

# SSL pretrain ablation
pm = PretrainMatrix(
    config_path="configs/pretrain/ablation_stage1_smoke.yaml",
    output_csv="ssl_results.csv",
)
print(f"Cells: {len(pm.expand())}")
pm.run(resume=True, on_error="continue")

# Downstream eval (linear probe)
rm = RunMatrix(
    config_path="configs/eval_matrix.yaml",
    output_csv="eval_results.csv",
)
rm.run(resume=True)
```

**YAML schema'ları dokümante** (`pretrain/run_matrix.py` ve `eval/run_matrix.py` docstring'lerinde).

### 3.6 Seviye 6 — Modüllerden cherry-pick (custom kullanım)

**Kütüphanenin parçalarını kullanma:**
```python
from yolo_contrastive.dense import (
    MultiScaleFeatureTap, FeatureQueue, MomentumEncoder,
    SpatialTwoViewAugmentation, multi_scale_dense_loss,
    saps_within_loss, saps_cross_loss,
)
from yolo_contrastive.augmentations import build_pipeline
from yolo_contrastive.pretext import CompositeTask, get_task
from yolo_contrastive.adapters import inject_lora, TaskRouter
from yolo_contrastive.data import LabelFractionSplitter, loaders_from_yolo_data_yaml
from yolo_contrastive.eval import LinearProbeTrainer
from yolo_contrastive.data.dedup import compute_pool_phashes, cross_set_leakage
```

**Bu seviye için dokümantasyon zayıf** — sadece docstring'lerden öğrenilebilir.

---

## 4. Hyperparameter Inventory

### 4.1 `DenseSSLPretrainer` (Modern hat) — 18 hyperparam

| Param | Type | Default | Range | Açıklama |
|---|---|---|---|---|
| `model` | str/Module | `"yolov8n.pt"` | — | Ultralytics spec veya DetectionModel |
| `out_dim` | int | 256 | >0 | Projection D |
| `queue_size` | int | 65536 | >0 | K (per level) |
| `momentum` | float | 0.999 | [0,1] | EMA m |
| `temperature` | float | 0.2 | >0 | τ |
| `n_query` | int | 256 | >0 | positions per image per level |
| `pos_radius` | float | 0.07 | [0,1] | coord threshold |
| `match_mode` | str | `"threshold"` | `"threshold"`/`"nearest"` | Match strategy |
| `weights` | dict | None | — | Per-level loss weights |
| `aug_kwargs` | dict | None | — | → SpatialTwoViewAugmentation |
| `imgsz` | int | 640 | — | Input size |
| `device` | str | None | auto-detect | Compute device |
| `logger` | obj | None | — | MultiLogger optional |
| `saps_mode` | str | `"none"` | `none/within/cross/both` | SAPS variant |
| `saps_t_scale` | float | 1.0 | >0 | Cross-level temperature |
| `saps_strict_negatives` | bool | False | — | Filter overlapping cross-scale |
| `saps_both_lambda` | float | 1.0 | ≥0 | λ in `within + λ·cross` (when mode=both) |
| `queue_update_strategy` | str | `"pooled"` | `pooled/per_position/subsample` | How keys are enqueued |
| `queue_subsample_n` | int | 16 | ≥1 | n random pos (when strategy=subsample) |

**Train method hyperparams (8):**
- `epochs` (default 100)
- `batch_size` (default 32)
- `lr` (default 1e-3)
- `weight_decay` (default 0.05)
- `warmup_epochs` (default 5)
- `num_workers` (default 4)
- `output` (default "dense_backbone.pt")
- `save_every` (default 25)
- `print_every` (default 10)

**Toplam: 27 hyperparam.**

### 4.2 `SSLPretrainer` (Legacy hat) — 16 hyperparam

| Param | Default | Açıklama |
|---|---|---|
| `model` | `"yolov8n.pt"` | Ultralytics spec |
| `aug_preset` | `"simclr_v2"` | one of `simclr_v1/v2/byol/aggressive` |
| `lambda_cl` | 1.0 | CL loss weight |
| `pretext_tasks` | None | list[str] (`["rotation", "solarization", ...]`) |
| `pretext_weights` | None | list[float] (per-task) |
| `lambda_pretext` | 0.0 | Composite pretext weight |
| `lambda_rot` | 0.0 | Legacy rotation weight |
| `temperature` | 0.2 | NT-Xent τ |
| `proj_dim` | 128 | Projection D |
| `proj_hidden` | 256 | Projection hidden |
| `rot_hidden` | 256 | Pretext head hidden |
| `imgsz` | 640 | Input size |
| `device` | None | auto |
| `adapter` | None | `"freq_gated"`/`"task_routed"`/None |
| `adapter_rank` | 4 | LoRA rank |
| `adapter_scale` | 1.0 | LoRA scale |
| `adapter_dropout` | 0.0 | LoRA dropout |

**Train method:** Aynı 8 train hyperparam (epochs, batch_size, lr, vs).

### 4.3 `FinetuneDetectionTrainer` (Modern hat) — Env var-driven

| Env Var | Type | Default | Açıklama |
|---|---|---|---|
| `YCL_PRETRAINED` | str | "" | SSL backbone .pt path |
| `YCL_FREEZE_BACKBONE` | int | 10 | Number of frozen layers |
| `YCL_UNFREEZE_EPOCH` | int | 0 | Epoch to unfreeze (0=never) |
| `YCL_BACKBONE_LR_SCALE` | float | 0.5 | LR multiplier for backbone |

**Ultralytics' train kwargs ortak** (data, epochs, imgsz, batch, device, project, name, vs).

### 4.4 `ContrastiveDetectionTrainer` (Legacy hat) — `CLConfig` env-driven

Yukarıdaki §2.3 `_config.py`'de detaylandı. **15+ env var.** Tam liste §5'te.

### 4.5 `PipelineConfig` (UX façade) — 35+ alan

Yukarıdaki §2.3'te tam liste. **UX redesign'ın asıl odağı** — bu dataclass'ın temizlenmesi ve dense hat için Seçenek Y forwarding eklenmesi.

### 4.6 `LinearProbeTrainer` (eval)

| Param | Default | Açıklama |
|---|---|---|
| `backbone` | "yolov8n.pt" | Ultralytics spec |
| `num_classes` | (zorunlu) | Output classes |
| `backbone_ckpt` | None | SSL .pt path |
| `feat_level` | `"P5"` | `"P3"`/`"P4"`/`"P5"` |
| `device` | None | auto |
| `normalize_features` | True | L2-norm before head |

**`fit` method:** `train_loader`, `val_loader`, `epochs=10`, `lr=1e-2`, `weight_decay=0`, `verbose=True`, `early_stopping_patience=None`.

### 4.7 `LabelFractionSplitter`

| Param | Default | Açıklama |
|---|---|---|
| `fractions` | (zorunlu) | List of fractions in (0, 1] |
| `seed` | 42 | RNG seed |
| `stratify_mode` | `"dominant"` | `"dominant"` or `"none"` |
| `min_per_class` | 2 | Min count for stratification |

### 4.8 `PretrainMatrix` + `eval/RunMatrix`

**YAML-driven**, programmatic config geçişi de mümkün. Detay §6 (YAML schemas).

---

## 5. Environment Variable Surface — Hidden State #1

UX redesign'ın en kritik alanı. **16+ env var** kütüphanenin davranışını yönetir. Plan v9'da bahsedilmedi.

### 5.1 FinetuneDetectionTrainer env var'ları (4)

| Env Var | Type | Default | Görev |
|---|---|---|---|
| `YCL_PRETRAINED` | str (path) | "" | SSL backbone .pt path; varsa load_backbone çağrılır |
| `YCL_FREEZE_BACKBONE` | int | 10 | layer index 0..N-1 freeze |
| `YCL_UNFREEZE_EPOCH` | int | 0 | Bu epoch'tan itibaren unfreeze (0=never) |
| `YCL_BACKBONE_LR_SCALE` | float | 0.5 | backbone params için LR multiplier |

**Set yeri:** Kullanıcı `os.environ` ile manuel, veya `pipeline.py::run_finetune` `_env_vars` context manager ile geçici.

### 5.2 ContrastiveDetectionTrainer / CLConfig env var'ları (15+)

`_config.py::CLConfig.from_env()` okuyor:

**CL loss:**
| Env Var | Type | Default | Görev |
|---|---|---|---|
| `YCL_LAMBDA` | float | 0.0 | CL loss weight (0=disabled) |
| `YCL_LOSS` | str | "ntxent" | Loss type (`ntxent`/`infonce`/`simclr`) |
| `YCL_TEMP` | float | 0.2 | NT-Xent temperature |
| `YCL_PRINT_EVERY` | int | 50 | Per-step logging interval |

**Two-view:**
| `YCL_TWO_VIEW` | bool ("0"/"1") | "0" | Real two-view augmentation |
| `YCL_PSEUDO_VIEW` | bool | "1" | Pseudo-view (noise) fallback |
| `YCL_NOISE_STD` | float | 1e-3 | Pseudo-view noise std |
| `YCL_AUG_PRESET` | str | "" | `simclr_v1/v2/byol/aggressive` |

**Pretext tasks (multi-task SSL):**
| `YCL_PRETEXT_TASKS` | csv strings | "" | `rotation,solarization,blur` |
| `YCL_PRETEXT_WEIGHTS` | csv floats | "" | per-task weights |
| `YCL_LAMBDA_PRETEXT` | float | 0.0 | Total pretext weight |

**Legacy rotation (backward compat):**
| `YCL_LAMBDA_ROT` | float | 0.0 | Rotation task weight (legacy) |
| `YCL_ROT_HIDDEN_DIM` | int | 256 | Rotation head hidden |

**Adapter (LoRA):**
| `YCL_ADAPTER_TYPE` | str | "" | `freq_gated`/`task_routed`/"" |
| `YCL_ADAPTER_RANK` | int | 4 | LoRA rank |
| `YCL_ADAPTER_SCALE` | float | 1.0 | LoRA scale |

**Legacy view2 augmentation params (≥10):**
| `YCL_FLIP_P` | float | 0.5 | Horizontal flip prob |
| `YCL_GRAY_P` | float | 0.2 | Grayscale prob |
| `YCL_BLUR_P` | float | 0.5 | Gaussian blur prob |
| `YCL_BLUR_K` | int | 5 | Blur kernel size |
| `YCL_BLUR_SIGMA` | float | 1.0 | Blur sigma |
| `YCL_BRIGHTNESS_LO/HI` | float | 0.6/1.4 | Brightness range |
| `YCL_CONTRAST_LO/HI` | float | 0.6/1.4 | Contrast range |

**Toplam env var: 22+.**

### 5.3 Hidden state riski

**`CLConfig.from_env()` ContrastiveDetectionTrainer'ın `_ensure_cl(batch)` metodunda **ilk batch'te** çağrılır.** Sonrasında env değişikliklerini görmez.

**UX risk senaryosu:**
```python
# Kullanıcı cell 1:
os.environ["YCL_LAMBDA"] = "0.1"
model.train(...)  # CL aktif

# Kullanıcı cell 2 (aynı runtime):
os.environ["YCL_LAMBDA"] = "0.3"
model.train(...)  # Yeni trainer yaratılırsa OK; aynı trainer ise eski lambda
```

UX redesign'da bu env-driven config **PipelineConfig forwarding** ile resmileştirilmeli (Aşama C iş).

---

## 6. YAML Schemas — Config Surface

### 6.1 `pretrain/run_matrix.py` (PretrainMatrix) YAML

```yaml
output_dir: /content/drive/.../pretrain_runs
output_csv: pretrain_results.csv

base:                          # fixed across every cell
  images_dir: /content/ssl_pool_local
  model: yolov8n.pt
  imgsz: 640
  epochs: 100
  batch_size: 64
  lr: 1.0e-3
  warmup_epochs: 5
  queue_size: 65536
  momentum: 0.999
  temperature: 0.2

grid:                          # varying ablation axes
  saps_mode: [none, within, cross, both]
  saps_both_lambda: [0.0, 0.5, 1.0, 2.0]
  queue_update_strategy: [pooled, per_position, subsample]
  saps_t_scale: [0.5, 1.0, 2.0, .inf]

seeds: [42]

exclude:                       # list-DSL with `in` filter
  - saps_mode: [none, within, cross]
    saps_both_lambda: [0.5, 1.0, 2.0]
```

**CSV columns:** `cell_id, seed, axes_json, metric, metric_value, status, elapsed_s, error, started_at, backbone_path`.

**Faz 5.1 ablation YAML'ları:** `configs/pretrain/ablation_stage{1,2,3}_*.yaml` (commit `bb6796d`).

### 6.2 `eval/run_matrix.py` (RunMatrix) YAML

```yaml
task: linear_probe         # or 'detection'
output_csv: results.csv

methods:
  - name: ours_a_d
    backbone_ckpt: /path/to/ours.pt
  - name: mocov3
    backbone_ckpt: /path/to/mocov3.pt

datasets:
  - name: pothole
    data_yaml: pothole.yaml
    num_classes: 1

fractions: [0.1, 0.25, 0.5, 1.0]
seeds: [42, 43]

hp:
  epochs: 30                # detection: 30, linear_probe: 10
  imgsz: 640
  batch: 16
  freeze: 10                # detection only
  unfreeze_epoch: 5
  backbone_lr_scale: 0.5

exclude:
  - {method: ours_a_d, fraction: 0.01}      # too few samples
```

**CSV columns:** `method, dataset, fraction, seed, task, metric, metric_value, status, elapsed_s, error, started_at`.

### 6.3 `data/unified_loader.py` SSL manifest YAML

```yaml
datasets:
  - name: bdd100k
    root: /data/bdd100k/images
    image_glob: "**/*.jpg"
    recursive: true
  - name: a2d2
    root: /data/a2d2
    image_glob: "**/cam_front_center/*.png"
```

### 6.4 YOLO data.yaml (Ultralytics standard, Roboflow compat)

```yaml
train: ../train/images       # Roboflow `..` quirk — _resolve_split() handles fallback
val: ../valid/images
test: ../test/images

nc: 4
names:
  - circular_cover
  - pothole
  - rectangular_cover
  - speed_bump
```

---

## 7. Data Formats — Hidden State #2

### 7.1 `.pt` backbone format (save/load via `pretrain.backbone_utils`)

```python
{
    "model": state_dict,          # Detection model state dict (backbone+head+...)
    "epoch": int,
    "best_loss": float,
    "type": str,                  # "dense_ssl" or other
    # Other metadata
}
```

**Save:** `save_backbone(model, path, epoch=N, extra={"loss": ..., "type": "dense_ssl"})`
**Load:** `load_backbone(model, path, strict=False, verbose=True, backbone_only=True)` — `backbone_only=True` ile sadece backbone layer'lar load.

### 7.2 `manifest.parquet` (SSL pool)

```
image_id        str (e.g. "bdd100k/train/abc-123")  unique
dataset         str (e.g. "bdd100k")
original_split  str (e.g. "train", "val", "test", "train_extra")
materialized_path  str (pool-relative)
materialized_w  int
materialized_h  int
image_hash      str (sha256, 64-char hex)
```

### 7.3 `phash.parquet` (dedup sidecar)

```
image_id   str
phash      str (16-char hex, 64-bit pHash)
```

Sidecar — manifest'i bozmaz. Cross-set leakage check için sorgulanır.

### 7.4 CSV — RunMatrix / PretrainMatrix results

**RunMatrix:** §6.2'deki schema.
**PretrainMatrix:** §6.1'deki schema (axes_json JSON-encoded).

### 7.5 `.txt` manifest — SSL pretrain pool

`build_ssl_manifest()` çıktısı:
```
/abs/path/to/img1.jpg
/abs/path/to/img2.jpg
...
```

UnlabeledImageDataset bunu line-by-line okur.

### 7.6 `.txt` label fraction split — `LabelFractionSplitter`

Output dir altında:
```
train_pct010.txt   ← %10 subset
train_pct025.txt   ← %25 subset
train_pct100.txt   ← %100 (full)
```

Her dosya bir image path per line. YOLO data.yaml `train: train_pct010.txt` ile referans edilebilir.

---

## 8. Test Coverage Map

Mevcut test dosyaları (commit `87d3e9d`, 726 passed):

| Test dosyası | Modül | Test sayısı (tahmin) | Tip |
|---|---|---|---|
| `tests/test_import.py` | top-level | 2 | smoke |
| `tests/test_contrastive.py` | `contrastive/losses` | ~15 | unit |
| `tests/test_feature_tap.py` | `feature_tap` | ~10 | unit |
| `tests/test_augmentations.py` | `augmentations/` | ~10 | parametric |
| `tests/test_pretext.py` | `pretext/` (6 task) | ~25 | parametric per-task |
| `tests/test_adapters.py` | `adapters/` (LoRA) | ~25 | unit |
| `tests/test_ssl_pretrainer.py` | `pretrain/trainer.py` | 5 + 1 slow | init modes |
| `tests/test_dense_*.py` | `dense/` (8 modül) | 213 | invariants |
| `tests/test_dense_ssl_pretrainer.py` | `pretrain/dense_trainer.py` | 48 | full smoke |
| `tests/test_dense_ssl_pretrainer_realyolo.py` | `pretrain/dense_trainer.py` | ~10 | real YOLOv8n |
| `tests/test_saps.py` | `dense/saps.py` | 37 | SAPS modes |
| `tests/test_pretrain_matrix.py` | `pretrain/run_matrix.py` | 34 | orchestrator |
| `tests/test_finetune_risk16.py` | `finetune/trainer.py` | 5 | regression |
| `tests/test_linear_probe.py` | `eval/linear_probe` | 28 | early stopping |
| `tests/test_run_matrix.py` | `eval/run_matrix` (orchestrator) | 26 | unit |
| `tests/test_run_matrix_detection.py` | `eval/run_matrix._run_detection` | 15 | mock-based (Adım 2 yeni) |
| `tests/test_label_fraction.py` | `data/label_fraction` | 30 | unit + invariants |
| `tests/test_unified_loader.py` | `data/unified_loader` | 32 | Roboflow path resolution |
| `tests/data/ssl_pool/test_bdd100k.py` | `data/ssl_pool/bdd100k` | ~20 | adapter |
| `tests/data/ssl_pool/test_a2d2.py` | `data/ssl_pool/a2d2` | ~?  | adapter |
| `tests/data/ssl_pool/test_cityscapes.py` | `data/ssl_pool/cityscapes` | ~?  | adapter |
| `tests/data/ssl_pool/test_mapillary.py` | `data/ssl_pool/mapillary` | ~?  | adapter |
| `tests/data/dedup/test_phash.py` | `data/dedup/phash` | ~? | unit |
| `tests/data/dedup/test_leakage.py` | `data/dedup/leakage` | 8+ | unit |

**Toplam (726 passed = commit 87d3e9d):**
- Modern hat: ~485 (dense ~213, dense_trainer ~58, finetune 5, eval 69, data ~140)
- Legacy hat: ~80 (contrastive 15, feature_tap 10, augmentations 10, pretext 25, adapters 25, ssl_pretrainer 5)
- UX hat: ~0 (sadece import test'i)
- SSL pool/dedup: ~160 (commit 557e151)

**Coverage gap — UX hat:** `pipeline.py`, `discovery.py`, `auto_train`, `_config.py` için **dedicated test yok**. Aşama B'nin asıl boşluğu burası.

---

## 9. Configs Directory

```
configs/
  pretrain/
    ablation_stage1_smoke.yaml      6 cells, 5K pool, 30 epoch, imgsz=320
    ablation_stage2_coarse.yaml     12 cells, 50K pool, 50 epoch, imgsz=640
    ablation_stage3_fine.yaml       9 cells, 186K pool, 100 epoch (Variant A template)
```

**Toplam 3 YAML** (commit `bb6796d`). Faz 5.2/5.3/5.4/5.5/5.6 için YAML'lar yapılacak (plan §4).

---

## 10. Bağımlılıklar (`pyproject.toml`)

```toml
dependencies = [
    "torch>=2.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
yolo     = ["ultralytics>=8.0.0"]
pretrain = ["opencv-python>=4.6.0"]
dev      = ["pytest", "ruff", "pre-commit"]
all      = ["yolo-contrastive[yolo,pretrain,dev]"]
```

**Hidden dependencies (paket içinden import edilen ama declared değil):**
- `imagehash` — `data/dedup/phash.py` (her Colab session install gerek)
- `pandas` — `data/dedup/phash.py` (parquet I/O)
- `pyarrow` — parquet support (transitive via pandas)
- `Pillow` — image I/O (transitive via opencv?)
- `numpy` — yaygın
- `sklearn` — `data/label_fraction.py` stratify (varsa daha hızlı, fallback var)

**UX risk:** Pip install sırasında bu hidden dependency'ler eksik kalabilir. Aşama B test'leri install state'ini kontrol etmeli.

---

## 11. Plan v9 Düzeltilmesi Gerekli Maddeleri

UX iyileştirmesinden önce plan v9'a yapılacak patch'ler:

### 11.1 §4 Modül Haritası — yanlış sınıflandırmalar
- ❌ `pretext/` "DONDURULDU" → ✅ "Legacy hat içinde AKTİF, README dokümanlı"
- ❌ `adapters/` "DONDURULDU" → ✅ "Legacy hat içinde AKTİF, LoRA sistemi"
- ❌ `_config.py` yok → ✅ Hidden hat'a ekle, env var inventory ile birlikte
- ❌ `trainer/_*` mixin'ler yok → ✅ Legacy hat'a ekle

### 11.2 §14 Paper Hikayesi — yeni soru
- FrequencyBandPrediction README "novel contribution" → Faz 5'te ablation mı yoksa "ayrı paper" mı?

### 11.3 §11 Sentinels — yeni
- §11.11 Top-level export sentinel: `__init__.py` modern hat ekleyene kadar `DenseSSLPretrainer` alt-paket import'tan gelir

---

## 12. Sonuç — UX iyileştirmesi öncesi durum

**Repo durumu:** Sağlam, 726 test passed, paper-grade.

**Gizli zenginlik:** 3 hat aktif, README'de dokümante ama plan v9'da kısmen kayıt dışı.

**UX redesign için yapılacaklar (sırayla):**

1. **Aşama A — Bu doküman ✅** (commit'lendiğinde repo'da kalıcı kayıt)
2. **Plan v9 patch** — §4 düzeltme + §14 FrequencyBandPrediction soru
3. **Aşama B — Integration smoke suite** — `tests/test_integration_smoke.py` (15-20 senaryo)
4. **Aşama C — Adım 3 Seçenek Y** — PipelineConfig forwarding + dense default

**Tahmini süre:**
- Plan v9 patch: 15-20 dk
- Aşama B (smoke suite): 1-2 saat
- Aşama C: 30-45 dk

**Toplam: 2-3 saat** kütüphanenin UX-ready hale gelmesi için.

---
**SON.**
