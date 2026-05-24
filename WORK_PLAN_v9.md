# YOLO-CONTRASTIVE — MASTER WORK PLAN v9

**Tez/Paper:** Scale-Aware Dense Contrastive Pretraining with Dual-Teacher Distillation for Real-Time Detection in Traffic Scenes

**Method codename (paper yazımında finalize):** DT-SAPS / HiTeC-YOLO / ScaDis-Net — bu evrede ad sabitlenmedi (§14.2).

**Hedef venue:** ITS-odaklı, CVPR-tier (T-ITS, ITSC, IV Symposium)

---

## Değişiklikler (v8 → v9)

**Method scope expansion — paper'ın asıl yenisi:**
- ✅ Stratejik karar: Saf SAPS pipeline tek başına paper-grade method DEĞİL; **C+D yaklaşımı** ile genişletilecek (§1.5)
- ✅ C+D karar: Dual-teacher consensus (COCO supervised + SSL momentum) + disagreement weighting eklenir
- ✅ Bilimsel hipotez: "Hybrid supervised+SSL teacher consensus + scale-aware dense distillation → COCO baseline'ı traffic detection'da eşit veya geç" (§5.3)
- ✅ Literature positioning: CoMAD (AAAI 2026), SimCLR-YOLO (Aug 2025), SSLKD (2024) — yakın işler taranmış, 4-axis kombinasyonumuz unique (§14.1)

**Closed (v8 → v9 arası kapatılanlar):**
- ✅ Risk 16 v2 kalıcı fix — taint cleanup + plain load_state_dict, EMA aliasing önlendi (§10.25)
- ✅ Risk 16 v1 → v2 forensics — paper supplementary'de "unit test correctness vs integration safety" lesson (§10.25)
- ✅ PretrainMatrix orchestrator — Faz 5 ablation grid expansion + list-DSL exclude (§10.24)
- ✅ pHash dedup modülü — exact-dup + cross-set leakage tooling (modül var, gerçek pool run §13.1)
- ✅ SSL pool indirme — 181,446 image (BDD+A2D2+Cityscapes+Mapillary, §2.1)
- ✅ Faz 5 ablation grid YAML'ları — 3-stage hiyerarşi (smoke/coarse/fine, 27 cells, §5.1)
- ✅ Risk 16 v2 production validation — Colab A100 smoke (§11.8)

**Faz 5 büyük revizyon:**
- ✅ Faz 5 alt-fazlara bölündü (5.1-5.6), her biri ayrı çıktı + ayrı bütçe
- ✅ Multi-backbone validation eklendi (Faz 5.2 — YOLOv8n, v9n, v10n, v11n, v12n, v26n)
- ✅ Dual-teacher ablation eklendi (Faz 5.3 — C+D yaklaşımının ablation grid'i)
- ✅ External baselines Faz 3'ten Faz 5.4'e taşındı (CoMAD-YOLO port, SimCLR-YOLO, MoCo-v3-YOLO)
- ✅ DINOv2/foundation karşılaştırma Faz 5.5 olarak en sona alındı (referans, "to beat" değil)

**Yeni §13 + §14 (akademik şeffaflık):**
- §13 — Bekleyen Geçmiş İşler: yapılmamış ama unutulmaması gereken adımlar
- §14 — Paper Hikayesi Şeffaflığı: paper supplementary'sinde "method evolution" anlatımı

**Test toplamı:** 710 (was 672 v8 sonunda; +34 PretrainMatrix +5 Risk16 v2 −1 v1 sentinel)

**Repo state:** `bb6796d` (commit `ff74127` Risk 16 v2 + `bb6796d` Faz 5 YAMLs + plan §11.8 §10.25 closure)

---

## 1. Stratejik Karar Özeti

### 1.1-1.4 (v1-v7 kayıtlı — ana noktalar)

**1.1 Method ailesi:** A (Foundation: COCO+CL joint) + D (SAPS: Scale-Aware Pretrained Specialization). B, C, E atıldı (akademik value zayıf).

**1.2 Backbone seçimi:** YOLOv8 ana hat. v9/v10/v11/v12/v26 nano varyantları multi-backbone validation eksen (Faz 5.2). Şu evrede s/m/l değil — hız önemli, gelecek work.

**1.3 Domain:** Traffic scenes (BDD100K + Mapillary + Cityscapes + A2D2). Pothole + 3 infrastructure class downstream. Paper domain-specific olarak konumlanır.

**1.4 Real-time deployment:** YOLOv8n primary; multi-backbone diğerleri nano. Detection latency paper'da raporlanır (gelecek), şu evrede odak mAP.

### 1.5 v9'da eklenen — DT-SAPS framework (yeni)

**Bilimsel hipotez:**

> *"Saf SSL (SAPS dahil) COCO supervised pretraining'i traffic detection downstream'inde GEÇEMEZ — bu Faz 4.7 + Yol 3 smoke'ta gözlendi (§11.7: COCO 0.4783 vs SSL-within 0.3754; §11.8: COCO 0.6266 vs SSL-both v2 0.5465). Buna karşı çözüm: SSL'in domain-specific bilgisini KORURKEN COCO'nun semantic anchor'ını da damıtmak. Dual-teacher consensus (supervised COCO + momentum SSL) + scale-aware dense distillation ile traffic detection'da COCO baseline'a **eşit veya geç**."*

**Method tasarımı (özet — detay §5.3'te):**

```
L_total = L_SAPS(student, ssl_teacher)              ← bizim mevcut katkı (Faz 1-2'den)
        + α · L_distill_B(student, w·f_coco + (1-w)·f_ssl)   ← yeni: learned-weighted L2
        + β · [KL(p_student||p_coco) + KL(p_student||p_ssl)] ← yeni: dual KL
        + γ · disagreement_weight · L_distill        ← yeni: hard-mining (Yaklaşım D)

disagreement_weight = exp(α_d · d(f_coco, f_ssl_ema))
                      α_d ∈ {0, 0.5, 1.0, 2.0}      ← ablation eksen, α_d=0 klasik
```

Hyperparametreler (α, β, γ, w_init, α_d) Faz 5.3 ablation eksenleri. Mevcut SAPS (`saps_mode`, `saps_both_lambda`, `saps_t_scale`, `queue_update_strategy`) korunur — DT-SAPS bunların üstüne inşa edilir.

**Akademik konumlanma (§14.1 literatür taraması ile doğrulanmış):**
- CoMAD (AAAI 2026): all-SSL teacher, ImageNet/ViT. Bizim hybrid (supervised+SSL) + CNN/detection farklı.
- SimCLR-YOLO (Aug 2025): single teacher, global pooling. Bizim multi-teacher + dense farklı.
- SSLKD (2024): road segmentation, generic distillation. Bizim scale-aware + detection farklı.
- 4-axis unique combination: supervised+SSL hybrid teacher + scale-aware dense + disagreement weighting + traffic specialization.

**Strateji sıralaması (kullanıcı kararı):**
1. **Faz 5.1** — Saf SAPS pipeline çalıştır, paper-grade baseline al
2. **Faz 5.2** — Multi-backbone (6 mimari) saf SAPS validation
3. **Faz 5.3** — C+D (dual-teacher) ablation, DT-SAPS en iyi config'i bul
4. **Faz 5.4** — External baselines (CoMAD-YOLO port, SimCLR-YOLO, MoCo-v3-YOLO)
5. **Faz 5.5** — DINOv2 foundation model karşılaştırma (referans, "geçme hedefi değil")
6. **Faz 5.6** — Eval matrix + paper writing

A ✅, D ✅, atılanlar dondurulmuş. DT-SAPS A+D framework'ün **doğal genişlemesi** (replace değil, extension).

---

## 2. Datasets

### 2.1 SSL pretrain pool — İNDİRİLDİ ✅ (~181K driving image)

| Kaynak | Image | Manifest | Status |
|---|---|---|---|
| BDD100K (train+val, unlabeled) | 99,995 | parquet | ✅ |
| A2D2 | 31,221 | parquet | ✅ |
| Cityscapes coarse | 20,000 | parquet | ✅ |
| Mapillary Vistas | 30,230 | parquet | ✅ |
| **Toplam** | **181,446** | unified `manifest.parquet` | ✅ |

Lokasyon: `/content/drive/MyDrive/yolo-contrastive/ssl_pool/` (Drive — Stage 2/3 öncesi `/content/ssl_pool_local`'a kopyalanır, §13.2).

Smoke alt küme: `/content/datasets/bdd100k_ssl_5k` (5000 image, seed=42 deterministik, Stage 1 smoke için). Bu **BDD pool'unun bir parçası**, separate dataset değil.

### 2.2 Downstream evaluation

**Ana eval dataset:** `0-no-dcs-no-aug v6` (Roboflow, custom)
- 3371 train / 548 valid / 0 test
- 4 class: `circular_cover`, `pothole`, `rectangular_cover`, `speed_bump`
- imgsz=640 native, 320 smoke
- Label fraction split'leri: `data_frac10.yaml`, `data_frac25.yaml`, `data_frac100.yaml` (seed=42)

**Eski interim referans (akademik kayıt için korunur):** Roboflow Pothole 1125 (Faz 4.7 + Yol 3 smoke referans rakamları, §11.7). v8'de tek downstream'di, v9'da artık secondary (akademik şeffaflık için silinmedi).

**Pothole 5K (yapılacak, §13.4):** Kullanıcı tarafında "yarı hazır". Tamamlanınca 4-class dataset ile **ek downstream eval** olur (paper'da "two downstream datasets" claim'i güçlenir). Şu anki 4-class yeterli, Pothole 5K bonus.

### 2.3 Data leakage kontrolü

pHash dedup modülü ✅ (commit `557e151`, 510 test). **Gerçek pool üzerinde run yapılmadı** (§13.1) — Stage 2/3 öncesi zorunlu. İçeride exact-dup detection + cross-set leakage check (pool ↔ downstream).

### 2.4 COCO teacher feature cache (yeni — Faz 5.3 için)

**Karar (K2 — §10.27):** Teacher feature'lar **augmente edilmemiş** orijinal image üzerinde **bir kez** çıkarılır, parquet/npz olarak diske cache'lenir. Eğitim sırasında student augment edilir (view_a, view_b), teacher cached feature ile distillation yapılır.

**Cache spesifikasyonu:**
```
/content/ssl_pool_local/teacher_cache/
  yolov8x_coco_p3p4p5/
    {image_id}.npz                   # P3 (80×80×128), P4 (40×40×256), P5 (20×20×512)
    metadata.json                    # teacher model, version, feature dims
```

**Disk:** 181K × ~2MB (compressed FP16, 3-scale) ≈ **~360 GB** — büyük. **Risk 18.** Mitigasyon:
- Alternatif 1: Sadece P5 cache (~30 GB) → Faz 5.3 ablation'da P3/P4 distillation marjinal yararsa
- Alternatif 2: Sadece subset (50K) cache + on-the-fly geri kalan → tradeoff
- Alternatif 3: Vast.ai instance üzerinde lokal NVMe SSD (genelde 100 GB+) → uygun

**Karar Stage 5.3 başlangıcında**, smoke ile P3/P4/P5 ayrı ayrı hangi seviyede distillation'ın yardımı var ölç, sonra cache stratejisi finalize.

---

## 3. Eval Matrix

```
                       0-no-dcs v6 (primary)    BDD val (secondary)    Pothole 5K (gelecek)
                       %10 %25 %100             %1 %5 %10 %25 %100     %10 %25 %100

Scratch                ▢   ▢   ▢                ▢  ▢  ▢   ▢   ▢        ▢   ▢   ▢
COCO baseline          ▢   ▢   ✓ (0.6266 §11.8) ▢  ▢  ▢   ▢   ▢        ▢   ▢   ▢
Saf SAPS (Faz 5.1)     ▢   ▢   ▢                ▢  ▢  ▢   ▢   ▢        ▢   ▢   ▢
DT-SAPS (Faz 5.3)      ▢   ▢   ▢                ▢  ▢  ▢   ▢   ▢        ▢   ▢   ▢
CoMAD-YOLO (Faz 5.4)   ▢   ▢   ▢                ▢  ▢  ▢   ▢   ▢        ▢   ▢   ▢
SimCLR-YOLO (Faz 5.4)  ▢   ▢   ▢                ▢  ▢  ▢   ▢   ▢        ▢   ▢   ▢
MoCo-v3 (Faz 5.4)      ▢   ▢   ▢                ▢  ▢  ▢   ▢   ▢        ▢   ▢   ▢
DINOv2 (Faz 5.5)       ▢   ▢   ▢                ▢  ▢  ▢   ▢   ▢        ▢   ▢   ▢
```

✓ = smoke ile yapıldı (§11.8). Tamam.

**Toplam:** ≈ 80-120 run (primary dataset, 8 method × 3 frac × N seed). Multi-backbone factor (Faz 5.2) eklendiğinde sadece "winning config" üzerinde 6× artar — toplam yine yönetilebilir (~120 + 30 multi-backbone = 150 max).

**Fair comparison invariant:** Tek değişen backbone init. Augmentation, finetune config, eval protokolü tüm satırlarda **bit-eşit aynı** (verify §11.10 sentinel).

---

## 4. Modül Haritası

Kütüphane **iki paralel API hattı** içerir: **Modern Hat** (Faz 1-2'nin paper'ın asıl katkısı için yazıldığı modüller) ve **Legacy Hat** (önceki versiyonlarda yazılmış, geriye uyumlu, paket içinde aktif). Plan v8'de yalnızca modern hat raporlanmıştı — v9'da legacy hat da haritalanır (bkz. §14.3 evolution narrative).

### 4.1 Modern Hat — Faz 1-2 ana iş + paper'ın kalbi

```
src/yolo_contrastive/

  dense/                              ✅ TAMAMLANDI (Faz 1+2, dokunulmaz)
    multi_scale_tap.py                ✓ 23 test    — P3/P4/P5 forward hook
    queue.py                          ✓ 32 test    — FeatureQueue + combine_queues
    momentum_encoder.py               ✓ 26 test    — EMA encoder
    spatial_aug.py                    ✓ 26 test    — coord-tracked 2-view aug
    dense_loss.py                     ✓ 29 test    — dense NT-Xent
    multi_scale_loss.py               ✓ 19 test    — weighted-sum multi-level
    projection.py                     ✓ 21 test    — MultiScaleProjectionHead
    saps.py                           ✓ 37 test    — within + cross variants

  pretrain/
    dense_trainer.py                  ✓ 48 test    — DenseSSLPretrainer (modern)
    run_matrix.py                     ✓ 34 test    — PretrainMatrix orchestrator
    dataset.py                        ✓             — UnlabeledImageDataset
    backbone_utils.py                 ✓             — save/load/freeze/unfreeze
    dual_teacher_trainer.py           ✅ TAMAMLANDI (42a74c4) — DenseSSLPretrainer'ı sarmalar (composition, §10.30)

  dual_teacher/                       ✅ TAMAMLANDI (Faz 5.3 — paper'ın asıl yenisi)
    __init__.py                       ✅ 5 modül export
    coco_teacher.py                   ✅ b31c79c — Frozen YOLO feature teacher + per-scale adapter (Risk 17), 17 test
    teacher_cache.py                  ✅ 5edc0c1 — FP16 npz feature cache I/O (§2.4), 16 test
    consensus_loss.py                 ✅ 381591e — Form B (learned weighted L2) + Form C (CWD dual KL), 22 test
    disagreement.py                   ✅ 008c5e7 — per-position cosine disagreement weighting, 24 test
    dual_teacher_trainer.py           ✅ 42a74c4 — DT-SAPS trainer (composition §10.30), 24 test

  finetune/
    trainer.py                        ✓ Risk 16 v2 fix devrede (§10.25)
                                      5 regression test (§11.8 production validated)

  eval/                               ⚠️  PARTIAL — detection runner STUB
    linear_probe.py                   ✓ 28 test    — early stopping
    run_matrix.py                     ⚠️  26 test (linear_probe runner OK)
                                      ⚠️  _run_detection STUB (raises NotImplementedError)
                                      §13.7'de Faz 5.1.1 öncesi implement edilecek
    leakage_check.py                  ✅ 502b069 — pool/downstream cross-set leakage CLI runner, 14 test

  baselines/                          ✅ TAMAMLANDI (Faz 5.4 — external SSL baseline portları)
    __init__.py                       ✅ 3 trainer export
    comad_yolo.py                     ✅ b2d2ae9 — CoMAD-YOLO (3 SSL teacher consensus gating + asymmetric masking), 17 test
    simclr_yolo.py                    ✅ 1c5299c — SimCLR-YOLO (in-batch NT-Xent, global-pooled), 13 test
    moco_v3.py                        ✅ 6211133 — MoCo-v3-YOLO (momentum + predictor, no queue), 15 test

  data/                               ✅ KAPSAMLI
    label_fraction.py                 ✓ 30 test    — fraction splits
    unified_loader.py                 ✓ 32 test    — Roboflow `..` fallback
    ssl_pool/                         ✓             — pool ingestion (paper supplementary)
      bdd100k.py                      ✓             — BDD100K adapter (99,995 image)
      a2d2.py                         ✓             — A2D2 adapter (31,221 image)
      cityscapes.py                   ✓             — Cityscapes coarse+fine adapter
      mapillary.py                    ✓             — Mapillary Vistas adapter
      common.py                       ✓             — shared resize/encode utilities
      manifest.py                     ✓             — parquet manifest schema
    dedup/                            ✓ 510 test (commit 557e151)
      phash.py                        ✓             — pHash compute + persistence
      leakage.py                      ✓             — exact-dup + cross-set leakage

  configs/
    pretrain/
      ablation_stage1_smoke.yaml      ✓ 6 cells, 5K pool, 30 epoch, imgsz=320
      ablation_stage2_coarse.yaml     ✓ 12 cells, 50K pool, 50 epoch, imgsz=640
      ablation_stage3_fine.yaml       ✓ 9 cells, 186K pool, 100 epoch (Variant A template)
      multi_backbone.yaml             ⬜ Faz 5.2 (winning config × 6 backbone)
      ablation_stage4_dt.yaml         ⬜ Faz 5.3 (DT-SAPS ablation, α/β/γ/w/α_d)
      ablation_stage5_baselines.yaml  ⬜ Faz 5.4 (CoMAD-YOLO, SimCLR-YOLO, MoCo-v3)
      ablation_stage6_foundation.yaml ⬜ Faz 5.5 (DINOv2 referans)
```

### 4.2 Legacy Hat — geriye uyumlu, hâlâ AKTİF (paper-grade modüller içerir)

Plan v8'de bu hat'ın bir kısmı "DONDURULDU" olarak işaretlenmişti — **v9 envanteri (INVENTORY.md §2.2) sonrası DÜZELTİLDİ**: `pretext/` ve `adapters/` aktif, README'de dokümante edilmiş modüllerdir (FrequencyBandPrediction paper-grade novelty claim'i içerir, §14.3).

```
src/yolo_contrastive/

  # ── High-level facade (UX) ────────────────────────────────────────────
  pipeline.py                         ✓ SSLFinetunePipeline + PipelineConfig + auto_train
                                      ⚠️  run_ssl şu an LEGACY SSLPretrainer kullanıyor;
                                      §13.8'de DenseSSLPretrainer'a rewire (Seçenek Y geriye uyumlu)
  discovery.py                        ✓ DatasetInfo + TrainMode + discover()
                                      Auto dataset structure detection (SSL_FINETUNE / DETECTION / SSL_ONLY)
  exceptions.py                       ✓ YoloContrastiveError + 4 alt-exception (FeatureTapError,
                                      ContrastiveLossError, ConfigError, PatchError)
  _config.py                          ✓ HIDDEN — CLConfig.from_env() (15+ YCL_* env var)
                                      ContrastiveDetectionTrainer'ın `_ensure_cl(batch)` ile
                                      ilk batch'te yüklediği config. Plan v8'de yoktu; v9
                                      envanteri sonrası eklendi (INVENTORY §5.2).

  # ── Pre-Faz 1 SSL hattı (saf global-pooling CL, korunur) ──────────────
  contrastive/
    losses.py                         ✓ NTXentLoss + build_contrastive_loss (ntxent/infonce/simclr)
  feature_tap.py                      ✓ Single-output FeatureTap [B, D] — auto-selects backbone layer
                                      (modern hat'ta dense MultiScaleFeatureTap kullanılır;
                                      legacy hat + LinearProbeTrainer bunu kullanır)
  trainer/                            ✓ Multi-inheritance trainer architecture
    _core.py                          ✓ ContrastiveDetectionTrainer = DetectionTrainer +
                                      AugmentationMixin + PatchingMixin + CSVLoggerMixin
    _helpers.py                       ✓ log, safe_scalar, extract_loss_from_out, replace_in_output
                                      is_main_process, preserve_bn_running_stats
    _augmentation.py                  ✓ AugmentationMixin — view2 generation (gaussian blur cache,
                                      flip/gray/color jitter — legacy torch-only)
    _patching.py                      ✓ PatchingMixin — _install_model_patches, forward override
                                      _patch_loss_for_compile (torch.compile uyumluluğu)
                                      _inject_all (det + CL + pretext)
    _csv_logger.py                    ✓ CSVLoggerMixin — thread-safe per-step CSV writer (_csv_lock)
  pretrain/
    trainer.py                        ✓ SSLPretrainer (legacy — pipeline.py + ContrastiveDetectionTrainer
                                      ile birlikte kullanılır). Composite pretext + LoRA adapter desteği.

  # ── Pretext task system (paper-grade — README'de "novel contribution" claim'i) ──
  pretext/                            ✅ AKTİF (plan v8'de yanlışlıkla "DONDURULDU" yazılmıştı)
    base.py                           ✓ BasePretextTask + register_task decorator + global registry
    heads.py                          ✓ ProjectionHead + PredictionHead
    rotation.py                       ✓ RotationTask — 4-class 0°/90°/180°/270° (legacy IE-Rot)
    tasks.py                          ✓ SolarizationTask (4-cls) + ColorPermutationTask (6-cls) +
                                      PatchShuffleTask (24-cls Jigsaw-lite) + BlurPredictionTask (4-cls)
    freq_band.py                      ✓ FrequencyBandPrediction (4-cls FFT band masking)
                                      README iddiası: "first in image SSL for detection"
                                      Bu paper'da gelecek work olarak işaret edilir (§14.3 — Seçenek B)
    composite.py                      ✓ CompositeTask — multi-task combiner
                                      transform sırayla uygulanır, tek embedding ile çok head

  # ── LoRA adapter system (parameter-efficient domain adaptation) ──────
  adapters/                           ✅ AKTİF (plan v8'de yanlışlıkla "DONDURULDU" yazılmıştı)
    conv_lora.py                      ✓ ConvLoRA — Conv2d için plain LoRA (rank-r decomp, initial-zero)
    freq_gate.py                      ✓ FreqGate — frequency-domain gating module
    freq_gated_lora.py                ✓ FreqGatedConvLoRA — ConvLoRA + FreqGate
    task_routed_lora.py               ✓ TaskRoutedConvLoRA + TaskRouter + LoRABranch
                                      (multi-task LoRA routing — pretext task'lar için)
    inject.py                         ✓ inject_lora + remove_lora + _freeze_backbone_non_lora
                                      SSLPretrainer + ContrastiveDetectionTrainer içinden çağrılır
    merge.py                          ✓ compute_merge_alphas + merge_task_routed_model
                                      (task-routed LoRA merging into single backbone)

  # ── Augmentation system (legacy ve modern hatlarda kullanılabilir) ────
  augmentations/                      ✓ Modular augmentation framework (20 primitif + 4 preset)
    registry.py                       ✓ BaseAugmentation + PerImageAugmentation + AugmentationPipeline
                                      + register decorator + get_augmentation + list_augmentations
    presets.py                        ✓ PRESETS = {simclr_v1, simclr_v2, byol, aggressive} + build_pipeline
    geometric.py                      ✓ HFlip, VFlip, Rotation90, Rotation, Affine
    color.py                          ✓ Brightness, Contrast, Saturation, Hue, ColorJitter,
                                          Grayscale, Solarize, Posterize, Equalize (9 primitif)
    erasing.py                        ✓ Cutout, Erasing, GridMask
    filtering.py                      ✓ GaussianBlur, GaussianNoise, Sharpen
                                      Faz 5.3 dual-teacher photometric aug eksen → §5.3'te
```

### 4.3 Durdurulan hatlar (DÜZELTME — v8'de yanlış sınıflandırıldı)

```
src/yolo_contrastive/
  # Plan v8 burada "adapters/, pretext/ DONDURULDU" yazmıştı — YANLIŞ.
  # v9 envanteri (INVENTORY.md §2.2): her ikisi de aktif, README dokümanlı.
  # Tek dondurulan: yok (şu anda durdurulmuş alt-modül yok).
```

**Sonuç:** `pretext/` ve `adapters/` AKTİF legacy hat içinde, §4.2'de listelendi. Plan v8'in "DONDURULDU" sınıflandırması v9 envanteri ile düzeltildi.

### 4.4 Test toplamı + dağılım

**Mevcut test toplam:** 726 (commit `a45c18b` sonrası — INVENTORY.md commit dahil hâlâ aynı)

Dağılım (INVENTORY.md §8 audit'inden):
- **Modern Dense Hat:** ~485 test
  - `dense/` 213 (multi_scale_tap 23 + queue 32 + momentum 26 + spatial_aug 26 + dense_loss 29 + multi_scale_loss 19 + projection 21 + saps 37)
  - `pretrain/dense_trainer` 48 (+ realyolo 10)
  - `pretrain/run_matrix` 34 (PretrainMatrix)
  - `finetune/trainer` 5 (Risk 16 v2 regression)
  - `eval/linear_probe` 28
  - `eval/run_matrix` 26 + `run_matrix_detection` 15 (Adım 2)
  - `data/label_fraction` 30 + `unified_loader` 32
  - `data/ssl_pool` + `data/dedup` ~160
- **Legacy Pretext Hat:** ~95 test (envanter sonrası ↑ revize)
  - `contrastive/losses` ~15
  - `feature_tap` ~10
  - `augmentations` ~10
  - `pretext/` ~25 (6 task + composite parametric)
  - `adapters/` ~25 (ConvLoRA + FreqGate + FreqGated + TaskRouted + inject + merge)
  - `pretrain/trainer` (SSLPretrainer) 5 (slow tag dahil)
  - `trainer/_core` (ContrastiveDetectionTrainer) ~5 indirekt
- **UX Façade Hat:** ~5 test (sadece import smoke)
  - `pipeline.py` + `discovery.py` + `auto_train` + `_config.py` için **dedicated test yok** (Aşama B'nin asıl boşluğu — INVENTORY §8 sonu)
- **Top-level smoke:** 2 (test_import.py)

**Coverage gap:** UX hat'ı + `_config.py` env var loading dedicated test eksik. Aşama B integration smoke suite bu boşluğu kapatacak.

**Faz 5 sonrası tahmini test sayısı:**
- §13.8 pipeline rewire: **+10 test** (Seçenek Y dispatcher + PipelineConfig forwarding)
- Faz 5.3 dual_teacher/ modülleri: **+60-80 test** (consensus loss + disagreement + cache I/O)
- Aşama B integration smoke (yeni — UX coverage gap için): **+15-20 test**
- **Toplam Faz 5.3 sonu:** ~810-840 test

### 4.5 Modern vs Legacy hat — kullanım yönergesi

| Kullanım | Doğru hat |
|---|---|
| Yeni paper deneyleri (Faz 5 tüm alt-fazlar) | **Modern** (`DenseSSLPretrainer`, `FinetuneDetectionTrainer`) |
| Hızlı UX/notebook tutorial (5 satırda SSL+finetune) | **Legacy** (`auto_train()`) — §13.8 sonrası **Modern** |
| Tek başına dense module kullanımı (custom trainer) | **Modern** (`dense/` modüllerinden import) |
| Single-output feature embedding (lineer prob, classification) | **Legacy** (`FeatureTap`) — modern hatta tap'ten manuel `nn.AdaptiveAvgPool2d` |
| Modular augmentation kullanımı | **Legacy** `augmentations/` (modern `dense/spatial_aug.py` sadece 2-view geometric) |

Plan v8'de UX hattı **paket içinde var** ama `dense/` modüllerine bağlı değil; v9'da §13.8 ile Modern + Legacy birleştirilir (Seçenek Y geriye uyumlu).

---

## 5. Faz-Faz Roadmap

### Faz 1 — Foundation ✅ KAPATILDI (201 test)
### Faz 2 — SAPS ✅ KAPATILDI (67 yeni)
### Faz 4 — Data ve eval altyapısı ✅ TAM İŞLEVSEL

| Adım | Dosya | Test | Durum |
|---|---|---|---|
| 4.4 | `data/label_fraction.py` | 30 | ✅ |
| 4.5 | `eval/linear_probe.py` | 28 | ✅ (early stopping) |
| 4.6 | `eval/run_matrix.py` | 26 | ✅ |
| 4.3 | `data/unified_loader.py` | 32 | ✅ |
| 4.7 smoke | Pretrain integration (Roboflow 1125) | — | ✅ |
| Yol 3 smoke | Detection finetune integration (Roboflow 1125) | — | ✅ |

### Yol 3 SMOKE SONUÇLARI (Roboflow Pothole 1125, 10-epoch finetune)

**Config:** imgsz=640, batch=16, freeze=10, unfreeze@5, bb_lr_scale=0.1.

| Backbone | mAP50 | mAP50-95 | Precision | Recall | crash post-train |
|---|---|---|---|---|---|
| COCO baseline | 0.4783 | 0.2182 | 0.475 | 0.548 | no |
| SSL-none (Faz 4.7) | 0.3709 | 0.1619 | 0.611 | 0.321 | no |
| SSL-within (Faz 4.7) | 0.3754 | 0.1577 | 0.438 | 0.371 | yes (Risk 16, caught) |

**Yorum:** COCO smoke ölçeğinde açık ara önde (beklenen, 900 image SSL yetersiz). SAPS-within marjinal kazanç. Faz 5'te ~186K pool ile **trend tersine dönmesi** gerekiyordu — v8 yazıldığında saf SSL'in COCO'yu geçeceği umuluyordu (Hipotez 15), v9'da bu hipotez revize edildi (Risk 15): **saf SSL geçmiyor, dual-teacher framework gerekli** (§1.5).

Detay paper supplementary §14.3'te.

### Faz 5 — DENEYLER (YENİ HİYERARŞİK YAPILANMA)

Faz 5 v9'da **6 alt-faza** bölündü. Her alt-faz ayrı bütçe + ayrı çıktı + ayrı paper section'a karşılık gelir.

#### Faz 5.1 — Saf SAPS ablation (paper-grade baseline)

**Amaç:** SAPS-only en iyi config'i bul. DT-SAPS'ın "baseline floor"u — paper'da "without dual-teacher" satırı.

**Hiyerarşik 3-stage (commit `bb6796d` YAML'ları):**

| Stage | Pool | Epoch | imgsz | Cells | A100 süre | $ (vast.ai @$1.5/h) |
|---|---|---|---|---|---|---|
| 5.1.1 — Smoke | 5K | 30 | 320 | 6 | ~1 saat | ~$1.5 |
| 5.1.2 — Coarse | 50K | 50 | 640 | 12 | ~50 saat | ~$75 |
| 5.1.3 — Fine | 186K | 100 | 640 | 9 | ~100 saat | ~$150 |
| **Toplam** | | | | **27** | **~150 saat** | **~$230** |

Eksenler: `saps_mode` × `saps_both_lambda` × `queue_update_strategy` × `saps_t_scale` (Stage 3'te). §10.24a list-DSL exclude ile redundant cell'ler elenir.

**Stage geçiş kriteri:**
- 5.1.1 → 5.1.2: Tüm 6 cell crash-free + acc@1 > 0.5 (smoke health)
- 5.1.2 → 5.1.3: Best (mode, queue) pair'i belirlenmiş, λ/t_scale Stage 3'te
- 5.1.3 → 5.2: En iyi 27-cell winner config sabitlenmiş

**Çıktı:** Paper Table 1: "Pure SAPS ablation results". Faz 5.1'in winner config'i Faz 5.2 ve Faz 5.3'ün **baseline argument'i** olur.

#### Faz 5.2 — Multi-backbone validation (yeni — generalization claim)

**Amaç:** Faz 5.1 winner config'in 6 farklı YOLO mimarisinde robust olduğunu göster. Paper'da "Our method generalizes across architectures" claim'i.

**Grid:**

| Backbone | Params | Faz 5.1 config | Cells |
|---|---|---|---|
| YOLOv8n | 3.0M | winning | 1 |
| YOLOv9n | 2.6M | winning | 1 |
| YOLOv10n | 2.7M | winning | 1 |
| YOLOv11n | 2.6M | winning | 1 |
| YOLOv12n | 2.6M | winning | 1 |
| YOLO26n | TBD | winning | 1 |
| **Toplam** | | | **6** |

Sabit: SAPS config Faz 5.1'den, 186K pool, 100 epoch.

**Süre:** ~6 cell × ~10-12 saat/cell = **~70 saat A100, ~$105**.

**Risk 21:** YOLOv9n/v10n/v11n/v12n/v26n batch_size/memory profileleri farklı — Stage 5.2 başlangıcında her backbone için batch_size auto-detect (smoke 1 epoch).

**Çıktı:** Paper Table 2: "Multi-backbone results". Reviewer "does this generalize?" sorusuna canlı cevap.

#### Faz 5.3 — Dual-teacher ablation (DT-SAPS, yeni — paper'ın asıl yenisi)

**Amaç:** C+D yaklaşımının (dual-teacher consensus + disagreement weighting) Faz 5.1 saf SAPS baseline'a katkısını ölç. DT-SAPS'ın en iyi config'i.

**Önkoşul:** COCO teacher feature cache hazır (§2.4). Smoke ile cache stratejisi finalize edilmiş (P3/P4/P5 vs P5-only).

**Augmentation pipeline kararı (§4.2 legacy/modern hat ayrımına bağlı):**

Faz 5.3'ün student augmentation pipeline'ı için iki seçenek var:

- **Seçenek A — Modern `dense/spatial_aug.py`:** Sadece geometric (random resized crop + hflip) + coord tracking. Dense SAPS loss'unun beklediği `view_a/view_b/coords_a/coords_b` çıktısı. **Sade, deterministik, paper-savunulabilir.**

- **Seçenek B — Modern + Legacy `augmentations/` hibrit:** SpatialTwoViewAugmentation geometric base + sonra `augmentations/` registry'sinden seçili photometric augmentation'lar (ColorJitter, GaussianBlur, Grayscale, Solarize) **post-step** olarak view_a ve view_b üzerine ayrı ayrı uygulanır. Coord tensors etkilenmez (photometric pixel-level). **Zengin, MoCo-v3/DINO standardı.**

**Karar:** **Seçenek B** — Faz 5.3 smoke (Stage 5.3.1) ile ablation eksen olarak test edilir:
- `aug_photometric` ∈ {`none`, `mocov3`, `dino`}
- `none` → saf geometric (Modern hat baseline)
- `mocov3` → preset `mocov3_v3_v2` (color_jitter[0.4,0.4,0.4,0.1] + grayscale[0.2] + blur[0.5])
- `dino` → preset `dino_v1` (color_jitter[0.4,0.4,0.4,0.1] + grayscale[0.2] + blur[1.0/0.1] + solarize[0.0/0.2])

Bu eksen Faz 5.1'in saf SAPS run'larında da ablation edilmiş olur, **iki kez harcanmaz** (Stage 5.3 SAPS-only cells'i Stage 5.1.3 winner'ı miras alır).

**Akademik gerekçe (§10.27 cache decision'la uyumlu):** Teacher cached features augmente edilmemiş orijinal image üzerinde; student'ın augmentation diversity'si tamamen view_a/view_b zincirinden (geometric + photometric). MoCo-v3/DINO-eşdeğer aug → contrastive signal güçlenir, distillation invariant teacher'a doğru.

**Implementation:** `dual_teacher_trainer.py` constructor `aug_photometric` parametresi alır, `dense/spatial_aug.py`'in geometric çıkışını `augmentations/presets.py::build_pipeline(preset_name)` ile post-process eder. View_a ve view_b **bağımsız** augment edilir.

**Ablation eksenleri:**

| Eksen | Değerler | Yorum |
|---|---|---|
| `teacher_combo` | none / coco_only / ssl_only / **both** | Hangi teacher(lar) aktif |
| `distill_form` | B / C / **B+C** | Loss formu (§10.28) |
| `α` (B weight) | 0.5 / 1.0 / 2.0 | Form B amplitude |
| `β` (C weight) | 0.5 / 1.0 / 2.0 | Form C amplitude |
| `γ` (D weight) | 0.0 / 1.0 | Disagreement on/off |
| `α_d` (D strength) | 0 / 0.5 / 1.0 / 2.0 | Disagreement hard-mining |
| `w_init` (B'nin w_coco başlangıcı) | 0.3 / 0.5 / 0.7 | Init bias |
| `aug_photometric` | none / mocov3 / dino | Photometric aug preset (§4.2 legacy `augmentations/`) |

Tam grid çok büyük (~700 cell). **List-DSL exclude** (§10.24a) ile redundant elenir:
- `teacher_combo=none` → α, β, γ, α_d, w_init irrelevant → 1 cell (saf SAPS regression test)
- `teacher_combo=coco_only` → β (dual KL) irrelevant, w irrelevant → 4 cell (α × γ × α_d kısmen)
- `teacher_combo=ssl_only` → benzer → 4 cell
- `teacher_combo=both`, `distill_form=B` → β irrelevant → α × γ × α_d × w_init = 3·2·4·3 = 72 cell
- `teacher_combo=both`, `distill_form=C` → α, w irrelevant → β × γ × α_d = 3·2·4 = 24 cell
- `teacher_combo=both`, `distill_form=B+C` → α × β × γ × α_d × w_init = 3·3·2·4·3 = 216 cell

**Toplam (after exclude):** ~320 cell. **Hâlâ büyük.** 3-stage hiyerarşi gerek:

| Stage | Pool | Epoch | Cells | A100 süre |
|---|---|---|---|---|
| 5.3.1 smoke | 5K | 30 | 8 (coarse exploration: teacher_combo × distill_form) | ~1.5 saat |
| 5.3.2 coarse | 50K | 50 | ~40 (winner combo × full α/β/γ/α_d, w_init=0.5) | ~150 saat |
| 5.3.3 fine | 186K | 100 | ~12 (winner combo + form + α/β/γ/α_d) | ~120 saat |
| **Toplam** | | | **~60** | **~270 saat** |

Bütçe: **~$405 vast.ai**. Maliyet Faz 5.1'den daha büyük çünkü dual-teacher ablation eksen sayısı fazla.

**Cache strategy karar noktası:** Stage 5.3.1 sonrası P3/P4/P5 ablation ile cache disk ihtiyacı netleşir, Stage 5.3.2 başlamadan finalize.

**Çıktı:** Paper Table 3 (DT-SAPS ablation grid) + Figure 2 (`w_coco` öğrenme eğrisi, paper'ın görsel hikayesi). Faz 5.3 winner config Faz 5.4'ün baseline'ı olur.

#### Faz 5.4 — External baselines (yeni — paper kabul için neredeyse zorunlu)

**Amaç:** DT-SAPS'i mevcut SOTA SSL/distillation method'larıyla karşılaştırmak.

**Baselines (literature taraması §14.1'den):**

| Method | Year | Port iş | A100 süre |
|---|---|---|---|
| MoCo-v3-YOLO | 2021/port | port + train | ~15 saat |
| SimCLR-YOLO | Aug 2025 | open source code → adapt | ~15 saat |
| CoMAD-YOLO | AAAI 2026 | port + train (en zor) | ~25 saat |
| COCO baseline | — | hâlihazırda mevcut (§11.8) | 0 |
| **Toplam** | | | **~55 saat** |

Hepsi Faz 5.1 winning backbone + config protokolüne uyacak (fair comparison).

**Bütçe:** ~$85 vast.ai.

**CoMAD-YOLO portu özellikle önemli** (paper'ın "vs SOTA" karşılaştırması). 3 SSL teacher → 1 YOLOv8n student, CoMAD'ın asymmetric masking + consensus gating mekanizması korunur. Bizim hybrid (supervised+SSL) bizim yaklaşımımızın 4-axis novelty'sini paper'da bu satırın **karşısında** gösterir.

**Çıktı:** Paper Table 4: "DT-SAPS vs SOTA SSL/distillation baselines".

#### Faz 5.5 — Foundation model karşılaştırması (referans, "to beat değil")

**Amaç:** DINOv2 (ViT-B, 1.4B image pretrain) ile karşılaştır. Paper'da "We don't aim to beat foundation models trained on 1000x more data; we propose a domain-specialized real-time alternative."

**Setup:**
- DINOv2-B ViT → feature linear probe ile YOLOv8n'e adapter
- Aynı downstream protokol (4-class dataset)
- 1 cell, paper'da reference satır

**Süre:** ~20 saat (DINOv2 inference + linear probe + finetune).

**Bütçe:** ~$30.

**Çıktı:** Paper Table 5: "Reference comparison with foundation models". Paper hikayesinin "honest positioning" parçası — DINOv2'yi geçme amacımız yok, bizim niche real-time + domain-specialized.

#### Faz 5.6 — Eval matrix + paper writing

**Süre:** 2-3 hafta (yazım yoğun, GPU sıfır).

**Çıktılar:**
- Tam eval matrix (§3 doldurulmuş)
- Paper draft (10-12 sayfa CVPR/T-ITS format)
- Paper supplementary:
  - §14 boyunca biriken "paper journey" anlatımı
  - Implementation lessons §10.x → supplementary B
  - Ablation full grid results → supplementary C
  - Code release (GitHub) — kütüphane paper-ready

### Faz 5 toplam bütçe (v9)

| Faz | GPU saat | Maliyet $ | Süre (calendar) |
|---|---|---|---|
| 5.1 — Saf SAPS | ~150 | ~$230 | 1-2 hafta |
| 5.2 — Multi-backbone | ~70 | ~$105 | 1 hafta |
| 5.3 — DT-SAPS | ~270 | ~$405 | 2-3 hafta |
| 5.4 — External baselines | ~55 | ~$85 | 1 hafta |
| 5.5 — Foundation comparison | ~20 | ~$30 | 1 hafta |
| 5.6 — Paper writing | 0 | $0 | 2-3 hafta |
| **Toplam** | **~565 saat** | **~$855** | **8-11 hafta** |

Submission target: bkz. §9.

---

## 6. Risk Listesi

| # | Risk | Durum |
|---|---|---|
| 1 | Spatial correspondence | ✅ Faz 1.4a |
| 2 | Multi-scale + AMP numerik | ✅ dense_loss autocast off |
| 3 | A2D2 lisans | Akademik izinli |
| 4 | Mapillary disk | Image listesi cache + 30K manifested |
| 5 | "X SSL methodu yok" | Related work tartış (§14.1 literatür tarama) |
| 6 | YOLOv9/v10/v11 neck | ✅ Faz 5.2 multi-backbone validation ile çözülecek |
| 7 | Queue update strategy | ✅ ÇÖZÜLDÜ — 3-strategy parametrik |
| 8 | Mock-encoder learning signal | Real YOLO testleri yeterli |
| 9 | SAPS-both α | ✅ ÇÖZÜLDÜ — `saps_both_lambda` parametresi |
| 10 | SAPS queue tagging memory | Negligible |
| 11 | mAP=0.79 yorumlama | ⏭ Atlandı, dataset basit (kullanıcı kararı). Pothole 5K geldiğinde tekrar bak. |
| 12 | Linear probe overfitting | ✅ early_stopping_patience |
| 13 | Roboflow `..` path | ✅ unified_loader fallback |
| 14 | Eval dataset değişimi | data.yaml swap hazır |
| 15 | SSL'in COCO'yu geçememesi | ⚠️ **GÖZLENDİ + KABUL EDİLDİ.** Faz 4.7 + Yol 3 smoke + §11.8 v2 smoke'ta tutarlı pattern: saf SSL geçmiyor. v9 strateji değişikliği: **dual-teacher framework (Faz 5.3 DT-SAPS) ile COCO'ya eşitlik/üstün**. Saf SSL paper'da "baseline floor", DT-SAPS asıl claim. |
| **16** | **PyTorch 2.x InferenceMode crash post-train** | ✅ ÇÖZÜLDÜ (v2) — `_safe_ema_sync` taint cleanup + plain `load_state_dict` (no `assign=True`). v1 production'da EMA aliasing → head collapse oluşturdu, §10.25'te root cause + v2 mechanism + paper-worthy methodological lesson. Regression: `tests/test_finetune_risk16.py` (5 test). Production validation §11.8. |
| **17** | **COCO teacher feature size mismatch** (yeni) | YOLOv8x teacher (P3: 80×80×128 → P5: 20×20×512), YOLOv8n student (P3: 80×80×64 → P5: 20×20×256) — channel mismatch. Mitigasyon: per-scale linear adapter (frozen teacher feature → student channel count). Faz 5.3 başlangıcında implement. |
| **18** | **Teacher cache disk space** (yeni) | 181K × 3-scale FP16 ≈ 360 GB. Mitigasyon: §2.4'te 3 alternatif (P5-only / subset cache / lokal NVMe). Stage 5.3.1 smoke sonrası karar. |
| **19** | **Dual-teacher trainer karmaşıklığı** (yeni) | `DualTeacherTrainer` `DenseSSLPretrainer`'ı sarmalar (composition, §10.30). 3 loss bileşeni (SAPS + B + C + D), 5+ hyperparameter. Debugging zorluğu artar. Mitigasyon: per-component unit test (60-80 yeni test), invariant assertions (each loss term > 0, gradients flow, no aliasing — Risk 16 lesson). |
| **20** | **Disagreement weighting (D) instability** (yeni) | `disagreement_weight = exp(α_d · d(f_coco, f_ssl))` — large α_d ile signal explosion riski. Mitigasyon: α_d ∈ {-1.0, -0.5, 0, 0.5, 1.0, 2.0} ablation (Stage 5.3.2; negatif = consensus yönü, §10.29). Numerik koruma: `clamp(weight, max=10.0)` + α_d clamp `[-3, 3]`. |
| **21** | **Multi-backbone batch_size scaling** (yeni) | YOLOv9n/v10n/v11n/v12n/v26n memory profileleri farklı. Aynı batch_size ile bazıları OOM yapabilir. Mitigasyon: Stage 5.2 başlangıcında her backbone için 1-epoch smoke ile batch_size auto-detect, sonra full run. |

---

## 7. Açık Kararlar

- ~~Pothole 5K eval split~~ → §13.4'te yapılacak iş olarak takip
- ~~SAPS t_scale~~ → ✅ Stage 5.1.3 ablation eksen ({0.5, 1.0, 2.0})
- ~~Multi-scale loss weights~~ → ✅ 1/3 sabit (Faz 1.5 kararı)
- ~~SAPS-both λ~~ → ✅ Stage 5.1.3 ablation ({0.5, 1.0, 2.0})
- ~~Queue update stratejisi~~ → ✅ Stage 5.1.2 ablation ({pooled, per_position, subsample})
- ~~Risk 16 kalıcı fix~~ → ✅ v2 çözüldü (§10.25)
- ~~Faz 3 reaktivasyon~~ → ✅ Faz 5.4'e merge edildi
- **Compute platform — vast.ai vs Colab Pro+** — Faz 5.1.1 smoke Colab'da, Stage 1.2+ vast.ai? Karar: 5.1.1 sonrası (smoke timing'i gerçek).
- **Multi-backbone listesi finalize** — 6 nano: v8n, v9n, v10n, v11n, v12n, v26n (kullanıcı kararı). YOLO26n teknik detaylar Stage 5.2 başlangıcında verify.
- **DT-SAPS hyperparam initial values** — α=1.0, β=1.0, γ=0.0 (D varsayılan kapalı), w_init=0.5, α_d=0.0 — bu Stage 5.3.1 smoke için. Coarse Stage 5.3.2'de eksenler açılır.
- **Teacher backbone** — YOLOv8x COCO (Karar K1 — büyük teacher, adapter ile size match) — Faz 5.3 öncesi indirme.

---

## 8. Önceki Benchmark Sonuçları (referans, korunur)

%10 etiket, BDD/pothole subset (eski deneyler):

| Yöntem | mAP50 | Status |
|---|---|---|
| B: COCO+CL (joint) | 0.6719 | Mevcut en iyi (eski reference) |
| A: COCO baseline | 0.6593 | Baseline (eski reference) |

**Yorum:** Bu rakamlar eski Yol 2 deneylerinden. Faz 5.6 final eval matrix'inde yeni baseline'larla karşılaştırma yapılacak. **Tarihsel kayıt, paper supplementary'sinde method-evolution context'i.**

**Yeni hedef X (v9):** DT-SAPS ≥ COCO baseline @ 0-no-dcs v6 %100 ≈ 0.6266 (§11.8'den). Yani DT-SAPS mAP50 ≥ 0.6266 traffic detection'da → paper'ın ana claim'i.

---

## 9. Submission planı

| Hafta | Faaliyet | Bağımlılık |
|---|---|---|
| 1 | §13 pending items: pool→local copy, pHash run, imagehash install, compute platform karar | Şimdi başlanabilir |
| 1-2 | Faz 5.1 Stage 1 (smoke 6 cell, 5K, 30 epoch) | Önkoşul: §13 done |
| 2-3 | Faz 5.1 Stage 2 (coarse 12 cell, 50K, 50 epoch) | Stage 1 yeşil |
| 3-5 | Faz 5.1 Stage 3 (fine 9 cell, 186K, 100 epoch) | Stage 2 winner belirlenmiş |
| 5-6 | Faz 5.2 Multi-backbone (6 cell, winning config) | Faz 5.1 winner |
| 6-9 | Faz 5.3 DT-SAPS (3-stage hiyerarşi, ~60 cell) | Teacher cache hazır, Faz 5.1 baseline mevcut |
| 9-10 | Faz 5.4 External baselines (CoMAD-YOLO port + SimCLR-YOLO + MoCo-v3) | DT-SAPS winner |
| 10 | Faz 5.5 DINOv2 reference | DT-SAPS winner |
| 10-13 | Faz 5.6 Paper writing | Tüm sonuçlar |

**Toplam:** 10-13 hafta. **Submission target:** §9'un sonu + 1 hafta revizyon = ~14-15 hafta sonra.

**Risk factor:** Bu plan **optimistik**. Gerçek pattern'lar paper writing 50%+ uzayabilir (ablation re-runs, reviewer feedback prep). Realist çerçeve: **14-17 hafta**.

**Compute kısıtı:** Vast.ai @ $1.5/saat senaryosu → toplam ~$855 (§5.6 son tablo). Eğer bütçe kısıtlıysa: Stage 5.1.3 + Stage 5.3.3 fine grid'leri **küçültülebilir** (örn. λ × t_scale eksenleri 2 değer × 2 değer = 4 cell yerine 3 × 3 = 9). Paper-grade'i etkilemez, sadece "fine granularity" azalır.

---

## 10. Implementation Lessons

Bu bölüm proje boyunca biriken **mühendislik ve metodoloji dersleri**. Paper supplementary §B olarak kullanılacak. Her lesson: (a) bağlam, (b) hangi versiyonda eklendi, (c) karar, (d) gerekirse alternatif düşünüldü mü.

### 10.1-10.18 (v1-v7 kayıtlı — özet)

V1-V7 boyunca biriken 18 lesson. Tam metinleri v1-v7 plan dokümanlarında — özet:

- **10.1** Multi-scale tap layer indexing (P3/P4/P5 = YOLO layer 4/6/9) sabit; gelecek backbone'larda revisit (Faz 5.2 multi-backbone validation buraya bakacak).
- **10.2** Queue size scaling (small dataset → small queue) — empirik 4-16× batch_size.
- **10.3** Momentum encoder EMA decay schedule — sabit 0.999 default, 0.99 küçük dataset.
- **10.4** Projection head 2-layer MLP, output L2-normalized.
- **10.5** Dense loss numerik koruma — InfoNCE'de log(softmax) yerine logsumexp.
- **10.6** Spatial augmentation reproducibility — `view_a` ve `view_b` aynı RNG state'ten paralel branch'lar.
- **10.7** AMP + dense loss — autocast OFF olması gerekiyor (loss FP32, gradient AMP).
- **10.8** Multi-scale loss weights 1/3 eşit; learned weights complexity ekleyip kazanım marjinal (Faz 1.5).
- **10.9** SAPS within-level scaling — feature spatial pyramid'ı correctness invariant.
- **10.10** Cross-level negative sampling — neighbor levels (P3↔P4, P4↔P5) — uzak level'lar (P3↔P5) zayıf signal.
- **10.11** Spatial pos_radius (cosine similarity threshold) — 0.07 sabit, dataset-agnostic.
- **10.12** Match mode strategies — `threshold` default, `top_k` queue dolma optimal değil.
- **10.13** Linear probe early stopping patience — 10 epoch, eval interval 5.
- **10.14** Eval matrix design — fair comparison invariant (tek değişen backbone init).
- **10.15** Data unified loader Roboflow `..` path fallback.
- **10.16** Test design pattern — invariant testler (math properties) > implementation testler (specific values).
- **10.17** Mock encoders development'ta zaman kazandırır, real backbone integration test ile validate edilir.
- **10.18** SAPS strict_neg parametresi — default False, ablation eksen.

### 10.19 Cross-trainer determinism (v8 yeni)

`torch.manual_seed(N)` iki ayrı `DenseSSLPretrainer` instance arasında **bit-eşit aynı** sonuç vermez. Sebep: constructor'daki `MomentumEncoder` deepcopy + augmentation construction + `_subsample_positions` her biri farklı miktarda RNG tüketir; iki instance arasındaki RNG state'leri ayrışır.

**Çıkarım:** Matematiksel invariant testleri **iki ayrı run karşılaştırması** üzerinden değil, **tek run'ın info dict'inden** doğrulanmalı.

### 10.20 λ-weighted vs convex combination (v8 yeni)

SAPS-both için `loss = α · loss_within + (1-α) · loss_cross` (convex) yerine `loss = loss_within + λ · loss_cross` (toplama+ağırlık) seçildi.

**Akademik gerekçe:** Default geriye uyumlu (λ=1 mevcut davranış), ablation kontrolü temiz (λ=0 = within-only), DINO/iBOT convention'a uyum, convex'in LR re-tuning ihtiyacından kaçınma.

### 10.21 Queue update strategy decision rationale (v8 yeni)

3 strategy: `pooled` (MoCo), `per_position` (DenseCL), `subsample` (PixPro). Default `pooled` (mevcut deneylerle tutarlı, geriye uyumlu). Faz 5.1.2 ablation'da hangisinin SAPS ile en iyi çalıştığı ölçülecek.

### 10.22 PyTorch 2.x InferenceMode crash pattern (v8 yeni — Risk 16)

**Pattern:** Ultralytics `model.train(...)` ile YOLOv8 detection finetune sonu, post-training `load_state_dict` çağrısında `'Inplace update to inference tensor outside InferenceMode is not allowed'` RuntimeError.

**Sebep zinciri:** Ultralytics validator forward'ı `torch.inference_mode()` içinde → BN running stats ve diğer buffer'lar InferenceMode flag taşır → sonraki `load_state_dict` in-place update yapmaya çalışıyor → crash.

**v8 önerilen fix (yanlış çıktı):** `with torch.no_grad():` + `.clone()` — sandbox forensics'te yetersiz olduğu kanıtlandı. **Detay §10.23'te.**

### 10.23 Risk 16 fix v1 — assign=True (v8 yeni, v9 REVERTED)

`_safe_ema_sync` helper'ı `load_state_dict(state, assign=True)` ile çağırıyor (PyTorch 2.1+). Destination tensor'ları referansla replace ediliyor → InferenceMode tainted tensor'lar drop ediliyor → crash önleniyor.

3 regression test eklendi (`test_baseline_crash_reproduces`, `test_assign_true_resolves_crash`, `test_safe_ema_sync_uses_assign_true`). Tüm testler geçti, sandbox'ta crash önlendi.

**REVERTED (v2 — §10.25):** Bu v1 fix isolated unit testlerde başarılı oldu (crash önlendi, target tensor'ları temizlendi) AMA production validation'da catastrophic — `assign=True` EMA aliasing yarattı, head weights collapse'una neden oldu. Yerine v2 strateji (taint cleanup + plain `load_state_dict`) geldi. Forensics §10.25'te, v1 commit `430cc75` repo'da kalır (akademik kayıt).

### 10.24 PretrainMatrix orchestrator design (v8 yeni)

`eval/run_matrix.py`'in **kız kardeşi** `pretrain/run_matrix.py`. YAML-driven cartesian grid expansion, CSV state machine (cell_id, seed, axes_json, metric, metric_value, status, elapsed_s, error, started_at, backbone_path). `cell_id = sha1[:12]` deterministik.

**Akademik kazanım — cell_id determinism (paper supplementary):** Stage 1 smoke'un 6 cell'i, Stage 2 coarse'un 12 cell'inin **ilk 6'sıyla aynı id'leri** taşır. Bu strong reproducibility claim'i — "Stage 1 results are a strict subset of Stage 2's grid".

### 10.24a List-DSL exclude (v8 yeni)

`exclude` parametresi list-of-dicts kabul eder, paper-quotable redundancy elimination. Örnek:

```yaml
exclude:
  - saps_mode: [none, within]
    saps_both_lambda: [0.5, 2.0]
```

Yorum: "λ irrelevant when mode≠both — keep only λ=1.0 representative". Reviewer'a savunulabilir.

### 10.25 Risk 16 fix v2 — taint cleanup, no aliasing (v9 yeni — paper supplementary'sinde lesson)

§10.23'te önerilen `load_state_dict(state, assign=True)` fix'i isolated unit testlerde başarılı oldu AMA production validation'da catastrophic — finetune sonrası `best.pt` model.22 head'inin tüm 31 weight'i = 0, mAP = 0.0000 (baseline aynı dataset'te 0.6266). Sandbox forensics + Colab production smoke ile root cause izole edildi.

**Root cause — EMA aliasing:**

`load_state_dict(state, assign=True)` destination Parameter wrapper'ının iç tensor'unu source referansı ile **REPLACE** eder (in-place copy değil, reference assignment). Sonuç: `ema.ema.param.data` ve `model.param.data` **aynı tensor objesidir** (kanıt: `data_ptr() == data_ptr()`).

Ultralytics' `ModelEMA.update()` döngüsü:
```python
for k, v in ema.ema.state_dict().items():
    v.mul_(d)                            # in-place: v *= d
    v.add_(msd[k].detach(), alpha=1 - d) # in-place: v += (1-d) * msd[k]
```

`v` ve `msd[k]` aynı tensor olduğunda:
- `v.mul_(d)` → tensor d ile çarpılır (model.param da scale-down, çünkü aynı tensor)
- `v.add_(msd[k] * (1-d))` → kendi (scaled) değerini kendine ekler
- Net: `v_new = d·v + (1-d)·(d·v) = d·v·(1 + (1-d)) ≈ 2d·v`

Ultralytics' decay schedule `decay(s) = 0.9999·(1 - exp(-s/2000))`, ilk step `d ≈ 0.0005` → `v_new ≈ 0.001·v` (**1000x scale-down per step**). 10 step'te `1.0 → 3.5e-24` (effective zero). Sandbox sim ile doğrulanmış (`tests/test_finetune_risk16.py::test_v2_survives_simulated_ema_updates`).

**Doğru çözüm (v2):** Destination'daki tainted tensor'ları rebuild et + plain `load_state_dict` (no `assign=True`):

```python
def _safe_ema_sync(self):
    # 1) Rebuild tainted params/buffers as detached clones
    for name, param in list(self.ema.ema.named_parameters()):
        if param.is_inference():
            *path, leaf = name.split(".")
            parent = self.ema.ema
            for p in path: parent = getattr(parent, p)
            parent._parameters[leaf] = nn.Parameter(param.detach().clone(),
                                                    requires_grad=param.requires_grad)
    for name, buf in list(self.ema.ema.named_buffers()):
        if buf.is_inference():
            *path, leaf = name.split(".")
            parent = self.ema.ema
            for p in path: parent = getattr(parent, p)
            parent._buffers[leaf] = buf.detach().clone()
    # 2) Plain load — copies VALUES into destination's existing tensors
    self.ema.ema.load_state_dict(self.model.state_dict())
```

v2 invariants (3 regression test ile guard'lı):
- **I1 — Crash prevention:** tainted destination → cleanup sonrası plain load çalışır
- **I2 — No aliasing:** `ema.param.data_ptr() != model.param.data_ptr()` (kritik)
- **I3 — EMA stable:** 10 update step sonrası weights kararlı (no collapse)

**Methodological lesson — paper-worthy:**

> Fix'in **unit test correctness'i** integration safety **garanti etmez**. v1 fix isolated bug-reproducing test'i geçti (`test_baseline_crash_reproduces` + `test_assign_true_resolves_crash`) ama production EMA update mechanism ile interaction'da catastrophic. Sandbox forensics'imde optimizer state invalidation'ı (Q3) test ettim AMA **EMA update mekanizmasının kendisi ile assign=True etkileşimi** test edilmedi — kritik blind spot.

> Generalize: bir fix'in unit test'i bug pattern'ını reprodüce ediyor olabilir, ama **fix'in yan etkilerini production-equivalent flow'da test etmeden** deploy etmek tehlikeli. Bizim case'de Risk 16 reprodüksiyonu izole bir minimal example (target taint + load_state_dict çağrısı) idi; **production'da bu çağrı bir Ultralytics ModelEMA update zincirinin içinde**, ve aliasing exploit edilebilir hâle geldi.

Future work: bu pattern'ı bir general principle olarak codify et — fix'lerin "post-fix invariant tests" (sadece original bug değil, fix'in yan etkileri) **production-representative integration test** ile validate edilmeli.

**Production validation closure (2026-05-12):** v2 fix Colab A100 üzerinde 12-epoch SSL + 12-epoch finetune E2E koşulunda doğrulandı. Head norm 4.438 (baseline 3.5), cls_loss 1.252 (baseline 1.05), mAP50 0.5465 (baseline 0.6266) — tam metrikler §11.8'de. Bu run hem v2 fix'in çalıştığının canlı kanıtı, hem de Faz 5 ablation matrix'ine geçiş için "pipeline-yeşil" damgası.

### 10.26 Hierarchical ablation strategy (v9 yeni)

Faz 5.1 3-stage hiyerarşi (smoke 5K/30ep → coarse 50K/50ep → fine 186K/100ep) **deterministik cell_id paylaşımı** sayesinde Stage 1 sonuçları Stage 2'nin **strict subset'i** olur. Aynı pattern Faz 5.3 dual-teacher ablation'da uygulanır.

**Akademik gerekçe:** Reviewer "neden tam grid değil?" sorusuna cevap — *"We progressively narrow the grid; Stage N is informed by Stage N-1's winner. Compute discipline + ablation interpretability."* PretrainMatrix orchestrator bunu zaten otomatik resume eder.

**Maliyet analizi:** Tam tek-stage grid (27 × full pool × 100 epoch) ≈ 540 saat A100. 3-stage hiyerarşi 150 saat. **3.6x tasarruf, ablation insight'ı aynı.**

### 10.27 Teacher feature caching tradeoff (v9 yeni)

Faz 5.3 DT-SAPS için COCO teacher feature'ları **augment edilmemiş orijinal image** üzerinde cache'lenir, augmentation diversity sadece student'ın view_a/view_b zincirinden gelir.

**Akademik gerekçe:**

1. **Compute economy:** Teacher inference O(N) bir kez, eğitim O(N × epochs) cache lookup. 100x tasarruf.
2. **Reproducibility:** Deterministik teacher feature → exact reproducibility.
3. **View diversity yeterli:** SAPS dense loss zaten 2-view kullanır (view_a, view_b student'lar augment'li); teacher "aug-invariant anchor" rol oynar (DINO/iBOT pattern'i).
4. **Trade-off:** Teacher'ın "augmentation'a tepkisi" öğrenilmez. Bizim için bu kayıp DEĞIL kazanç — invariant teacher istiyoruz.

**Karar:** Augmente teacher signal'i akademik olarak **gerek yok**; cache stratejisi paper-savunulabilir.

### 10.28 B+C kombinasyon distillation rationale (v9 yeni)

Faz 5.3 distillation loss'u **B (learned weighted L2) + C (dual KL)** kombinasyonu. 4 alternatif düşünüldü:

| Form | Mekanizma | Akademik value |
|---|---|---|
| A | Geometric mean target | Düşük (basit, novelty yok) |
| **B** | Learned weighted L2: `w·f_coco + (1-w)·f_ssl` | Orta (interpretable w) |
| **C** | Dual KL: KL(s‖coco) + KL(s‖ssl) | Orta (Hinton classic, dual extension) |
| D | Contrastive consensus (InfoNCE with consensus positive) | Yüksek (yeni) ama unstable |
| **B+C** | Hem feature-space (B) hem logit-space (C) | **Yüksek** (multi-level transfer) |

B+C kombinasyon seçimi:
1. **Multi-level distillation** literature'da güçlü pattern (DINO, BYOL multi-level loss).
2. **`w` öğrenilebilirliği** paper'ın görsel hikayesini güçlendirir (Figure: `w_coco` vs epoch).
3. **Ablation zenginliği:** α, β, w_init, scale (B-only vs C-only vs B+C) — paper'ın yarısı bu ablation'dan çıkar.
4. **Implementation maliyeti hafif** — birkaç loss term, mevcut framework'e fits.

### 10.29 Disagreement weighting mechanism (v9 yeni — implementasyonda genişletildi)

Yaklaşım D (`L_distill *= exp(α_d · d(f_coco, f_ssl))`) iki teacher'ın uyuşma/uyuşmazlık derecesine göre per-position signal'i modüle eder.

**İmplementasyon kararları (web literatür taraması ile, modül 3 `disagreement.py`):**

- **Metrik = per-position cosine distance** `d = 1 - cos(f_coco, f_ssl)`, kanal ekseninde → `[B,H,W]`. Feature distillation literatürü L2 yerine cosine tercih ediyor (CSKD, ReviewAT); ölçek-bağımsız, iki teacher'ın magnitude farkını değil semantik yön farkını ölçer. Detection KD'de spatial (per-position) weighting standardı (FGD/SAMKD/CSKD).

- **α_d işaret-agnostik — yön ablation'a bırakıldı.** Literatür bölünmüş: UEKD uyuşmazlığı *vurgular* (hard-mining), PAD hard-mining'in KD'de *zararlı* olduğunu deneysel gösterir. Tek formül her ikisini kapsar: α_d>0 uyuşmazlığı vurgular (UEKD/hard-mining), α_d=0 uniform (klasik), **α_d<0 uzlaşıyı vurgular (PAD/CoMAD consensus yönü)**. Hangi yönün traffic detection'da COCO'yu geçtiğine ablation karar verir.

- **AblATION EKSENİ GÜNCELLENDİ:** α_d ∈ {0, 0.5, 1.0, 2.0} → **α_d ∈ {-1.0, -0.5, 0, 0.5, 1.0, 2.0}** (negatif değerler consensus yönünü test eder). Risk 20 tablosu da bu eksene göre güncellendi.

**İki özgün katkı (literatürde yok — paper contribution):**
1. **Öğrenilebilir α_d** — sabit hyperparameter yerine `nn.Parameter`; model eğitimde uyuşmazlığa ne kadar ağırlık vereceğini kendisi öğrenir. α_d-vs-epoch trajektörü paper Figure; negatife yakınsama PAD'i, pozitife UEKD'yi deneysel doğrular.
2. **Per-scale α_d** — FPN seviyesi başına ayrı α_d (P3/P4/P5). Uzak küçük nesneler vs büyük yüzeyler için uyuşmazlık farklı işlev görebilir — weighter scale-aware, SAPS çekirdeğiyle tutarlı.

2×2 ablation alt-ekseni: `disagreement_mode ∈ {fixed, learnable}` × `per_scale ∈ {shared, per_scale}`. `fixed-shared` literatürün ayarı, `learnable-per_scale` bu paper'ın genişletmesi.

**Risk 20 — instability:** `exp(α_d · d)` patlayabilir. Mitigasyon: `clamp(weight, max=10.0)` + α_d clamp `[-3, 3]`.

**Akademik value:** Bu CoMAD'ın "consensus gating" mekanizmasından farklı — CoMAD uyumsuzluğu sabit şekilde filtreler; biz α_d işaretiyle yönü ablation'a bırakırız. Paper'ın "vs CoMAD" defansının kalbi.

### 10.31 DT-SAPS modül implementasyon kararları (v9 yeni — Faz 5.3 build)

`dual_teacher/` paketi 5 modül halinde inşa edildi (commit zinciri b31c79c..42a74c4, 103 yeni test, tam suite 952). Plana yazılı olmayan, web literatür taraması ile verilen implementasyon kararları:

1. **CocoTeacher adapter trainable.** Teacher backbone frozen, ama teacher→student kanal projeksiyonu (Risk 17 adapter) öğrenilebilir. Random frozen projeksiyon distillation sinyalini bozar; alignment öğrenilecek bir şeydir (linear-probe deseni). Cache ham (un-adapted) feature tutar — adapter eğitimde değişir.

2. **Form C = CWD channel-wise KL.** Dense feature map'te logit yok; CWD (Channel-wise KD, Shu et al. 2021) her kanalın H×W haritasını spatial softmax ile dağılıma çevirir. Dense prediction'da SOTA; softmax magnitude-scale farkını siler (teacher-teacher). T² ölçekleme.

3. **Form C dual KL ayrı, füzyonsuz.** `KL(s‖coco) + KL(s‖ssl)` — Form B (feature-space füzyon, w-ağırlıklı) ile Form C (logit-space dual) gerçek anlamda farklı mekanizmalar; B/C/B+C ablation ayrımı anlamlı kalır.

4. **SSL teacher = ayrı frozen pretrained backbone (Faz 5.1 SAPS winner), COCO ile simetrik.** Momentum encoder DEĞİL — o student'ın EMA'sı, bağımsız teacher değil; 'dual-teacher' + 'supervised+SSL hybrid' iddiası (§14.1, vs CoMAD) ancak ayrı frozen SSL teacher ile tutar. Bootstrap: SAPS pretrain → winner → DT-SAPS'ın SSL teacher'ı. SSL teacher YOLOv8n = student mimarisi → adapter gerekmez (Risk 17 sadece COCO YOLOv8x için).

5. **İki teacher feature modu.** Cache modu (TeacherCache, Faz 5.3 gerçek run 181K img × ~60 cell) ve canlı mod (frozen teacher forward, smoke/test). DualTeacherTrainer her ikisini de destekler; COCO adapter her iki modda uygulanır.

6. **teacher_combo tek kod yolu.** `none/coco_only/ssl_only/both` — tek-teacher combolarda ConsensusLoss'un iki teacher slotu aynı feature'ı alır (Form B target tek teacher'a iner, disagreement self-comparison → no-op). Ayrı kod yolu gerekmez.

---

### 10.32 External baseline port kararları (v9 yeni — Faz 5.4 build)

`baselines/` paketi 3 modül halinde inşa edildi (commit 1c5299c..b2d2ae9, 45 yeni test). Sadakat seviyesi kararı ve modül kararları:

**Sadakat seviyesi — çekirdek-mekanizma fair baseline.** Her baseline, bilinen bir SSL yönteminin ayırt edici mekanizmasını YOLOv8n backbone + aynı pool + aynı protokol ile uygular — orijinal kod tabanının birebir portu değil. Gerekçe: CoMAD ViT/ImageNet/classification, biz CNN/traffic/detection — birebir port zaten mimari olarak imkânsız; "X-YOLO" baseline'ları için literatürdeki standart pratik, X'in prensibini yeni backbone'a uygulamaktır.

**Modül kararları:**

1. **SimCLR-YOLO** — SimCLR çekirdeği: tek encoder, momentum/queue yok, iki view, global-pooled embedding, in-batch NT-Xent. Mevcut `NTXentLoss` + `ProjectionHead` yeniden kullanıldı.

2. **MoCo-v3-YOLO** — MoCo-v3 çekirdeği (MoCo-v2 farkıyla): momentum encoder + query-only prediction head (asimetrik), queue YOK, symmetric InfoNCE. `2τ` ölçekleme MoCo-v3 official'e sadık.

3. **CoMAD-YOLO** — CoMAD çekirdeği: 3 SSL teacher (repo'nun SimCLR/MoCo-v3/dense-SAPS backbone'ları — CoMAD'ın MAE+MoCo+iBOT diversity'sine analog), asymmetric masking (student yüksek-oran, teacher'lar hafif/farklı), joint consensus gating (parameter-free, per-position gate = cosine affinity × inter-teacher agreement), CWD channel-wise KL.

4. **CoMAD-YOLO bilinçli olarak tek-scale (P5).** Multi-scale + scale-aware reweighting DT-SAPS'ın kendi 4-axis novelty eksenlerinden biri (§14.1); baseline'a verilmedi. CoMAD da ViT olarak tek-scale — hem CoMAD'a sadık hem bizim ekseni korur.

5. **Baseline'lar `dual_teacher`'dan bağımsız.** Kavramsal ayrım: baseline'lar bizim yöntemimizi import etmez. CWD-KL ve cosine helper'ları her baseline kendi içinde tutar (küçük duplikasyon kabul — paket bağımsızlığı için).

---

### 10.30 Composition over inheritance for trainer extension (v9 yeni — Karar K5)

`DualTeacherTrainer` `DenseSSLPretrainer`'dan **inherit etmiyor**, onu **member** olarak tutuyor (composition).

```python
class DualTeacherTrainer:
    def __init__(self, ...):
        self.ssl_trainer = DenseSSLPretrainer(...)  # mevcut, dokunulmaz
        self.coco_teacher = CocoTeacher(...)         # yeni
        self.distill_loss = ConsensusLoss(...)       # yeni
    
    def train(self, ...):
        # SSL trainer'ın train loop'unu çağır + distill loss ekle
        # Hook pattern veya custom training loop
```

**Akademik gerekçe:**
- **Risk mitigation:** Mevcut 710 test yeşil kalır (DenseSSLPretrainer dokunulmazsa).
- **Independent ablation:** "Without dual-teacher" baseline = sadece `self.ssl_trainer.train()` çağrısı.
- **Code clarity:** Distillation logic ayrı dosya, ayrı test suite (60-80 yeni test).
- **Future-proof:** Distillation framework değişikliği SSL trainer'ı etkilemez.


### 10.33 Gerçek-run bug avı — mock testing kör noktaları (v9 yeni — paper supplementary'sinde lesson)

Kütüphane kullanılabilirlik audit'i + tam repo bug avı sırasında, 1019 yeşil birim/integration testine rağmen 2 gizli bug yalnızca **gerçek-run testing** ile tespit edildi. Her iki bug da aynı kök desende — bu bölüm o deseni methodological lesson olarak kaydeder.

**Bulunan 2 bug:**

| Bug | Modül | Kök sebep |
|---|---|---|
| CocoTeacher SSL checkpoint yükleyemiyor | dual_teacher | String weight `YOLO(weights).model` ile yükleniyordu; SAPS checkpoint `{model_state_dict}` formatında, `model` anahtarı yok → `KeyError` |
| `mocov3_v3_v2` / `dino_v1` preset'leri eksik | augmentations | Plan §5.3 bu preset isimlerine atıfta bulunuyor, `presets.py` sağlamıyordu → Faz 5.3'te `build_pipeline` `KeyError` verecekti |

**Ortak kök desen — mock testing kör noktası:**

Her iki modülün de birim testleri vardı ve geçiyordu. Ama:
- CocoTeacher testleri `weights=` parametresine **nn.Module** (mock encoder) veriyordu — gerçek kullanım yolu (string SSL-checkpoint path) hiç test edilmemişti.
- Augmentation preset testleri yalnızca **var olan** preset'leri parametrize ediyordu — plan'ın atıfta bulunduğu ama eksik isimler test kapsamı dışındaydı.

İki durumda da test "modül çalışıyor" diyordu, ama "kullanıcının/planın gerçekte varsayacağı yol çalışıyor" demiyordu.

**§10.25 ile ilişki — genelleme:**

§10.25 şunu kaydetmişti: bir fix'in unit testi bug pattern'ını reprodüce edebilir, ama fix'in yan etkilerini production flow'da test etmeden deploy etmek tehlikeli. §10.33 bunun kardeş dersi: bir feature'ın unit testi feature'ı **izole** doğrulayabilir, ama gerçek kullanım yolunu (gerçek girdi formatı, gerçek dosya, gerçek API tüketicisi) test etmeden "kapsanmış" saymak tehlikeli. Her ikisi de "mock/izole test ≠ entegre/gerçek test" prensibinin iki yüzü.

**Uygulanan metodoloji — gerçek-run bug avı:**

Tüm repo, gerçek YOLOv8n + gerçek dummy veri ile sistematik tarandı: DT-SAPS framework, baselines, `run_matrix` orchestrator'ları (grid/CSV/resume/exclude), `ssl_pool` ingestion, `dedup` pHash, ve kütüphanenin ilk dönem SSL hattı — `SSLPretrainer` 3 modu (CompositeTask / legacy rotation / saf CL), her pretext task tek tek (`FrequencyBandPrediction` dahil), `ContrastiveDetectionTrainer` gerçek YOLO `train()` döngüsü. Mock'un göremediği iki bug bu taramada çıktı; geri kalan tüm yüzey temiz doğrulandı.

**Paper-worthy lesson:** Test sayısı metriği (1019 test) modül-içi correctness'i ölçer ama API-tüketim yollarını ve plan-kod tutarlılığını ölçmez. Engineering rigor için: (a) feature testleri **gerçek girdi formatlarını** kapsamalı (mock nesneler değil), (b) plan dokümanının atıfta bulunduğu her API ismi koda karşı doğrulanmalı (plan-kod drift kontrolü), (c) release öncesi gerçek-run smoke — kullanıcının gerçekte koşturacağı yollar — birim test paketini tamamlar.

Her iki bug'ın kör noktası teste bağlandı: CocoTeacher'a `TestSSLCheckpointLoading` (gerçek SSL-checkpoint yükleme), `test_augmentations.py` parametrize 6 preset.

**Devam — Faz 5 eğitim smoke kampanyası (v9 ek).** Yukarıdaki repo bug avından sonra, her eğitim fazı (Faz 0/5.1/5.2/5.3/5.4/5.5) tam eğitimden önce gerçek A100'de küçük-ölçek smoke olarak koşuldu. Amaç aynı: birim testlerin kaçırdığı mantık hatalarını gerçek-run'da yakalamak. Kampanya 2 kök bug daha çıkardı — toplam 4:

| Bug | Modül | Faz | Commit | Kök sebep |
|---|---|---|---|---|
| CocoTeacher SSL checkpoint | dual_teacher | (audit) | `1a30142` | string weight `YOLO().model` ile yükleniyordu; SSL checkpoint `model` anahtarı taşımıyor |
| Augmentation preset eksik | augmentations | (audit) | `9973baa` | plan §5.3'ün atıfta bulunduğu preset isimleri `presets.py`'de yoktu |
| MultiScaleFeatureTap mimari-bağımlı | dense | 5.2 | `6b53873` | P3/P4/P5 hardcoded `{15,18,21}` v8 indeksleriyle bulunuyordu; v10/v11/v12/v26 farklı indeks kullanıyor — v12 gürültülü patladı, v10/v11/v26 SESSİZCE yanlış katmandan tap'lendi |
| ConsensusLoss Form C negatif KL | dual_teacher | 5.3 | `76c989a` | `_cwd_kl` KL'yi H×W spatial ekseni üzerinden toplamıyordu; KL'nin ≥0 garantisi yalnızca dağıtım ekseni toplamı için geçerli — per-position bırakılınca negatife sapıyor, B+C içine sızıyordu |

**İki sessiz bug — neden kritik.** MultiScaleFeatureTap ve ConsensusLoss Form C bug'larının ortak özelliği: **crash etmiyorlardı.** Faz 5.2 smoke'unda 6 mimariden 5'i "✓" verdi — ama gerçekte sadece 2'si (v8/v9) doğru katmandan tap'leniyordu; 3'ü sessizce yanlış feature üretiyordu. Faz 5.3'te `both/C` senaryosu "✓" verdi — ama distill loss 8 epoch boyunca negatife (`-0.0436`) sapıyordu. İkisi de "test geçti / smoke ✓" deyip geçilebilirdi. Yakalanmaları iki ayrı disipline bağlıydı: (a) smoke çıktısındaki sayısal sentinel'lere bakmak (param sayısı tutarlılığı, loss işareti), (b) şüpheli her sinyali varsayımla kapatmak yerine izole teşhisle kovalamak.

**Metodolojik ders — warmup'sız kısa smoke loss eğrisini gizler.** Faz 5.1'de ilk smoke 2-epoch, warmup=0 idi; loss ARTIYOR göründü — alarm. 15-epoch warmup=3 diagnostik koşusu bunun warmup-fazı gürültüsü olduğunu, epoch 3-4 sonrası loss'un düzgün düştüğünü gösterdi. Sonuç: warmup'sız 2-epoch smoke'lar loss eğrisini **okunamaz** kılar. ConsensusLoss Form C bug'ı tam bu yüzden ilk 2-epoch smoke'ta `-0.0042` olarak görünüp kaçabilirdi; 8-epoch + 2-warmup re-smoke onu net negatif trend olarak gösterdi. **Karar: tüm eğitim smoke'ları ≥8 epoch + ≥2 warmup — loss eğrisi gözlemlenebilir olmalı.**

**Teşhis aracı disiplini.** Kampanyada 2 kez (v12 ilk teşhisi, CoMAD teacher teşhisi) bir YOLO modeli `MultiScaleFeatureTap` olmadan düz çağrıldı; tam forward Detect head çıktısı (`[B,84,2100]`) verdi ve yanlış "bug" alarmı üretti. Kütüphane doğruydu, teşhis aracı kusurluydu. Ders: YOLO feature teşhisi her zaman tap üzerinden yapılmalı; düz model çağrısı backbone feature değil head çıktısı verir.

**Reproducibility scaffold.** Colab runtime reset `/content`'i siliyor (kampanya boyunca 3 kez); açılmış havuz + smoke alt-kümesi + SAPS checkpoint kayboluyor. `scripts/smoke_setup.py` (`93a2f6b`) üçünü kalıcı kaynaklarından (Drive part-tar'ları + repo kodu) idempotent yeniden kurar — reset tek-komut kurtarmaya iner.

---

### 10.34 SSL veri havuzu — parçalı arşivleme stratejisi (v9 yeni — Faz 0 build)

SSL pretraining havuzu (181,446 işlenmiş görüntü; BDD100K + Cityscapes + Mapillary + A2D2) 5 ham arşivden (273 GB) inşa edildi. İnşa mimarisi iki kez başarısız oldu, üçüncü stratejiyle oturdu — kayda değer bir lesson:

**Başarısız 1 — havuzu doğrudan Drive'a yazmak.** On binlerce küçük JPEG'i Google Drive mount'una yazmak dramatik yavaş (her dosya ayrı API çağrısı) ve oturum kopmasına açık.

**Başarısız 2 — tek büyük tar.** Havuz hızlı yerel diskte (`/mnt/local-scratch`) inşa edilip tek `.tar` (9.84 GB) olarak Drive'a kopyalandı. Drive senkronizasyonu büyük tek dosyada güvenilmez — `.tar` tam yüklenmeden oturum koparsa kısmi/bozuk dosya kalıyor, ve "yarısı inmiş" durumu programatik tespit edilemiyor.

**Oturmuş strateji — dataset-başına parçalı tar.** Havuz dataset başına ayrı `.tar`'a bölündü (`bdd100k.tar` 4.73 GB, `cityscapes.tar` 1.16 GB, `mapillary.tar` 1.91 GB, `a2d2.tar` 2.03 GB) + ayrı `manifest.parquet`. Her parça Drive'a yazıldıktan **hemen sonra** içeriği doğrulanır (jpg sayısı ↔ manifest). Avantaj: küçük parça = Drive senkronu hızlı ve doğrulanabilir biter; bir parça eksik inerse sadece o yeniden yazılır (318 dakikalık ingest değil); "yarısı inmiş" belirsizliği biter — her parça ya tam ya yok.

**Lesson:** Büyük türetilmiş veri ürünleri (dataset havuzları, cache'ler) için Drive gibi senkronizasyonu asenkron/opak depolara yazarken: (a) çok-küçük-dosya I/O'dan kaçın (arşivle), ama (b) tek dev arşivden de kaçın (parçala) — orta nokta dataset/shard-başına arşiv, her biri yazımdan sonra bütünlük-doğrulamalı. `scripts/build_ssl_pool.py` `--archive-to` ile her dataset sonrası checkpoint-arşivleme yapar.

---

## 11. Architecture Sentinels

Sentinel'ler: "bu değer bir daha bu mertebede çıkmazsa kütüphanede regression var" tripwire'ları.

### 11.1-11.6 (v1-v7 kayıtlı — özet)

- **11.1** Multi-scale tap feature shapes: P3 (80×80×64), P4 (40×40×128), P5 (20×20×256) — YOLOv8n. Channel count backbone'a göre değişir, spatial fixed.
- **11.2** Queue tensor dtype FP32, L2-normalized rows. Verify: `queue.norm(dim=1).std() < 0.01`.
- **11.3** Momentum encoder gradient leak yok: `param.requires_grad=False` her zaman.
- **11.4** Dense loss `pos > neg` invariant (random init bile, en az %20 gap).
- **11.5** SAPS scale-aware reweighting matrisi: diagonal dominant, satır toplamı ≈ 1.
- **11.6** Linear probe early stopping reproducible: aynı seed → aynı best epoch ± 1.

### 11.7 Yol 3 detection finetune reference numbers (Roboflow Pothole 1125, v8 kayıt)

10-epoch finetune, imgsz=640, batch=16, freeze=10, unfreeze@5, bb_lr_scale=0.1:

```python
{
    "coco_baseline":   {"mAP50": 0.4783, "mAP50_95": 0.2182, "P": 0.475, "R": 0.548},
    "ssl_none":        {"mAP50": 0.3709, "mAP50_95": 0.1619, "P": 0.611, "R": 0.321},
    "ssl_within":      {"mAP50": 0.3754, "mAP50_95": 0.1577, "P": 0.438, "R": 0.371},
}
```

**Pattern observation:** SSL pretrained backbone'ları COCO'dan precision higher / recall lower üretiyor (konservatif). Bu pattern §11.8 v2 smoke'da da gözlendi — paper'a "SSL produces more discriminative low-recall features" gözlemi.

### 11.8 Risk 16 v2 fix production validation (v9 yeni)

v2 fix'in gerçek koşullar altında doğrulandığı smoke run. Roboflow 4-class infrastructure dataset (3371 train / 548 val), 12-epoch SSL pretrain (5K BDD subset) + 12-epoch detection finetune, imgsz=320, batch=32, freeze=10, unfreeze@5, A100-40GB:

```python
{
    "coco_baseline":     {"mAP50": 0.6266, "mAP50_95": 0.3676,
                          "head_norm": 3.5,     "cls_loss_final": 1.05},
    "v1_aliased":        {"mAP50": 0.0000, "mAP50_95": 0.0000,
                          "head_norm": 0.00,    "cls_loss_final": 19480,
                          "note": "§10.25 EMA aliasing collapse — head→0"},
    "v2_ssl_both":       {"mAP50": 0.5465, "mAP50_95": 0.2832,
                          "head_norm": 4.438,   "cls_loss_final": 1.252,
                          "note": "v2 fix production-validated"},
}
```

**Pattern observation (v1 → v2):** Aynı dataset + config + epoch budget altında, v1 (aliased EMA) tüm metriklere "0" sonucunu verir; v2 (taint cleanup + plain load) baseline'la **aynı mertebede** sonuç verir (mAP50 farkı 0.08, cls_loss farkı %20). Bu fark **SSL'in henüz mAP'i geçemediğinin** göstergesi (5K image + 12 epoch yetersiz), **fix'in çalışmadığının değil**. SSL'in baseline'ı geçmesi için Faz 5.1.3 fine (full pool, 100 epoch) + Faz 5.3 DT-SAPS gerekir.

**Faz 5 readiness sentinel:** İleride yeni bir EMA-touching kod değişikliği yapıldıktan sonra, yukarıdaki üç metrik (mAP50, head_norm, cls_loss_final) ±20% aralıkta yeniden üretilebilmeli. Aksi takdirde regression var demektir — sandbox forensics (tests/test_finetune_risk16.py I3 invariant) ek olarak fail edebilir.

### 11.9 DT-SAPS smoke reference numbers (v9 yeni — Faz 5.3 öncesi placeholder)

Faz 5.3.1 smoke (DT-SAPS, 5K pool, 30 epoch, smoke config) tamamlandığında bu satıra reference numbers eklenecek. Şu an placeholder.

**Beklenen aralık (Faz 5.3.1 smoke için):**
- `dt_saps_smoke.mAP50`: ≥ Faz 5.1.1 saf SAPS smoke + 0.02 (en az marjinal kazanç olmalı)
- `learned_w_coco_final`: 0.3 < w < 0.7 (uç değerler → bir teacher tamamen baskın → ablation insight zayıf)
- `consensus_loss_decay`: monotonic decrease (training healthy)

### 11.10 Multi-backbone consistency sentinel (v9 yeni — Faz 5.2 öncesi placeholder)

Faz 5.2 her 6 backbone için saf SAPS sonucu çıktığında bu sentinel'a doldurulacak.

**Beklenen pattern:**
- Tüm 6 backbone içinde mAP50 std/mean < 0.1 (yöntem backbone-robust)
- COCO baseline ile karşılaştırmada **aynı yön** (tüm 6 backbone'da SSL > scratch veya SSL < COCO baseline) — yön tutarsızlığı = yöntem unstable

**Fair comparison invariant:** Her backbone için **aynı finetune config** (imgsz, batch, freeze, lr_scale, unfreeze_epoch), sadece pretrain backbone değişir.

### 11.11 Top-level `__init__.py` export gap sentinel (v9 yeni — INVENTORY §1 bulgusu)

INVENTORY.md §1.1 envanteri ortaya çıkardı: `src/yolo_contrastive/__init__.py` v0.2.0 modern hat'ı top-level export ETMİYOR. Kullanıcı `DenseSSLPretrainer`, `FinetuneDetectionTrainer`, `multi_scale_dense_loss`, `saps_within_loss` gibi paper'ın asıl modüllerine **alt-paket import** ile erişebiliyor:

```python
# ❌ from yolo_contrastive import DenseSSLPretrainer  # ImportError
# ✅ from yolo_contrastive.pretrain import DenseSSLPretrainer  # OK
```

**Mevcut top-level export'lar (sadece 14 sembol):**
```python
__all__ = [
    "__version__",
    "NTXentLoss", "build_contrastive_loss", "FeatureTap",        # Legacy hat
    "SSLFinetunePipeline", "PipelineConfig", "auto_train",       # UX façade
    "discover", "DatasetInfo", "TrainMode",                       # Discovery
    "YoloContrastiveError", "FeatureTapError",
    "ContrastiveLossError", "ConfigError", "PatchError",          # Exceptions
]
```

**Modern hat top-level değil.**

**Sentinel:** İleride `__init__.py` güncellenip modern hat top-level export eklendiğinde (§13.8 pipeline rewire sonrası anlamlı olur), şu invariant kontrol edilmeli:
- `from yolo_contrastive import auto_train` → modern hat default kullanır (Seçenek Y)
- `from yolo_contrastive import DenseSSLPretrainer` → ImportError VERMEMELI (eklendiyse)
- Mevcut tüm import'lar geriye uyumlu olmalı (3 hat aynı anda erişilebilir)

**Test bağlantısı:** `tests/test_import.py` şu an sadece 2 test içeriyor (`test_import_version`, `test_convenience_imports` — sadece legacy). §13.8 sonrası bu test dosyası genişletilmeli.

---

## 12. Şu Anki Durum

**Kütüphane: TAM İŞLEVSEL ✅**
**Risk 16 v2 ile production-validated ✅**
**Faz 5 ablation grid YAML'ları hazır ✅**
**INVENTORY.md ile kütüphanenin tüm yüzeyi haritalandı ✅** (commit `a45c18b`)

Kanıtlar:
- **726 test geçer durumda** (commit `87d3e9d` sonrası: 710 + 15 yeni `_run_detection` + 1 toplam fark)
- Foundation + SAPS + Eval altyapı + SSL pretrain E2E + Detection finetune E2E + PretrainMatrix orchestrator
- Ablation parametreleri hazır: `saps_mode`, `saps_both_lambda`, `saps_t_scale`, `queue_update_strategy`, `queue_subsample_n`, `early_stopping_patience`
- 3-stage ablation YAML (configs/pretrain/) commit `bb6796d`
- Risk 16 v2 fix smoke-validated (mAP50=0.5465, head_norm=4.438)

**Faz 5 alt-faz status:**

| Faz | Status | Sonraki adım |
|---|---|---|
| 5.1.1 Smoke | ⬜ Hazır, çalıştırılmadı | §13 pending → smoke run (~1 saat A100) |
| 5.1.2 Coarse | ⬜ YAML hazır | Önkoşul: 5.1.1 yeşil + 50K subset hazır |
| 5.1.3 Fine | ⬜ YAML template hazır | Önkoşul: 5.1.2 winner + pool→local copy |
| 5.2 Multi-backbone | ⬜ YAML yapılacak | Önkoşul: 5.1.3 winner |
| 5.3 DT-SAPS | ⬜ Modüller yazılacak | Önkoşul: 5.1.3 winner + teacher cache |
| 5.4 External baselines | ✅ Portlar yazıldı (b2d2ae9) | 3 trainer: CoMAD/SimCLR/MoCo-v3-YOLO, 45 test. Eğitim Faz 5.4 run'da |
| 5.5 DINOv2 reference | ⬜ Hazır kalır | Önkoşul: DT-SAPS winner |
| 5.6 Paper writing | ⬜ | Önkoşul: tüm sonuçlar |

**Sırada (öncelikli):**
- (Şimdi) §13 pending items: pHash run, pool→local copy, compute platform karar
- (Sonra) Faz 5.1.1 smoke (1 saat A100, $1.5)
- (Sonra) Karar: vast.ai vs Colab Pro+ Stage 5.1.2 için

**Faz 5 öncesi yapılacaklar (önkoşul):**
- ✅ Risk 16 kalıcı fix (v2, §10.25)
- ✅ SSL pretrain pool — 181,446 image (§2.1)
- ✅ Ablation grid YAML'ları (configs/pretrain/, §4)
- §13 pending items
- Pothole 5K dataset (kullanıcı paralel, §13.4)

---

## 13. Bekleyen Geçmiş İşler (yeni — yapılmamış ama unutulmaması gereken)

Akademik şeffaflık: proje boyunca bilerek atlanan veya başlandı-bitirilmedi olan adımlar. Faz 5 başlamadan veya devam ederken kapatılmalı.

### 13.1 pHash dedup gerçek pool run

**Status:** Modül var (commit `557e151`, 510 test), **gerçek pool üzerinde run yapılmadı**.

**Yapılacak:**
1. `pip install imagehash` (Colab session başlatıldığında her kez)
2. `python -m yolo_contrastive.data.dedup.compute --pool /content/ssl_pool_local --out hashes.parquet`
3. `python -m yolo_contrastive.data.dedup.check_exact --hashes hashes.parquet` — exact-dup raporu
4. **Sonuçta uygulanacak filter:** Manifest dedup'u — exact duplicates removed → `manifest_clean.parquet`

**Süre:** ~15 dakika (181K image hash compute + dedup).

**Önkoşul:** Pool lokal SSD'de (§13.2 önce).

**Bağımlılık:** Faz 5.1.2 coarse stage'inden önce zorunlu (50K subset üretilirken clean manifest gerek).

### 13.2 Pool→local SSD copy

**Status:** Pool Drive'da (`/content/drive/MyDrive/yolo-contrastive/ssl_pool/`). Drive read-throughput Stage 2/3 ölçeğinde **çok yavaş** (~10-20 MB/s vs lokal NVMe ~3000 MB/s, 150-300x). 50K-186K image × 100 epoch = milyarlarca read, Drive'da imkansız.

**Yapılacak:**
- Colab: `/content/ssl_pool_local/` klasörüne rsync-style kopya
- Vast.ai: instance NVMe'sine wget/aria2 ile

**Disk:** ~10 GB image + ~360 GB teacher cache (eğer P3/P4/P5 full) → vast.ai instance 100 GB+ olmalı.

**Süre:** ~15-30 dakika Drive→Colab; vast.ai indirme bandwidth'e bağlı.

**Bağımlılık:** Faz 5.1.2 coarse + Faz 5.1.3 fine + Faz 5.3 dual-teacher öncesi zorunlu.

### 13.3 Cross-set leakage check (pool ↔ downstream)

**Status:** `eval/leakage_check.py` CLI runner yazıldı (commit `502b069`, 14 test) — pool↔downstream cross-set leakage. Gerçek karşılaştırma pool lokalken yapılacak (§13.1/13.2 önkoşul).

**Yapılacak:**
1. SSL pool hash'leri (§13.1'in çıktısı)
2. Downstream dataset hash'leri (`0-no-dcs-no-aug v6` + Roboflow Pothole 1125 + (gelecek) Pothole 5K)
3. Cross-match → leakage raporu
4. Eğer leakage > %1 → pool'dan leaking image'lar **çıkar** (paper'da rapor edilir)

**Süre:** ~10 dakika (small downstream sets, hızlı match).

**Bağımlılık:** Faz 5.1.3 fine ile birlikte (paper-grade results öncesi). Smoke için kritik değil.

### 13.4 Pothole 5K finalize (kullanıcı)

**Status:** "Yarı hazır" (kullanıcı sözü). Detaylar belirsiz.

**Yapılacak (kullanıcı tarafı):**
- 5000 image collection complete
- Annotation (pothole class)
- Train/val split
- YOLO format export
- Roboflow upload veya direct repo

**Bağımlılık:** Faz 5.6 paper writing'de "two downstream datasets" claim'i için. Eğer Pothole 5K hazır değilse 4-class dataset tek başına yeterli (paper savunulabilir).

**Karar:** Pothole 5K **bloklayıcı değil**, bonus. Faz 5.6 öncesi hazırsa eklenir, değilse 4-class ile devam.

### 13.5 imagehash kütüphane install (operasyonel) ✅ KALICI ÇÖZÜM (commit `d3910a7`)

**Status:** ✅ ÇÖZÜLDÜ. `imagehash` (+ `pandas`, `pyarrow`) `pyproject.toml` `[project.optional-dependencies]` `pretrain` grubuna pinlendi (commit `d3910a7`). Artık `pip install -e ".[pretrain]"` hepsini kalıcı çeker — Colab session reset sonrası manuel kurulum gerekmez.

**Arka plan:** Aşama B Turn 3'te Colab session reset `imagehash`'i silmiş, `data/dedup/` collection ERROR vermişti. Tek seferlik `pip install imagehash` yerine dependency declaration kalıcı çözüm.

**Vast.ai:** `pip install -e ".[pretrain]"` aynı şekilde çalışır; ayrı `requirements.txt` gerekmez.

### 13.6 §11.7 v1 references preserved (akademik kayıt)

v8'den korunur: Roboflow Pothole 1125 + Yol 3 smoke reference numbers (§11.7'de). Paper supplementary §14.3'te "method evolution" hikayesinde geçer:

> "Initial experiments used Roboflow Pothole 1125 (interim dataset), demonstrating SSL methods' baseline behavior. Pothole 1125 was replaced by 4-class infrastructure dataset (3371 train / 548 val) in v8→v9 transition for: (a) more class diversity, (b) larger training set, (c) Pothole 5K parallel dataset preparation."

**Aksiyon:** Sadece dokümante etme, §11.7 silinmez.

### 13.7 `eval/run_matrix.py::_run_detection` STUB implement ✅ TAMAMLANDI (commit `87d3e9d`)

**Status:** ✅ TAMAMLANDI (Adım 2, commit `87d3e9d`). `_run_detection` Ultralytics YOLO + `FinetuneDetectionTrainer` ile implemente edildi; 15 yeni test (`tests/test_run_matrix_detection.py`). Env var lifecycle (set→run→restore), Risk 16 v2 fix-safe. Aşağıdaki orijinal plan akademik kayıt için korunur.

**Status (orijinal):** `eval/run_matrix.py` Faz 4.6'da yazıldı (26 test linear_probe için). Detection runner **stub**:

```python
def _run_detection(cell, hp):
    raise NotImplementedError(
        "Detection runner is a stub. Faz 5 will integrate Ultralytics YOLO.train."
    )
```

Plan v8 §4 ve v9 §4'te eval/run_matrix.py "✅ tamamlandı" olarak işaretlenmişti — **bu yanıltıcıydı**: linear_probe runner ✅, detection runner ⬜. v9'da düzeltildi (§4.1).

**Yapılacak (Faz 5.1.1 smoke öncesi):**

1. `_run_detection` implementation:
   ```python
   def _run_detection(cell, hp):
       """Detection runner — YOLO + FinetuneDetectionTrainer integration."""
       from ultralytics import YOLO
       from ..finetune import FinetuneDetectionTrainer
       
       backbone_ckpt = cell["method"]["backbone_ckpt"]
       data_yaml = cell["dataset"]["data_yaml"]
       
       # Env var pattern (FinetuneDetectionTrainer sözleşmesi)
       env_backup = {}
       env_overrides = {
           "YCL_PRETRAINED": backbone_ckpt,
           "YCL_FREEZE_BACKBONE": str(hp.get("freeze", 10)),
           "YCL_UNFREEZE_EPOCH": str(hp.get("unfreeze_epoch", 5)),
           "YCL_BACKBONE_LR_SCALE": str(hp.get("backbone_lr_scale", 0.5)),
       }
       for k, v in env_overrides.items():
           env_backup[k] = os.environ.get(k)
           os.environ[k] = v
       
       try:
           model = YOLO(hp.get("base_model", "yolov8n.pt"))
           results = model.train(
               data=data_yaml,
               epochs=hp.get("epochs", 30),
               imgsz=hp.get("imgsz", 640),
               batch=hp.get("batch", 16),
               device=hp.get("device", 0),
               trainer=FinetuneDetectionTrainer,
               project=hp.get("project", "/content/runs/eval_matrix"),
               name=f"cell_{cell['cell_id'][:8]}",
               exist_ok=True,
               verbose=False,
               plots=False,
           )
           return {
               "metric": "mAP50-95",
               "metric_value": float(results.box.map),
               "mAP50": float(results.box.map50),
               "precision": float(results.box.mp),
               "recall": float(results.box.mr),
           }
       finally:
           # Restore env
           for k, v in env_backup.items():
               if v is None:
                   os.environ.pop(k, None)
               else:
                   os.environ[k] = v
   ```

2. Test eklemeleri (~15 test):
   - `test_detection_runner_env_var_set_and_restored` — env state isolation
   - `test_detection_runner_invokes_finetune_trainer` — mock-based call validation
   - `test_detection_runner_captures_map_metrics` — return dict shape
   - `test_detection_runner_with_failed_train` — error propagation to run_matrix's continue-on-error
   - `test_detection_runner_cell_id_in_run_name` — output dir naming
   - Diğer 10 invariant test (Risk 16 v2 fix-safe integration kontrolleri)

3. RunMatrix integration test — gerçek bir 2-cell mock matrix detection cell ile koşar.

**Süre:** ~45-60 dakika implementation + test. Sandbox'ta refactor edilebilir.

**Bağımlılık:** Faz 5.1.1 smoke öncesi zorunlu — yoksa eval matrix manuel cell ile koşturulur (her 27 cell tek tek), ki bu hata kaynağıdır.

**Akademik gerekçe:** Reviewer "ablation grid kaç runda toplandı?" sorusuna "PretrainMatrix orchestrator otomatik resume ile tek komutla 27 cell" cevabı — sadece detection runner kurulduğunda mümkün.

### 13.8 `pipeline.py::run_ssl` modern hatla rewire ✅ TAMAMLANDI (commit `db6faf4`)

**Status:** ✅ TAMAMLANDI (Adım 3, commit `db6faf4`). `PipelineConfig.ssl_method` seçici eklendi (default `"dense"`); `run_ssl` dense/legacy dispatch + bilinmeyen değer→`ConfigError`. Production-validated 10 SAPS/dense alanı `PipelineConfig`'e eklendi. 11 yeni test (`tests/test_pipeline_ssl_method.py`, 2 slow). Cerrahi değişiklik — `run_finetune/run_detection/run/summary/auto_train` dokunulmadı. Tam suite 849 passed; Hat C C11/C13 dense default ile yeşil (regresyon yok). Aşağıdaki orijinal plan akademik kayıt için korunur.

**Status (orijinal):** `pipeline.py` v0.2.0'da yazıldı (legacy hat). `auto_train()` ve `SSLFinetunePipeline.run_ssl()` `SSLPretrainer` (legacy) kullanıyor. Modern `DenseSSLPretrainer` paket içinde mevcut ama UX'e bağlı değil.

**Karar (kullanıcı): Seçenek Y — geriye uyumlu rewire**

`PipelineConfig`'e yeni alan:
```python
@dataclass
class PipelineConfig:
    # ... mevcut alanlar ...
    ssl_method: Literal["dense", "legacy"] = "dense"  # ← yeni, default modern
    
    # Modern hat (dense) için yeni alanlar:
    ssl_saps_mode: str = "both"                   # SAPS variant
    ssl_saps_both_lambda: float = 1.0             # within+cross weight
    ssl_saps_t_scale: float = 1.0                 # cross-level temperature
    ssl_queue_strategy: str = "pooled"             # queue update strategy
    ssl_out_dim: int = 128                         # projection dim
    ssl_queue_size: int = 4096                     # K
    ssl_momentum_coef: float = 0.99                # m
    ssl_temperature: float = 0.2                   # τ
    ssl_n_query: int = 128                         # positions per image
    ssl_pos_radius: float = 0.07                   # match threshold
    ssl_match_mode: str = "threshold"              # match strategy
```

`run_ssl()` dispatcher:
```python
def run_ssl(self, images_dir=None, output=None):
    if self.cfg.ssl_method == "dense":
        from .pretrain import DenseSSLPretrainer
        pretrainer = DenseSSLPretrainer(
            model=self.cfg.model,
            out_dim=self.cfg.ssl_out_dim,
            queue_size=self.cfg.ssl_queue_size,
            momentum=self.cfg.ssl_momentum_coef,
            temperature=self.cfg.ssl_temperature,
            n_query=self.cfg.ssl_n_query,
            pos_radius=self.cfg.ssl_pos_radius,
            match_mode=self.cfg.ssl_match_mode,
            saps_mode=self.cfg.ssl_saps_mode,
            saps_both_lambda=self.cfg.ssl_saps_both_lambda,
            saps_t_scale=self.cfg.ssl_saps_t_scale,
            queue_update_strategy=self.cfg.ssl_queue_strategy,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
        )
    else:  # "legacy"
        from .pretrain import SSLPretrainer
        pretrainer = SSLPretrainer(
            model=self.cfg.model,
            aug_preset=self.cfg.ssl_aug_preset,
            lambda_cl=self.cfg.ssl_lambda_cl,
            lambda_rot=self.cfg.ssl_lambda_rot,
            temperature=self.cfg.ssl_temperature,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
        )
    
    # ... train kwargs ortak ...
```

**Geriye uyumluluk:** Mevcut `auto_train(ssl_method="legacy", ...)` çağrıları çalışmaya devam eder. Yeni `auto_train(...)` (default `ssl_method="dense"`) modern hat kullanır.

**Test eklemeleri (~10 test):**
- `test_pipeline_default_uses_dense` — default modern hat
- `test_pipeline_legacy_fallback` — `ssl_method="legacy"` SSLPretrainer
- `test_pipeline_dense_saps_params_forwarded` — SAPS kwargs DenseSSLPretrainer'a doğru gider
- `test_pipeline_dense_config_validation` — invalid SAPS değer → ConfigError
- `test_pipeline_legacy_compat_smoke` — eski API çağrısı yine çalışır
- Diğer 5 invariant test (env var isolation, output paths, dataset_info chain)

**Süre:** ~30-45 dakika refactor + test.

**Bağımlılık:** §13.7 sonrası. Bloklayıcı değil — Faz 5.1.1 smoke `pipeline.py` kullanmadan da koşturulabilir (direct `DenseSSLPretrainer.train()`). Bu UX iyileştirme paper sonrası community release için önemli.

**Akademik gerekçe:** Paper supplementary "code release" kısmında "5-line tutorial" örneği:
```python
from yolo_contrastive import auto_train
auto_train(data="data.yaml", unlabeled="/path/to/pool", epochs=100)
# → otomatik SSL pretrain (modern dense SAPS) + finetune
```
Bu Ultralytics-seviyesinde kolaylık — paper'ın "real-time deployment" + "easy to adopt" konumlanmasını destekler.

---

### 13.9 Aşama B — Integration Smoke Suite ✅ TAMAMLANDI (commit `206f3b8`..`151aca7`)

**Status:** ✅ TAMAMLANDI. INVENTORY §8 "UX coverage gap" kapatıldı — kütüphanenin 4 paralel API hattı uçtan uca smoke kapsamına alındı.

**Kapsam:** 64 senaryo / 112 test, `tests/integration/` altında 4 dosya + paylaşılan `conftest.py`:

| Hat | Modül grubu | Senaryo | Test | Commit |
|---|---|---|---|---|
| D — Data Infra | `data/` (label_fraction, unified_loader, ssl_pool, dedup) | 15 | 15 | `206f3b8` |
| A — Modern Dense | `dense/`, `pretrain/dense_trainer`, `eval/` | 17 | 33 | `b862a84` |
| B — Legacy Pretext | `pretext/`, `adapters/`, `pretrain/trainer` | 15 | 43 | `b4100db` |
| C — UX Façade | `pipeline.py`, `discovery.py`, top-level `__init__` | 17 | 21 | `151aca7` |

**Sonuç:** Tam suite 726→838 passed (+112). Sandbox API kalibrasyonu (project_knowledge_search) her turn öncesi yapıldı — yanlış varsayımlar (MANIFEST 10-sütun schema, pretext 6 task vs plandaki 9) teslim öncesi yakalandı.

**Akademik gerekçe:** Reviewer "kütüphane test edildi mi" sorusuna birim testlerin ötesinde 4-hat whole-path integration ağı gösterilebilir. C16 §11.11 sentinel testi modern hat top-level export gap'ini kasıtlı pinler.

### 13.10 Integration Smoke API Bulguları (akademik kayıt — paper supplementary §A)

Aşama B + Adım 3 sırasında keşfedilen, plana yazılmamış kütüphane davranışları. Paper supplementary §A "honest research process" damgasına katkı + kütüphane bakım kaydı:

1. **`print_every >= 1` zımni constraint.** `DenseSSLPretrainer.train()` ve `SSLPretrainer.train()` `print_every=0` verilince `dense_trainer.py:603` `epoch % print_every` → `ZeroDivisionError`. Docstring'de belirtilmemiş. Test ve kullanım kodu `print_every >= 1` vermeli. (Aşama B Turn 2'de yakalandı.)

2. **`MultiScaleProjectionHead` "caller normalizes" sözleşmesi.** Head L2-normalize ETMEZ — ham embedding döndürür. `dense/projection.py` docstring + `test_projection.py::test_output_not_normalized` ile doğrulanmış. Tüketen kod `F.normalize` uygular. (İlk test taslağı yanlış varsaymıştı; düzeltildi.)

3. **`_run_detection` device kaynağı.** Device `hp.get("device")` ile okunur — `cell`'den DEĞİL. Default `0` (CUDA); CPU runtime'da `ValueError`. Testler runtime-bağımsız olmak için `hp["device"]="cpu"` vermeli. (A15 dersi: GPU runtime default `0`'ı maskelemişti, CPU runtime bug'ı ortaya çıkardı.)

4. **Dense vs legacy checkpoint schema farkı.** `DenseSSLPretrainer` checkpoint: `model_state_dict` anahtarı + `extra.type="dense_ssl"`, top-level `type="ssl_pretrained"`. Legacy `SSLPretrainer` farklı schema. Checkpoint tüketen kod hat-spesifik olmalı.

5. **chore — `pytest.mark.slow` registration + dependency pin (commit `d3910a7`).** `[tool.pytest.ini_options] markers` ile `slow` marker kaydedildi (önceden `PytestUnknownMarkWarning`). `imagehash`/`pandas`/`pyarrow` `pretrain` extras'a pinlendi (§13.5).

---

## 14. Paper Hikayesi Şeffaflığı (yeni — paper supplementary §A)

Bu paper'ın asıl bilimsel hikayesi method'un kendisi kadar **yolculuğu**dur. v8'den v9'a evrim, Risk 16 v1→v2 forensics, ablation hiyerarşi rationale — hepsi paper'ın "honest research process" damgası. Bu §14 paper supplementary §A olarak yer alır.

### 14.1 Literature positioning özet (v9'da yapılan tarama)

Faz 5.3 DT-SAPS framework önerilmeden önce 3 literatür araması yapıldı (project_knowledge_search kullanılarak). 6 yakın çalışma incelendi:

| Yıl | Çalışma | Yakınlık | Bizim fark |
|---|---|---|---|
| 2026 AAAI | **CoMAD** | ★★★★★ | Multi-teacher SSL consensus + asymmetric masking. 3 SSL teacher (MAE+MoCo+iBOT). **All-SSL**, supervised teacher yok. ViT/ImageNet/classification. Bizim hybrid (supervised+SSL) + CNN/detection/traffic. Disagreement weighting **filtreliyor** (consensus gating), biz **vurguluyor** (hard mining). |
| 2025 Aug | **SimCLR-YOLO** | ★★★ | YOLO + SimCLR, **single teacher**, **global pooling** (dense değil), bizim multi-scale yok. Bizim multi-teacher + dense farklı. Faz 5.4 baseline. |
| 2024 Feb | **SSLKD road detection** | ★★★ | Road **segmentation** + dual teacher (büyük + SSL). Detection değil, scale-aware değil. Bizim scale-aware + detection farklı. |
| 2024 | **MSSD (YOLO self-distill)** | ★★ | YOLO multi-scale **self**-distillation (kendi katmanları arası). External teacher yok. Bizim external dual teacher farklı. |
| 2024 | **DTSKD** | ★★ | Dual teacher self-distillation, NLP/general. CV/detection değil. |
| Feb 2024 | SSLKD road | ★★ | Yukarıda. |

**Bizim 4-axis unique combination (literature'da olmayan):**
1. Supervised + SSL hybrid teacher (CoMAD all-SSL)
2. Dense multi-scale + scale-aware reweighting (MSSD self-distill; CoMAD ImageNet)
3. Disagreement weighting (CoMAD'ın tersine — vurguluyor, filtrelemiyor)
4. Real-time detection + traffic domain specialization

**Reviewer "CoMAD ile karşılaştırma var mı?" sorusuna cevap:** ✅ Faz 5.4'te CoMAD-YOLO portu (YOLOv8n student, 3 SSL teacher → CoMAD consensus gating mekanizması). Bizim hybrid teacher + scale-aware + disagreement vurgulama farklı.

### 14.2 Method naming evolution

v8'de paper başlığı: "Scale-Aware Dense Contrastive Pretraining for Real-Time Detection in Traffic Scenes" — saf SAPS odaklı.

v9'da DT-SAPS framework eklenince başlık genişledi: "Scale-Aware Dense Contrastive Pretraining **with Dual-Teacher Distillation** for Real-Time Detection in Traffic Scenes".

3 codename adayı (paper writing aşamasında finalize):
- **DT-SAPS** — "Dual-Teacher Scale-Aware Pretrained Specialization" (SAPS-centric, en akademik tutarlılık)
- **HiTeC-YOLO** — "Hybrid-Teacher Consensus YOLO" (novelty-centric, dual-teacher asıl)
- **ScaDis-Net** — "Scale-aware Dissensus-weighted Distillation" (en yenilikçi, en riskli)

**Karar:** Faz 5.6 paper writing'inde, asıl sonuçlar geldikten sonra ad finalize. Kullanıcı sözü: "İsim şu an önemsiz en son değiştiririz."

### 14.3 Method evolution narrative (paper supplementary için)

Paper supplementary §A'da, kronolojik olarak:

1. **Faz 1-2 (v1-v6):** Foundation + SAPS dense + multi-scale loss kuruldu. Saf SSL claim'i.
2. **Faz 4.7 + Yol 3 smoke (v7-v8):** Roboflow Pothole 1125 ile saf SSL'in COCO baseline'ı geçmediği gözlendi (Δ = -0.107 mAP50). Hipotez 15 sallandı.
3. **Risk 16 v1 fix attempt (v8):** PyTorch 2.x InferenceMode crash için `assign=True` fix. Isolated tests yeşil, production'da catastrophic EMA aliasing. Methodological lesson §10.25.
4. **Risk 16 v2 fix (v9):** Taint cleanup + plain load_state_dict. Sandbox + production validated. mAP50=0.5465 (baseline 0.6266) — saf SSL hâlâ baseline'ın altında ama crash'siz pipeline.
5. **Literatür tarama (v9):** CoMAD AAAI 2026 keşfi → bizim niche'in hâlâ unique olduğunu doğrulama (4-axis combination).
6. **DT-SAPS framework (v9):** C+D yaklaşım — dual-teacher consensus + disagreement weighting. Faz 5.3'ün asıl claim'i.
7. **External baselines (Faz 5.4):** CoMAD-YOLO + SimCLR-YOLO + MoCo-v3 portları — "vs SOTA" karşılaştırması paper tablosu.
8. **Foundation reference (Faz 5.5):** DINOv2 referans satır — "we don't aim to beat foundation models on 1000x more data; we propose a real-time domain-specialized alternative" konumlanması.
9. **Multi-backbone validation (Faz 5.2):** Generalization claim — 6 nano YOLO mimarisinde (v8/v9/v10/v11/v12/v26) tutarlı kazanım.
10. **INVENTORY.md tam library audit (v9):** UX iyileştirmesi öncesi 4 API hattı haritalandı (commit `a45c18b`). Plan v8'in `pretext/` ve `adapters/` "DONDURULDU" yanlış sınıflandırması düzeltildi (§4.2'de detaylı).

**Honest claim:** "Our pure SSL baseline does not surpass COCO supervised pretraining in our traffic detection setting. We propose DT-SAPS to bridge this gap via dual-teacher knowledge transfer, demonstrating that domain-specialized SSL+ can match or exceed general-purpose supervised pretraining when augmented with proper teacher distillation."

#### 14.3a Scope kararı — FrequencyBandPrediction (v9'da karar verildi)

Library audit (INVENTORY.md §2.2) repository'de `pretext/freq_band.py` modülünün **README'de "novel contribution" claim'i** içerdiğini ortaya çıkardı:

> *"FrequencyBandPrediction — novel frequency domain pretext task (first in image SSL for detection)."* — README.md, "Features" section.

Modül implement edilmiş + test edilmiş + dokümante (`tests/test_pretext.py` parametric coverage). Bu, paper'a ikinci bir novelty olarak dahil edilebilir veya gelecek work'e bırakılabilir.

**Karar (kullanıcı, 2026-05-14): Seçenek B — bahset ama eksen değil.**

**Gerekçe:**
- Paper'ın asıl yenisi **DT-SAPS dual-teacher framework**. Anlatım hatları temiz: 4-axis novelty (§14.1), Risk 16 forensics (§10.25), method evolution (Faz 1→DT-SAPS).
- FrequencyBandPrediction'ı eklemek bu odağı dağıtır. İki novelty'yi bir paper'a sıkıştırmak hikayeyi zayıflatır.
- Modül **gelecek work** olarak konumlanır. Bu paper kabul olduktan sonra ayrı bir kısa paper'a malzeme olur ("FreqBandPretext for YOLO Detection — Standalone Frequency-Domain SSL").

**Paper supplementary §A için anlatım:**

> *"During library development we implemented a frequency-domain pretext task (FrequencyBandPrediction — FFT band masking + IFFT reconstruction + band-id prediction; module `pretext/freq_band.py`). To the best of our knowledge this is the first application of frequency-band pretext to image SSL for object detection, building on time-series SSL precedents (TF-C, TRLS, FreMixer). However, ablating this orthogonal axis would dilute the present paper's focus on DT-SAPS dual-teacher distillation. We defer the FrequencyBandPrediction evaluation to a follow-up study, where it can be cleanly compared against standard pretext baselines (rotation, jigsaw, colorization) without entanglement with our hybrid-teacher framework."*

**Aksiyon planı:**
- Faz 5'te FrequencyBandPrediction ablation **eksen olarak yok**
- README'deki claim korunur (gelecek work çağrısı)
- Paper supplementary §A.3'te yukarıdaki paragraf yer alır
- Repo'da modül **bozulmadan korunur** — paper kabul sonrası ayrı paper için hazır

### 14.4 Compute budget transparency

Paper'da raporlanacak compute budget (vast.ai @ A100 $1.5/saat):

| Faz | GPU saat | $ tahmini |
|---|---|---|
| 5.1 — Saf SAPS | ~150 | ~$230 |
| 5.2 — Multi-backbone | ~70 | ~$105 |
| 5.3 — DT-SAPS | ~270 | ~$405 |
| 5.4 — External baselines | ~55 | ~$85 |
| 5.5 — Foundation reference | ~20 | ~$30 |
| **Toplam** | **~565 saat** | **~$855** |

"Single-academic-researcher accessible budget" — paper'da reviewer "neden tek seed?" sorusuna cevap: "Total compute budget < $1000 on commercial A100 cloud; multi-seed reserved for winning configuration only."

### 14.5 v8 → v9 doc evolution

v8 plan (302 satır, commit `bb6796d` sonrası) → v9 plan (bu doküman, ~900 satır). Akademik kayıt için git history korunur. v9 plan'ın "Değişiklikler" bölümü (en üstteki) paper supplementary §A.2 olarak kullanılır.

---

**SON.**
