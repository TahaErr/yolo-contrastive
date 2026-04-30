# YOLO-CONTRASTIVE — MASTER WORK PLAN v6

**Tez/Paper:** Scale-Aware Dense Contrastive Pretraining for Real-Time Detection in Traffic Scenes
**Hedef venue:** ITS-odaklı, CVPR-tier (T-ITS, ITSC, IV Symposium)
**Tahmini süre:** ~3.5 ay (Faz 1-5) + 3-4 hafta paper writing

**Değişiklikler (v5 → v6):**
- Faz 4'ün veri-bağımsız kısmı ✅ KAPATILDI (4.3, 4.4, 4.5, 4.6)
- Real-dataset smoke ✅ Roboflow Pothole 1125 üzerinde end-to-end yeşil
- Eval dataset değişikliği: Pothole 5K → Roboflow Pothole 1125 (interim), Pothole 5K paralel hazırlanmaya devam
- Faz 4.1, 4.2 askıya alındı (Pothole 5K geldiğinde yapılır)
- §10'a Faz 4 implementation lessons (Roboflow `..` pattern, mAP yorumu, multi-label binary)
- §11'e Roboflow dataset stats sentinel
- Şu anki sıra: **Faz 3 → Faz 5**

---

## 1. Stratejik Karar Özeti

### 1.1 Geçmiş yöntemlerin durumu

| Bileşen | Karar | Gerekçe |
|---|---|---|
| LoRA ailesi (ConvLoRA, FreqGate, FreqGated, TaskRouted) | ❄ DONDUR | Repo'da kalır, hiç çağrılmaz. Önceki deneyler hep baseline'ın altında kaldı. |
| 6 pretext task + CompositeTask | ❄ DONDUR | Repo'da kalır, ana pipeline'da kullanılmaz. |
| NT-Xent, FeatureTap, augmentation registry | ✅ ALTYAPI | Mevcut; yeni Dense CL altyapısının yanında yaşıyor. |
| ContrastiveDetectionTrainer | ✅ TUT | Joint training ana akış (B deneyi için). |

### 1.2 Final yöntem paketi (A + D)

**A — Foundation** ✅ TAMAMLANDI:
1. MoCo memory queue (K=65536) — `dense/queue.py`
2. Momentum encoder (m=0.999, EMA) — `dense/momentum_encoder.py`
3. Dense Spatial CL (per-position embedding, FPN P3-P5) — `dense/dense_loss.py`
4. Multi-scale FPN CL (her seviyede shared queue) — `dense/multi_scale_loss.py`

**D — Novel katkı (ulaşıma özel)** ✅ TAMAMLANDI:
- **Scale-Aware Positive Sampling (SAPS)**:
  - Within-image: cross-scale konumlar arası negatif olarak kullanılır (`dense/saps.py::saps_within_loss`)
  - Cross-image: queue elemanlarına scale tag, scale-similarity weighting (`dense/saps.py::saps_cross_loss`)
  - Trainer flag: `saps_mode in {none, within, cross, both}`

### 1.3 A bileşenlerinden atılanlar

| Atıldı | Sebep |
|---|---|
| Hard negative mining | Queue zaten 64K negatif → mining redundant |
| Multi-positive (4-8 view) | Compute 2-4x, marjinal kazanç |
| Detection-aware positives | Pothole 5K labeled için opsiyonel; ana pipeline'da yok |
| Temperature schedule | Marjinal, hikaye değeri yok |

### 1.4 D bileşenlerinden atılanlar

| Atıldı | Sebep |
|---|---|
| Temporal consistency (BDD video) | Dense CL ile registration sorunu, scope dağıtır. İleride bonus. |
| Metadata-driven invariance | "Scale" temasını sulandırır |
| Spatial-context priors | Tanımı muğlak, reviewer saldırı yüzeyi |

---

## 2. Datasets

### 2.1 SSL pretrain pool (~186K driving image) — HENÜZ İNDİRİLMEDİ

| Dataset | Boyut | Karakter |
|---|---|---|
| BDD100K | ~100K | ABD, çeşitli weather/time |
| Mapillary Vistas | ~25K | Global, geographic diversity |
| Cityscapes coarse | ~20K | Avrupa, urban |
| A2D2 (Audi) | ~41K | Almanya, highway+urban |

A2D2 lisansı: akademik kullanım izinli, atıf zorunlu.

**Disk indirme paralel işi — kullanıcı yan tarafta yürütüyor. Faz 5 öncesi ready olmalı.**

### 2.2 Downstream evaluation — INTERIM DATASET KULLANIMINDA

**Şu anki interim eval dataset:**
- **Roboflow Pothole 1125** (Roboflow workspace `taha-yasin-er-phd-studies-99wyc`)
- 1125 image: train 900 / valid 169 / test 56
- 640×640, auto-orient, no augmentation (Roboflow export)
- 2 class: `pothole` (id=0), `uncertain` (id=1)
- YOLO format, `data.yaml` provided
- Path: `/content/datasets/roboflow/data.yaml`

**Hedef eval dataset (paralel hazırlanıyor):**
- **Pothole 5K** (custom, 4K train + 1K test)
- Hazır olunca interim'i bu replace edecek; pipeline path swap ile çalışır
- Pothole 5K geldiğinde `uncertain` class drop edilir (`drop_classes=[1]` parametresi `MultiLabelImageDataset`'te hazır)

**Generic AD detection (genellik kanıtı):**
- BDD val — standart split
- Pretrain pool indirildikten sonra eval matrix'e eklenir

### 2.3 Data leakage kontrolü

Pretrain pool ↔ eval split arasında perceptual hash (pHash) dedup zorunlu. **Faz 4.2'de implementasyon (Pothole 5K + SSL pool hazır olunca).**

---

## 3. Eval Matrix

```
                    Roboflow 1125 (interim)   BDD val
                    ────────────────────────  ─────────────
                    %10 %25 %50 %100          %1 %5 %10 %25 %100

Scratch             ▢   ▢   ▢   ▢             ▢  ▢  ▢   ▢   ▢
COCO baseline       ▢   ▢   ▢   ▢             ▢  ▢  ▢   ▢   ▢
DINOv2 / MoCo-v3*   ▢   ▢   ▢   ▢             ▢  ▢  ▢   ▢   ▢
Ours (A+D)          ▢   ▢   ▢   ▢             ▢  ▢  ▢   ▢   ▢

Linear probe %100   ▢ (4 method × 2 dataset = 8 run)
Ablation A only     ▢ (label-fraction sweep, 1 dataset)
Ablation D only     ▢ (label-fraction sweep, 1 dataset)
```

*External baselines (DINOv2/MoCo-v3) **self-implemented in our pipeline** — not off-the-shelf checkpoint loader.

**Toplam:** 4 method × 9 fraction × 2 dataset = 72 detection run + linear probe + ablations ≈ 85-90 run.

### 3.1 Fair comparison protokolü

Her baseline için fine-tune hyperparameters AYNI:
- Aynı epoch, batch, augmentation
- Aynı LR schedule
- Aynı seed (42)
- Değişen tek değişken: backbone init

---

## 4. Modül Haritası

```
src/yolo_contrastive/
  dense/                         ✅ TAMAMLANDI (Faz 1+2)
    __init__.py
    multi_scale_tap.py           ✓ (Faz 1.1, 23 test)
    queue.py                     ✓ (Faz 1.2, 32 test)
    momentum_encoder.py          ✓ (Faz 1.3, 26 test)
    spatial_aug.py               ✓ (Faz 1.4a, 26 test)
    dense_loss.py                ✓ (Faz 1.4b, 29 test)
    multi_scale_loss.py          ✓ (Faz 1.5, 19 test)
    projection.py                ✓ (Faz 1.6a, 21 test)
    saps.py                      ✓ (Faz 2.1+2.2, 37 test)

  pretrain/
    trainer.py                   ✓ (mevcut SSLPretrainer dokunulmuyor)
    dense_trainer.py             ✓ (Faz 1.6b + 2.3, 36 + 8 test)

  data/                          ✅ KISMEN TAMAMLANDI (4.3, 4.4)
    __init__.py
    label_fraction.py            ✓ (Faz 4.4, 30 test)
    unified_loader.py            ✓ (Faz 4.3, 32 test) ← Roboflow `..` fallback dahil

  eval/                          ✅ TAMAMLANDI (4.5, 4.6)
    __init__.py
    linear_probe.py              ✓ (Faz 4.5, 24 test)
    run_matrix.py                ✓ (Faz 4.6, 26 test)
    leakage_check.py             ⬜ (Faz 4.2, Pothole 5K bekliyor)

  baselines/                     ⬜ Faz 3 (ŞİMDİ SIRADA)
    mocov3_pipeline.py
    dinov2_pipeline.py

  adapters/                      ❄ DONDURULDU (kalır, çağrılmaz)
  pretext/                       ❄ DONDURULDU (kalır, çağrılmaz)
```

---

## 5. Faz-Faz Roadmap

### Faz 1 — Foundation ✅ KAPATILDI

| Adım | Dosya | Test |
|---|---|---|
| 1.1 | `dense/multi_scale_tap.py` | 23 |
| 1.2 | `dense/queue.py` | 32 |
| 1.3 | `dense/momentum_encoder.py` | 26 |
| 1.4a | `dense/spatial_aug.py` | 26 |
| 1.4b | `dense/dense_loss.py` | 29 |
| 1.5 | `dense/multi_scale_loss.py` | 19 |
| 1.6a | `dense/projection.py` | 21 |
| 1.6b | `pretrain/dense_trainer.py` | 18 + 7 (real YOLO) |
| | **Total** | **201** |

### Faz 2 — SAPS ✅ KAPATILDI

| Adım | Dosya | Test |
|---|---|---|
| 2.1 | `dense/saps.py::saps_within_loss` | 21 |
| 2.2 | `dense/saps.py::saps_cross_loss` | 16 |
| 2.3 | `pretrain/dense_trainer.py` SAPS flag | 18 + 1 (real YOLO) |
| | **Total** | **56 yeni** (cumulative: 268) |

### Faz 4 — Data ve eval altyapısı ✅ VERI-BAĞIMSIZ KISIM KAPATILDI

| Adım | Dosya | Test | Durum |
|---|---|---|---|
| 4.4 | `data/label_fraction.py` | 30 | ✅ |
| 4.5 | `eval/linear_probe.py` | 24 | ✅ |
| 4.6 | `eval/run_matrix.py` | 26 | ✅ |
| 4.3 | `data/unified_loader.py` | 32 | ✅ (Roboflow `..` fix dahil) |
| smoke | Roboflow Pothole 1125 end-to-end | — | ✅ Real-data smoke yeşil |
| 4.1 | `data/pothole_split.py` | — | ⏳ Pothole 5K bekliyor |
| 4.2 | `eval/leakage_check.py` | — | ⏳ Pothole 5K + SSL pool bekliyor |
| | **Faz 4 yapılan total** | **112 yeni** | (cumulative: 380) |

**Faz 4 smoke özeti (real Roboflow Pothole 1125):**
- COCO-pretrained YOLOv8n → 3-epoch linear probe → val mAP=0.7906
- 25% subset probe (LabelFractionSplitter) → val mAP=0.7686
- RunMatrix 2-cell (fraction × seed): both ok, CSV writeup yeşil

⚠ **Smoke gözlemi (Faz 5'te dikkat):** mAP=0.79 random-init head ile yüksek görünüyor. İki olası sebep:
1. COCO weights pothole'a yakın road/asphalt features taşıyor
2. Class imbalance — constant-1 baseline AP'si dataset positive rate'e yakın çıkar

Faz 5 başında **constant prediction baseline** + **class distribution analizi** yapılacak; mAP yorumu o zaman kalibre edilir.

### Faz 3 — External baselines (≈1.5-2 hafta) ⬜ ŞİMDİ SIRADA

**Karar:** Self-implement (option b). Vanilla checkpoint loader DEĞİL.

| Adım | Dosya | Çıktı |
|---|---|---|
| 3.1 | `baselines/mocov3_pipeline.py` | MoCo-v3 our-pipeline'da (queue + momentum + global pooled CL) |
| 3.2 | `baselines/dinov2_pipeline.py` | DINOv2-style: student-teacher self-distillation, multi-crop |

**Reviewer cevabı:** "We re-implement these baselines in the YOLO setting for fair comparison; vanilla checkpoints are architecture-incompatible."

**Faz 3 tasarım soruları (kararlaştırılacak):**
- MoCo-v3'te ViT yerine YOLO backbone — convention'dan sapma, justify gerekir
- DINOv2 self-distillation kayba ne kadar simplifiy edilecek (full multi-crop pahalı)
- Her ikisi için pretrain pool aynı SSL veri seti olacak

### Faz 5 — Deneyler (≈2 hafta compute + 1 hafta analiz)

```
1. Class distribution analiz (Roboflow Pothole 1125 + sonra Pothole 5K)
   + constant prediction baseline (mAP yorumlama kalibrasyonu)
2. Leakage check (Faz 4.2'de implementasyon)
3. Pothole 5K split (Faz 4.1'de implementasyon, dataset gelince)
4. SSL pretrain — Ours full A+D (~186K, 100 epoch)
5. SSL pretrain — Ours ablations (A-only, no SAPS, P3-only, etc.)
6. SSL pretrain — MoCo-v3 baseline (Faz 3.1 hazır)
7. SSL pretrain — DINOv2 bridge (Faz 3.2 hazır)
8. Eval matrix — 72 detection run
9. Linear probe (8 run)
10. Sonuç toplama + grafik + analiz
```

---

## 6. Risk Listesi

| # | Risk | Durum / Önlem |
|---|---|---|
| 1 | Spatial correspondence augmentation | ✅ Çözüldü Faz 1.4a |
| 2 | Multi-scale + AMP numerik problem | ✅ Çözüldü — `dense_loss` autocast off + force_fp32 |
| 3 | A2D2 lisans | Akademik izinli, paper'da Audi atfı ekle |
| 4 | Mapillary disk maliyeti | Image listesi cache'le |
| 5 | Reviewer "X SSL methodu yok" | Related work'te tartış, scope dışı |
| 6 | YOLOv9/v10/v11 neck farkları | Faz 1.6'da v8 ana, ek tabloda v10/v11 |
| 7 | Queue strategy (pooled push) | **Faz 5'te incele:** `_step` mean-pooled key push yapıyor (B vec/step). |
| 8 | Mock-encoder learning signal | Bilgi: Real YOLOv8 testleri kanıt için yeterli. |
| 9 | SAPS-both `α=0.5` varsayılanı | **Faz 5 ablation:** `both_alpha` parametresi eklenebilir. |
| 10 | SAPS queue tagging memory | Negligible (~512KB at K=65536). |
| 11 | mAP=0.79 anlamsızlık riski (yeni) | **Faz 5 başında çöz:** class distribution analizi + constant baseline. Eğer dataset pozitif oranı yüksekse mAP yanıltıcı, F1@τ veya per-class breakdown ekle. |
| 12 | Linear probe overfitting (subset, yeni) | Step 3 smoke'da epoch=2 best, epoch=3 düşmüş. Faz 5 eval matrix'inde küçük subset için early stopping eklenebilir. |
| 13 | Roboflow `..` path quirk (yeni) | ✅ Çözüldü `unified_loader._resolve_split` fallback ile. |
| 14 | Eval dataset değişikliği (yeni) | Pothole 5K geldiğinde data.yaml path swap ile geçiş. `drop_classes=[1]` parametresi `uncertain`'i çıkarmak için hazır. |

---

## 7. Açık Kararlar (sonra)

- **Pothole 5K eval split tam protokolü** — Pothole 5K ulaşınca Faz 4.1'de
- **SAPS t_scale değeri** — Faz 5 ablation belirler ({0.5, 1.0, 2.0, ∞})
- **Multi-scale loss weights** — başlangıç 1/3 eşit, ablation ile fine-tune
- **Queue update stratejisi** — pooled vs per-position vs adaptive (Faz 5 ablation)
- **SAPS-both α** — `α·loss_within + (1-α)·loss_cross`, default α=0.5 (toplam), Faz 5'te parametrize edilebilir
- **mAP yorumlama** (yeni) — class distribution analizi sonrası karar; gerekirse F1@τ ek metrik
- **Faz 3 baseline mimarisi** (yeni) — MoCo-v3'te YOLO backbone vs ViT, Faz 3.1 öncesi karar

---

## 8. Önceki Benchmark Sonuçları (referans)

%10 etiket, BDD/pothole subset:

| Yöntem | mAP50 | vs A | Status |
|---|---|---|---|
| B: COCO+CL (joint) | 0.6719 | +1.26pp | ✅ Mevcut en iyi |
| A: COCO baseline | 0.6593 | — | Baseline |
| G: Plain ConvLoRA | 0.5431 | -11.6pp | ❄ Atıldı |
| F: TaskRouted | 0.5232 | -13.6pp | ❄ Atıldı |
| E: FreqGated | 0.4881 | -17.1pp | ❄ Atıldı |
| C/D: Full SSL | <0.49 | <-17pp | ❄ Atıldı (forgetting) |

**Yeni hedef X:** B (0.6719). A+D'nin B'yi geçmesi → paper'ın ana ölçütü.

**NOT:** Bu rakamlar **detection mAP50** (object detection IoU=0.5). Faz 4 smoke'taki **mAP=0.79** **classification mAP** (multi-label image-level). İki metric farklı ölçek, kıyaslanamaz.

---

## 9. Submission planı

| Hafta | Faaliyet | Durum |
|---|---|---|
| 1-3 | Faz 1 (Foundation) | ✅ Tamamlandı |
| 3-5 | Faz 2 (SAPS) | ✅ Tamamlandı |
| 5-6 | Faz 4 (Data/eval, veri-bağımsız) | ✅ Tamamlandı + real smoke yeşil |
| 6-8 | **Faz 3 (External baselines)** | ⏳ Sırada |
| 8-9 | SSL veri indirme (Faz 5 prep) | 🟡 Kullanıcı paralel yürütüyor |
| 9-11 | Faz 5 pretrain compute | ⬜ |
| 11-12 | Faz 5 eval + analiz | ⬜ |
| 12-15 | Paper writing | ⬜ |

**Aday venue deadlines:** T-ITS (rolling), ITSC (Mart-Nisan), IV (Aralık-Ocak)

---

## 10. Implementation Lessons

Foundation + SAPS + Faz 4 implementation'ından çıkan bilgi havuzu:

### 10.1 Queue dolma artefaktı (Faz 1)

İlk birkaç step'te queue boş veya yarı dolu. Loss "bu dönemde düşük" görünür çünkü negatif sayısı az. Eğitim metrikleri grafiklerinde ilk N step'i "warmup" olarak işaretle.

### 10.2 mean_pos_sim trend (Faz 1)

Eğitimin başında düşük (~0.0-0.2), sonunda yüksek (0.4-0.7). 0.9+ ise overfitting. `acc_top1` ile birlikte loglanmalı.

### 10.3 Projection head EMA (Faz 1)

MoCo-v3 / BYOL standardı: encoder + projection head **ikisi de** EMA'lı. `MomentumEncoder` helper dict input desteklemediği için projection head EMA manuel.

### 10.4 Mock vs real encoder (Faz 1)

Real YOLOv8 smoke testleri convergence sinyali için kullanılıyor; mock 23-layer Sequential learning signal üretmiyor.

### 10.5 Drop-in compatibility (Faz 1)

DenseSSLPretrainer'ın kaydettiği backbone'u `FinetuneDetectionTrainer.load_backbone(backbone_only=True)` sorunsuz yüklüyor. Faz 5 eval matrix için kritik. Aynı load_backbone Faz 4.5 LinearProbeTrainer'da da kullanıldı, ikisinde de doğru çalışıyor.

### 10.6 SAPS info schema farklılığı (Faz 2.3)

`saps_mode` değerine göre `_step` info dict yapısı farklı:
- `none/within/cross` → flat: `info[level]["acc_top1"]`
- `both` → nested: `info["within"][level]`, `info["cross"][level]`

Trainer loop bunu `info.get("within", info)` fallback ile çözüyor.

### 10.7 Cross-mode queue tagging (Faz 2.3)

`saps_mode in {cross, both}` → queue'lar `with_tags=True`. None/within modunda eski tagsız davranış korunur (regression-safe).

### 10.8 SAPS-both α=0.5 default (Faz 2.3)

`both` modunda `loss = loss_within + loss_cross` — eşit ağırlık. Faz 5 ablation'da `both_alpha` parametresi eklenebilir.

### 10.9 LabelFractionSplitter "B + fallback C" (Faz 4.4 — yeni)

- **B (default):** dominant class per image → stratified prefix-ordering
- **C (manuel):** `stratify_mode="none"` → uniform shuffle
- **B içinde gömülü 2 fallback:** tiny class merging (`-999` bucket), unlabeled images (`-1` class id, stratified bir sınıf gibi davranır)

### 10.10 LinearProbeTrainer empty loader (Faz 4.5 — yeni)

`DataLoader(empty_dataset, shuffle=True)` → `RandomSampler` patlar (`num_samples=0`). Boş loader durumunu test etmek için **`shuffle=False`** zorunlu (`SequentialSampler` boş datasetı kabul eder). Modülün `evaluate()` empty input için safe-zero döner.

### 10.11 RunMatrix resume invariant (Faz 4.6 — yeni)

CSV'de `status="ok"` olan cell'ler resume modunda atlanır. `status="failed"` veya `"skipped"` olanlar yeniden denenir. Bu, kısmen koşmuş bir matrix'i devam ettirmenin temel davranışı.

### 10.12 Roboflow `..` path quirk (Faz 4.3 — yeni)

Roboflow exports `train: ../train/images` yazar ama dataset genelde data.yaml'ın **kardeşidir** (parent değil child). `_resolve_split` standart resolution fail ederse `..` segment'lerini drop ederek tekrar dener. Bu Pothole 5K geldiğinde de işe yarayacak (aynı Roboflow toolchain).

### 10.13 mAP yorumlama riski (Faz 4 smoke — yeni)

Real Roboflow 1125'te 3-epoch random-init head → mAP=0.79. **Yüksek ama yanıltıcı olabilir**:
- COCO weights yol/asfalt features içeriyor olabilir → pothole detection kolay
- Class imbalance varsa constant-1 baseline AP'si dataset positive rate'e eşittir

Faz 5 başında **class distribution + constant baseline** ile mAP'ın gerçek baseline'ı kalibre edilmeli.

### 10.14 Multi-label binary works at nc=1 (Faz 4.5 — yeni)

`MultiLabelImageDataset` multi-class için tasarlandı; nc=1 ve nc=2 her ikisinde de aynı kod çalışıyor. `multilabel_average_precision` per-class AP + mean döner — nc=1 için mean = AP. Pothole 5K geldiğinde `drop_classes=[1]` ile uncertain çıkarılınca nc effectively 1 olur, kod değişikliği gerek yok.

---

## 11. Architecture Sentinels

### 11.1 YOLOv8 FPN layer indices

```python
YOLOV8_FPN_LAYERS = {"P3": 15, "P4": 18, "P5": 21}
YOLOV8_FPN_STRIDES = {"P3": 8, "P4": 16, "P5": 32}
```

### 11.2 YOLOv8n channel widths (sentinel test)

```python
{"P3": 64, "P4": 128, "P5": 256}
```

`tests/test_dense_ssl_pretrainer_realyolo.py::test_yolov8n_known_channels` doğrular.

### 11.3 SAPS level_to_id mapping (Faz 2.3)

```python
{"P3": 0, "P4": 1, "P5": 2}
```

`infer_in_channels` `OrderedDict` döndürür; enumeration P3→P4→P5. Test: `test_level_to_id_stable`.

### 11.4 Roboflow Pothole 1125 dataset stats (Faz 4 smoke — yeni)

```python
{
    "total": 1125,
    "splits": {"train": 900, "valid": 169, "test": 56},
    "imgsz": 640,             # Roboflow auto-resize
    "preprocessing": "auto-orient + stretch to 640x640",
    "augmentations": "none",
    "nc": 2,
    "names": ["pothole", "uncertain"],
    "data_yaml": "/content/datasets/roboflow/data.yaml",
}
```

Pothole 5K hazır olunca yeni sentinel eklenir.

### 11.5 Eval pipeline smoke baseline (Faz 4 — yeni)

Random-init head + COCO YOLOv8n → 3-epoch linear probe → **val mAP ~0.79** Roboflow Pothole 1125 üzerinde. Bu, eval pipeline'ın çalıştığını doğrular ama **performans sayısı değil** — Faz 5 calibration sonrası gerçek baseline belirlenir.

---

## 12. Şu Anki Aksiyon

**Sırada:** Faz 3 — External baselines (MoCo-v3 + DINOv2 self-implementation)

**Paralel iş (kullanıcı tarafında):**
1. SSL pretrain pool indirme (BDD100K, A2D2, Mapillary, Cityscapes coarse) — Drive mount + download
2. Pothole 5K dataset toplama / annotation devamı

Bu ikisi hazır olunca Faz 4.1, 4.2 + Faz 5 başlayabilir. Bu arada Faz 3 implement ediliyor.
