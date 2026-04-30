# YOLO-CONTRASTIVE — MASTER WORK PLAN v7

**Tez/Paper:** Scale-Aware Dense Contrastive Pretraining for Real-Time Detection in Traffic Scenes
**Hedef venue:** ITS-odaklı, CVPR-tier (T-ITS, ITSC, IV Symposium)
**Tahmini süre:** ~3.5 ay (Faz 1-5) + 3-4 hafta paper writing

**Değişiklikler (v6 → v7):**
- Faz 4.7 ✅ End-to-end SSL pretrain smoke yeşil (Roboflow Pothole 1125)
  - Both `saps_mode="none"` + `saps_mode="within"` koştu, beklenen learning trajectory
- LinearProbeTrainer'a `early_stopping_patience` parametresi eklendi (Adım A)
- Risk 12 (linear probe overfitting) ✅ ÇÖZÜLDÜ
- §10'a 3 yeni lesson (10.15, 10.16, 10.17): SSL trajectory health, target-domain pretraining caveat, early stopping default
- §11'e Faz 4.7 SSL pretrain sentinel sayıları eklendi
- Faz 3 askıya alındı: kütüphane tam çalışana kadar bekliyor (artık çalışıyor)
- Test toplam: 383 (was 380; +4 early stopping; -1 muhasebe düzeltmesi)
- Şu anki sıra: **kütüphane tam işlevsel ✅, sırada Faz 5 prep (data download) + opsiyonel Faz 3**

---

## 1. Stratejik Karar Özeti

### 1.1 Geçmiş yöntemlerin durumu

| Bileşen | Karar | Gerekçe |
|---|---|---|
| LoRA ailesi | ❄ DONDUR | Repo'da kalır, hiç çağrılmaz. |
| 6 pretext task + CompositeTask | ❄ DONDUR | Repo'da kalır, ana pipeline'da kullanılmaz. |
| NT-Xent, FeatureTap, augmentation registry | ✅ ALTYAPI | Yeni Dense CL altyapısının yanında yaşıyor. |
| ContrastiveDetectionTrainer | ✅ TUT | Joint training ana akış. |

### 1.2 Final yöntem paketi (A + D)

**A — Foundation** ✅ TAMAMLANDI:
1. MoCo memory queue (K=65536) — `dense/queue.py`
2. Momentum encoder (m=0.999, EMA) — `dense/momentum_encoder.py`
3. Dense Spatial CL — `dense/dense_loss.py`
4. Multi-scale FPN CL — `dense/multi_scale_loss.py`

**D — Novel katkı** ✅ TAMAMLANDI:
- **Scale-Aware Positive Sampling (SAPS)**:
  - Within-image: `dense/saps.py::saps_within_loss` (Faz 2.1)
  - Cross-image: `dense/saps.py::saps_cross_loss` (Faz 2.2)
  - Trainer flag: `saps_mode in {none, within, cross, both}` (Faz 2.3)

### 1.3-1.4 A + D bileşenlerinden atılanlar

| Atıldı | Sebep |
|---|---|
| Hard negative mining | Queue zaten 64K negatif redundant |
| Multi-positive (4-8 view) | Compute 2-4x, marjinal kazanç |
| Detection-aware positives | Pothole 5K labeled için opsiyonel |
| Temperature schedule | Marjinal, hikaye değeri yok |
| Temporal consistency | Dense CL ile registration sorunu |
| Metadata-driven invariance | "Scale" temasını sulandırır |
| Spatial-context priors | Tanımı muğlak |

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

**Paralel iş — kullanıcı yan tarafta yürütüyor. Faz 5 öncesi ready olmalı.**

### 2.2 Downstream evaluation — INTERIM DATASET KULLANIMINDA

**Şu anki interim eval dataset:**
- **Roboflow Pothole 1125** (Roboflow workspace `taha-yasin-er-phd-studies-99wyc`)
- 1125 image: train 900 / valid 169 / test 56
- 640×640, auto-orient, no augmentation
- 2 class: `pothole` (id=0), `uncertain` (id=1)
- Path: `/content/datasets/roboflow/data.yaml`

**Hedef eval dataset (paralel hazırlanıyor):**
- Pothole 5K (custom). Hazır olunca path swap. `drop_classes=[1]` ile uncertain çıkarılabilir.

**Generic AD detection:** BDD val (pretrain pool indirilince eklenir).

### 2.3 Data leakage kontrolü

pHash dedup zorunlu. Faz 4.2'de implementasyon (Pothole 5K + SSL pool hazır olunca).

---

## 3. Eval Matrix

```
                    Roboflow 1125 (interim)   BDD val
                    %10 %25 %50 %100          %1 %5 %10 %25 %100

Scratch             ▢   ▢   ▢   ▢             ▢  ▢  ▢   ▢   ▢
COCO baseline       ▢   ▢   ▢   ▢             ▢  ▢  ▢   ▢   ▢
DINOv2 / MoCo-v3*   ▢   ▢   ▢   ▢             ▢  ▢  ▢   ▢   ▢
Ours (A+D)          ▢   ▢   ▢   ▢             ▢  ▢  ▢   ▢   ▢
```

*Faz 3 askıda — kütüphane tam çalıştığında implement edilecek (artık çalışıyor).

**Toplam:** ≈ 85-90 run.

### 3.1 Fair comparison protokolü

Her baseline için fine-tune hyperparameters AYNI. Değişen tek değişken: backbone init.

---

## 4. Modül Haritası

```
src/yolo_contrastive/
  dense/                         ✅ TAMAMLANDI (Faz 1+2)
    multi_scale_tap.py           ✓ 23 test
    queue.py                     ✓ 32 test
    momentum_encoder.py          ✓ 26 test
    spatial_aug.py               ✓ 26 test
    dense_loss.py                ✓ 29 test
    multi_scale_loss.py          ✓ 19 test
    projection.py                ✓ 21 test
    saps.py                      ✓ 37 test

  pretrain/
    trainer.py                   ✓ (mevcut SSLPretrainer dokunulmuyor)
    dense_trainer.py             ✓ Faz 1.6+2.3, ~52 collected (with parametrize)

  data/                          ✅ KISMEN (4.3, 4.4)
    label_fraction.py            ✓ 30 test
    unified_loader.py            ✓ 32 test (Roboflow `..` fallback dahil)

  eval/                          ✅ TAMAMLANDI (4.5+early stopping, 4.6)
    linear_probe.py              ✓ 28 test (Adım A: early stopping eklendi)
    run_matrix.py                ✓ 26 test
    leakage_check.py             ⬜ (Faz 4.2, Pothole 5K bekliyor)

  baselines/                     ⬜ Faz 3 (askıda, kütüphane tam → reactivate edilebilir)
  adapters/                      ❄ DONDURULDU
  pretext/                       ❄ DONDURULDU
```

---

## 5. Faz-Faz Roadmap

### Faz 1 — Foundation ✅ KAPATILDI (201 test)

### Faz 2 — SAPS ✅ KAPATILDI (67 yeni)

### Faz 4 — Data ve eval altyapısı ✅ TAM İŞLEVSEL

| Adım | Dosya | Test | Durum |
|---|---|---|---|
| 4.4 | `data/label_fraction.py` | 30 | ✅ |
| 4.5 | `eval/linear_probe.py` | **28** | ✅ (early stopping eklendi) |
| 4.6 | `eval/run_matrix.py` | 26 | ✅ |
| 4.3 | `data/unified_loader.py` | 32 | ✅ |
| smoke | Roboflow 1125 end-to-end (Faz 4 smoke) | — | ✅ Real-data probe |
| **4.7** | **Pretrain integration smoke** (Faz 4.7 — yeni) | — | ✅ End-to-end SSL pretrain → save → load → probe |
| 4.1 | `data/pothole_split.py` | — | ⏳ Pothole 5K bekliyor |
| 4.2 | `eval/leakage_check.py` | — | ⏳ Pothole 5K + SSL pool bekliyor |

### Faz 4.7 SMOKE SONUÇLARI (Roboflow Pothole 1125 train pool, 900 image)

**Pretrain config:** D=128, K=2048, m=0.99, τ=0.2, n_query=128, imgsz=224, lr=1e-3, 7 epoch, batch=8.

**SAPS-none run:**
| Metric | Epoch 1 | Epoch 7 |
|---|---|---|
| loss | 2.77 | 2.68 |
| acc@1 | 0.259 | 0.583 |
| mean_pos_sim | 0.440 | 0.854 |

**SAPS-within run:**
| Metric | Epoch 1 | Epoch 7 |
|---|---|---|
| loss | 3.33 | 2.88 |
| acc@1 | 0.213 | 0.541 |
| mean_pos_sim | 0.598 | 0.866 |

Within başlangıç loss'u (3.33 > 2.77) Faz 2.1'de matematik olarak kanıtlanan **`saps_loss >= multi_scale_loss`** invariant'ını gerçek YOLO + gerçek görüntülerde de doğrular.

**Linear probe (3 epoch, frozen backbone, multi-label BCE):**

| Backbone | val mAP | Best epoch |
|---|---|---|
| COCO baseline (Faz 4 smoke) | 0.7906 | 3/3 |
| SSL-none pretrain | 0.7720 | 1/3 |
| SSL-within pretrain | 0.7646 | 2/3 |

**Yorum:** SSL pretrained mAP'lar COCO-init'in altında. Beklenen — 900 image SSL için yetersiz, 7 epoch çok kısa, target-domain pretraining (aynı dataset hem pretrain hem eval). Faz 5'te ~186K image + 100 epoch ile gerçek değerlendirme yapılacak. **Smoke amacı: pipeline yaşıyor mu? → ✅ EVET.**

### Faz 3 — External baselines (askıda)

Karar: kütüphane tam işlevsel oldu (Faz 4.7 ✅). Faz 3 reaktif edilebilir.

Bekleyen kararlar (Faz 3 başlatılırsa):
- Backbone seçimi: YOLO-native (önerim) vs ViT vs hibrit
- Scope: full faithful vs stripped-down essential

### Faz 5 — Deneyler (sıra)

1. SSL pretrain pool indirme (kullanıcı paralel)
2. Class distribution analizi + constant baseline (mAP yorumu)
3. Leakage check
4. Pothole 5K split (gelirse)
5. SSL pretrain — Ours full A+D (~186K, 100 epoch)
6. SSL pretrain — Ours ablations
7. (Opsiyonel Faz 3 yapılırsa) MoCo-v3 + DINOv2 baselines
8. Eval matrix (~85-90 run)
9. Sonuç toplama + grafik + analiz

---

## 6. Risk Listesi

| # | Risk | Durum |
|---|---|---|
| 1 | Spatial correspondence augmentation | ✅ Çözüldü Faz 1.4a |
| 2 | Multi-scale + AMP numerik | ✅ Çözüldü dense_loss autocast off |
| 3 | A2D2 lisans | Akademik izinli |
| 4 | Mapillary disk maliyeti | Image listesi cache'le |
| 5 | Reviewer "X SSL methodu yok" | Related work tartış |
| 6 | YOLOv9/v10/v11 neck farkları | v8 ana, ek tabloda diğerleri |
| 7 | Queue strategy (pooled push) | Faz 5 ablation |
| 8 | Mock-encoder learning signal | Real YOLO testleri yeterli |
| 9 | SAPS-both α=0.5 | Faz 5 ablation |
| 10 | SAPS queue tagging memory | Negligible (~512KB) |
| 11 | mAP=0.79 anlamsızlık riski | Faz 5 başında class distribution + constant baseline |
| 12 | Linear probe overfitting | ✅ ÇÖZÜLDÜ — `early_stopping_patience` eklendi (Adım A) |
| 13 | Roboflow `..` path quirk | ✅ Çözüldü `_resolve_split` fallback |
| 14 | Eval dataset değişikliği | data.yaml path swap ile geçiş hazır |
| 15 | SSL'in COCO baseline'ı geçememesi (yeni) | Faz 4.7'de smoke ölçeğinde **gözlendi**: SSL mAP COCO-init'in altında. Beklenen sebepler: az veri (900), kısa pretrain (7 epoch), target-domain. Faz 5'te ~186K + 100 epoch + cross-domain ile bu trend tersine dönmeli; aksi halde paper hipotezi tehlikede — fallback: domain-specific pretrain hikayesi. |

---

## 7. Açık Kararlar

- Pothole 5K eval split protokolü — Faz 4.1
- SAPS t_scale değeri — Faz 5 ablation ({0.5, 1.0, 2.0, ∞})
- Multi-scale loss weights — başlangıç 1/3 eşit
- Queue update stratejisi — pooled vs per-position
- SAPS-both α — default 0.5 (toplam), Faz 5'te parametrize edilebilir
- mAP yorumlama — class distribution sonrası karar
- Faz 3 reaktivasyon — kütüphane çalıştığına göre artık başlatılabilir, kullanıcı kararı

---

## 8. Önceki Benchmark Sonuçları

%10 etiket, BDD/pothole subset:

| Yöntem | mAP50 | vs A | Status |
|---|---|---|---|
| B: COCO+CL (joint) | 0.6719 | +1.26pp | Mevcut en iyi |
| A: COCO baseline | 0.6593 | — | Baseline |
| LoRA family, full SSL, etc. | <0.55 | <-10pp | ❄ Atıldı |

**Yeni hedef X:** B (0.6719). A+D'nin B'yi geçmesi → paper'ın ana ölçütü.

**NOT:** Bu rakamlar **detection mAP50** (object detection IoU=0.5). Faz 4.7'deki **classification mAP** (multi-label image-level) ile kıyaslanamaz.

---

## 9. Submission planı

| Hafta | Faaliyet | Durum |
|---|---|---|
| 1-3 | Faz 1 (Foundation) | ✅ |
| 3-5 | Faz 2 (SAPS) | ✅ |
| 5-6 | Faz 4 (Data/eval) + 4.7 smoke | ✅ |
| 6-9 | SSL veri indirme + Faz 5 prep | 🟡 paralel |
| 9-11 | Faz 5 pretrain compute | ⬜ |
| 11-12 | Faz 5 eval + analiz | ⬜ |
| ?-? | Faz 3 (opsiyonel reactive) | ⏸ askıda |
| 12-15 | Paper writing | ⬜ |

**Aday venue deadlines:** T-ITS (rolling), ITSC (Mart-Nisan), IV (Aralık-Ocak)

---

## 10. Implementation Lessons

(Önceki §10.1-10.14 v6'da kayıtlı; aşağıda yeni eklenenler.)

### 10.15 SSL pretrain trajectory health (Faz 4.7 — yeni)

7-epoch real-YOLO pretrain'de gözlenen sağlıklı sinyaller:
- `acc@1`: 0.26 → 0.58 (rastgele üzerinde monotonik artış)
- `mean_pos_sim`: 0.44 → 0.85 (encoder pozitiflerl daha yakın projecte ediyor)
- `mean_neg_sim`: -0.18 → -0.04 (queue eskirken hafif artar — normal)
- `loss`: 2.77 → 2.68 (queue dolma artefaktına rağmen düşüyor)

`mean_pos_sim` 0.95+ ise overfit eşiği. 0.85 sağlıklı band içinde.

### 10.16 Target-domain pretraining caveat (Faz 4.7 — yeni)

Faz 4.7 smoke'unda aynı 900 image hem SSL pretrain hem linear probe için kullanıldı. Bu paper protokolü değil — gerçek SSL'in gücü cross-domain transfer'de görünür. Bu setup smoke amacı için yeterli ("pipeline yaşıyor mu?") ama **rapor değeri yok**, sadece iç doğrulama.

Faz 5'te protokol: pretrain pool ≠ eval dataset. Pretrain pool ↔ eval split'te pHash leakage check zorunlu (Faz 4.2).

### 10.17 Early stopping default semantik (Adım A — yeni)

`LinearProbeTrainer.fit(early_stopping_patience=...)`:
- `None` (default) → mevcut davranış aynen, regression-safe
- `N >= 1` → val_mAP `N` epoch boyunca iyileşmediyse durur, `early_stopped=True` döner
- `0` veya negatif → `ValueError`

Default'ta None bırakılması kritik — mevcut çağrıların (Faz 4 smoke, Faz 4.7 smoke) hiçbirini kırmadı.

### 10.18 Drop-in compat load_backbone log (Faz 4.7 — yeni)

`load_backbone(strict=False, backbone_only=True)` typical YOLOv8n için:
```
[ycl] Backbone loaded: 162/355 params (0 skipped, 193 filtered by backbone_only)
```

- 162 = bizim backbone-side params
- 193 filtered = detection head params (atılır, doğru davranış)
- 0 skipped = incompatible param yok (drop-in compat sağlıklı)

Faz 5'te tüm 4 method için aynı sayılar görülmeli — farklı çıkarsa pipeline'da tutarsızlık var demek.

---

## 11. Architecture Sentinels

### 11.1-11.5: önceki sentinel'ler v6'da kayıtlı.

### 11.6 Faz 4.7 SSL pretrain reference numbers (yeni)

Yukarıdaki §5 Faz 4.7 SMOKE SONUÇLARI tablosundaki rakamlar Faz 5 pretrain "anormal mı normal mi?" kontrolü için referans. Özellikle:
- Epoch 7 acc@1 ~0.55-0.60 aralığında olmalı (mock encoder'da çok daha düşük çıkıyordu — Faz 1 lessons §10.4)
- Epoch 7 mean_pos_sim 0.85±0.05 aralığında olmalı
- Linear probe mAP'i COCO-init'in (~0.79) etrafında olmalı (yukarı veya aşağı)

Faz 5 pretrain'de bu rakamlardan **belirgin sapma** (örn. acc@1<0.3, pos_sim<0.5, mAP<0.5) → pipeline'da bug ihtimali, debug.

---

## 12. Şu Anki Durum

**Kütüphane: TAM İŞLEVSEL ✅**

Kanıtlar:
- 383 test geçer durumda
- Foundation matematik + integration testleri yeşil
- SAPS matematik + integration testleri yeşil
- Eval altyapısı (Roboflow 1125 üzerinde smoke yeşil)
- End-to-end SSL pretrain → save → load → probe akışı yeşil (Faz 4.7)
- Drop-in compat (DenseSSL → FinetuneDetection + LinearProbe) yeşil

**Sırada:**
- Mevcut durumun kullanıcı tarafından değerlendirilmesi
- Sonra kullanıcı kararına göre devam: Faz 3 reactivate, ya da Faz 5 prep'e geç
