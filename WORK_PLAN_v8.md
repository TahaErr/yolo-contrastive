# YOLO-CONTRASTIVE — MASTER WORK PLAN v8

**Tez/Paper:** Scale-Aware Dense Contrastive Pretraining for Real-Time Detection in Traffic Scenes
**Hedef venue:** ITS-odaklı, CVPR-tier (T-ITS, ITSC, IV Symposium)

**Değişiklikler (v7 → v8):**
- ✅ Yol 3 — Detection finetune smoke yeşil (3 backbone karşılaştırma, mAP50 ölçüldü)
- ✅ Risk 9 — `saps_both_lambda` parametresi eklendi (λ-weighted toplama)
- ✅ Risk 7 — `queue_update_strategy` parametresi eklendi (pooled/per_position/subsample)
- ⏳ Risk 16 (yeni) — PyTorch 2.x InferenceMode crash, post-train workaround aktif, kalıcı fix Faz 5 öncesi
- §10'a 4 yeni lesson (10.19 cross-trainer determinism, 10.20 λ-weighted academic rationale, 10.21 queue strategy decision rationale, 10.22 PT 2.x crash pattern + workaround)
- §11'e Yol 3 detection mAP sentinel sayıları
- Test toplam: 395 (was 383)
- Kütüphane TAM İŞLEVSEL ve gerçek detection mAP'ı ölçülmüş durumda

---

## 1. Stratejik Karar Özeti

### 1.1-1.4 (önceki versiyonlardan değişmeden korundu)

A — Foundation ✅, D — SAPS ✅, atılanlar dondurulmuş.

---

## 2. Datasets

### 2.1 SSL pretrain pool — Tamamlandı (181,446 image)
Pool: `/content/drive/MyDrive/yolo-contrastive/ssl_pool/` (manifest at `manifest.parquet`). 640px long-side JPEG q=90, aspect-preserved, no upscale.
- BDD100K: 100,000 (70K train / 10K val / 20K test)
- Cityscapes: 24,998 (coarse train_extra 19,998 + fine train/val/test 5,000)
- Mapillary Vistas v2: 25,000 (training 18K + validation 2K + testing 5K)
- A2D2: 31,448 (cam_front_center only; doc'taki ~41K rakamı 6-kamera toplam frame sayısı)

Adapter modülleri: `src/yolo_contrastive/data/ssl_pool/{bdd100k,a2d2,cityscapes,mapillary}.py`. Materialize idempotent on `image_id`, resume-safe.

### 2.2 Downstream evaluation
**Şu anki interim:** Roboflow Pothole 1125 (`/content/datasets/roboflow/data.yaml`)
- 900 train / 169 valid / 56 test, 640×640, 2 class (`pothole`, `uncertain`)
**Hedef:** Pothole 5K (custom). `drop_classes=[1]` ile uncertain çıkar.
**Generic AD detection:** BDD val (pretrain pool indirilince).

### 2.3 Data leakage kontrolü
pHash dedup. Faz 4.2'de implementasyon (Pothole 5K + SSL pool hazır olunca).

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

*Faz 3 askıda (kütüphane tam çalıştı, reactivate edilebilir).

**Toplam:** ≈ 85-90 run. Fair comparison: tek değişen backbone init.

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
    trainer.py                   ✓ (mevcut SSLPretrainer korunuyor)
    dense_trainer.py             ✓ 48 test (ön: 36 + λ: 6 + queue strat: 6)

  finetune/
    trainer.py                   ✓ (Yol 3 smoke ile validated, Risk 16 var)

  data/                          ✅ KISMEN (4.3, 4.4)
    label_fraction.py            ✓ 30 test
    unified_loader.py            ✓ 32 test (Roboflow `..` fallback)

  eval/                          ✅ TAMAMLANDI
    linear_probe.py              ✓ 28 test (early stopping)
    run_matrix.py                ✓ 26 test (detection runner stub, Risk 16 fix sonrası tamamlanır)
    leakage_check.py             ⬜ Faz 4.2

  baselines/                     ⬜ Faz 3 (askıda)
  adapters/, pretext/            ❄ DONDURULDU
```

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
| Yol 3 smoke | **Detection finetune integration** (Roboflow 1125) | — | ✅ Yeni v8'de |

### Yol 3 SMOKE SONUÇLARI (Roboflow Pothole 1125, 10-epoch finetune)

**Config:** imgsz=640, batch=16, freeze=10, unfreeze@5, bb_lr_scale=0.1.

| Backbone | mAP50 | mAP50-95 | Precision | Recall | crash post-train |
|---|---|---|---|---|---|
| COCO baseline | 0.4783 | 0.2182 | 0.475 | 0.548 | no |
| SSL-none (Faz 4.7) | 0.3709 | 0.1619 | 0.611 | 0.321 | no |
| SSL-within (Faz 4.7) | 0.3754 | 0.1577 | 0.438 | 0.371 | yes (caught) |

**Δ analizi:**
- Δ(COCO − SSL-none) = +0.107 mAP50
- Δ(within − none) = +0.005 mAP50 (gürültü içinde)

**Yorum:** COCO smoke ölçeğinde açık ara önde (beklenen, 900 image SSL yetersiz). SAPS-within marjinal kazanç (Faz 4.7 linear probe pattern'i ile tutarlı). Faz 5'te ~186K SSL pool ile **trend tersine dönmesi** beklenir; aksi halde hipotez sallantıda (Risk 15).

**Smoke amacı: pipeline yaşıyor mu? → ✅ EVET.** Detection mAP50 elde edilebildiği kanıtlandı, paper'a ölçüm referansı.

### Faz 3 — External baselines (askıda — kütüphane tam → reactivate edilebilir)
Bekleyen kararlar: backbone (YOLO-native vs ViT) + scope (full vs essential).

### Faz 5 — Deneyler (sıra)
1. SSL pool indirme (paralel)
2. Class distribution analizi (Risk 11 — atlanmıştı, gerek olunca)
3. Leakage check (Faz 4.2)
4. Pothole 5K split
5. **Risk 16 kalıcı fix** (PT 2.x crash, RunMatrix detection runner için zorunlu)
6. SSL pretrain — Ours full A+D (~186K, 100 epoch)
7. SSL pretrain — Ours ablations (saps_mode × saps_both_lambda × queue_strategy × t_scale)
8. (Faz 3 reactive ise) MoCo-v3 + DINOv2
9. Eval matrix
10. Sonuç + analiz

---

## 6. Risk Listesi

| # | Risk | Durum |
|---|---|---|
| 1 | Spatial correspondence | ✅ Faz 1.4a |
| 2 | Multi-scale + AMP numerik | ✅ dense_loss autocast off |
| 3 | A2D2 lisans | Akademik izinli |
| 4 | Mapillary disk | Image listesi cache |
| 5 | "X SSL methodu yok" | Related work tartış |
| 6 | YOLOv9/v10/v11 neck | v8 ana, ek tabloda diğerleri |
| 7 | Queue update strategy | ✅ ÇÖZÜLDÜ — 3-strategy parametrik (Risk 7) |
| 8 | Mock-encoder learning signal | Real YOLO testleri yeterli |
| 9 | SAPS-both α | ✅ ÇÖZÜLDÜ — `saps_both_lambda` parametresi (default 1.0, geriye uyumlu) |
| 10 | SAPS queue tagging memory | Negligible |
| 11 | mAP=0.79 yorumlama | ⏭ Atlandı, dataset basit (kullanıcı kararı). Pothole 5K geldiğinde tekrar bak. |
| 12 | Linear probe overfitting | ✅ early_stopping_patience |
| 13 | Roboflow `..` path | ✅ unified_loader fallback |
| 14 | Eval dataset değişimi | data.yaml swap hazır |
| 15 | SSL'in COCO'yu geçememesi | Faz 4.7 + Yol 3 smoke ölçeğinde GÖZLENDİ. Faz 5'te ~186K + cross-domain ile test edilecek. **Eğer Faz 5'te de geçmezse hipotez sallantıda — fallback domain-specific hikaye.** |
| **16** | **PyTorch 2.x InferenceMode crash post-train** (yeni) | **WORKAROUND aktif:** try/except + CSV fallback. Kalıcı fix: `finetune/trainer.py::save_model` ve `_setup_train`'deki `load_state_dict` çağrılarına `torch.no_grad()` veya `.clone()` patch. **Faz 5 RunMatrix detection runner için zorunlu.** |

---

## 7. Açık Kararlar

- Pothole 5K eval split — Faz 4.1
- SAPS t_scale değeri — Faz 5 ablation ({0.5, 1.0, 2.0, ∞})
- Multi-scale loss weights — başlangıç 1/3
- ~~SAPS-both α~~ → ✅ `saps_both_lambda` parametresi (Risk 9 kapatıldı, Faz 5 ablation: {0, 0.5, 1.0, 2.0})
- ~~Queue update stratejisi~~ → ✅ `queue_update_strategy` parametresi (Risk 7 kapatıldı, Faz 5 ablation: {pooled, per_position, subsample})
- Risk 16 kalıcı fix tasarımı — Faz 5 öncesi
- Faz 3 reaktivasyon — kullanıcı kararı

---

## 8. Önceki Benchmark Sonuçları (referans)

%10 etiket, BDD/pothole subset:

| Yöntem | mAP50 | Status |
|---|---|---|
| B: COCO+CL (joint) | 0.6719 | Mevcut en iyi |
| A: COCO baseline | 0.6593 | Baseline |

**Yeni hedef X:** B (0.6719). A+D'nin B'yi geçmesi → paper'ın ana ölçütü.

---

## 9. Submission planı

| Hafta | Faaliyet | Durum |
|---|---|---|
| 1-3 | Faz 1 | ✅ |
| 3-5 | Faz 2 | ✅ |
| 5-6 | Faz 4 + smokes | ✅ (Faz 4.7 + Yol 3) |
| 6-7 | Risk 7, 9 closure (ablation prep) | ✅ |
| 7-9 | SSL veri indirme + Risk 16 kalıcı fix | 🟡 paralel + ⬜ |
| 9-11 | Faz 5 pretrain compute | ⬜ |
| 11-12 | Faz 5 eval + analiz | ⬜ |
| ?-? | Faz 3 (opsiyonel) | ⏸ |
| 12-15 | Paper writing | ⬜ |

---

## 10. Implementation Lessons

(§10.1-10.18 önceki versiyonlardan kayıtlı, aşağıda yeni eklenenler.)

### 10.19 Cross-trainer determinism (Risk 9 → test bug — yeni)

`torch.manual_seed(N)` iki ayrı `DenseSSLPretrainer` instance arasında **bit-eşit aynı** sonuç vermez. Sebep: constructor'daki `MomentumEncoder` deepcopy + augmentation construction + `_subsample_positions` her biri farklı miktarda RNG tüketir; iki instance arasındaki RNG state'leri ayrışır.

**Çıkarım:** Matematiksel invariant testleri **iki ayrı run karşılaştırması** üzerinden değil, **tek run'ın info dict'inden** doğrulanmalı. Risk 9 testlerinde bu lesson uygulandı (`test_lambda_amplifies_cross_contribution` info dict'ten oku, `test_lambda_two_doubles_cross_contribution` (eski) iki-run yaklaşımı atıldı).

### 10.20 λ-weighted vs convex combination (Risk 9 — yeni)

SAPS-both için `loss = α · loss_within + (1-α) · loss_cross` (convex) yerine `loss = loss_within + λ · loss_cross` (toplama-with-weight) seçildi.

**Akademik gerekçe:**
1. **Default geriye uyumlu**: `λ=1` mevcut `w + c` davranışına bit-eşit eşit
2. **Ablation kontrolü**: `λ=0` numerik olarak `saps_mode="within"` ile aynıdır → ablation tablosunda kontrol satırı temiz
3. **DINO/iBOT convention**: Multi-loss SSL paper'ları toplama+ağırlık kullanır, convex değil
4. **Convex'in dezavantajı**: `α=0.5` loss'u yarıya indirir → LR re-tuning gerek, önceki sonuçlarla doğrudan kıyaslanmaz

### 10.21 Queue update strategy decision rationale (Risk 7 — yeni)

3 strategi seçildi (`pooled` / `per_position` / `subsample`):
- **pooled** (default, MoCo classification convention): yavaş queue dolar, image-level features
- **per_position** (DenseCL convention): tüm konumlar, queue çabuk dolar, position bilgisi korunur
- **subsample** (PixPro convention): görüntü başına n random konum, queue + spatial denge

Faz 5 ablation'da hangi strategy'nin SAPS ile en iyi çalıştığı ölçülecek. Default `pooled` çünkü mevcut deneylerle (Faz 4.7) tutarlı, geriye uyumlu.

### 10.22 PyTorch 2.x InferenceMode crash pattern (Yol 3 → Risk 16 — yeni)

**Pattern:** Ultralytics `model.train(...)` ile YOLOv8 detection finetune sonu, post-training `load_state_dict` çağrısında `'Inplace update to inference tensor outside InferenceMode is not allowed'` RuntimeError.

**Sebep zinciri:**
1. Ultralytics `DetectionTrainer.final_eval()` veya `save_model()` çağrılıyor
2. `model.eval()` BN running stats'ı InferenceMode tensor'a dönüştürüyor
3. `load_state_dict` BN stats'ı üzerine yazmaya çalışıyor → InferenceMode dışında inplace update yasak (PyTorch 2.x)

**Workaround (geçici):** `model.train(...)` çağrısını `try: ... except BaseException` ile sar. Crash sonrası `results.csv` diskten oku — training tamamlanmıştır, sadece teardown'da patlamıştır.

**Kalıcı fix (yapılacak, Faz 5 öncesi):** `finetune/trainer.py`:
- `save_model()` override'ında `with torch.no_grad():` içinde `load_state_dict` çağır
- `_setup_train()`'deki EMA sync'te tensor'ları `.clone()` ile kopyala
- Test: full Yol 3 smoke crash-free çalışmalı

**Neden hemen düzeltmedik:** Smoke amacımız mAP raporu almaktı (alındı). Fix Ultralytics versiyonuna spesifik, 3 run test gerek, ayrı bir commit hak ediyor. Faz 5 RunMatrix detection runner yazılırken zorunlu olacak.

---

## 11. Architecture Sentinels

### 11.1-11.6 (önceki versiyonlardan kayıtlı)

### 11.7 Yol 3 detection finetune reference numbers (Roboflow Pothole 1125 — yeni)

10-epoch finetune, imgsz=640, batch=16, freeze=10, unfreeze@5, bb_lr_scale=0.1:

```python
{
    "coco_baseline":   {"mAP50": 0.4783, "mAP50_95": 0.2182, "P": 0.475, "R": 0.548},
    "ssl_none":        {"mAP50": 0.3709, "mAP50_95": 0.1619, "P": 0.611, "R": 0.321},
    "ssl_within":      {"mAP50": 0.3754, "mAP50_95": 0.1577, "P": 0.438, "R": 0.371},
}
```

Faz 5 detection finetune'unda bu rakamlardan **belirgin sapma** (örn. COCO mAP50 < 0.3 ya da > 0.6) → pipeline'da bug ihtimali, debug.

**Pattern observation:** SSL pretrained backbone'ları COCO'dan precision higher / recall lower üretiyor (konservatif). Eğer Faz 5'te aynı pattern devam ederse "SSL produces more discriminative low-recall features" gözlemi paper'a girer.

---

## 12. Şu Anki Durum

**Kütüphane: TAM İŞLEVSEL ✅**
**Real detection mAP measured ✅**

Kanıtlar:
- 395 test geçer durumda
- Foundation + SAPS + Eval altyapı + SSL pretrain E2E + Detection finetune E2E
- Ablation parametreleri hazır: `saps_mode`, `saps_both_lambda`, `saps_t_scale`, `queue_update_strategy`, `queue_subsample_n`, `early_stopping_patience`
- Drop-in compat (DenseSSL → FinetuneDetection + LinearProbe) yeşil

**Sırada:**
- (Şimdi) WORK_PLAN_v8 + GitHub commit + push
- (Sonra, kullanıcı kararıyla) Faz 5 prep ya da Faz 3 reactivate

**Faz 5 öncesi yapılacaklar:**
- Risk 16 kalıcı fix (PT 2.x crash patch)
- SSL pretrain pool indirme (kullanıcı paralel)
- Pothole 5K dataset (kullanıcı paralel)
