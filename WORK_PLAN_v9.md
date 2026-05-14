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

```
src/yolo_contrastive/
  dense/                              ✅ TAMAMLANDI (Faz 1+2, dokunulmaz)
    multi_scale_tap.py                ✓ 23 test
    queue.py                          ✓ 32 test
    momentum_encoder.py               ✓ 26 test
    spatial_aug.py                    ✓ 26 test
    dense_loss.py                     ✓ 29 test
    multi_scale_loss.py               ✓ 19 test
    projection.py                     ✓ 21 test
    saps.py                           ✓ 37 test

  pretrain/
    trainer.py                        ✓ (mevcut legacy SSLPretrainer korunur)
    dense_trainer.py                  ✓ 48 test (Faz 1-2)
    run_matrix.py                     ✓ 34 test (Faz 5.1 PretrainMatrix orchestrator)
    dual_teacher_trainer.py           ⬜ YENİ (Faz 5.3) — DenseSSLPretrainer'ı sarmalar, dual teacher ekler

  dual_teacher/                       ⬜ YENİ (Faz 5.3)
    __init__.py
    coco_teacher.py                   ⬜ Frozen COCO YOLOv8x wrapper + size adapter
    teacher_cache.py                  ⬜ Feature cache I/O (parquet/npz)
    consensus_loss.py                 ⬜ Form B (learned weighted L2) + Form C (dual KL)
    disagreement.py                   ⬜ Yaklaşım D — exp(α_d · d) sample weighting

  finetune/
    trainer.py                        ✓ Risk 16 v2 fix devrede (§10.25), 5 regression test (§11.8)

  data/                               ✅ KISMEN
    label_fraction.py                 ✓ 30 test
    unified_loader.py                 ✓ 32 test (Roboflow `..` fallback)
    dedup/                            ✓ pHash (commit 557e151, 510 test)

  eval/                               ✅ TAMAMLANDI
    linear_probe.py                   ✓ 28 test (early stopping)
    run_matrix.py                     ✓ 26 test (detection runner, Risk 16 v2 fix sonrası temiz)
    leakage_check.py                  ⬜ Faz 4.2 (pHash modülü hazır, runner şart)

  baselines/                          ⬜ Faz 5.4 (Faz 3'ten taşındı)
    comad_yolo.py                     ⬜ CoMAD-YOLO port (multi-SSL teacher consensus)
    simclr_yolo.py                    ⬜ SimCLR-YOLO (global pooling baseline, 2025 reference)
    moco_v3.py                        ⬜ MoCo-v3 port

  configs/
    pretrain/
      ablation_stage1_smoke.yaml      ✓ 6 cells, 5K pool
      ablation_stage2_coarse.yaml     ✓ 12 cells, 50K pool
      ablation_stage3_fine.yaml       ✓ 9 cells, 186K pool (Variant A template)
      multi_backbone.yaml             ⬜ Faz 5.2 (winning config × 6 backbone)
      ablation_stage4_dt.yaml         ⬜ Faz 5.3 (DT-SAPS ablation, α/β/γ/w/α_d)
      ablation_stage5_baselines.yaml  ⬜ Faz 5.4 (CoMAD-YOLO, SimCLR-YOLO, MoCo-v3)
      ablation_stage6_foundation.yaml ⬜ Faz 5.5 (DINOv2 referans)

  adapters/, pretext/                 ❄ DONDURULDU
```

**Mevcut test toplam:** 710. Faz 5.3 dual_teacher/ modülleri eklendiğinde tahmini +60-80 test (consensus loss invariants + disagreement weighting + cache I/O).

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
| **20** | **Disagreement weighting (D) instability** (yeni) | `disagreement_weight = exp(α_d · d(f_coco, f_ssl))` — large α_d ile signal explosion riski. Mitigasyon: α_d ∈ {0, 0.5, 1.0, 2.0} ablation (Stage 5.3.2). α_d=0 default (D kapalı, fallback). Numerik koruma: `clamp(weight, max=10.0)`. |
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

### 10.29 Disagreement weighting hard-mining mechanism (v9 yeni)

Yaklaşım D (`L_distill *= exp(α_d · d(f_coco, f_ssl))`) iki teacher'ın **uyuşmadığı sample'larda** signal'i artırır. α_d=0 klasik (D kapalı), α_d→∞ çok agresif hard mining.

**Sezgi:** Easy examples'da iki teacher consensus'a varır (signal düşük zaten redundant). Hard examples'da uyuşmaz (signal yüksek — burada öğrenecek bir şey var). Hard mining'in dual-teacher versiyonu.

**Risk 20 — instability:** `exp(α_d · d)` patlayabilir. Mitigasyon: `clamp(weight, max=10.0)`, α_d ablation ile optimal değer.

**Akademik value:** Bu CoMAD'ın "consensus gating" mekanizmasından **ters yön** — CoMAD uyumsuzluğu filtreler, biz vurgularız. Bu fark paper'ın "vs CoMAD" defansının kalbi.

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

---

## 12. Şu Anki Durum

**Kütüphane: TAM İŞLEVSEL ✅**
**Risk 16 v2 ile production-validated ✅**
**Faz 5 ablation grid YAML'ları hazır ✅**

Kanıtlar:
- 710 test geçer durumda
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
| 5.4 External baselines | ⬜ Portlar yapılacak | Önkoşul: DT-SAPS yapılırken paralel |
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

**Status:** pHash modülünün `leakage_check` fonksiyonu var, gerçek karşılaştırma yapılmadı.

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

### 13.5 imagehash kütüphane install (operasyonel)

Her Colab session başlatıldığında manuel:
```bash
pip install imagehash
```

Vast.ai instance'larda Dockerfile veya `requirements.txt` ile:
```
imagehash>=4.3.1
```

**Bağımlılık:** §13.1, §13.3 öncesi.

### 13.6 §11.7 v1 references preserved (akademik kayıt)

v8'den korunur: Roboflow Pothole 1125 + Yol 3 smoke reference numbers (§11.7'de). Paper supplementary §14.3'te "method evolution" hikayesinde geçer:

> "Initial experiments used Roboflow Pothole 1125 (interim dataset), demonstrating SSL methods' baseline behavior. Pothole 1125 was replaced by 4-class infrastructure dataset (3371 train / 548 val) in v8→v9 transition for: (a) more class diversity, (b) larger training set, (c) Pothole 5K parallel dataset preparation."

**Aksiyon:** Sadece dokümante etme, §11.7 silinmez.

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

**Honest claim:** "Our pure SSL baseline does not surpass COCO supervised pretraining in our traffic detection setting. We propose DT-SAPS to bridge this gap via dual-teacher knowledge transfer, demonstrating that domain-specialized SSL+ can match or exceed general-purpose supervised pretraining when augmented with proper teacher distillation."

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
