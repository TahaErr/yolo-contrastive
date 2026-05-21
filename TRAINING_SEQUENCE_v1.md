# Backbone Eğitim Sıralama Planı — Faz 5.1 → 5.4

**Durum:** Kütüphane kod tarafında tamamlandı (commit `c527df5`, 1011 test, 0 fail).
DT-SAPS framework + 3 external baseline + leakage runner hazır. Bu doküman,
paper-grade backbone eğitimlerinin hangi sırayla, hangi önkoşulla ve hangi
çıktıyı bir sonrakine besleyerek koşturulacağının haritasıdır.

Bu **kod değil, operasyonel plan**. Her faz bir pretraining üretir; pretraining
çıktıları (backbone checkpoint'leri) en sonda toplu fine-tune + detection eval'e
girer.

---

## 0. Bağımlılık grafiği

```
[ÖN] Pool hazırlık  (§13.1 pHash → §13.2 SSD copy → §13.3 leakage check)
  │
  ▼
Faz 5.1  Saf SAPS ablation  (5.1.1 smoke → 5.1.2 coarse → 5.1.3 fine)
  │
  ├── winner CONFIG ──────────────┐
  │                               │
  ├── winner BACKBONE ckpt ────────┼──────────────┐
  │                               │              │
  ▼                               ▼              ▼
Faz 5.2                     Teacher cache    Faz 5.4 baselines
multi-backbone              build            (SimCLR → MoCo-v3 → CoMAD)
(6 mimari)                    │
                              ▼
                        Faz 5.3  DT-SAPS ablation
                        (5.3.1 → 5.3.2 → 5.3.3)
                              │
                              ▼
                        DT-SAPS winner config
  
Tüm faz çıktıları ──▶ Final eval (fine-tune + detection mAP50 matrix)
```

**Kritik bağımlılıklar:**

1. **Faz 5.1 winner** üç şeyi birden besler: (a) Faz 5.2'nin sabit config'i,
   (b) Faz 5.3'ün "baseline floor" argümanı, (c) Faz 5.3'ün **SSL teacher
   checkpoint'i** — DualTeacherTrainer kararı gereği SSL teacher = Faz 5.1 saf
   SAPS winner backbone'u (momentum encoder değil). Faz 5.1 bitmeden Faz 5.3
   başlayamaz.
2. **Faz 5.3 önkoşulu** teacher feature cache: COCO YOLOv8x + SSL teacher
   (Faz 5.1 winner) feature'ları 186K pool üzerinde önceden cache'lenir.
3. **Faz 5.4 CoMAD-YOLO** üç SSL teacher gerektirir: SimCLR-YOLO + MoCo-v3-YOLO
   (Faz 5.4 içinde önce eğitilir) + dense-SAPS backbone (= Faz 5.1 winner).
   Bu yüzden Faz 5.4 iç sırası sabittir: SimCLR → MoCo-v3 → CoMAD.
4. **Faz 5.2 ⊥ Faz 5.3** — birbirine bağlı değil; ikisi de yalnız Faz 5.1
   winner'a bağlı. Tek GPU varsa sıralı, çoklu GPU varsa paralel koşturulabilir.

---

## ÖN — Pool hazırlık (eğitim başlamadan zorunlu)

Hiçbir eğitim, pool temizliği bitmeden başlamaz. Aksi halde SSL pretrain
pool'u ile downstream eval seti çakışır ve sonuçlar şişer.

| Adım | İş | Modül | Çıktı |
|---|---|---|---|
| §13.1 | Tüm pool'un pHash'i hesaplanır | `data/dedup` | `pool_phash.parquet` |
| §13.2 | Pool → yerel SSD'ye kopyalanır | (manuel/script) | `/content/ssl_pool_local/` |
| §13.3 | Cross-set leakage kontrolü | `eval/leakage_check.py` | leakage raporu |

`eval/leakage_check.py` çalıştırma:
```
python -m yolo_contrastive.eval.leakage_check \
    --pool-phash pool_phash.parquet \
    --downstream <eval_train_dir> <eval_valid_dir> \
    --hamming-threshold 5 \
    --output leaking_ids.txt
```
Leakage oranı %1'in üzerindeyse `leaking_ids.txt`'teki pool görüntüleri pool
manifest'inden çıkarılır, ondan sonra eğitim başlar.

---

## Faz 5.1 — Saf SAPS ablation

**Amaç:** SAPS-only en iyi config'i bulmak. DT-SAPS'ın "baseline floor"u,
paper Table 1.

**3-stage hiyerarşi** (YAML'lar commit `bb6796d`):

| Stage | Pool | Epoch | imgsz | Cell | A100 | $ |
|---|---|---|---|---|---|---|
| 5.1.1 Smoke | 5K | 30 | 320 | 6 | ~1 sa | ~$1.5 |
| 5.1.2 Coarse | 50K | 50 | 640 | 12 | ~50 sa | ~$75 |
| 5.1.3 Fine | 186K | 100 | 640 | 9 | ~100 sa | ~$150 |
| **Toplam** | | | | **27** | **~150 sa** | **~$230** |

Ablation eksenleri: `saps_mode` × `saps_both_lambda` × `queue_update_strategy`
× `saps_t_scale`. Redundant cell'ler §10.24a list-DSL exclude ile elenir.

**Stage geçiş kriterleri** — bir sonraki stage'e ancak bu sağlanınca geçilir:
- 5.1.1 → 5.1.2: 6 cell de crash-free **ve** acc@1 > 0.5 (smoke sağlık testi).
- 5.1.2 → 5.1.3: en iyi `(saps_mode, queue_update_strategy)` çifti sabitlenir;
  `λ` ve `t_scale` Stage 3'e bırakılır.
- 5.1.3 → sonraki: 27-cell winner config kilitlenir.

**Çıktı:** Paper Table 1 + **winner config** + **winner backbone checkpoint**.
Bu checkpoint Faz 5.3'ün SSL teacher'ı olarak da kullanılacağından ayrıca
saklanır (örn. `faz51_saps_winner.pt`).

---

## Teacher cache build — Faz 5.3 önkoşulu

Faz 5.1 bitince, Faz 5.3'e geçmeden önce iki teacher'ın feature'ları 186K pool
üzerinde cache'lenir. `TeacherCache.build()` kullanılır; cache FP16 npz.

| Teacher | Kaynak | Cache tag | Not |
|---|---|---|---|
| COCO | YOLOv8x (`yolov8x.pt`) | `yolov8x_coco` | Risk 17 — student'a adapter eğitimde |
| SSL | Faz 5.1 winner backbone | `saps_winner` | YOLOv8n = student → adapter yok |

Cache, augmente edilmemiş orijinal görüntü üzerinde alınır (§10.27). Tek-geçiş
forward, backprop yok — maliyet düşük (~birkaç saat A100).

**Cache strateji karar noktası:** P3/P4/P5 tam cache mi, P5-only mi — Faz 5.3.1
smoke'tan sonra netleşir, Faz 5.3.2'den önce finalize edilir. Disk ihtiyacı
buna göre planlanır.

---

## Faz 5.2 — Multi-backbone validation

**Amaç:** Faz 5.1 winner config'in 6 farklı YOLO mimarisinde robust olduğunu
göstermek. Paper Table 2, "generalizes across architectures" claim'i.

| Backbone | Params | Config | Cell |
|---|---|---|---|
| YOLOv8n / v9n / v10n / v11n / v12n / YOLO26n | 2.6–3.0M | Faz 5.1 winner | 6 |

Sabit: SAPS config Faz 5.1'den, 186K pool, 100 epoch. **Süre:** ~70 saat
A100, ~$105.

**Risk 21:** v9n/v10n/v11n/v12n/v26n'in batch_size / bellek profilleri farklı.
Her backbone için Stage başında 1-epoch smoke ile batch_size auto-detect.

**Konumlandırma:** Faz 5.3'e bağlı değil; Faz 5.1 bitince başlayabilir. Tek GPU
varsa Faz 5.3'ten önce ya da sonra; çoklu GPU varsa paralel.

---

## Faz 5.3 — DT-SAPS ablation (paper'ın asıl yenisi)

**Amaç:** Dual-teacher consensus + disagreement weighting'in Faz 5.1 saf SAPS
baseline'a katkısını ölçmek. Paper Table 3 + Figure 2.

**Önkoşul:** Teacher cache hazır (yukarıdaki bölüm).

**Ablation eksenleri:** `teacher_combo` × `distill_form` × `α` × `β` × `γ` ×
`α_d` × `w_init` × `aug_photometric`. Tam grid ~700 cell; list-DSL exclude ile
~320 cell'e iner. 3-stage hiyerarşi:

| Stage | Pool | Epoch | Cell | A100 |
|---|---|---|---|---|
| 5.3.1 Smoke | 5K | 30 | 8 (teacher_combo × distill_form keşfi) | ~1.5 sa |
| 5.3.2 Coarse | 50K | 50 | ~40 (winner combo × α/β/γ/α_d, w_init=0.5) | ~150 sa |
| 5.3.3 Fine | 186K | 100 | ~12 (winner combo + form + α/β/γ/α_d) | ~120 sa |
| **Toplam** | | | **~60** | **~270 sa** |

Bütçe: **~$405**. Faz 5.1'den pahalı — dual-teacher eksen sayısı fazla.

**Notlar:**
- `α_d` ekseni v9'da güncellendi: `{-1, -0.5, 0, 0.5, 1, 2}` — negatif değerler
  consensus yönünü (PAD/CoMAD) test eder (§10.29).
- `aug_photometric` ekseni Faz 5.1'in saf SAPS run'larıyla paylaşılır; iki kez
  harcanmaz (Stage 5.3 SAPS-only cell'leri 5.1.3 winner'ı miras alır).
- Cache strateji (P3/P4/P5 vs P5-only) 5.3.1 sonrası finalize.

**Çıktı:** Paper Table 3 + Figure 2 (`w_coco` öğrenme eğrisi) + **DT-SAPS
winner config**.

---

## Faz 5.4 — External baselines

**Amaç:** DT-SAPS'i SOTA SSL yöntemleriyle karşılaştırmak. Paper Table 4.

Üçü de YOLOv8n + Faz 5.1 winner protokolü (fair comparison). **İç sıra
sabittir** — CoMAD-YOLO diğer ikisini teacher olarak kullanır:

| Sıra | Baseline | Trainer | A100 | Bağımlılık |
|---|---|---|---|---|
| 1 | SimCLR-YOLO | `SimCLRYOLOTrainer` | ~15 sa | — |
| 2 | MoCo-v3-YOLO | `MoCoV3YOLOTrainer` | ~15 sa | — |
| 3 | CoMAD-YOLO | `CoMADYOLOTrainer` | ~25 sa | adım 1 + 2 + Faz 5.1 winner |
| — | COCO baseline | (mevcut, §11.8) | 0 | — |
| **Toplam** | | | **~55 sa** | **~$85** |

CoMAD-YOLO'nun 3 SSL teacher'ı: SimCLR-YOLO (adım 1) + MoCo-v3-YOLO (adım 2) +
dense-SAPS backbone (Faz 5.1 winner). Bu yüzden CoMAD-YOLO Faz 5.4'ün son
adımıdır.

---

## Final eval — fine-tune + detection mAP50 matrix

Tüm pretraining çıktıları (Faz 5.1 winner, Faz 5.2'nin 6 backbone'u, Faz 5.3
DT-SAPS winner, Faz 5.4'ün 3 baseline'ı + COCO baseline) downstream traffic
detection dataset'inde fine-tune edilir ve mAP50 ölçülür.

**Hedef X:** DT-SAPS mAP50 ≥ COCO baseline (≈ 0.6266, §11.8) — paper'ın ana
claim'i. Karşılaştırma `eval/run_matrix.py` detection runner ile yürütülür.

---

## Bütçe ve süre özeti

| Faz | A100 saat | Tahmini $ (@$1.5/sa) |
|---|---|---|
| Pool hazırlık | ~birkaç saat (CPU ağırlıklı) | düşük |
| Faz 5.1 SAPS | ~150 | ~$230 |
| Teacher cache build | ~birkaç saat | düşük |
| Faz 5.2 multi-backbone | ~70 | ~$105 |
| Faz 5.3 DT-SAPS | ~270 | ~$405 |
| Faz 5.4 baselines | ~55 | ~$85 |
| Final eval (fine-tune matrix) | (downstream, ayrı bütçe) | — |
| **Toplam (pretraining)** | **~545+** | **~$825+** |

---

## Karar / risk noktaları (gate'ler)

Her gate'te durup devam kararı verilir:

1. **Pool leakage gate** — leakage > %1 ise pool temizlenmeden eğitim başlamaz.
2. **5.1.1 smoke gate** — 6 cell crash-free + acc@1 > 0.5 değilse Faz 5.1.2'ye
   geçilmez (config/pipeline hatası sinyali).
3. **Teacher cache gate** — cache build sonrası birkaç örnekte feature sağlık
   kontrolü; bozuk cache Faz 5.3'ü baştan zehirler.
4. **5.3.1 smoke gate** — teacher_combo=both'un saf SAPS'ı geçtiğine dair erken
   sinyal; geçmiyorsa eksen/loss gözden geçirilir.
5. **Final eval gate** — DT-SAPS mAP50 < COCO baseline ise paper'ın ana claim'i
   sallanır; §13 dürüst raporlama notları devreye girer.

---

## Notlar

- Faz 5.2 ile Faz 5.3'ün sırası esnek (ikisi de yalnız Faz 5.1'e bağlı). Tek
  GPU senaryosunda önerilen sıra: 5.1 → cache build → 5.3 → 5.2 → 5.4, çünkü
  Faz 5.3 paper'ın asıl katkısı ve en uzun kuyruğu; erken başlaması toplam
  takvimi kısaltır.
- Bütün eğitimler Colab + vast.ai A100 üzerinde, mevcut `pretrain/run_matrix.py`
  CSV state machine ile checkpoint-resume destekli yürütülür.
- Bu plan WORK_PLAN_v9.md §5.1–5.4 + §13'ten türetildi; rakamlar plan tahminidir
  ve smoke stage'lerinde kalibre edilir.
