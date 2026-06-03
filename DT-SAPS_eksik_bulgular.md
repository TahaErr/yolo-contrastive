# DT-SAPS — Raporda Eksik Kalan Bulgular (Tam Detay)

> **Kapsam.** Bu dosya, **DT-SAPS (Dual-Teacher SAPS)** çalışmasının final raporda
> (CMP-719 / ICLR draft) yalnızca birkaç cümleye sıkıştırılmış olan negatif-sonuç
> teşhis zincirinin tüm detaylarını toplar. Standalone SAPS pilotu (linear-probe
> top-1 ≈ 0.91 @ ep43, Tablo 7'de mAP50 0.335 @100%) raporda zaten yer aldığı için
> "eksik bulgu" kapsamına girmez; burada **DT-SAPS** ele alınır.
>
> **Kaynaklar.** `WORK_PLAN_v9.md` §10.35–§10.36, §10.28–§10.29; kod:
> `src/yolo_contrastive/dual_teacher/{mgd_distiller,consensus_loss,disagreement,coco_teacher,dual_teacher_trainer}.py`;
> ve "Work plan 10 improvements" / GASP collapse-debug oturumları (2026-05).
>
> **Tek cümlelik özet.** Frozen heterojen-teacher feature distillation, YOLOv8n↔YOLOv8x
> kapasite uçurumunda öğrenci→teacher **R² ≈ 0.27**'de yapısal olarak doyar; bu sonuç,
> 8 kontrollü deneyle (vanilla MGD, DAMS-cosine, DAMS-L2, RAMS, λ-taraması, generator
> derinliği, generator-baypas) ve CKA + lineer-açıklanabilirlik ölçümleriyle kanıtlanmıştır.

---

## 1. Neden bu bulgu önemli ve neden "eksik"

Final raporda DT-SAPS §3.4'te mekanizma düzeyinde (Form B/C/D, iki teacher, GradNorm)
ve §5.2'de sonuç düzeyinde ("sekiz kontrollü deney", "R²≈0.27", üç ders) **doğru ama
yüzeysel** biçimde anlatılır. Raporu okuyan biri o sekiz deneyin "dual KL +
disagreement weighting" varyasyonları olduğunu sanır. Gerçekte sekiz deneyin
**çoğu Masked Generative Distillation (MGD) + adaptif maske örnekleme** üzerinde koştu,
ve teşhisi sağlam kılan ölçümler (CKA, generator-baypas, anlaşmazlık-haritası
istatistikleri) raporda **hiç yok**.

Kendi çalışma notların bu zinciri *"paper'ın en sağlam asset'i (WACV/BMVC/TIP)"* diye
nitelendirmişti — yani burada bilinçli/farkında olmadan ciddi bir değer feda edilmiş.
Özellikle **RAMS**, kurulmuş + test edilmiş + literatürden ayrışan özgün bir mekanizma
olduğu halde raporda adı dahi geçmiyor; bu, FrequencyBandPrediction ve Task-Routed
Conv-LoRA gibi *üçüncü bir özgün bileşen* niteliğinde.

### Raporda VAR olanlar (referans)
- Form B (learned-weighted feature distillation), Form C (dual KL), Form D
  (disagreement weighting; `exp(α_d · d(f_coco, f_ssl))`), teacher çifti (YOLOv8x COCO +
  momentum SSL), GradNorm "in spirit". (§3.4)
- "Sekiz kontrollü deney", "R² ≈ 0.27 doygunluğu", üç ders. (§5.2)
- Final matris satırı (Tablo 7) ve localization (Tablo 8) — aşağıda §9.

### Raporda EKSİK olanlar (bu dosyanın konusu)
1. **MGD (Masked Generative Distillation)** — 8 deneyin fiili çekirdeği; raporda hiç geçmiyor.
2. **RAMS (Reconstruction-Adaptive Mask Sampling)** — özgün, kodlanmış, test edilmiş mekanizma; raporda hiç geçmiyor.
3. **CKA(ep5, ep15) = 0.999** — "ep5'te fiilen donmuş" tezinin somut kanıtı.
4. **Generator-baypas kontrolü** — "mekanizmadan bağımsız" ifadesini hak ettiren deney.
5. **DAMS-cosine / DAMS-L2 adımları** ve anlaşmazlık-haritası teşhisleri (ortogonal teacher yönleri; statik-yapılı harita).
6. **DT-SAPS v1 erken-doyma** spesifikleri (ep2'de %75–92 çöküş + ölü taban).
7. **Maske-sinyali yapı/dinamiklik istatistikleri** (cosine CV, L2 CV, reconstruction CV + sıralama-korelasyonu).

---

## 2. DT-SAPS nedir (mimari)

DT-SAPS, dense SAPS ön-eğitimine **iki donuk teacher'dan** distillation ekler:

- **COCO teacher:** COCO-pretrained **YOLOv8x** (süpervizeli bilgi). Öğrenci YOLOv8n ile
  kanal uyumsuzluğu var (teacher P3 80×80×128 → P5 20×20×512; öğrenci P3 80×80×64 →
  P5 20×20×256), bu yüzden **per-scale lineer adapter** (frozen teacher feature →
  öğrenci kanal sayısı) kullanılır.
- **SSL teacher:** Faz 5.1 saf-SAPS kazananı (öz-denetimli bilgi), momentum encoder.

İki sinyal consensus + disagreement weighting ile birleştirilir. Modül haritası
(`src/yolo_contrastive/dual_teacher/`):

| Modül | Görev |
|---|---|
| `coco_teacher.py` | Frozen YOLO feature teacher + per-scale adapter |
| `teacher_cache.py` | FP16 npz feature cache I/O (augmentsiz orijinal imge üzerinde) |
| `disagreement.py` | Per-position cosine disagreement weighting (§10.29) |
| `consensus_loss.py` | Form B (learned-weighted L2) + Form C (CWD dual KL) (§10.28) |
| `dual_teacher_trainer.py` | DT-SAPS trainer; `DenseSSLPretrainer` üzerine composition (§10.30) |
| `mgd_distiller.py` | **MGD + RAMS** (ayrı modül; ConsensusLoss'a dokunmadan, commit `0b00483`) |

**Teacher cache kararı (§10.27):** Teacher feature'ları **augmentsiz orijinal imge**
üzerinde cache'lenir; augmentation çeşitliliği yalnızca öğrencinin view_a/view_b
zincirinden gelir. Teacher "aug-invariant anchor" rolü oynar (DINO/iBOT paterni).
Compute ~100× ekonomi + exact reproducibility.

### Loss formları (DT-SAPS v1)
- **Form B — learned-weighted L2 (feature-space):**
  `target = w·f_coco + (1−w)·f_ssl`, `L_B = ‖f_student − target‖²` (per-position, kanal-ortalamalı).
  `w` **öğrenilebilir** skaler (sigmoid ile [0,1]); `w_init ∈ {0.3, 0.5, 0.7}` ablation ekseni;
  `w_coco` vs epoch eğrisi planlı bir paper figürüydü.
- **Form C — channel-wise dual KL (logit-space), CWD (Shu et al. 2021):** her kanalın
  H×W haritası spatial dağılıma softmax-normalize edilir;
  `L_C = KL_cwd(student‖coco) + KL_cwd(student‖ssl)`. İki teacher **ayrı tutulur** (füzyon yok),
  böylece B / C / B+C ablation'ı anlamlı kalır. Softmax, teacher/öğrenci magnitude-ölçek
  farkını siler (cosine'i disagreement metriği olarak seçmenin aynı gerekçesi).
- **B+C:** `α·L_B + β·L_C` (multi-level transfer).
- **Form D — disagreement weighting (§10.29):** `L_distill *= exp(α_d · d(f_coco, f_ssl))`.
  CoMAD'ın consensus gating'inin **tersine** (CoMAD anlaşmazlığı *filtreler*), DT-SAPS
  anlaşmazlık bölgelerini **vurgular** (hard-mining). `α_d ∈ {−1, −0.5, 0, 0.5, 1, 2}`;
  sayısal koruma `clamp(weight, max=10)`, `α_d ∈ [−3, 3]`.
- **Loss dengeleme:** GradNorm (Chen et al. 2018) ruhunda.

---

## 3. Bulgu A — DT-SAPS v1 erken-doyma teşhisi (§10.35)

**Olgu.** DT-SAPS v1'in distillation kaybı **erken doydu**: **ep2'de %75–92 çöküş +
ölü taban** (dead floor). Loss keskin bir duvara çarpıp sabitleniyordu.

**Kök neden (kontrollü teşhisle kesinleşti):**
- Random-init öğrenci ep0 distill = 1.02.
- SAPS **kapalı** saf-distill koşusunda da erken doyma görüldü → sorun SAPS baskısı değil,
  **donuk teacher'ın STATİK feature hedefi.**

**Çözüm yönü.** "Teacher feature'ını kopyala" yerine "maskeli öğrenci feature'ından
teacher'ı yeniden üret" — maske her batch yeniden örneklenir, hedef tek bir noktaya
çökmeyip kombinatoryal büyüklükte bir reconstruction görev kümesine dönüşür. Bu, MGD
(Yang et al., ECCV 2022, arXiv:2205.01529) fikridir. DT-SAPS Improved = MGD + DAMS + ADS
hibridi olarak tasarlandı.

---

## 4. Bulgu B — DT-SAPS Improved: MGD→RAMS teşhis merdiveni (§10.36)

**Ortak ölçüm protokolü:** 5K havuz, 15 epoch (λ-taraması 10 epoch), dual-teacher
(COCO YOLOv8x + SSL = SAPS pilot), öğrenci YOLOv8n. **Tüm sayılar tek koşu.**

### 4.1 — Vanilla MGD (Aşama 0)
Form B/C yerine maskeli üretim; `λ_mask = 0.65` sabit, tek-teacher.
- **Sonuç:** v1'in **ep2 duvarını kaldırdı** — loss ep1→15 monoton iniyor, ölü tabana
  çakılmıyor (epoch-epoch düşüş hep pozitif, +12.6% → +0.13%).
- **Ama:** ep5 sonrası **yumuşak plato** kaldı. Düşüşün ~%90'ı ilk 2 epoch'ta
  (0.0326 → 0.0285); ep5–15 toplam düşüş **%3.3** (`ep15/ep10 = 0.990`).
- **Yorum:** Her-batch-farklı-maske hedefi tüketilmez yaptı, ama **tek-nokta
  yeniden-üretim görevi yine ep5'te doyuyor.** Vanilla MGD gerekli ama yeterli değil.

### 4.2 — DAMS / cosine disagreement
Maske, iki teacher'ın **cosine-anlaşmazlık haritasından** örneklenir.
- **Sonuç:** Platoyu **kırmadı**; disagreement 15 epoch boyunca sabit **~1.0038**.
- **Teşhis:** COCO/SSL teacher feature **yönleri her yerde ortogonal** (cosine CV ~0.05–0.12)
  → anlaşmazlık haritası düz → DAMS **uniform maskeye çöktü** (DAMS = vanilla MGD'ye eşdeğer hale geldi).

### 4.3 — DAMS / ham L2 disagreement
Metrik cosine → **ham L2 + per-image min-max norm** (yapı magnitude'de).
- **Sonuç:** Platoyu **kırmadı**; disagreement yine sabit **0.1545**.
- **Teşhis:** L2 haritası **yapılı ama statik** (L2 CV ~0.24–0.47; cosine ve norm-L2 düz).
  İki teacher donuk → anlaşmazlık öğrenciden bağımsız → **statik maske kaynağı = başka
  bir statik hedef**, öğrenci yetişince plato yine oluşur.

### 4.4 — RAMS (Reconstruction-Adaptive Mask Sampling) — **ÖZGÜN MEKANİZMA**
Maske sinyali teacher-anlaşmazlığından değil, **öğrencinin kendi reconstruction
hatasından** örneklenir: `err = max(err_coco, err_ssl)`, EMA-hafızalı (`m = 0.9`).

- **Reconstruction teşhisi:** sinyal hem **yapılı** (CV ~0.35) hem **dinamik**
  (sıralama-korelasyonu 5 epoch boyunca **0.55–0.72** — "en kötü" pozisyonlar kayıyor).
  Statik anlaşmazlık haritasının aksine, sinyal öğrenciyle birlikte **evriliyor.**
- **Sonuç:** Platoyu yine **kırmadı** (ep10→15 %1.28); **ama** `recon_err` eğitim
  boyunca gerçekten **hareket etti** — yani sinyal dinamikti.
- **Kritik bulgu:** **Dinamik sinyal bile platoyu kırmıyor → sorun maske KONUMU değil.**
  Bu, sonraki adımın (generator/plato teşhisi) önünü açtı.

**RAMS mekaniği (`mgd_distiller.py`, per-level per-batch):**
1. Maske, `err_memory[level]` üzerinden örneklenir (EMA of past per-position recon error).
   İlk batch: `err_memory` boş → maske uniform (cold-start).
2. `masked_student = student * mask`; `gen_coco` / `gen_ssl` teacher feature'ını yeniden üretir.
3. Loss sonrası `err = max(‖gen_coco − coco‖, ‖gen_ssl − ssl‖)` per-position (no-grad),
   per-image min-max normalize, batch-ortalaması ile `[H,W]`, ve EMA ile `err_memory`'ye katılır.
   EMA hem chicken-and-egg'i çözer (maske GEÇMİŞ hatayı kullanır, mevcut maskeden bağımsız)
   hem de gürültülü tek-batch sinyalini düzleştirir.

**RAMS'in literatürden farkı (kodun kendi dokümantasyonundan):** AMD / DMKD / SAMKD ailesi
maske ipucunu **statik teacher attention'dan** alır; **RAMS evrilen öğrenci reconstruction
durumundan** alır. Hedef öğrenciyle birlikte hareket eder, tüketilemez — bu, "statik hedef
tükenir" teşhisine doğrudan yanıt veren tek mekanizma tasarımıydı.

### 4.5 — λ-taraması
RAMS sabit, `λ_mask ∈ {0.5, 0.65, 0.8}`, adil karşılaştırma (aynı seed).
- **Sonuç:** Plato **λ'ya duyarsız** — geç-düşüş yelpazesi yalnızca **%1.27.**
- Üstelik **λ↑ geç-düşüşü hafifçe KÖTÜLEŞTİRİYOR** (λ=0.5 → %3.58; λ=0.8 → %2.31).
- **Yorum:** Maske **oranı** da kök neden değil; yüksek maske öğrenmeyi sürdürmüyor,
  daha erken doygunluk veriyor.

### 4.6 — Plato gerçekliği + generator teşhisleri (kök nedene iniş)
**(a) Plato gerçek mi? — CKA.** ep5 ve ep15 backbone snapshot'ları, pothole imgelerinde CKA.
- **CKA(ep5, ep15) = 0.999**, efektif rank değişimi **±0.1%** → **backbone ep5'te FİİLEN DONMUŞ.**
- Plato bir distill-loss artefaktı değil; **gerçek öğrenme durması.** (Raporun "static
  target… fixed point the student reaches early" dersinin somut kanıtı budur.)

**(b) Generator kapasitesi.** Derin generator (4-conv + residual) vs sığ (2-conv MGD simple block).
- Derin generator platoyu **kırmadı** — yalnızca daha yüksek bir seviyeden daha yavaş
  yakınsadı (ep10'da hâlâ 0.046; sığ 0.029). **Kapasite kök neden değil.**

**(c) Generator-baypas.** Generator tümden kaldırılır → **saf öğrenci→teacher MSE feature
imitation.**
- Generator'lı ve generator'sız kurulum **AYNI**: öğrenci donuyor, **R² ~0.27**'de tavan.
- **Generator suçlu değil** — bu kontrol, "mekanizmadan bağımsız" ifadesini hak ettiren deneydir.

---

## 5. Kök neden — temsil uyumsuzluğu (R² ≈ 0.27)

Öğrenci→COCO-teacher **lineer açıklanabilirlik (R²)**:
- ep5'te **0.27**, ep15'te hâlâ **0.27** — kazanç **sıfır.** Saf feature imitation'da bile öyle.

**Yorum.** Bir YOLOv8n SSL öğrencisi, YOLOv8x COCO-süpervizeli teacher'ın feature uzayının
**%73'ünü lineer olarak BİLE temsil edemiyor** — ve temsil edebildiği %27'yi ep5'te alıp
orada kalıyor. Plato bir **mekanizma kusuru değil**; öğrenci–teacher **temsil
uyumsuzluğunun ölçüsü.** 8 bağımsız geliştirme (4 maske varyantı, 3 λ, generator derinliği)
hepsi maske/oran/generator tarafını ayarladı; kök neden ise **frozen heterojen-teacher
feature distillation'ının kendisinde.**

---

## 6. 8-deney özet tablosu (§10.36'dan)

| # | Metod | Sonuç (kanıt) | Neden işe yaramadı |
|---|---|---|---|
| 1 | Vanilla MGD | duvar kalktı, plato kaldı (%0.97) | tek-nokta yeniden-üretim görevi ep5'te doyuyor |
| 2 | DAMS cosine | kırmadı; disag sabit ~1.0038 | teacher yönleri her yerde ortogonal — harita düz, maske uniform'a çöktü |
| 3 | DAMS L2 | kırmadı; disag sabit 0.1545 | harita yapılı ama statik — donuk teacher, öğrenciden bağımsız |
| 4 | **RAMS** | kırmadı (%1.28); sinyal dinamik (CV~0.35, sıralama-kor. 0.55–0.72) | sorun maske konumu değil — dinamik sinyal bile kırmıyor |
| 5 | λ-taraması | duyarsız (%1.27 yelpaze) | maske oranı kök neden değil; λ↑ erken doygunluğu artırıyor |
| 6a | derin generator | kırmadı; daha yavaş yakınsama | generator kapasitesi değil |
| 6b | generator-baypas | generator'lı = generator'sız | generator suçlu değil |
| — | **KÖK NEDEN** | öğrenci→teacher R² 0.27'de tavan (ep5 = ep15) | YOLOv8n, YOLOv8x feature'ının %73'ünü temsil edemiyor — temsil uyumsuzluğu |

---

## 7. Üç ders ve GASP'a bağlantı

Kampanyadan üç ders kristalize oldu (bunlar raporda **var**, ama yukarıdaki kanıtlar
olmadan):
1. **Statik hedef tükenir** — donuk teacher'ın feature'ı sabit bir nokta; öğrenci erken
   yetişir. *(Kanıt: CKA = 0.999 — raporda yok.)*
2. **Heterojen kapasite uçurumu mekanizma ayarıyla aşılamaz.** *(Kanıt: 8 deney.)*
3. **Mekanizma ayarı kök nedeni adreslemiyor** — deneyler "nasıl aktaralım"ı ayarladı,
   sorun "ne aktarıyoruz"daydı. *(Kanıt: generator-baypas — raporda yok.)*

Bu üç ders GASP'ı doğrudan şekillendirdi: **teacher yok** (statik hedef yok), **tek ağ**
(kapasite uçurumu yok), **hedef sahnenin kendi geometrisinden** (tükenmez). Yani DT-SAPS'in
negatif sonucu, GASP'ın varlık gerekçesidir — bu nedensel bağ raporda kısa geçer, ama bu
dosyadaki kanıt zinciriyle çok daha güçlü kurulabilir.

---

## 8. Final matris sayıları (referans — raporda VAR)

DT-SAPS backbone'u final 10-kaynak LOSO frozen-probe matrisine de girdi:

| Metrik | @10% | @50% | @100% |
|---|---|---|---|
| mAP50 (mean ± std) | 0.227 ± 0.083 | 0.326 ± 0.128 | 0.343 ± 0.124 |
| mAP50:95 @100% | — | — | 0.137 |

(Tablo 7 / Tablo 8.) Not: bu sayılar **final** protokoldendir; yukarıdaki teşhis sayıları
**5K/15-epoch dev** protokolündendir ve absolute değerde kıyaslanamaz — ayrı okunmalıdır.

---

## 9. Önem, yayın değeri ve öneri

- **RAMS** tek başına bir katkı: öz-denetimli/distillation maskelemeyi statik teacher
  attention'dan değil, **evrilen öğrenci reconstruction hatasından** süren özgün bir
  mekanizma. Kurulmuş + test edilmiş (`TestGASP...`/MGD test setleri) + literatürden
  (AMD/DMKD/SAMKD) açıkça ayrışıyor. Negatif sonuç çerçevesinde bile en azından bir
  cümle + bir ablation satırı hak ediyor.
- **Kök-neden teşhisi (R²=0.27 + CKA + generator-baypas)**, "frozen heterojen-teacher
  distillation küçük detektörde neden başarısız" sorusuna **mekanizmadan bağımsız** bir
  cevap veriyor — diagnostic katkı olarak güçlü ve yayınlanabilir.
- **Önerilen rapor eklentisi:** §3.4'e kısa bir "DT-SAPS Improved" paragrafı (MGD + RAMS
  + ADS hibridi) + §5.2'ye 8-satırlık ablation tablosu (bu dosyanın §6'sı) + CKA ve
  generator-baypas'ın birer cümlelik aktarımı. Headline değiştirmeden, negatif sonucu
  *daha sağlam* kılar.

---

## 10. Caveat'ler (dürüstlük notları)

- **Tüm teşhis ölçümleri 5K havuz / 15-epoch / tek koşu.** Patern (ep5 plato, R²=0.27
  tavanı) 8 deneyde tutarlı ve büyük ölçüde ölçek-bağımsız görünüyor; kesin doğrulama
  tam-koşu `loss_history`'sinden gelir.
- **Test edilmemiş tek varyant:** öğrenciye yakın ölçekli teacher (YOLOv8s) — uyumsuzluğu
  azaltabilir, ama DT-SAPS'in "güçlü süpervizeli teacher" değer önermesini zayıflatır.
- Bu dosyadaki sayılar çalışma planı + oturum kayıtlarından derlendi; rapora aktarmadan
  önce repo'daki güncel `loss_history` / metrik çıktılarıyla **bire bir teyit** önerilir.

---

*Sonraki dosyalar (planlanan, her biri ayrı): GASP v8 dev bulguları (eff_rank 21.94,
fine-tune Δ −0.029, ölçekle collapse derinleşmesi 10.48→7.97, eff_rank yetersizliği) — ve
istenirse standalone SAPS pilotu ve diğer eksikler.*
