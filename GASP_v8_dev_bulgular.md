# GASP v8 — Raporda Eksik Kalan Dev Bulguları (Tam Detay)

> **Kapsam.** Bu dosya, **GASP v8** (cross-scale reconstruction) çalışmasının
> **30K dev protokolünde** ölçülmüş ama final rapora (CMP-719 / ICLR draft)
> girmemiş — ve bazı noktalarda raporla **çelişen** — bulgularını toplar.
> Rapor, v8'in **final 10-kaynak LOSO frozen-probe** sayısını (mAP50 0.257 @100%,
> matrisin dibinde) doğru biçimde raporlar; ama v8'in dev-protokol teşhislerini
> "pending" / "not measured" diye bırakır. Oysa o teşhisler **ölçülmüştü** ve
> daha nüanslı bir hikâye anlatıyorlar.
>
> **Kaynaklar.** "Work plan 10 improvements" ve GASP collapse-debug oturumları
> (2026-05-30, commit mesajı + 4-sonuç tablosu + 5-parça hikâye özeti);
> `WORK_PLAN_v10_gasp_v6_v7.md` (v3→v7 trajesi, dev protokol sabitleri);
> `GASP_DESIGN_PLAN.md`; kod: `src/yolo_contrastive/gasp/{reconstruction,trainer}.py`.
>
> **Tek cümlelik özet.** v8'in cross-scale reconstruction'ı collapse'ı **kırdı**
> (feature effective rank 3.09 → **21.94**, ~10 tavanını aşan ilk yöntem) ve
> fine-tune altında COCO'ya açığı neredeyse **yarıladı** (frozen Δ −0.063 →
> finetune Δ −0.029); ama enjekte edilen bilgi "detection-hazır" biçimde olmadığı
> için (common-mode cos~1) frozen-probe bunu olduğundan zayıf gösteriyor —
> bu da **eff_rank'in tek başına yetersiz bir SSL metriği** olduğunu ortaya koyuyor.

---

## 1. Neden bu bulgu önemli ve neden "eksik"

Raporun GASP anlatısı (§3.5, §5.3) v8'i şöyle bırakır: *"This version is training at
the time of writing; its standalone SSL diagnostics and detection result are pending."*
Tablo 6'da v8 satırı **mAP50 (dev) = "not measured (dev)"**, **eff rank = "not measured"**
der; yalnızca final-matris sonucuna (0.257) atıf yapar. §3.5 dürüst-konumlandırma da
v8'i "frozen feature olarak zayıf transfer" diye özetler — ki final matris için **doğru.**

**Eksik olan**, bu final-matris hükmünün arkasındaki **dev-protokol kanıtları**dır; bunlar
raporun *kendi merkezî açık sorusuna* ("does scale-equivariance help under full fine-tuning?
→ left to future work") kısmi bir cevap veriyor. Dahası, Tablo 6'daki bir satır
(v6r 181K = 10.48) **muhtemelen yanlış etiketli** — gerçek 181K sayısı (7.97) ve onun
taşıdığı "ölçekle collapse derinleşiyor" bulgusu eksik.

### Raporda VAR olanlar (referans)
- Final matris (Tablo 7): GASP v8 = **0.257 @100%** (matrisin dibinde, scratch hariç en alt).
- Localization (Tablo 8): GASP v8 mAP50:95 @100% = **0.094**.
- §3.5: collapse teşhisi (single-axis → content ekseni zorlanmadı), VICReg/isotropy/projection
  head'in neden çözmediği (variance bir **floor**'dur, covariance rank'a **kör**, projector
  collapse baskısını backbone'a sızdırır), cross-scale reconstruction mekanizması, ve
  "frozen feature olarak zayıf" dürüst konumlandırması.
- Tablo 6: COCO (dev) 0.2341 / eff_rank 60.43; v3 very low; v5 9.36; v6r (181K) **10.48**; v7 3.09.

### Raporda EKSİK olanlar (bu dosyanın konusu)
1. **v8 eff_rank = 21.94** — collapse'ı kıran ilk yöntem. Rapor "not measured" diyor. *(Bulgu A)*
2. **Dev fine-tune karşılaştırması** — frozen Δ −0.063 → finetune Δ −0.029. Raporun "future work" dediği sorunun ön-cevabı. *(Bulgu B)*
3. **Ölçekle collapse derinleşmesi** — v6r 30K = 10.48 → 181K = **7.97** (daha kötü). + Tablo 6 etiket sorunu. *(Bulgu C)*
4. **eff_rank yetersiz metrik** — v8 bilgi taşıyor (standardize 13.6→22) ama detection-hazır değil (cos~1). *(Bulgu D)*
5. Ufak: v6iso (isotropy) = 8.78 başarısız; P5-BN denemesi başarısız; koşulmamış teklifler. *(§8)*

---

## 2. GASP v8 nedir (cross-scale reconstruction)

**Çekirdek fikir.** Önceki GASP sürümleri **tek-eksenliydi**: yalnızca *ölçek* eksenini
kurdu (FiLM-T equivariance + in-image consistency), *içerik* (kimlik) eksenini hiç
zorlamadı. "Bu bir araba mı?" hiç sorulmadığı için encoder, ölçek-tutarlılığını
düşük-rank bir alt-uzayda sağlayıp **collapse** etti. Yüksek rank'ı süren tek kuvvet
içerik ayrımıdır, o da yoktu.

**v8 çözümü.** İçerik eksenini **denetimsiz, yapısal** bir kuvvetle ekle: **reconstruction.**
Bir patch'i ölçek *a*'da kodla, görünümünü ölçek *b*'de **P5 spatial haritasından**
yeniden üret. Reconstruction içerik korumayı zorunlu kılar (düşük-rank darboğazdan piksel
üretilemez); cross-scale olması onu ölçek eksenine bağlar. En yakın öncül: **Scale-MAE**
(Reed et al. 2023). **Spatial bottleneck**, "resize'ı tersine çevir" kısayolunu kırar:
decoder view_a piksellerini hiç görmez, yalnızca kodlanmış haritayı görür.

**Mimari (`reconstruction.py`, YENİ):**
- `ScaleConditionedDecoder`: hafif conv upsampler, **~0.4M parametre** (backbone'dan **7× küçük**),
  `log_ratio` ile **FiLM-conditioned** (T ile aynı mekanizma → birleşik ölçek işleme).
  `gap_bottleneck` flag: girişi 1×1'e pool eder (downstream'in kullandığı GAP vektörü) →
  daha güçlü içerik baskısı.
- `cross_scale_reconstruction_loss`: photometric-jitter'sız, **sadece-ölçek** view'ler;
  `_augment_patch_scale_aware`'i yeniden kullanır (tek render kaynağı).

**Trainer entegrasyonu (`trainer.py`):**
- `use_reconstruction` flag (default `False` = v6/v7 davranışı, backward compat).
- `_encode_spatial`: GAP öncesi P5 haritası (decoder girişi, backbone'a gradient).
- `lambda_recon`, `recon_base_dim`, `recon_gap_bottleneck` parametreleri.
- `L_recon` → `loss_history`'ye yazılır + print edilir.
- decoder optimizer + resume checkpoint'e eklenir; **final checkpoint sadece backbone**
  (decoder atılır → COCO ile birebir kıyaslanabilir).
- **Test:** `TestGASPReconstructionV8` (7 test); tüm GASP suite geçiyor.

**v8 config (ölçülen):** `lambda_recon = 30`, P5 darboğaz **2×2** (64-px patch için).

---

## 3. Bulgu A — v8 collapse'ı kırdı (eff_rank 21.94)

**30K dev, pothole valid backbone üzerinde feature effective rank** (100 pothole valid imge,
P5 global-average-pooled; COCO referans ≈ 60.43):

| Sürüm | Konfigürasyon | eff_rank | Not |
|---|---|---|---|
| COCO (dev) | — | 60.43 | referans |
| GASP v3 | VICReg, 2-scale, vanilla matcher | very low | failed |
| GASP v5 | F-variant (InfoNCE on log r) + stratified + 4-scale | 9.36 | anizotropi/norm patlaması |
| GASP v6r (30K) | + L_ctrl scale-aware augmentation fix | 10.48 | controlled term öğreniyor; collapse sürüyor |
| GASP v6iso (30K) | + isotropy regularizer | 8.78 | başarısız |
| GASP v7 | + projection head | 3.09 | projector backbone'u izole etmedi |
| **GASP v8 (30K)** | **+ cross-scale reconstruction (lambda_recon=30, 2×2)** | **21.94** | **~10 tavanını kıran İLK yöntem** |

- v8, eff_rank'i v7'nin **~7 katına** çıkardı (3.09 → 21.94).
- **`L_iso ~0.25` sabit kaldı** (norm patlaması YOK) — yani reconstruction, collapse'ı
  norm-patlamasıyla takas etmedi; gerçek bir yapısal kazanım.

**Raporla çelişki.** Tablo 6 v8 satırı eff_rank için **"not measured"** der. Oysa ölçüldü
(21.94) ve bu, v8'in (ve genel olarak reconstruction-tabanlı çözümün) **asıl teşhis
başarısıydı.** Bu satır en azından doldurulmalı.

---

## 4. Bulgu B — Dev fine-tune karşılaştırması (raporun açık sorusunun ön-cevabı)

Rapor full fine-tune'u tekrar tekrar "future work" diye bırakır (§3.5, §6, §7):
*"whether scale-equivariance helps under full fine-tuning remains genuinely open."*
Ama 30K dev protokolünde **frozen vs finetune** karşılaştırması yapıldı:

| Protokol (30K dev) | COCO | GASP v8 | Δ |
|---|---|---|---|
| frozen probe | 0.345 | 0.282 | **−0.063** |
| fine-tune | 0.394 | 0.365 | **−0.029** |

(Karşılaştırma için: v7 frozen ≈ 0.269 → v8 frozen 0.282.)

**Okuma.** Fine-tune altında GASP'ın COCO'ya açığı **neredeyse yarıya iniyor** (−0.063 →
−0.029). Yani frozen probe, GASP'ı **olduğundan zayıf** gösteriyor; adaptasyon, v8'in
taşıdığı bilgiyi kısmen kullanışlı hale getiriyor. Bu, raporun "frozen feature olarak
zayıf" hükmünü çürütmez, ama onu önemli ölçüde **niteler** — ve tam da raporun "gelecek
work" dediği sorunun elimizdeki ön-cevabıdır.

**Senin kendi dürüst yorumun (oturumdan):** "tek seedde 0.029 fark, ±0.016 gürültü payını
aşıyor ama çok da değil — *küçük ama muhtemelen gerçek* bir fark. v8 finetune'da COCO'yu
yakalamadı, ama uçurum da değil (~%7 görece)."

> **Önemli comparability uyarısı:** Bu frozen/finetune sayıları (COCO 0.345/0.394) **dev-B
> protokolünden**; Tablo 6'nın direct-finetune COCO'su (0.2341) **dev-A protokolünden**;
> final matrisin v8'i (0.257) **10-kaynak LOSO**'dan. Üçü farklı protokol — absolute değerde
> birbirleriyle kıyaslanamaz. Yalnızca **aynı protokol içindeki Δ'lar** anlamlıdır.
> Bu yüzden bu bulgu rapora **headline değil, dipnot/ön-bulgu** olarak girmeli.

---

## 5. Bulgu C — Ölçekle collapse DERİNLEŞİYOR (10.48 → 7.97)

- GASP **v6r, 30K**'da eff_rank **10.48**.
- GASP **v6r, 181K**'da (6× veri, ~11 saat eğitim) eff_rank **7.97** — yani **DAHA KÖTÜ.**

**Yorum.** Daha çok veri + daha uzun eğitim collapse'ı **azaltmadı, derinleştirdi.** Bu,
collapse'ın bir **veri-ölçeği artefaktı değil, yöntemsel** bir patoloji olduğunu kanıtlayan
güçlü bir argümandır (single-axis objective, ölçek ne olursa olsun düşük-rank alt-uzaya
çöküyor). Bu kanıt, GASP'ın "neden reconstruction gerekliydi" gerekçesini sağlamlaştırır.

**Tablo 6 etiket sorunu (teyit edilmeli).** Raporun Tablo 6'sı v6r satırını **"(181K run) …
10.48"** diye yazıyor. Oysa oturum kaydı net: **10.48 = 30K**, **7.97 = 181K**. Yani Tablo
6'daki "181K = 10.48", görünüşe göre aslında 30K sayısı; gerçek 181K sayısı (7.97) ve
"ölçekle kötüleşme" yorumu eksik. **Repo çıktısıyla bire bir teyit edip düzeltmeni öneririm.**

---

## 6. Bulgu D — eff_rank tek başına yetersiz bir SSL metriği

v8'in eff_rank'i 7 katına çıktı (3→22) **ama** frozen mAP yalnızca 0.269 → 0.282 arttı ve
Δ hâlâ −0.063. Yani eff_rank sıçraması downstream'e **kısmen** yansıdı, tam değil.

**Teşhis (feature geometri ölçümü):** v8 backbone'u **bilgiyi enjekte etti** —
standardize edilince eff_rank **13.6 → 22** (gizli ölçek dengesizliği var) — **ama bu bilgi
detection-hazır biçimde değil**: **common-mode cosine ~1** (güçlü ortak mod). Yani
reconstruction "collapse'ı çöz" işini yaptı, ama "COCO-kalitesinde detection feature'ı üret"
işini tam yapmadı.

**Metodolojik çıkarım (yayın değeri yüksek):** **Effective rank, SSL temsil kalitesi için
tek başına yetersiz bir metriktir** — yüksek rank, detection-kullanışlılığını garanti etmez.
Bu, eff_rank'i tek gösterge olarak kullanan SSL diagnostic literatürüne dürüst bir uyarı.
Rapor bu çıkarımı yapmıyor.

> Not: Raporun §3.5'i variance-floor / covariance-rank-körlüğü / projection-head-başarısızlığı
> bulgularını **zaten içeriyor**. Eksik olan, bu listeye eklenmesi gereken **iki** kalem:
> (i) reconstruction'ın collapse'ı kırdığı pozitif sonuç (eff_rank 21.94), ve
> (ii) eff_rank'in *yine de* yetersiz kaldığı nüanslı teşhis (cos~1, detection-hazır değil).

---

## 7. Dürüst, çok-parçalı hikâye (oturumdaki 5-parça özet)

Bu beş parça birlikte, başlı başına yayınlanabilir bir **negatif-sonuç + metodoloji +
kısmi-çözüm** anlatısı oluşturuyor:

1. **Çekirdek fikir:** ölçek-eşdeğişir SSL (detection için ölçeği silmemek).
2. **Negatif sonuç + kök neden:** GASP tek-eksenliydi (içerik eksenini zorlamadı) → collapse.
   Dört müdahale — **VICReg, isotropy, projection head, +181K veri** — çözemedi. 181K,
   collapse'ın **yöntemsel** olduğunu kanıtladı (eff_rank yine 7.97, hatta daha kötü).
3. **Metodolojik bulgular:** `L_ctrl→0` collapse-proxy'si; covariance'ın rank'a körlüğü;
   variance'ın taban-değil-eşitleyici olması; **frozen-protokolün, finetune'un maskelediği
   farkı ortaya çıkarması.**
4. **Çözüm + kısmi başarı:** cross-scale reconstruction (içerik ekseni) collapse'ı kırdı
   (eff_rank 10→22, L_iso patlaması durdu, frozen Δ −0.076→−0.063, finetune −0.029).
5. **İnce teşhis:** v8 bilgiyi enjekte etti ama detection-hazır değil (cos~1 ortak mod,
   standardize 13.6→22 gizli ölçek dengesizliği) → eff_rank yetersiz metrik.

---

## 8. Ufak ek bulgular ve koşulmayanlar

- **v6iso (isotropy regularizer), 30K:** eff_rank **8.78** — başarısız (collapse sürdü).
  Tablo 6'da satırı yok; "dört müdahale" anlatısının bir parçası.
- **P5-BN denemesi:** başarısız (kayda geçmiş bir ek negatif; feature geometri düzeltme denemesi).
- **Koşulmayan teklifler (teklif aşamasında kaldı, hiç uygulanmadı):**
  `lambda_recon` 30 → **60** + darboğazı **daraltma** (örn. 2×2 → 1×1). v8'i daha ileri
  itmek için iki kaldıraçtı; garantisi olmadığı ve proje "konuyu kapat" yönünde ilerlediği
  için koşulmadı. (Gelecek-work adayı.)

---

## 9. Önem, öneri

- **v8 dev sonuçları, raporun GASP hükmünü adil biçimde nitelendirir:** "frozen feature
  olarak zayıf" doğru, ama (i) collapse kırıldı, (ii) fine-tune açığı yarıladı, (iii)
  zayıflığın *nedenini* biliyoruz (detection-hazır olmayan bilgi). Bu üçü, perspektifin
  "ölü" değil "kısmen doğrulanmış + nereye gideceği belli" olduğunu gösterir.
- **Önerilen rapor eklentisi:** Tablo 6'da v8 eff_rank (21.94) satırını doldur; §5.3'e bir
  dipnot/alt-paragraf — frozen vs finetune dev tablosu (§4) + eff_rank-yetersizliği teşhisi
  (§6). v6r 181K = 7.97 etiketini teyit edip düzelt. Hepsi **headline değiştirmeden**
  negatif sonucu daha dürüst ve daha sağlam kılar; aynı zamanda raporun kendi "future work"
  sorusuna ilk veri noktasını verir.

---

## 10. Caveat'ler (dürüstlük notları)

- **Tüm dev sayıları tek koşu / tek seed.** Frozen seed-std ≈ ±0.016. Δ −0.029 bu gürültüyü
  aşıyor ama dar bir marjla — "küçük ama muhtemelen gerçek."
- **Protokol comparability (kritik):** Dev-A (Tablo 6 direct-finetune, COCO 0.2341), dev-B
  (frozen/finetune, COCO 0.345/0.394) ve final (10-kaynak LOSO, v8 0.257) **üç ayrı protokol**.
  Yalnızca aynı protokol içindeki Δ'lar yorumlanmalı; absolute değerler kıyaslanamaz.
- **Dev fine-tune ≠ final fine-tune.** §4'teki finetune sonucu 30K/5-epoch/imgsz-320/tek-seed
  dev protokolünden; raporun "future work" dediği **final 10-kaynak LOSO full-fine-tune** hâlâ
  koşulmadı. Yani bu bir **ön-bulgu**, kesin cevap değil.
- Bu dosyadaki sayılar oturum kayıtları + çalışma planlarından derlendi; rapora aktarmadan
  önce repodaki güncel `loss_history` / feature-diag çıktılarıyla **bire bir teyit** önerilir
  (özellikle §5'teki 10.48 vs 7.97 etiketi).

---

*Önceki dosya: `DT-SAPS_eksik_bulgular.md`. Sonraki adaylar (istenirse, her biri ayrı):
standalone SAPS pilotu detayları; RAMS'in bağımsız bir katkı olarak ayrı dokümantasyonu;
veya raporda kısıtlı geçen diğer kalemler (ör. çoklu-mimari/DINOv2 planlananları "future work"
çerçevesi).*
