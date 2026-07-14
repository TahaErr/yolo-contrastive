# B1 Design — Own From-Scratch Monocular Depth (the geometry half of "beat COCO without COCO")

The geometry half of the program. B1 trains our OWN monocular depth net FROM SCRATCH (no
ImageNet / DINOv2 / Depth-Anything) on REAL sensor GT (A2D2 LiDAR + Cityscapes stereo),
infers depth over the ~181K pool, and feeds geometry pothole mining (**M1**) and the
geometry∧appearance consensus (**M4** — the headline "beat" bet). See
[`METHOD_OPTIONS.md`](METHOD_OPTIONS.md) and the appearance half in `roadrecon/`.

---

## 1. GO / NO-GO — position taken

**Conditional GO, but write no B1 ingestion or net code until two cheap gates pass — and both
run on code that already exists, with zero GT ingestion and zero training.** Ingestion is
genuinely feasible (A2D2 `lidar/cam_front_center/*.npz` ships points **pre-projected** to
`row`/`col`; Cityscapes `disparity=(p-1)/256` is deterministic), so feasibility is *not* what
kills B1. The two things that kill B1 are (a) whether a plane-residual can localize few-cm
potholes **at all** from depth, and (b) whether depth-geometry **decorrelates** from
`roadrecon` — because M4 consensus is the entire reason B1 is worth its large net-new cost.
Both are measurable **today** with the committed `examples/09_terra_e1_fidelity.py` fed a
Depth-Anything-V2 cache as a **decision oracle** (never shipped; DA-V2 is an *optimistic upper
bound* on scratch-net quality — if it fails, scratch fails harder). Run Gate 0 and Gate 1
first. If **either** fails → **NO-GO**, redirect to hardening `roadrecon` + a cheap
DA-V2-free geometry proxy. If **both** pass → GO, build in order
Cityscapes-GT → depth-net → A2D2-GT → DepthCache seam → M1 → M4, with a mid-build recall
tripwire so a too-smooth scratch net dies in GPU-days, not weeks.

> **Purity note.** The gates use DA-V2 as a *throwaway decision oracle* only — it never enters
> the shipped method, which stays 100% from-scratch. If even that is unacceptable, the
> fully-pure alternative is the zero-GPU Cityscapes-GT smoke test → build Cityscapes ingestion
> → depth-net-v0 → the mid-build recall checkpoint as the gate (slower, but no external model
> even in the decision).

---

## 2. If GO — the concrete plan

### (a) GT-ingestion (net-new; a parallel GT branch, manifest schema untouched)

**Cityscapes disparity — FIRST (deterministic, quasi-dense, ~25K frames).**
- Add `data/ssl_pool/cityscapes_disparity.py`, mirroring `cityscapes.py`'s zip stream with
  `CANONICAL_PREFIX="disparity/"`, `_disparity.png`, + a sibling read of
  `camera/<split>/<city>/<id>_camera.json`.
- `cv2.IMREAD_UNCHANGED`, assert uint16; `disp=(p-1)/256` for `p>0` else invalid;
  `valid=(p>0)&(disp>0.1)`. Metric `Z=baseline*fx/disp` (fx≈2262, baseline≈0.209; fall back to
  those constants if a JSON is missing, log the count). The affine-invariant consumer needs no
  Z — cache `inv=disp` directly.
- Coverage ≈ **24,997** (trainvaltest 5,000 + trainextra 19,997, minus one corrupt frame).
- **Downsample trap:** pool RGB is long-side-640; downsample disparity with **masked pooling**
  (never bilinear across the `p==0` sentinel).

**A2D2 LiDAR — SECOND (sparse, metric near-field anchor).**
- **Modify** `data/ssl_pool/a2d2.py`: relax `_is_canonical_image` (a2d2.py:68) to also extract
  `lidar/cam_front_center/*.npz` + `label/cam_front_center/*.png` **in the same
  `tarfile.open(..., "r|")` pass** (a2d2.py:162) — do NOT re-scan the 164 GB tar.
- Add `data/ssl_pool/a2d2_lidar.py`: load `cams_lidars.json` → undistort the **original
  1928×1208** frame (not the resized pool JPEG), scatter `1/depth` at `(round(row),round(col))`
  into an HxW map + mask, z-buffer nearest, clip ~[1.5, 60] m, and **drop points on dynamic
  classes** via the A2D2 semantic mask so vehicles don't poison the road plane.
- **Verify one npz first** (`np.load`, 1 line) to confirm the range/depth key.

**GT store — NOT plain `DepthCache`** (it carries no per-pixel validity mask; sparse LiDAR /
holed disparity need one). Add `geoteach/depth_gt.py`: uint16 inverse-depth PNG + a bit-packed
mask, keyed by `image_id`, with audit counters (frames, dropped, median valid-fraction inside
`trapezoid_mask`, camera-json fallbacks).

### (b) From-scratch depth net + training

**Reuse the `roadrecon` scaffold** — `RoadReconNet` (`roadrecon/recon_net.py`) is already a
scratch YOLOv8n encoder + light upsampling decoder with tap/probe/`up_steps` done.
- `geoteach/depth_net.py` → `OwnDepthNet`: same encoder + a **1-channel** decoder head
  (`Conv2d(...,1,...)` + softplus → positive **relative inverse depth**; `plane_fit` is
  affine-invariant so no metric head is needed, and a scale-invariant loss lets Cityscapes and
  A2D2 co-train). **Bonus:** `net.encoder` is `save_backbone`-compatible → a *second* M2 init.
- `geoteach/depth_losses.py`: **masked SILog** (supervise only at `valid`) + **multi-scale
  gradient matching** (preserves road-crown relief — the term that stops the net smoothing away
  the very depression M1 needs).
- `geoteach/depth_trainer.py`: mirror `RoadReconstructor.train` (AdamW, warmup+cosine, AMP,
  best-epoch `save_backbone`). Mix Cityscapes:A2D2 ≈ 3:1; heavy aug (the lever against EU→US
  shift). ~50–60K frames ≈ a few GPU-days.

### (c) DepthCache seam (reuses `plane_fit`/`residual_labels`/`TerraChannel` UNCHANGED)

`run_depth_anything(images, cache, ..., pipe=None)` (`depth_cache.py:213`) calls
`pipe(pil_images, batch_size)` and reads `output["predicted_depth"]` — that `pipe=` is the seam.
- `geoteach/own_depth_pipe.py` → a callable wrapping `OwnDepthNet` returning
  `[{"predicted_depth": tensor}, ...]`; then
  `run_depth_anything(pool, DepthCache(root, tag="own_depth_b1"), pipe=OwnDepthPipe(net))`
  caches B1 depth in the **identical** format. `fit_road_plane`, `standardized_residual`,
  `residual_labels.*`, `TerraChannel` consume it **with no change**. Distinct `tag` so the DA-V2
  oracle cache and the B1 cache coexist.

### (d) M1 mining + M4 consensus

- **M1:** `labels_from_inverse_depth` over the `own_depth_b1` cache → the Stage-0 label dir
  (`images/`, `labels/*.png`, `boxes/*.txt`) that `TerraChannel` / M3 already read. **Zero new
  consumer code.**
- **M4 (net-new but tiny):** `geoteach/consensus.py` — per image, intersect geometry boxes
  (`residual_labels.mine_boxes`) with appearance boxes (`roadrecon.mining.mine_image_boxes`),
  both `MinedBox`, via the existing `roadrecon.mining.box_iou_xywh`; keep only where geometry
  AND appearance agree. Shadow → no depression → geometry vetoes; vehicle → not a recon-anomaly
  → appearance vetoes. Write survivors as the M3 set (reuse `write_yolo_txt` + `_write_data_yaml`).

### Files to add / modify

| Action | Path (`src/yolo_contrastive/` unless noted) | Purpose |
|---|---|---|
| modify | `data/ssl_pool/a2d2.py` | also extract `lidar/`+`label/` in the same tar pass |
| add | `data/ssl_pool/a2d2_lidar.py` | undistort + scatter + semantic-mask LiDAR → sparse GT |
| add | `data/ssl_pool/cityscapes_disparity.py` | disparity + camera-json → inverse-depth GT (masked) |
| add | `geoteach/depth_gt.py` | sparse GT store (inv-depth PNG16 + validity mask) |
| add | `geoteach/depth_net.py` | `OwnDepthNet` (scratch encoder + 1-ch depth decoder) |
| add | `geoteach/depth_losses.py` | masked SILog + multi-scale gradient matching |
| add | `geoteach/depth_trainer.py` | from-scratch trainer (mirrors `roadrecon/reconstructor.py`) |
| add | `geoteach/own_depth_pipe.py` | `pipe=`-compatible adapter → `run_depth_anything` |
| add | `geoteach/consensus.py` | M4: intersect geometry × appearance via `box_iou_xywh` |
| add | `examples/16_m4_consensus_killgate.py` | Gate 1 (mirrors `12b_roadrecon_killgate.py`) |
| reuse | `examples/09_terra_e1_fidelity.py` | Gate 0 + mid-build recall checkpoint (depth-agnostic) |
| untouched | `geoteach/plane_fit.py`, `residual_labels.py`, `channel.py`, `heads.py` | the reused consumer |

---

## 3. The kill-gates FIRST (cheapest falsification)

**Zero-GPU smoke (~minutes, DA-V2-free):** feed a few Cityscapes GT disparity maps into
`fit_road_plane` → `standardized_residual` on frames with a visible curb/ramp; confirm flat-road
`|z|≈0` and correct polarity (depression `z<0`, elevation `z>0`) at the defect. If perfect stereo
can't show correct polarity at a known step, the premise is dead.

**Gate 0 — geometry premise (GPU-hours, example 09 + DA-V2 oracle):**
```
python examples/09_terra_e1_fidelity.py --dataset <pothole_val> --compute \
    --model depth-anything/Depth-Anything-V2-Small-hf --class-map "0:depression" --out runs/b1_gate0
```
GO: **`recall_near ≥ 0.50` and `wrong_polarity < 0.10`.** Fail → NO-GO (scratch will be worse).

**Gate 1 — M4 decorrelation value (GPU-hours, `examples/16`):** on the same labeled val, mine
geometry boxes from the DA-V2 residual and appearance boxes from a trained `roadrecon`, intersect,
and compare **consensus** vs **roadrecon-alone** with `roadrecon.mining.mining_fidelity`. GO:
consensus must either (i) raise **precision at matched recall** by ≥ a pre-registered margin, or
(ii) recover a **decorrelated recall tail** (potholes roadrecon misses that geometry catches at
correct polarity). If consensus ≈ roadrecon → NO-GO, double down on roadrecon.

---

## 4. Sequencing (a dead B1 dies in GPU-days)

1. **Gate 0** (example 09 + DA-V2) — GPU-hours. ← run first.
2. **Gate 1** (consensus vs roadrecon-alone) — GPU-hours. *(needs a trained roadrecon.)*
3. **Cityscapes disparity GT** — ~a day, no GPU.
4. **Depth-net v0 on Cityscapes-only** — ~1–2 GPU-days.
5. **Mid-build recall tripwire:** cache v0 predictions (`own_depth_pipe` → `run_depth_anything`,
   tag `own_depth_b1`) and re-run example 09 on the pothole val with the net's cache; compare
   `recall_near` to the DA-V2 upper bound. Collapse → **kill B1 here**, before A2D2.
6. **A2D2 LiDAR GT** only if step 5 clears.
7. Retrain (Cityscapes+A2D2) → pool inference → M1 label factory → M3 → **M4 consensus** →
   downstream mAP vs the COCO baseline (the final arbiter).

---

## 5. Risk ledger — the 3 ways B1 ties, and the measurement that resolves each

1. **Scratch depth too smooth for shallow potholes** (a 5 cm dip at 10 m ≈ 0.25 px disparity —
   below SGM noise; VLP-16 ground hits are 0.5–2 m apart at ±3 cm). → **Measure:** step-5
   `recall_near` of the Cityscapes-only net vs DA-V2 upper bound. The gradient-matching loss is
   the lever; if it can't close the gap, kill.
2. **Domain shift EU→US/global** (train A2D2-DE + Cityscapes-EU; ~130K of the pool is BDD-US /
   Mapillary-global). → **Measure:** example 09 `recall_near` stratified by domain, or (no
   off-domain labels) the `fit.trusted` fraction + `sigma_mad` distribution per domain.
3. **Monocular depth is appearance-derived → NOT decorrelated from `roadrecon` (the real tie).**
   A scratch depth net infers "depression" from the same dark-textured-patch cue `roadrecon`
   flags, so consensus may agree only where they'd already agree. → **Measure:** Gate 1's
   precision-at-matched-recall gain **and** the decorrelated recall tail. This is the
   load-bearing number for the entire B1 bet.

**Cheapest alternative if NO-GO:** keep `roadrecon` and add a **frozen weak-geometry veto** — the
`trapezoid_mask` vertical-position prior (a road pixel's expected disparity is monotone in row),
used only to reject `roadrecon` boxes whose "depression" contradicts the row-order prior. Most of
the shadow-veto value at ~none of the net-new cost, purity-clean (no depth model at all).
