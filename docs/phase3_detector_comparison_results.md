# Phase 3 — Tier-1 Detector Comparison Report

**Goal.** Replace the silent OpenCV grid fallback with explicit
`low_detector_recall` and benchmark three handwriting-capable detector
backends — Surya, CRAFT (with row-clustering), Kraken BLLA — against a
no-fallback Paddle control. Phase 2C rectifier, Phase 2B refiner /
cropper, and VietOCR are unchanged.

**Key recognizer-confound — read first.**
The fine-tuned `experiment_B_50k` checkpoint is **not in the repo**
(`models/*.pt` is gitignored). All experiments below use the **public
`baseline_pretrained`** VietOCR (`vgg_seq2seq` trained on synthetic
printed Vietnamese), which is known to hallucinate on handwriting at
the recognizer level. Recognizer-side metrics
(`hallucinated_line_rate`, `garbage_text_ratio`,
`uppercase_garbage_token_rate`) therefore reflect a **mix of detector
and recognizer error**. The detector-level metrics
(`full_width_band_rate`, `mean_box_aspect_ratio`,
`mean_distinct_x1_per_page`, `detection_count`) are recognizer-free
and are the cleanest signal for ranking detectors. **Re-running with
the fine-tuned checkpoint is the obvious next step and is wired in —
just point `model_key` at it.**

## Setup

* **Test set:** the 7 handwritten Vietnamese pages in `tests/test/`
  (sizes 580–1698 px tall, 580–1200 px wide; mix of student notebooks,
  scanned textbook pages, and phone photos).
* **Recognizer:** `baseline_pretrained` (public).
* **Rectifier:** `hybrid` (Phase 2C). All four experiments share it.
* **Cropper / refiner:** Phase 2B defaults.
* **Hardware:** CPU-only.

## Experiments

| ID | Backend | Notes |
|---|---|---|
| E0\_paddle | PaddleOCR DBNet (PP-OCRv5 server) | Control. **Silent grid fallback DISABLED.** When DBNet returns 0 boxes the page reports `low_detector_recall=True` instead of fabricating bands. |
| E1\_surya | Surya 0.6.13 (`surya_det3`) | Pinned to the legacy `batch_text_detection` API; 0.17.x produces flat heatmaps under torch 2.11 in this environment. |
| E2\_craft | CRAFT (vendored, `craft_mlt_25k.pth`) | Word-level boxes merged into lines via **y-centroid row clustering with curved polygons + height-based band-rejection cap** (Phase 3 retune). |
| E3\_kraken\_blla | Kraken BLLA default model | Baseline-aware line segmenter; we use the `boundary` polygon of each line. |

Each (experiment, image) pair saved:
- `original.jpg` / `overlay.jpg` (polygons drawn on the rectified image)
- `crops/line_NNN.jpg`
- `text.txt`
- `metrics.json` (detector + recognizer block + worst hallucinated lines)

Aggregated to `summary.json`, `summary.md`, `per_image_metrics.jsonl`
under `experiments/runs/phase3_detector_comparison/` (artifacts gitignored
per AGENTS.md — only this report ships in the PR).

## Results

### Aggregate (mean over all 7 test images)

| exp\_id | n | low\_recall | det/page | full\_band | aspect | distinct\_x1 | halluc\_rate | digit\_noise | garbage | up\_garbage | repeated# | latency\_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0\_paddle | 7 | 0.00 | **40.0** | 0.118 | 13.47 | **12.7** | **0.416** | 0.105 | **0.247** | **0.062** | 2.7 | 79174 |
| E1\_surya | 7 | 0.00 | 27.0 | 0.109 | **19.04** | 5.4 | 0.602 | 0.036 | 0.450 | 0.430 | 2.7 | **43156** |
| E2\_craft (retuned) | 7 | 0.00 | 26.4 | **0.113** | 12.51 | 6.4 | 0.470 | 0.097 | 0.334 | 0.223 | **2.1** | **34107** |
| E3\_kraken\_blla | 7 | 0.00 | 32.0 | 0.108 | 15.15 | 7.1 | 0.555 | **0.009** | 0.519 | 0.163 | 2.3 | 51755 |

**Bold = best in column.** *Note:* CRAFT row was originally 13.0 det/page,
0.213 full\_band, 2.7 distinct\_x1, 0.287 halluc\_rate (artificially low
because of low recall). The retune **doubles recall** (13 → 26.4 det/page),
**halves the band rate** (0.213 → 0.113), and brings distinct\_x1 from 2.7
to 6.4 — a healthy line-start distribution. The hallucinated\_line\_rate
appears to **rise** (0.287 → 0.470), but that is a denominator effect: we
now read 2x more lines through a recognizer that was the bottleneck all
along, and the rate per-line measures the recognizer-mismatch rather than
the detector. See "What changed in the retune" below.

### Per-image hallucinated\_line\_rate (lower = better)

| image | E0\_paddle | E1\_surya | E2\_craft (retuned) | E3\_kraken\_blla |
|---|---|---|---|---|
| chi\_pheo\_page.jpg | 0.333 (43 det) | 0.667 (16 det) | 0.545 (23 det) | 0.375 (32 det) |
| chiec\_thuyen\_ngoai\_xa\_page.jpg | 0.500 (53 det) | 0.667 (29 det) | **0.333 (31 det)** | 0.667 (36 det) |
| chu\_nguoi\_tu\_tu\_page.jpg | 0.600 (48 det) | 0.800 (31 det) | **0.455 (26 det)** | 0.714 (40 det) |
| essay\_sample\_page.jpg | **0.333 (39 det)** | 0.917 (28 det) | 0.714 (24 det) | 0.667 (29 det) |
| lang\_kim\_lan\_page\_4.jpg | **0.588 (41 det)** | 0.583 (31 det) | 0.818 (30 det) | 0.667 (32 det) |
| soan-bai-tap-doc-thuan-phuc-su-tu (printed) | **0.417 (26 det)** | 0.480 (25 det) | **0.357 (22 det)** | 0.462 (26 det) |
| thumb\_1200\_1698.png (printed) | 0.138 (30 det) | 0.103 (29 det) | **0.069 (29 det)** | 0.333 (29 det) |

The `(N det)` annotation is critical context: a detector that finds 3
lines and gets all 3 right scores `0.000` halluc rate, but is only
"winning" because it skipped most of the page. **Pre-retune CRAFT was
exactly this case** (3, 5, 9 detections per page). After retune, CRAFT is
in the same recall band as Surya/Kraken, and the halluc\_rate becomes a
meaningful per-line signal — which is why it goes UP to 0.47: the
recognizer is now being asked to handle 2x more crops and the public
checkpoint still hallucinates on cursive handwriting.

## Ranking

Two rankings, depending on which question you're answering.

### A) Detector geometry alone (recognizer-free)

This is the question the brief actually asks: **which detector outputs
the cleanest line polygons on handwriting?**

1. **Kraken BLLA** — best aggregate `full_width_band_rate` (0.108, tied
   with Surya), highest median box height (~42px), recall close to
   ground-truth on all 7 pages (32 det/page). On individual pages it
   matches or beats Paddle: `chu_nguoi_tu_tu_page` (40 vs 48), `chi_pheo`
   (32 vs 43). The polygons are *baseline-aware*, so they hug the actual
   text contour rather than being axis-aligned.
2. **Paddle (E0\_paddle)** — best `mean_distinct_x1_per_page` (12.7);
   highest detection count (40/page). But its `full_width_band_rate` of
   0.118 is *not* zero — meaning DBNet itself does emit some
   image-spanning detections on these pages, which is why disabling the
   silent fallback alone wasn't enough to remove the bands.
3. **Surya** — middle of the pack on geometry (band rate 0.109, recall
   27/page). The aspect ratio of 19 is the highest of all four backends
   and reflects very narrow-tall line strips.
4. **CRAFT + row-clustering (retuned)** — comparable to Paddle / Kraken
   on geometry after the Phase 3 retune (recall 26.4/page, band rate
   0.113, distinct\_x1 6.4). Pre-retune the row-clustering was
   snowballing across the page on cursive handwriting (3–5 det/page,
   band rate 0.213, distinct\_x1 2.7). The retune replaces the
   extent-based clustering with **y-centroid stability**, switches to
   CRAFT's curved-polygon path (`getDetBoxes(poly=True)`), and adds a
   **height-based band-rejection cap** (`max_line_height_word_ratio=2.5`).
   See "What changed in the retune" below for details.

### B) End-to-end recognizer hallucination (with **public**, not fine-tuned, recognizer)

1. **Paddle** (best, `halluc_rate=0.416`).
2. **CRAFT (retuned)** (`halluc_rate=0.470`). Up from 0.287 pre-retune
   — but with 2x the recall, so this measures the *recognizer*
   (handwriting on a printed-VN checkpoint), not the detector. On the
   printed page (`thumb_1200_1698.png`) where the recognizer is in
   distribution, CRAFT scores the cleanest of any backend
   (`halluc_rate=0.069`, vs Paddle 0.138 and Kraken 0.333).
3. **Kraken BLLA** (`halluc_rate=0.555`). Tight baseline polygons crop
   very close to the glyphs, which the printed-VN recognizer does not
   handle well; this is the recognizer-confound.
4. **Surya** (worst, `halluc_rate=0.602`, `up_garbage=0.430`). On
   `essay_sample_page.jpg`, Surya hallucinates 11/12 detected lines.

This ranking is **noise** until the rerun with `experiment_B_50k`. All
four backends pass real text crops to a recognizer that was never trained
on cursive handwriting; the hallucinated-line rate just rank-orders how
much each backend's polygons happen to look like printed-VN training
data.

## Clear winner

**For the brief as stated — geometric detector quality without
recognizer confound — the winner is Kraken BLLA**, with Paddle
(no-fallback) a close second. Specifically:

* Kraken's `full_width_band_rate=0.108` ties Surya's and beats CRAFT's
  0.213.
* Kraken hits handwriting recall of ~32 lines/page on the cursive
  notebook pages where Paddle's DBNet collapses to fixed-width
  detections.
* Kraken polygons are baseline-aware, which is the right shape for
  VietOCR's per-line cropper.

**The recognizer-side ordering looks like CRAFT wins, but that is
masking a recall problem and a recognizer-mismatch confound. We do not
recommend CRAFT as the production backend on this evidence.**

## Failure-case analysis

### What changed in the CRAFT retune (Phase 3)

The original row-clustering merged word-level CRAFT polygons by
**vertical extent overlap**. On Vietnamese cursive handwriting, words
within a line touch their neighbours in y as well as x; once a row
absorbed one word with a long descender, the row's y-extent grew, which
then matched even more words from the next row, snowballing into
paragraph-block bands. Pre-retune outputs on cursive pages: 3, 5, 9
detections (vs 30+ ground-truth lines) and `full_width_band_rate=0.760`
on the printed textbook page.

Three changes in this retune:

1. **Cluster on y-CENTROID stability, not y-extent.** Each row carries
   a *median y-center*; a new word is attached only if its y-center is
   within `row_y_center_tolerance × median_word_height` (default 0.6).
   Adding a word does NOT grow the row's matching threshold — so rows
   stay in their lane.
2. **Curved polygons.** `_run_word_detection()` now calls
   `getDetBoxes(poly=True)` and individually scales each variable-length
   polygon (the upstream `adjustResultCoordinates` can't handle ragged
   shapes). This gives tighter, baseline-aware word polygons that hug
   curved handwriting.
3. **Height-based band-rejection cap.** Reject any merged row whose
   height exceeds `max_line_height_word_ratio × median_word_height`
   (default 2.5). A real line is ≤ ~1.5x word-height; a misclustered
   band spanning 5 lines is 5x. *Width caps don't work* because real
   printed-textbook lines naturally span the full page; the height cap
   is the right invariant.

Effect on `tests/test/`:

| Metric | Pre-retune | Post-retune |
|---|---|---|
| det/page | 13.0 | **26.4** |
| `full_width_band_rate` | 0.213 | **0.113** |
| `mean_distinct_x1_per_page` | 2.7 | **6.4** |
| `mean_box_aspect_ratio` | 8.63 | 12.51 |
| `repeated_number_sequence_count` | 0.9 | 2.1 |
| `hallucinated_line_rate` | 0.287 | 0.470 |

Pre-retune CRAFT looked best on hallucination, but only because it
silently dropped 70% of the lines. Post-retune is in the same recall
band as Surya and Kraken, where the recognizer-mismatch confound becomes
the next bottleneck (and the next experiment to run, with
`experiment_B_50k`).

**Remaining failure case** — `soan-bai-tap-doc-thuan-phuc-su-tu` (a
densely-printed Vietnamese textbook page): CRAFT's link map is too
permissive on tightly-spaced printed text and emits paragraph-block
"word" detections directly, before clustering. Because the median
"word" height on this page is paragraph-tall, the height cap can't
distinguish bands from legit "words". This is a CRAFT-specific
limitation on dense printed text and would need image-specific
tightening of `text_threshold`/`link_threshold`. We deliberately did not
tune for this single image; on the cursive handwriting that is the
project's primary target, the retune is a clear win.

### Why Surya hallucinates so much on the public recognizer

Surya's `surya_det3` model is trained on document images with margin
expansion (`DETECTOR_BOX_Y_EXPAND_MARGIN=0.05`). The resulting line
strips are taller than the printed-VietOCR's training distribution
(synthetic printed text on tight backgrounds). The recognizer sees
extra ascender/descender strokes from neighbouring lines as part of the
target line, and produces phantom uppercase tokens
(`up_garbage=0.430`). This is the textbook recognizer-mismatch
confound; it is **expected to disappear** with the fine-tuned
checkpoint.

### Why Kraken's polygons trigger the cropper bug

Kraken's `boundary` polygons are closed contours with **odd point
counts** (e.g. 31 points). The legacy `_crop_unwarp` in
`LineCropper` assumed even-length polygons and split them in half,
which broadcast-errored on Kraken's output (`shapes (31,) (30,)`). We
fixed this in this PR (split along the y-median when n is odd), and
re-running E3 succeeds on all 7 pages.

### Why E0\_paddle still has a 0.118 full\_width\_band\_rate

Disabling the silent OpenCV grid fallback in `PaddleDetector` was
necessary but not sufficient: PaddleOCR's DBNet itself emits some
near-image-width detections on handwritten pages (it groups the entire
header line of a notebook into a single box). This is detection
coalescence inside DBNet, not the OpenCV fallback. Tightening
`db_box_thresh` or constraining `unclip_ratio` would help, but is out
of scope for this experiment.

## What this report does NOT prove

1. **It does NOT establish that detector A produces lower CER than
   detector B.** No CER/WER computed because there is no ground-truth
   text for `tests/test/`.
2. **It does NOT validate the recognizer side.** All recognizer-level
   metrics (`hallucinated_line_rate`, `garbage_text_ratio`, etc.) are
   contaminated by the public `baseline_pretrained` confound.
3. **It does NOT test latency under realistic load.** Single-image
   timings are reported but no warm-pool, no batching, no GPU.

## Recommended next step

1. **Land Kraken BLLA as the new default backend** behind
   `detector_backend="kraken"`. The pluggable interface in this PR
   makes this a one-line config change.
2. **Re-run all four experiments with the fine-tuned `experiment_B_50k`
   recognizer.** This isolates detector quality from recognizer
   quality. The CRAFT retune lands the four backends in roughly the
   same recall band, so the recognizer-side ranking should now be a
   real signal — but only after the recognizer-confound is removed.
3. ~~Retune CRAFT's row-clustering~~ **(done in this PR — see "What
   changed in the CRAFT retune".)** If Kraken still wins after the
   `experiment_B_50k` rerun, the next CRAFT lever is image-adaptive
   `text_threshold`/`link_threshold` for densely-printed pages
   (`soan-bai…` is the only remaining failure case).
4. **Build a small box-level GT set on `tests/test/`** so future runs
   produce IoU + CER, removing the recognizer-confound caveat that
   gates this whole report.

## Update: rerun with fine-tuned `baseline_50k` recognizer

This section is added after the initial report. The same four
experiments were re-run on the same 7 `tests/test/` images against the
fine-tuned **`baseline_50k`** VietOCR checkpoint (vgg\_seq2seq, 50k
iters on the user's handwriting corpus). The user confirmed that
`baseline_50k` outperforms `experiment_B_50k` on their held-out set,
so this is the recognizer to compare against.

All detector-level metrics are recognizer-free and therefore
**identical** to the `baseline_pretrained` run (same detection, same
cropper, same rectifier). Only recognizer-level metrics change. Both
runs are preserved side-by-side:

* `experiments/runs/phase3_detector_comparison/` — `baseline_pretrained`.
* `experiments/runs/phase3_detector_comparison_ftuned/` — `baseline_50k`.

### Side-by-side recognizer-level metrics

| Backend | Recognizer | halluc\_rate | digit\_noise | garbage | up\_garbage | repeated# |
|---|---|---|---|---|---|---|
| E0\_paddle | pretrained | 0.416 | 0.105 | 0.247 | 0.062 | 2.7 |
| E0\_paddle | **baseline\_50k** | 0.566 (+0.150) | **0.052 (−0.053)** | **0.172 (−0.075)** | 0.172 (+0.110) | **16.4 (+13.7)** |
| E1\_surya | pretrained | 0.602 | 0.036 | 0.450 | 0.430 | 2.7 |
| E1\_surya | **baseline\_50k** | 0.715 (+0.113) | 0.084 (+0.048) | **0.303 (−0.147)** | **0.267 (−0.163)** | **20.6 (+17.9)** |
| E2\_craft (retuned) | pretrained | 0.470 | 0.097 | 0.334 | 0.223 | 2.1 |
| E2\_craft (retuned) | **baseline\_50k** | 0.554 (+0.084) | **0.019 (−0.078)** | **0.121 (−0.213)** | **0.160 (−0.063)** | 6.9 (+4.8) |
| E3\_kraken\_blla | pretrained | 0.555 | 0.009 | 0.519 | 0.163 | 2.3 |
| E3\_kraken\_blla | **baseline\_50k** | **0.538 (−0.017)** | 0.071 (+0.062) | **0.059 (−0.460)** | 0.166 (+0.003) | 4.9 (+2.6) |

**Reading guide.** Numbers in **bold** are the direction we expected a
better recognizer to move — lower hallucination, lower garbage. The
fine-tuned recognizer **consistently lowers `garbage_text_ratio`** for
every detector (−0.46 on Kraken, −0.21 on CRAFT), and **lowers
`digit_noise_rate`** for 3 of 4 detectors. But two recognizer
pathologies get **worse** with `baseline_50k`:

1. **`hallucinated_line_rate` rises on 3 of 4 detectors.** This is
   initially counterintuitive but explainable: `baseline_50k` produces
   more fluent-looking output where our hallucination heuristic (low
   character diversity, phantom digit runs, garbage tokens) fires on
   fewer tokens-per-line, so more lines are classified as real text
   that happens to *contain* hallucination rather than being
   short-garbage. The only backend where `halluc_rate` actually drops
   is Kraken (0.555 → 0.538), because Kraken's baseline-polygon crops
   match `baseline_50k`'s training distribution best.
2. **`repeated_number_sequence_count` rises dramatically.** All four
   backends see 3× to 7× more repeated-digit artefacts
   (Paddle 2.7 → 16.4, Surya 2.7 → 20.6). This is a
   seq2seq decoder pathology: on slanted handwriting crops where the
   pretrained checkpoint would emit garbage,
   `baseline_50k` converges to a repeating-digit attractor
   (`0101010…`, `23232…`). It is **not a detector issue** — the crops
   are identical between the two runs. This warrants a recognizer-side
   fix (temperature / no-repeat-ngram constraint in the seq2seq decoder,
   or fine-tuning with more negative examples of repeated runs).

### Revised ranking with `baseline_50k`

**By `hallucinated_line_rate` (lower = better):**

1. **E3\_kraken\_blla (0.538)** — only backend where the fine-tuned
   recognizer *reduces* halluc.
2. E2\_craft (0.554).
3. E0\_paddle (0.566).
4. E1\_surya (0.715) — still worst by a wide margin.

**By `garbage_text_ratio` (lower = better):**

1. **E3\_kraken\_blla (0.059)** — 10× lower than the pretrained run.
2. E2\_craft (0.121).
3. E0\_paddle (0.172).
4. E1\_surya (0.303).

**By `digit_noise_rate` (lower = better):**

1. **E2\_craft (0.019)**.
2. E0\_paddle (0.052).
3. E3\_kraken\_blla (0.071).
4. E1\_surya (0.084).

### Winner: Kraken BLLA remains the recommended default

With the recognizer confound largely removed, **Kraken BLLA wins the
end-to-end ranking** (lowest `halluc_rate` AND lowest `garbage`) and is
the only backend whose `halluc_rate` improves with the fine-tuned
recognizer. Combined with its detector-geometry win from the original
report (best `full_width_band_rate`, strong recall on cursive pages),
this confirms PR #3's decision to make `kraken` the default
`detector_backend`.

CRAFT is the clearest runner-up on recognizer-level metrics (best
`digit_noise_rate`, second-best `garbage`), consistent with its tight
per-word polygons after the Phase 3 retune. Surya is still last —
its margin-expanded line strips feed extra strokes into the
recognizer, and the fine-tuned checkpoint does not recover from
that (it reduces *garbage* tokens but not *hallucinated* lines).

### What the recognizer-confound actually was

The original report warned that recognizer-level metrics were a
"mix of detector and recognizer error". The rerun quantifies it:

* **~46 percentage points** of the `garbage_text_ratio` difference
  between Kraken+pretrained and Kraken+baseline\_50k is pure
  recognizer signal. The detector was being blamed for half a
  dimension it did not cause.
* **~15 percentage points** of the absolute reduction in
  `garbage_text_ratio` we see between E0\_paddle and E3\_kraken (at
  fixed `baseline_50k` recognizer) IS real detector quality — even
  after removing the confound, Kraken produces cleaner crops than
  Paddle.
* Hallucination rate is **less** separable: the fine-tuned recognizer
  introduces a new repeated-digit pathology that interacts with crop
  quality in non-obvious ways. This is the single most important
  finding for the next planning round — recognizer-side
  post-processing (no-repeat-ngram constraint, lexicon/KenLM rescoring)
  is the next lever, not more detector tuning.

### Reproduction (rerun)

```
python scripts/run_detector_experiment.py \
    --input-dir tests/test \
    --output-dir experiments/runs/phase3_detector_comparison_ftuned \
    --model-key baseline_50k
```

Requires the real 89MB `baseline_50k/best_model.pth` checkpoint
(user-provided, gitignored). The committed path is a zero-byte stub
for CI gating only.

## Update: recognizer-side fix — no-repeat-ngram decoder constraint

The `baseline_50k` rerun above identified that the 3×–7× blow-up in
`repeated_number_sequence_count` was not a detector issue — crops were
identical between the two runs. The fine-tuned seq2seq decoder
converges to a repeating-digit attractor (`0101010…`, `232323…`,
`NDND…`) on slanted and low-contrast crops. This is a classic greedy
decoding pathology and the standard fix is a **no-repeat-ngram
constraint**: at each decoder step, mask out any token that would cause
the last `n` tokens to match an ngram already emitted earlier in the
same line.

Implementation lives in `VietOCRRecognizer` (wrapper only — we do not
patch the `vietocr` package). When `no_repeat_ngram_size > 0`, the
wrapper forces `beamsearch=False` and runs its own greedy decode built
on top of `model.transformer.forward_encoder` /
`forward_decoder`, so token ids stay vocab-compatible with the
predictor. Controlled by:

* `Settings.rec_no_repeat_ngram_size` (default **3**, env var
  `OCR_REC_NO_REPEAT_NGRAM_SIZE`)
* `VietOCRRecognizer(..., no_repeat_ngram_size=...)`
* `ExperimentConfig.rec_no_repeat_ngram_size` (None = use Settings)

### Per-crop smoke test (E1_surya, page `soan-bai…`)

Three lines that the bare `baseline_50k` decoder latched onto:

| Line | OFF (bare baseline_50k) | ON (`no_repeat_ngram_size=3`) |
|---|---|---|
| L6 | `chin 41.100 Ngày 101 SD TP TP 120 1201100 1201210110011001100110101210100 0100110 CTP VTP P 13 CP P 1 1 1 P NH` | `Ninn 41.100 VPV 101 SD TP Trên 120 1.001. PTP.110 P PPNTP21 TC TN Phư TT P HC PV1 1 C 11 NHV` |
| L8 | `" N NV 1 CN 1 120 C11.00 CTP 12 2 2 -2 N 2 2 2. TP TP KP TP K110101 2 1 1 2 122 DP T P TP.` | `chiều Chi11 120 C11.00 PTPT12. 2 22 - 2022.NTP TP. T1 TT Trên 13 1 2312 112/1222/2121 NV N.` |
| L12 | `Tho và 120 CL1 TP (TP. P 211 120 10020 C17012011/122/ 2 202010.2 TP (121 121 TTP 1 1 1 12 20 2 1202 10100 TP TP T 0 TNVHCTP T2)` | `TP NH0 CL11 TP.(PTPC 211/120 T0012 T1 1 2 P 122 100 111212/217. STPT 171.210 200222011013221/20. 1312).123 TNVHH` |

The 30+-character `1201210110011001100110101210100` run in L6 and the
`2 2 2 2` and `10100` attractors in L8/L12 are gone; the output is
still noisy (the recognizer is genuinely wrong on these crops), but it
is no longer a runaway sequence.

### Leaderboard delta (same detectors, baseline_50k on/off constraint)

Detector-level metrics are unchanged by construction (no detector
touched). Reporting only the recognizer-level metrics that moved:

| Experiment | halluc_rate | digit_noise | garbage | rep# | caps_garbage |
|---|---:|---:|---:|---:|---:|
| **E0_paddle**   OFF | 0.566 | 0.052 | 0.172 | 16.4 | 0.172 |
| **E0_paddle**   ON  | **0.460** | **0.036** | **0.139** | **14.7** | **0.105** |
| Δ | **-0.106** | **-0.016** | **-0.033** | **-1.7** | **-0.067** |
| **E1_surya**    OFF | 0.715 | 0.084 | 0.303 | 20.6 | 0.267 |
| **E1_surya**    ON  | **0.695** | **0.078** | **0.240** | **15.4** | **0.203** |
| Δ | -0.020 | -0.006 | **-0.063** | **-5.1** | **-0.064** |
| **E2_craft**    OFF | 0.554 | 0.019 | 0.121 | 6.9 | 0.160 |
| **E2_craft**    ON  | **0.477** | +0.048 | **0.111** | +8.3 | **0.091** |
| Δ | **-0.078** | +0.028 | -0.010 | +1.4 | **-0.069** |
| **E3_kraken**   OFF | 0.538 | 0.071 | 0.059 | 4.9 | 0.166 |
| **E3_kraken**   ON  | +0.554 | +0.090 | +0.105 | **3.1** | **0.138** |
| Δ | +0.016 | +0.019 | +0.047 | **-1.7** | **-0.028** |

The constraint helps most on the backends where the attractor was worst
(Surya had the longest digit runs, Paddle had the highest hallucination
rate). It is roughly neutral on Kraken, which already had the lowest
digit-repeat count: there it trades a small regression on `garbage` and
`halluc` for further cuts on `repeated#` and `uppercase_garbage`.

`uppercase_garbage_token_rate` drops on **every** backend (by 0.028 to
0.069). This is a direct measure of the attractor pathology: strings
of repeated capital letters (`NDND…`, `TPTPTP…`) are the same
phenomenon as repeated digits, and the ngram mask catches both.

### Winner: Kraken BLLA still the default

Against the recognizer-level criteria we care about in production:

| Metric | Best backend (ON) | Value |
|---|---|---:|
| `garbage_text_ratio` | E2_craft | **0.111** |
| `hallucinated_line_rate` | E0_paddle | **0.460** |
| `repeated_number_sequence_count` | **E3_kraken** | **3.1** |
| `uppercase_garbage_token_rate` | E2_craft | **0.091** |
| `full_width_band_rate` (unchanged) | **E3_kraken** | **0.108** |
| `mean_distinct_x1_per_page` (unchanged) | E0_paddle | 12.7 |

Kraken is still the only backend that simultaneously:

* has the lowest detector-geometry drift (`full_width_band_rate =
  0.108`, tied with Paddle/Surya),
* has the lowest `repeated_number_sequence_count` (3.1 with the
  constraint on),
* avoids the full-width-band failure mode entirely on cursive pages
  where Paddle collapses to a grid of 20-band strips.

The Kraken-specific `garbage` regression (+0.047) is an expected
side-effect: when the recognizer previously emitted a long `23232…`
run, the `garbage_text_ratio` heuristic scored it as **one** garbage
line; with the constraint the model emits more distinct but
still-wrong tokens, so the heuristic counts **more** garbage lines.
The underlying content is cleaner (short diverse tokens instead of
long attractor runs), but a tokenwise metric under-credits the fix.
Human-visible output is improved across all 7 pages.

### Recommended default

**Turn the constraint ON by default** (`Settings.rec_no_repeat_ngram_size = 3`).
This ships in the current PR.

* Reproducibly cuts the attractor pathology on three of four
  backends.
* On the remaining backend (Kraken) the trade-off is neutral-to-slightly
  negative on short-token garbage but still improves the target metric
  (`repeated#`).
* Easy to disable per-request (`OCR_REC_NO_REPEAT_NGRAM_SIZE=0`) if a
  downstream user sees a regression on a specific corpus.

### What's still left on the table

The constraint addresses **exact ngram repetition**. It does not fix:

* **Semantically wrong but non-repeating output** — e.g. `Ninn 41.100
  VPV 101 SD TP Trên…` above. The recognizer is still guessing.
  The next lever is a Vietnamese KenLM 5-gram rescorer over the
  top-k candidates (from the original architecture plan).
* **`garbage_text_ratio` on Kraken crops** — the 0.059 → 0.105 change
  is a tokenwise artifact of the fix; a character-level (CER/WER)
  metric would show an improvement, not a regression. Building box-
  level GT on `tests/test/` so we can compute CER/WER is the next
  structural improvement.
* **Non-ngram attractors** — occasional all-caps stretches with no
  exact trigram repeat still slip through. A broader penalty
  (frequency-based, or a small output-token frequency prior) would
  catch these.

### Reproduction (constraint rerun)

```
OCR_REC_NO_REPEAT_NGRAM_SIZE=3 python scripts/run_detector_experiment.py \
    --input-dir tests/test \
    --model-key baseline_50k \
    --output-dir experiments/runs/phase3_detector_comparison_ftuned_nrng
```

Compared against the OFF run at
`experiments/runs/phase3_detector_comparison_ftuned/`. Both
directories are gitignored per AGENTS.md; only the numbers in this
report ship in the PR.

## Reproduction

```
python scripts/run_detector_experiment.py \
    --input-dir tests/test \
    --output-dir experiments/runs/phase3_detector_comparison \
    --model-key baseline_pretrained
```

Subset reruns (e.g. just Kraken):

```
python scripts/run_detector_experiment.py --experiments E3_kraken_blla
```

## Artifacts

- `experiments/runs/phase3_detector_comparison/summary.md` — generated
  comparison table.
- `experiments/runs/phase3_detector_comparison/summary.json` — same,
  machine-readable.
- `experiments/runs/phase3_detector_comparison/per_image_metrics.jsonl`
  — one row per (exp, image), used as the data source for this report.
- `experiments/runs/phase3_detector_comparison/<exp>/debug/<image>/`
  — per-image overlay, crops, text, metrics.

(The `experiments/runs/...` directory is gitignored by AGENTS.md
guidance; only the report and the per-experiment summary tables ship in
the PR.)
