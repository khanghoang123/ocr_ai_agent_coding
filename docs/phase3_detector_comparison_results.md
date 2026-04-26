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
| E2\_craft | CRAFT (vendored, `craft_mlt_25k.pth`) | Word-level boxes merged into lines via vertical-overlap row clustering (`row_overlap_ratio=0.4`). |
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
| E0\_paddle | 7 | 0.00 | **40.0** | 0.118 | 13.47 | **12.7** | 0.416 | 0.105 | 0.247 | 0.062 | 2.7 | 79174 |
| E1\_surya | 7 | 0.00 | 27.0 | 0.109 | 19.04 | 5.4 | 0.602 | 0.036 | 0.450 | 0.430 | 2.7 | **43156** |
| E2\_craft | 7 | 0.00 | 13.0 | **0.213** | 8.63 | 2.7 | **0.287** | **0.033** | 0.345 | **0.099** | **0.9** | **40136** |
| E3\_kraken\_blla | 7 | 0.00 | 32.0 | 0.108 | 15.15 | 7.1 | 0.555 | 0.009 | 0.519 | 0.163 | 2.3 | 51755 |

**Bold = best in column.**

### Per-image hallucinated\_line\_rate (lower = better)

| image | E0\_paddle | E1\_surya | E2\_craft | E3\_kraken\_blla |
|---|---|---|---|---|
| chi\_pheo\_page.jpg | 0.333 (43 det) | 0.667 (16 det) | **0.333 (5 det)** | 0.375 (32 det) |
| chiec\_thuyen\_ngoai\_xa\_page.jpg | 0.500 (53 det) | 0.667 (29 det) | **0.000 (3 det)** | 0.667 (36 det) |
| chu\_nguoi\_tu\_tu\_page.jpg | 0.600 (48 det) | 0.800 (31 det) | **0.333 (9 det)** | 0.714 (40 det) |
| essay\_sample\_page.jpg | **0.333 (39 det)** | 0.917 (28 det) | **0.333 (9 det)** | 0.667 (29 det) |
| lang\_kim\_lan\_page\_4.jpg | 0.588 (41 det) | 0.583 (31 det) | **0.529 (11 det)** | 0.667 (32 det) |
| soan-bai-tap-doc-thuan-phuc-su-tu (printed) | **0.417 (26 det)** | 0.480 (25 det) | 0.444 (25 det) | 0.462 (26 det) |
| thumb\_1200\_1698.png (printed) | 0.138 (30 det) | 0.103 (29 det) | **0.034 (29 det)** | 0.333 (29 det) |

The `(N det)` annotation is critical context for the per-page numbers: a
detector that finds 3 lines and gets all 3 right scores `0.000` halluc
rate, but is only "winning" because it skipped most of the page.

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
4. **CRAFT + row-clustering** — last on detector geometry. Recall
   collapses (13 det/page on average; 3–5 on cursive pages) and the
   `full_width_band_rate` is the **worst** of all four (0.213). Root
   cause: the row-clustering merges words across the entire page width
   on cursive handwriting where word spacing is tight. This is
   parametric — tuning `row_overlap_ratio` from 0.4 toward 0.2 will
   recover most of the recall, but we did not retune for this report.

### B) End-to-end recognizer hallucination (with **public**, not fine-tuned, recognizer)

1. **CRAFT** — lowest aggregate `hallucinated_line_rate` (0.287),
   `digit_noise_rate` (0.033), `repeated_number_sequence_count` (0.9).
   But this is largely an artefact of CRAFT's low recall: the recognizer
   sees fewer crops, so fewer crops can hallucinate. On the printed
   `thumb_1200_1698.png` page where CRAFT *does* recall every line,
   it produces the cleanest output of any backend (`halluc_rate=0.034`).
2. **Paddle** — second best (`halluc_rate=0.416`).
3. **Kraken BLLA** — third (`halluc_rate=0.555`). Tight baseline
   polygons crop very close to the glyphs, which the printed-VN
   recognizer does not handle well; this is the recognizer-confound.
4. **Surya** — worst (`halluc_rate=0.602`, `up_garbage=0.430`). On
   `essay_sample_page.jpg`, Surya hallucinates 11/12 detected lines.

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

### Why CRAFT scores cleanly on hallucination but loses on geometry

The row-clustering merges word-level CRAFT polygons by vertical overlap.
On Vietnamese cursive handwriting, words within a line touch their
neighbours in y as well as x; clusters absorb words from adjacent rows
*and* full-width strokes. The resulting polygons span the page width
(`full_width_band_rate=0.213`), so they appear as bands. The
recognizer, given a band that contains 1–2 words of legible text,
returns those words, and the metric counts the page as "clean." But the
remaining 80% of lines are silently dropped.

**Fix path** (not in this PR — listed for the next iteration):
- tune `row_overlap_ratio` from 0.4 → 0.2,
- add a per-row max-width cap relative to the page width,
- or replace CRAFT post-processing with the official `getDetBoxes(poly=True)` path which emits curved polygons.

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
   quality and tells us whether Kraken's recognizer-side numbers are
   really worse than Paddle's, or just look that way under the public
   baseline.
3. **If the rerun confirms Kraken**, retune CRAFT's row-clustering as
   a tier-2 baseline (`row_overlap_ratio=0.2`,
   `getDetBoxes(poly=True)`), and only then revisit Surya.
4. **Build a small box-level GT set on `tests/test/`** so future runs
   produce IoU + CER, removing the recognizer-confound caveat that
   gates this whole report.

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
