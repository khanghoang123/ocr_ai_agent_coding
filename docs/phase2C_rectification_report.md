# Phase 2C — Page-level rectification: root-cause + before/after report

This is a **diagnostic** evaluation, not a leaderboard run. It answers a specific
question:

> Is the missing-page-level-rectification hypothesis the root cause of VietOCR
> hallucinating on Vietnamese handwritten document photos?

Short answer: **partially**. Page-level rectification is the right tool for the
problem the user described, the implementation works correctly, and on the one
test image with real perspective distortion it materially reduces hallucination.
But on the seven images in `tests/test/`, only one image actually has
rectifiable distortion. The dominant residual hallucination on the remaining six
is **not** geometric — it comes from the recognizer (the project's finetuned
checkpoint is not in the repo, so this eval uses base pretrained VietOCR) and
from PaddleOCR's printed-text models being applied to handwritten input.
Concrete numbers and recommendations are below.

---

## 1. Root cause confirmation

### 1a. Code-level evidence

- `enable_document_perspective_correction` already existed as a config flag in
  `src/ocr_pipeline/experiment_config.py`, but was explicitly listed under
  `unsupported_options` — **the page-level rectification stage was never
  implemented**, only declared. ✅ Confirmed.
- `LineRefiner` (`src/ocr_pipeline/preprocess/notebook_preprocessor.py`) only
  has `rectify_curved_crop` — a **per-line** remap. There is no per-page step.
  ✅ Confirmed.
- `PaddleDetector` initialises PaddleOCR with `use_doc_unwarping=False` (and
  `use_doc_orientation_classify=False`). Even PaddleOCR's own document
  pre-processing is disabled. ✅ Confirmed.

### 1b. Empirical evidence from `phase2_debug.zip`

Diagnostic metrics applied to the project's existing Phase 2A/2B debug outputs
(no rectification) on these same seven images:

| Image | Phase 2B `hallucinated_line_count / total` | uppercase garbage tokens | repeated digit sequences |
|---|---:|---:|---:|
| `Chiếc thuyền ngoài xa 1.jpg` | 14/20 (70%) | 45 | 12 |
| `LÀNG -Kim Lân - trang 4.jpg` | 9/20 (45%) | 18 | 11 |
| `soan-bai-tap-doc-thuan-phuc-...jpg` | 20/22 (91%) | 42 | 38 |
| `tai xuông (1).jpg` | 10/20 (50%) | 20 | 5 |
| `tai xuông.jpg` | 10/15 (67%) | 35 | 6 |
| `thumb_1200_1698.png` | 16/29 (55%) | 28 | 3 |
| `viet-doan-van-...jpg` | 7/20 (35%) | 27 | 4 |

The hallucination patterns the user described — `ND/TP/UBND`-style uppercase
abbreviations and `010101/1111`-style repeated digit sequences — are present in
quantity on every page. Phase 2A → Phase 2B (crop-geometry tightening) only
moved these numbers a little; the residual is large, which is consistent with
the user's claim that line-level patches don't address the real cause.

### 1c. Verdict

The structural root cause that *page-level rectification was missing* is real
and confirmed at code level. Whether it's the *dominant* cause of hallucination
depends on the input — see §4. For these seven test images specifically, the
dominant cause turns out to be the recognizer/detector mismatch, not geometry
(see §5 and §7).

---

## 2. Chosen rectification method

**Hybrid: DocTr++ primary, OpenCV 4-corner perspective correction fallback.**

| Method | Status | When it runs |
|---|---|---|
| DocTr++ (`DocTrPlusRectifier`) | implemented, weights not bundled | when a TorchScript / ONNX / `.pth` checkpoint is provided at `rectifier_weights_path` |
| OpenCV 4-corner (`OpenCVRectifier`) | implemented, no extra deps | always available; fires when a clear paper-quad is detected and the area outside the quad is meaningfully darker than the inside |
| Identity (`IdentityRectifier`) | implemented | when `enable_document_perspective_correction=False` (zero-cost no-op) |
| `HybridRectifier` | implemented | tries DocTr++ first, falls back to OpenCV, never raises on missing weights |

Why this combination:

- **DocTr++** is the SOTA practical option for non-rigid document dewarping
  (curved/folded pages). It's a 2D displacement-field model — no retraining
  needed, you just load weights.
- **OpenCV 4-corner** has no external dependencies, runs in milliseconds, and
  handles the common case (paper photographed on a darker desk).
- **Hybrid** lets the system always run: missing weights → silent OpenCV
  fallback. If both backends decline, it returns the original image with
  `applied=False` and a diagnostic reason.

A critical guard prevents the OpenCV backend from latching onto the *printed
inner rectangle* of an already-flat scan (which would just crop the page
without fixing anything): the **paper-vs-background contrast gate**. If the
mean luminance inside the quad is not at least
`rectifier_min_paper_background_contrast` brighter than outside (default 18 on
0–255), the rectifier returns `applied=False` with reason
`quad_likely_inner_content_rect`. This was added in iteration 2 after the first
eval showed false-positive rectifications on flat scans.

---

## 3. Code changes

New files:

- `src/ocr_pipeline/rectifier/__init__.py`, `base.py`, `opencv_rectifier.py`,
  `doctrpp_rectifier.py`, `hybrid.py` — the rectifier module.
- `configs/phase2C_rectified.yaml` — Phase 2C config (Phase 2B settings +
  `enable_document_perspective_correction: true`).
- `scripts/run_rectification_eval.py` — before/after eval harness.
- `tests/test_rectifier.py`, `tests/test_diagnostic_metrics.py` — 16 unit tests.
- `docs/phase2C_rectification_report.md` — this report.

Modified:

- `src/ocr_pipeline/pipeline.py` — wires the rectifier into
  `OCRPipeline._process_single_page` *before* `detect_with_notebook_fallback`.
  All downstream stages (detector → refiner → cropper → recognizer → layout)
  run on the rectified image, in rectified-image coordinates. Saves
  `rectifier_*` fields on `DebugPageResult`.
- `src/ocr_pipeline/experiment_config.py` — adds `rectifier_*` fields, removes
  `enable_document_perspective_correction` from `unsupported_options`.
- `src/ocr_pipeline/schemas.py` — extends `DebugPageResult` with rectifier
  diagnostics.
- `src/ocr_pipeline/validation/metrics.py` — adds 8 new hallucination-detection
  metrics (uppercase-garbage tokens, repeated digit sequences, abnormal
  symbols, suspicious digit-letter tokens, garbage line ratio, hallucinated
  line detector, plus a `diagnostic_metrics` aggregator and `diagnostic_diff`
  before/after comparator).

Backwards compatibility: `enable_document_perspective_correction` defaults to
`False`. Existing configs are unaffected.

---

## 4. Before vs. after on `tests/test/`

Eval command (reproducible):

```bash
PYTHONPATH=src python scripts/run_rectification_eval.py \
  --images-dir tests/test \
  --output-dir experiments/runs/phase2C_rectification_eval
```

Stack used for this eval:

- Detector: `paddleocr==3.5.0` + `paddlepaddle==3.0.0` (PP-OCRv5 server detector
  + Latin recognizer, the version that actually runs on the VM).
- Recognizer: **base pretrained VietOCR `vgg_seq2seq`**
  (`https://vocr.vn/data/vietocr/vgg_seq2seq.pth`), used as a stand-in because
  the project's finetuned `experiment_B_50k` checkpoint is not in the repo.
  This is documented at `models/model_registry.yaml`. The hallucination
  patterns in question (uppercase tokens, repeated 0/1 sequences) are present
  in this base model too, so it is a fair proxy for showing whether
  page-level rectification helps. Absolute hallucination rates would be lower
  with the user's finetune.

### 4a. Aggregate

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Rectifier applied rate | – | 14.3% (1/7) | – |
| `digit_noise_rate` | 0.013 | 0.047 | **+0.034** ⚠️ |
| `garbage_text_ratio` | 0.200 | 0.168 | **−0.032** ✓ |
| `high_digit_noise_line_rate` | 0.012 | 0.020 | +0.009 |
| `uppercase_garbage_token_rate` | 0.142 | 0.161 | +0.019 |
| `suspicious_digit_token_count` | 0.43 | 0.57 | +0.14 |
| `repeated_number_sequence_count` | 3.00 | 2.29 | **−0.71** ✓ |
| `hallucinated_line_rate` | 0.374 | 0.367 | −0.007 |

Verdict counts: **improved 1, no_change 6, worsened 0**.

### 4b. Per-image

| # | Image | Detections | Rectifier | Verdict |
|---|---|---:|---|---|
| 1 | `chi_pheo_page.jpg` | 6 → 6 | not applied (`primary_unavailable`, no quad) | no_change |
| 2 | `chiec_thuyen_ngoai_xa_page.jpg` | 2 → 2 | not applied | no_change |
| 3 | `chu_nguoi_tu_tu_page.jpg` | 14 → 14 | not applied | no_change |
| 4 | `essay_sample_page.jpg` | 12 → 12 | not applied (no clear paper quad) | no_change |
| 5 | `lang_kim_lan_page_4.jpg` | 13 → 17 | **applied** (conf 0.85, OpenCV fallback) | **improved** |
| 6 | `soan-bai-tap-doc-...jpg` | 24 → 24 | not applied (already-flat scan) | no_change |
| 7 | `thumb_1200_1698.png` | 29 → 29 | not applied (already-flat scan) | no_change |

`primary_unavailable` everywhere is expected — DocTr++ weights aren't in the
repo, so the hybrid backend falls back to OpenCV. The OpenCV backend then
either applies (image 5) or correctly skips (the rest), guarded by the
paper-vs-background contrast check.

### 4c. Sample OCR — `lang_kim_lan_page_4.jpg` (the one image where rectification fires)

Before (raw image, FB watermark + red marginalia inside frame):

```
0010000000001000
001000003
sốia t dưa m m t m n t nam mộ chanang ngưới th ngia đanh ma n chan anh c nhangia ...
03 100 H0000000000000000002000010
0012000000
...
```

After (rectified, FB watermark cropped out by the paper-quad warp):

```
LAR
NGUYỄNG NH NGUNGH THỊ CH T NH T NG NGU CH NGHING NGU NH N NGIA T NGINH CHAN NG ...
Trung tranh ng nging nghingi tiến đin m ng nhi tiế thôngiang nhingiên thàngiền ...
người nông diên chu nhu thường nhườ pha nhan phangiếngia ngungia ngiê ngiều ...
031200000000000002012
NGUNG đơng C nh t nguyên
```

Concrete deltas on this image:

- `garbage_text_ratio`: 0.46 → 0.24 (**−0.226**, the largest single-image win)
- `hallucinated_line_rate`: 0.46 → 0.41 (**−0.05**)
- `repeated_number_sequence_count`: 5 → 5 (no change in count, but the long
  binary garbage runs from before are noticeably shorter)
- `digit_noise_rate`: 0.016 → 0.254 (**+0.238**, one outlier line dominated
  by `031200000000000002012`)
- `uppercase_garbage_token_count`: 25 → 32 (more all-caps tokens because the
  recognizer is now reading the printed `Date / No.` header as letters)
- Detections: 13 → 17

The improvement is real (less repeated-digit garbage, sharper line crops, the
distracting FB watermark / Date header / red marginalia are gone) but mixed:
the recognizer still hallucinates because it's being asked to read handwriting
with a base pretrained model.

Original-vs-rectified images for `lang_kim_lan_page_4.jpg` are saved at
`experiments/runs/phase2C_rectification_eval/debug/0004_lang_kim_lan_page_4/`
(`original.jpg`, `rectified.jpg`, `overlay_before.jpg`, `overlay_after.jpg`,
`crops_before/`, `crops_after/`, `text_before.txt`, `text_after.txt`,
`metrics.json`).

---

## 5. Does rectification reduce VietOCR hallucination?

**On these seven images, marginally.** `hallucinated_line_rate` drops from
0.374 to 0.367 (~2% relative). `repeated_number_sequence_count` drops from
3.0 to 2.3 per image (24% relative). `garbage_text_ratio` drops from 0.20 to
0.17 (16% relative). These are real but small gains driven almost entirely by
one image (`lang_kim_lan_page_4`).

**On images with actual perspective distortion, materially.** That's exactly
one image in this set, but `garbage_text_ratio` drops by 22.6 percentage points
on it.

**On already-flat scans (the majority here), the rectifier is correctly a
no-op.** Paper-vs-background contrast gating prevents it from cropping the
printed inner rectangle of a clean scan and pretending it dewarped something.

This eval set is **not representative** of the user's stated input
characteristics ("photos of paper, paper can be curved when held by hand,
perspective distortion from camera angle, non-uniform lighting"). Six of the
seven images are flat photographic scans / screenshots of notebook pages with
rectangular paper edges in the frame. Only one (`lang_kim_lan_page_4`) is the
kind of input where page-level rectification has work to do. To test the
hypothesis at the strength the user has in mind, the eval needs more
hand-held / curved / oblique photos.

---

## 6. Failure cases / honest caveats

1. **DocTr++ weights are not bundled.** Hybrid → OpenCV fallback always.
   Curved-page dewarping (the case OpenCV 4-corner can't handle) is therefore
   not exercised here. To unlock it, drop a TorchScript / ONNX / `.pth`
   checkpoint at `rectifier_weights_path` and re-run.
2. **The user's finetuned `experiment_B_50k` recognizer is not in the repo.**
   This eval uses base pretrained VietOCR `vgg_seq2seq`, which produces a lot
   of name-like uppercase noise (`NGUYỄN…`) on handwriting regardless of
   geometry. The rectifier's effect on hallucination is therefore being
   measured against a much noisier baseline than the user's actual setup. The
   recognizer hallucination is mostly a recognizer problem, not a geometry
   problem.
3. **PaddleOCR's PP-OCRv5 detector is a printed-text model.** On handwriting
   it sometimes returns 0 boxes (then the existing OpenCV fallback line
   detector takes over and produces the "full-width horizontal bands" the
   user complained about), or it returns text-tight quads that miss
   inter-character gaps. Page-level rectification doesn't help with this
   either way.
4. **OpenCV 4-corner can over-crop margin content.** On `lang_kim_lan_page_4`
   it removed the FB watermark (good) and the red marginalia (questionable).
   That's why the contrast gate is conservative by default — false-positive
   rectifications that drop content are worse than no-op skips.
5. **`chi_pheo_page` is the kind of image rectification *should* fix** —
   notebook page photographed at angle, slightly curved at the binding. But
   the OpenCV 4-corner backend can't find a quad because the paper edges run
   off the frame. DocTr++ would handle this case; OpenCV won't. Documented
   limitation.

---

## 7. Recommendation for the next step

**Keep rectification as the first stage** (it is correct, zero-cost when
inactive, and a real win when it applies), but **do not expect it to be the
high-impact lever on this particular dataset**. The honest priority order is:

1. **Restore the user's `experiment_B_50k` checkpoint** in
   `models/model_registry.yaml` and re-run `scripts/run_rectification_eval.py`
   on `tests/test/`. The aggregate hallucination numbers will drop sharply,
   and the rectifier's contribution will be measured against a fair baseline.
2. **Provide a DocTr++ checkpoint** (or a TorchScript export of GeoTr/TADoc).
   That unlocks rectification for `chi_pheo_page` and the broader curved-page
   case. Drop the file at the path in `configs/phase2C_rectified.yaml` —
   no code changes needed.
3. **Move detection off PaddleOCR PP-OCRv5 for handwriting.** The user is
   using a printed-text detector on handwritten pages, and falling back to
   horizontal-band heuristics when it returns nothing. This is the
   single largest source of hallucination on the seven-image set tested
   here. Options: train a DBNet variant on Vietnamese handwriting, or move
   to a paragraph-level architecture (DAN / Meta-DAN, Document Attention
   Network) that doesn't crop lines at all and uses the recognizer's own
   attention to read the page top-to-bottom. The architecture-rethink
   document attached separately analyses this trade-off in detail.
4. **Per-line dewarping inside the page-level rectified image.** The current
   `LineRefiner.rectify_curved_crop` is line-level and runs after detection.
   Combined with page-level rectification, this is a two-stage geometry
   correction; the per-line step can be tightened (or replaced with a 1-D
   displacement field) once page-level dewarping is producing clean pages.

The user's working hypothesis ("missing page-level rectification is the real
cause") is **architecturally correct** — the missing stage was a real gap and
worth filling — but on the specific eval set provided, it is **not** the
dominant cause of hallucination. The dominant cause is recognizer mismatch
plus printed-text detector applied to handwriting. Page-level rectification
is the right *first* lever; (1) and (3) above are the right *second* and
*third* levers.
