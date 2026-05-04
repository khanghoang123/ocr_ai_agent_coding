# Box-level / line-level ground truth for `tests/test/`

This directory contains transcription ground truth (GT) for the seven fixture
images under `tests/test/`. The GT is consumed by
`scripts/run_eval_cer_wer.py` to compute character / word error rates on the
detector-leaderboard runs in `experiments/runs/phase3_*/`.

## File layout

One JSON file per image, named `<image_stem>.json`. Schema:

```json
{
  "image": "thumb_1200_1698.png",
  "ground_truth_quality": "verified",
  "notes": "Clean printed Vietnamese page, transcribed character-for-character.",
  "lines": [
    {"text": "A. Môi trường vĩ mô"},
    {"text": "1. Nhân khẩu học:"}
  ]
}
```

Fields:

- `image` — file name in `tests/test/` (must match exactly).
- `ground_truth_quality` — one of `verified` (machine-readable / printed page
  transcribed character-for-character), `best_effort` (handwritten page,
  human-transcribed but minor errors possible), `partial` (only sentinel-prefix
  lines populated; full-page CER/WER will be unreliable).
- `notes` — free-form human-readable note about the GT quality and any caveats.
- `lines` — list of per-line objects in **reading order**. Each object must
  have a `text` field; an optional `polygon` field
  (`[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`) can be added later for IoU evaluation.

## Evaluation

```sh
python scripts/run_eval_cer_wer.py \
    --run-dir experiments/runs/phase3_detector_comparison_ftuned \
    --gt-dir  tests/test_gt \
    --out     experiments/runs/phase3_detector_comparison_ftuned/cer_wer.json
```

This walks `<run-dir>/<experiment>/per_image_outputs.jsonl` (or the equivalent
`debug/<image>/<image>.txt` line dumps), joins to GT by image name, and reports
two metrics per image and per experiment:

- **Page-level CER / WER** — concatenate all detected lines (in reading order)
  and the GT lines into a single page string, then compute CER / WER. This is
  detection-aware: a missed line raises both numbers.
- **Matched line-level CER / WER** — Hungarian matching by edit distance
  between detected lines and GT lines, then per-pair CER / WER. This isolates
  recognition quality from detection coverage.

Both metrics use NFC normalisation + collapsed whitespace, the same convention
already used in `src/ocr_pipeline/validation/metrics.py`.

## Adding GT for new fixtures

1. Drop a new image in `tests/test/`.
2. Add a corresponding `<stem>.json` here.
3. Set `ground_truth_quality` honestly. `partial` is fine — the eval CLI will
   skip page-level metrics for partial GTs and only report line-level CER/WER
   on the populated lines.
