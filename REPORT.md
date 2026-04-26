# AutoResearch OCR Report

## Goal

Build an AutoResearch loop for Vietnamese handwritten OCR and use it to compare conservative detection/crop hypotheses without changing datasets, ground truth, model weights, or FastAPI response format.

## Current Pipeline

Image/PDF input flows through PaddleOCR line detection, `LineRefiner`, `LineCropper`, VietOCR recognition, reading-order reconstruction, optional safe Vietnamese postprocess, and experiment-only debug artifact export.

## Runtime Debug Update

`configs/baseline.yaml`, `configs/crop_pad_008.yaml`, and `configs/refine_combined.yaml` use the same detector runtime settings:

- `detect_on_upscaled_image: true`
- `upscale_factor: 1.0`
- same Paddle detector thresholds from global settings

`refine_combined` only adds Phase 1 crop/refine options over `crop_pad_008`: `enable_noise_box_filter`, tiny-box thresholds, `enable_rotated_crop`, and `crop_padding_ratio: 0.08`. These options do not select a different PaddleOCR model/runtime.

In the current Python environment, PaddleOCR 3.x / Paddle static CPU inference fails before returning detections:

```text
ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]
```

The same failure reproduced with the current `crop_pad_008` smoke run, so the invalid `detection_count = 0` was caused by the Paddle runtime path, not by `refine_combined` geometry options.

## Fix Applied

The detector now supports both PaddleOCR APIs explicitly:

- Modern objects with `.predict()` use `.predict()`.
- Legacy objects without `.predict()` use `.ocr(...)`.
- A missing `.predict()` is not treated as detector runtime failure.
- The OpenCV fallback is only used for real inference/runtime exceptions such as `ConvertPirAttribute2RuntimeAttribute`.

When PaddleOCR works, the previous PaddleOCR path remains unchanged.

The experiment runner also handles a missing VietOCR dependency by running detection/crop/debug-only mode, marking `vietocr_recognizer_missing_torchvision` in `unsupported_options`, and setting `score: null` so a blank-recognition run does not produce a fake `0.0` score. In the current environment, `torchvision` was installed for the active Python and VietOCR now loads successfully.

## How To Run

Baseline:

```bash
python scripts/run_ocr_experiment.py \
  --config configs/baseline.yaml \
  --dataset tests/test \
  --limit 7 \
  --output-dir experiments/runs/baseline_test
```

Crop pad v4:

```bash
python scripts/run_ocr_experiment.py \
  --config configs/crop_pad_008.yaml \
  --dataset tests/test \
  --limit 7 \
  --output-dir experiments/runs/crop_pad_008_v4
```

Refine combined v4:

```bash
python scripts/run_ocr_experiment.py \
  --config configs/refine_combined.yaml \
  --dataset tests/test \
  --limit 7 \
  --output-dir experiments/runs/refine_combined_test_v4
```

AutoResearch:

```bash
python scripts/auto_research.py \
  --dataset tests/test \
  --limit 7 \
  --iterations 5
```

Debug analyzer:

```bash
python scripts/analyze_ocr_debug.py \
  --debug-dir experiments/runs/baseline_test/debug \
  --output experiments/runs/baseline_test/debug_analysis.json
```

## Current Comparison

| Run | Recognition available | Score | Digit noise rate | Detection count | Geometry metrics |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline_test` | yes in old artifact | 0.0866 | 0.0866 | 19.43 | not present in old metrics |
| `crop_pad_008` leaderboard best | yes in old artifact | 0.0262 | 0.0262 | 19.43 | not present in old metrics |
| `crop_pad_008_v4` | yes | 0.0792 | 0.0792 | 21.43 | present |
| `refine_combined_test_v4` | yes | 0.0792 | 0.0792 | 21.43 | present |
| `phase2A_test` | yes | 0.0528 | 0.0528 | 20.86 | present |
| `phase2B_crop_geometry_test` | yes | 0.0560 | 0.0560 | 20.86 | present |

Both v4 runs pass the detection and recognition gates: detection count is not zero, `recognition_available` is true, and score is computed from recognized output rather than an empty-detection placeholder.

Geometry metrics for both `crop_pad_008_v4` and `refine_combined_test_v4`:

- `tiny_box_rate`: 0.0
- `merged_box_rate`: 0.06
- `abnormal_crop_ratio`: 0.18
- `high_digit_noise_line_rate`: 0.08
- `average_crop_aspect_ratio`: 47.0382

## Phase 2A Result

Phase 2A targets downstream quality while keeping detector fallback output fixed. It adds conservative line regrouping/splitting, ink-aware horizontal crop trimming, dynamic vertical padding, crop flags, and before/after crop debug artifacts. It does not add rotated crop, deskew, postprocess changes, AutoResearch strategy changes, or projection-split strategy work.

Before/after comparison:

| Metric | `refine_combined_test_v4` | `phase2A_test` | Change |
| --- | ---: | ---: | ---: |
| `detection_count` | 21.43 | 20.86 | -2.7% |
| `score` | 0.0792 | 0.0528 | improved |
| `digit_noise_rate` | 0.0792 | 0.0528 | improved |
| `average_crop_aspect_ratio` | 47.04 | 39.36 | improved |
| `abnormal_crop_ratio` | 0.1800 | 0.1712 | improved |
| `merged_box_rate` | 0.0600 | 0.0137 | improved |
| `high_digit_noise_line_rate` | 0.0800 | 0.0753 | slightly improved |
| `valid_line_ratio` | unavailable | 0.2397 | added |
| `garbage_text_ratio` | unavailable | 0.0890 | added |

Debug artifacts:

- Before crops: `experiments/runs/phase2A_test/debug/before_crops/`
- After crops: `experiments/runs/phase2A_test/debug/crops/`
- Overlay files: `experiments/runs/phase2A_test/debug/*_overlay.jpg`
- Mapping with crop flags: `experiments/runs/phase2A_test/debug/debug_mapping.json`

Visual crop improvement is visible in before/after dimensions. Example from the first page:

- line 0: before `720x53`, after `728x73`
- line 2: before `720x13`, after `705x17`, flags `ink_horizontal_trimmed`, `too_thin_crop`
- line 3: before `720x12`, after `707x16`, flags `ink_horizontal_trimmed`, `too_thin_crop`

The detection count stayed stable enough for this phase. The count changed from 150 total lines in v4 to 146 total lines in Phase 2A because overlapping/near-duplicate fallback bands were merged conservatively.

## Phase 2B Crop Geometry Result

Phase 2B adds validated crop geometry behind config flags:

- `crop_strategy: validated`
- `enable_line_deskew: true`
- `min_crop_height_ratio: 0.60`
- `vertical_padding_ratio: 0.18`
- `horizontal_padding_ratio: 0.25`
- `max_deskew_angle: 8.0`

It estimates foreground components, validates vertical boundaries, enforces a practical minimum crop height, preserves margins, and writes before/after crop debug artifacts. A first aggressive run over-expanded vertical crops and worsened digit noise; the final config is intentionally more conservative.

Phase 2A vs Phase 2B:

| Metric | `phase2A_test` | `phase2B_crop_geometry_test` | Change |
| --- | ---: | ---: | ---: |
| `detection_count` | 20.86 | 20.86 | stable |
| `score` | 0.0528 | 0.0560 | slightly worse |
| `digit_noise_rate` | 0.0528 | 0.0560 | slightly worse |
| `average_crop_aspect_ratio` | 39.36 | 35.93 | improved |
| `abnormal_crop_ratio` | 0.1712 | 0.0000 | improved |
| `merged_box_rate` | 0.0137 | 0.0137 | stable |
| `high_digit_noise_line_rate` | 0.0753 | 0.0616 | improved |
| `valid_line_ratio` | 0.2397 | 0.8425 | improved |
| `garbage_text_ratio` | 0.0890 | 0.1096 | worse |
| `too_thin_crop_rate` | unavailable | 0.0000 | added |
| `deskew_applied_rate` | unavailable | 0.0000 | added |

Debug artifacts:

- Before crops: `experiments/runs/phase2B_crop_geometry_test/debug/before_crops/`
- After crops: `experiments/runs/phase2B_crop_geometry_test/debug/crops/`
- Debug analysis: `experiments/runs/phase2B_crop_geometry_test/debug_analysis.json`

Visual crop geometry improved: no crops are flagged as too thin after validation, abnormal aspect ratio is zero, and the average aspect ratio drops from `39.36` to `35.93`. Deskew did not activate on this run because estimated angles stayed below the conservative threshold or were not reliable enough after masking.

## Remaining Issues

- PaddleOCR `.predict()` still fails in this environment with `ConvertPirAttribute2RuntimeAttribute`; v4 uses the runtime-specific OpenCV fallback after the real inference exception.
- `crop_pad_008_v4` and `refine_combined_test_v4` tie because the detector fallback produces the same line boxes for both configs.
- Phase 2A improves crop geometry and digit noise, but many crops are still too thin; flags are currently diagnostic only and do not drop crops.
- Phase 2B fixes thin/abnormal crop geometry but is not a clear OCR-quality win over Phase 2A yet; digit noise is still slightly worse than Phase 2A and first-page handwriting remains difficult.
- CER/WER are still unavailable for `tests/test` because no ground truth text is attached.
- Projection split, curved unwarp, and VietOCR training/fine-tuning remain deferred Phase 2 work.

## Validation

Completed:

```bash
pytest tests/test_debug_analyzer.py tests/test_strategy_config.py tests/test_autoresearch_runner.py tests/test_pipeline.py::TestEdgeCases tests/test_line_refiner.py
```

Result: 18 passed.

Latest targeted validation:

```bash
pytest tests/test_pipeline.py::TestEdgeCases tests/test_line_refiner.py tests/test_autoresearch_runner.py
```

Result: 13 passed.

Phase 2A validation:

```bash
pytest tests/test_pipeline.py::TestEdgeCases tests/test_line_refiner.py tests/test_autoresearch_runner.py tests/test_debug_analyzer.py
```

Result: 15 passed.

```bash
python scripts/run_ocr_experiment.py \
  --config configs/refine_combined.yaml \
  --dataset tests/test \
  --limit 7 \
  --output-dir experiments/runs/phase2A_test
```

Result: `processed_count=7`, `error_count=0`, `recognition_available=true`.

Phase 2B validation:

```bash
pytest tests/test_crop_validation.py tests/test_strategy_config.py tests/test_line_refiner.py tests/test_debug_analyzer.py
```

Result: 13 passed.

```bash
python scripts/run_ocr_experiment.py \
  --config configs/phase2B_crop_geometry.yaml \
  --dataset tests/test \
  --limit 7 \
  --output-dir experiments/runs/phase2B_crop_geometry_test
```

Result: `processed_count=7`, `error_count=0`, `recognition_available=true`.

```bash
python scripts/analyze_ocr_debug.py \
  --debug-dir experiments/runs/phase2B_crop_geometry_test/debug \
  --output experiments/runs/phase2B_crop_geometry_test/debug_analysis.json
```

Aggregate: `valid_line_ratio=0.8425`, `abnormal_crop_ratio=0.0`, `too_thin_crop_rate=0.0`.
