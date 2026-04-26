# Vietnamese Handwritten OCR Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![VietOCR](https://img.shields.io/badge/VietOCR-0.3.6-green?style=flat-square)
![Gemini API](https://img.shields.io/badge/Gemini-Batch_API-orange?style=flat-square&logo=google)

**An end-to-end pipeline for Vietnamese handwritten text recognition, featuring automated pseudo-labeling via Google Gemini Batch API and cross-validation with VietOCR.**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset Statistics](#dataset-statistics)
- [Results](#results)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Project Structure](#project-structure)

---

## Overview

This project builds a production-quality OCR system for Vietnamese handwritten text by combining:
1. **Supervised training** on curated handwritten datasets (UIT-HWDB, VNOnDB, VN Handwritten Images).
2. **Self-supervised data augmentation** via an automated pseudo-labeling pipeline using **Google Gemini Batch API** and **VietOCR cross-validation**.
3. **Comparative experiments** to measure the impact of pseudo-labeled data on model accuracy.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                       │
│                                                             │
│  Dataset (16,363 images) ──► 01_eda ──► 02_data_prep       │
│                                                   │         │
│               ┌───────────────────────────────────┘         │
│               ▼                                             │
│        train: 13,090 │ val: 1,636 │ test: 1,637            │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                  BASELINE TRAINING                         │
│                                                             │
│   03_train_baseline (ResNet + Transformer / VietOCR)       │
│              ▼                                              │
│   04_evaluate_baseline ──► models/baseline/best_model.pth  │
└─────────────────────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────────────────────┐
│              PSEUDO-LABEL DATA PIPELINE                    │
│                                                             │
│  🌐 Crawl Images (internet)                                │
│         │ 9,187 images crawled                             │
│         ▼                                                   │
│  05_crawl_and_detect (PaddleOCR text detection)            │
│         │ 9,187 detected line crops                        │
│         ▼                                                   │
│  06_pseudo_label — Stage 1: Quality Filter                 │
│         │ 9,187 → 5,103 passed (55.5% kept)               │
│         ▼                                                   │
│  06_pseudo_label — Stage 2: Gemini Batch API               │
│         │ 5,103 → 4,040 labeled (Gemini Flash)             │
│         ▼                                                   │
│  06_pseudo_label — Stage 3: VietOCR Cross-Validation       │
│         │ 4,040 → 2,503 final clean labels (61.9% kept)    │
│         │ (similarity threshold: 0.70, mean sim: 0.73)     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              AUGMENTED TRAINING (Experiments)              │
│                                                             │
│  07_prepare_comparison_data                                │
│         │                                                   │
│  Experiment B: Ground Truth + Filtered Pseudo Labels       │
│    ├── Ground truth:    13,090 samples                     │
│    ├── Pseudo-labeled:   2,503 samples                     │
│    └── TOTAL:           15,593 samples                     │
│         ▼                                                   │
│  08_train_experiment_B (Kaggle/Colab — GPU)                │
│         ▼                                                   │
│  models/experiment_B/best_model.pth                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset Statistics

### Source Datasets (Baseline)

| Dataset | # Samples | Split |
|---|---|---|
| UIT-HWDB (line-level) | 7,229 | Train: 5,783 / Val: 723 / Test: 723 |
| VNOnDB (line-level) | 7,296 | Train: 5,837 / Val: 729 / Test: 730 |
| VN Handwritten Images | 1,838 | Train: 1,470 / Val: 184 / Test: 184 |
| **Total** | **16,363** | **Train: 13,090 / Val: 1,636 / Test: 1,637** |

### Pseudo-Labeling Pipeline

| Stage | Input | Output | Retention |
|---|---|---|---|
| Web Crawl → Text Detection | — | 9,187 line crops | — |
| Stage 1: Quality Filter | 9,187 | 5,103 | 55.5% |
| Stage 2: Gemini Flash Labeling | 5,103 | 4,040 labeled | 79.2% |
| Stage 3: VietOCR Cross-Validation (≥0.70) | 4,040 | 2,503 | 61.9% |

### Training Sets

| Experiment | Ground Truth | Pseudo Labels | Total |
|---|---|---|---|
| Baseline | 13,090 | 0 | 13,090 |
| Experiment B | 13,090 | 2,503 | **15,593** |

---

## Results

> See [models/comparison_results.md](models/comparison_results.md) for full evaluation details.

| Model | CER ↓ | WER ↓ | Full-match Acc ↑ |
|---|---|---|---|
| Pretrained VietOCR (zero-shot) | — | — | — |
| Baseline (13K samples) | 4.38% | 11.76% | 32.50% |
| Experiment B (+2,503 pseudo) | **2.98%** | **7.92%** | **46.55%** |

*Weights available on request (Google Drive / HuggingFace) — see Setup below.*

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/vietnamese-handwritten-ocr.git
cd vietnamese-handwritten-ocr
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your API keys and paths
```

### 5. Download datasets
- **UIT-HWDB**: [Request from UIT](https://uitnlp.github.io/)
- **VNOnDB**: [Available on request]
- **VN Handwritten Images**: Included in this repo (small subset) or download from Drive.

### 6. Download model weights *(optional)*
```bash
# Baseline model
# gdown <DRIVE_LINK> -O models/baseline/best_model.pth

# Experiment B model  
# gdown <DRIVE_LINK> -O models/experiment_B/best_model.pth
```

---

## Running the Pipeline

Run notebooks **in numbered order**:

| Order | Notebook | Environment | Purpose |
|---|---|---|---|
| 01 | `01_eda_dataset.ipynb` | Local | Explore dataset |
| 02 | `02_data_preparation.ipynb` | Local | Prepare train/val/test splits |
| 03 | `03_train_baseline.ipynb` | Colab/Kaggle (GPU) | Train baseline model |
| 04 | `04_evaluate_baseline.ipynb` | Local/Colab | Evaluate baseline |
| 05 | `05_crawl_and_detect_CLEANED.ipynb` | Local | Crawl & detect text lines |
| 06 | `06_pseudo_label.ipynb` | Colab (needs API key) | Pseudo-label pipeline |
| 07 | `07_prepare_comparison_data.ipynb` | Local | Merge data for experiments |
| 08 | `08_train_experiment_B.ipynb` | Kaggle (GPU) | Train Experiment B |
| 09 | `09_ocr_post_processing.ipynb` | Local/Colab | Evaluate Error Correction (SOTA Model & Dictionary) |

---

## AutoResearch

AutoResearch runs OCR hypotheses from YAML configs, records metrics, updates a leaderboard, and generates the next candidate config.

### Install
```bash
pip install -r requirements.txt
```

For CI/lightweight unit tests:
```bash
pip install -r requirements-ci.txt
```

### Run Baseline
```bash
python scripts/run_ocr_experiment.py \
  --config configs/baseline.yaml \
  --dataset data/processed/val.txt \
  --limit 10 \
  --output-dir experiments/runs/baseline_smoke
```

### Run AutoResearch
```bash
python scripts/auto_research.py \
  --dataset data/processed/val.txt \
  --limit 10 \
  --iterations 5
```

### View Results
- Metrics: `experiments/runs/<run>/metrics.json`
- Predictions: `experiments/runs/<run>/predictions.jsonl`
- Leaderboard: `experiments/leaderboard/leaderboard.csv` and `experiments/leaderboard/leaderboard.json`
- Report: `REPORT.md`

### View Debug Crops And Overlays
Each experiment run writes debug artifacts under:

```text
experiments/runs/<run>/debug/
```

Look for line crops in `debug/crops/`, overlay images ending in `_overlay.jpg`, and the line mapping in `debug/debug_mapping.json`.

### Use The Best Config
After AutoResearch finishes, pick the rank-1 config path from `experiments/leaderboard/leaderboard.csv`. Use that YAML with `scripts/run_ocr_experiment.py` for further validation. The API/frontend default response format is unchanged; these configs are intended for experiment runs unless explicitly wired into deployment.

---

## Project Structure

```
ocr_ai_agent_coding/
├── notebooks/
│   ├── 01_eda_dataset.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_train_baseline.ipynb
│   ├── 04_evaluate_baseline.ipynb
│   ├── 05_crawl_and_detect_CLEANED.ipynb
│   ├── 06_pseudo_label.ipynb
│   ├── 07_prepare_comparison_data.ipynb
│   ├── 08_train_experiment_B.ipynb
│   ├── 09_ocr_post_processing.ipynb
├── data/
│   ├── configs/                     # Model config files
│   ├── crawled/                     # Crawled data & pseudo-labels
│   │   ├── detected_lines/          # [LARGE - not in Git] 115MB crops
│   │   ├── pseudo_labels_final_clean.txt  # 2,503 clean pseudo labels
│   │   └── phase2_clean_labels.txt  # 4,040 Gemini-labeled samples
│   └── processed/                   # Annotation text files
│       ├── train_annotation.txt     # 13,090 training samples
│       ├── val_annotation.txt       # 1,636 validation samples
│       ├── test.txt                 # 1,637 test samples
│       ├── experiment_B_train.txt   # 15,593 samples (B)
│
├── Dataset/
│   ├── data/                        # [LARGE - not in Git] 1.3GB source images
│   └── labels.txt                   # 16,363 ground-truth labels
│
├── models/
│   ├── baseline/                    # Weights NOT in Git (too large)
│   │   ├── metrics.csv
│   │   ├── loss_curve.png
│   │   └── test_predictions.csv
│   ├── experiment_B/
│   │   ├── metrics.csv
│   │   └── loss_curve.png
│   ├── comparison_results.csv
│   └── comparison_results.md
│
├── docs/
│   └── VSCODE_COLAB_SETUP.md
│
├── .env.example                     # ← Copy to .env and fill secrets
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| OCR Model | [VietOCR](https://github.com/pbcquoc/vietocr) (ResNet + Transformer) |
| Text Detection | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| Pseudo-Labeling | Google Gemini Flash (Batch API) |
| Cross-Validation | VietOCR similarity scoring |
| Training Platform | Kaggle / Google Colab (GPU) |
| Framework | PyTorch |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
