# VSCode Colab Extension Setup Guide

## 📦 Prerequisites

1. **VSCode Extension**: Google Colab
   ```bash
   # Install from VSCode marketplace
   # Extension ID: ms-toolsai.vscode-jupyter-cell-tags
   ```

2. **Python packages**:
   ```bash
   pip install jupyter notebook ipykernel
   ```

3. **VietOCR & dependencies**:
   ```bash
   pip install vietocr jiwer
   ```

## 🚀 How to Run

### 1. Open Notebook in VSCode

```bash
cd /home/khang/Projects/OCR_project/ocr_ai_agent_coding
code notebooks/07_train_baseline_base.ipynb
```

### 2. Select Kernel

- Click on kernel selector (top right)
- Choose: **Google Colab**
- Sign in with your Google account
- Select runtime: **GPU (T4 or better)**

### 3. Verify Paths

All paths are now **local** (no Google Drive mount needed):

```python
CONFIG = {
    "train_path": "/home/khang/Projects/OCR_project/ocr_ai_agent_coding/data/processed/train.txt",
    "image_root": "/home/khang/Projects/OCR_project/ocr_ai_agent_coding/Dataset/data",
    "checkpoint_dir": "/home/khang/Projects/OCR_project/ocr_ai_agent_coding/models/baseline_img_32/",
    ...
}
```

### 4. Run Cells

- Run all cells: `Ctrl+Alt+Enter` (or click "Run All")
- Training will use Colab GPU but save outputs locally

## 📂 Expected Directory Structure

```
ocr_ai_agent_coding/
├── Dataset/
│   └── data/           # Images here
├── data/
│   └── processed/      # train.txt, val.txt, test.txt
├── models/
│   └── baseline_img_32/  # Checkpoints saved here
└── notebooks/
    └── 07_train_baseline_base.ipynb
```

## ⚙️ Key Changes from Colab Web

| Aspect | Colab Web | VSCode Extension |
|--------|-----------|------------------|
| Drive mount | `drive.mount()` | ❌ Not needed |
| Paths | `/content/drive/MyDrive/...` | Local absolute paths |
| GPU | Colab cloud | Colab cloud (same) |
| Outputs | Saved to Drive | Saved locally |
| Code editing | Web UI | VSCode |

## 🔧 Troubleshooting

**GPU not available:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show Tesla T4 or similar
```

**Paths not found:**
- Verify Dataset is extracted: `ls Dataset/data/ | head`
- Check processed files exist: `ls data/processed/`

**VietOCR import error:**
```bash
pip install vietocr --upgrade
```

## 📊 Monitoring Training

Training logs saved to:
- `models/baseline_img_32/training_log`
- `models/baseline_img_32/metrics.csv`
- `models/baseline_img_32/loss_curve.png`

View in VSCode or terminal:
```bash
tail -f models/baseline_img_32/training_log
```

## 💡 Tips

1. **Auto-save enabled**: Checkpoints save every 200 iters
2. **Keyboard interrupt safe**: `Ctrl+C` will save last checkpoint
3. **Resume training**: Load from `last_checkpoint.pth`
4. **Multiple experiments**: Change `checkpoint_dir` for each run
