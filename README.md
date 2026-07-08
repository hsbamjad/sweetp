# Multispectral Sweet Potato Grading System

Automated sweet potato quality grading using multispectral vision (RGB + NIR1 + NIR2) and deep learning instance segmentation.

## Project Overview

- **Objective**: ≥90% grading accuracy at instance and sample level
- **Classes (3)**:
  - `0` = **Normal** (no visible defect)
  - `1` = **Moderate defect** (minor surface defects)
  - `2` = **Severe defect** (major rot, cracks, large blemishes)
- **Bands available**: RGB (Source0) + NIR1 (Source1) + NIR2 (Source2)
- **Task**: Instance segmentation → Tracking → Multi-view grading

## Data Overview

### 2024 Data (`2024/data/`)
- `ExtractedFrames/ConvertSet/` — Source0 (RGB) JPG + YOLO polygon labels (.txt)
- `ExtractedFrames1/` — Source1 (NIR1) JPG frames
- `ExtractedFrames2/` — Source2 (NIR2) JPG frames
- **5 unique potato sessions**, ~257 frames each
- **1124 annotated instances**: Normal=516, Moderate=488, Severe=120

### 2024 Models (`2024/DetectModels/`)
| Folder   | Models available |
|----------|-----------------|
| RGB      | bestLRGB_grade.pt, bestSRGB_grade.pt (+ track variants) |
| RGNIR1   | bestLNIR1_grade.pt, bestSNIR1_grade.pt |
| RGNIR2   | bestSNIR2_grade.pt |
| RNIR12   | bestLNIR12_grade.pt, bestMNIR12_grade.pt, bestSNIR12_grade.pt |

### 2026 Data (`2026/`) — To be provided

## Directory Structure

```
sweetpotatoes/
├── 2024/
│   ├── Code/           # Original 2024 grading scripts (reference)
│   ├── data/
│   │   ├── ExtractedFrames/ConvertSet/  # RGB images + YOLO labels
│   │   ├── ExtractedFrames1/            # NIR1 images
│   │   └── ExtractedFrames2/            # NIR2 images
│   └── DetectModels/   # Trained 2024 model checkpoints
├── scripts/            # Data preparation and audit
├── models/             # Training and evaluation
├── configs/            # Experiment YAML configs
├── grading/            # Sample-level grading pipeline
├── processed_data/     # Prepared multispectral datasets (generated)
├── runs/               # Training outputs (generated)
└── results/            # Evaluation outputs (generated)
```

## Classes

- **0**: Normal — good quality sweet potato
- **1**: Moderate defect — minor surface defects, acceptable
- **2**: Severe defect — major rot, cracks, or large blemishes

## Setup

```bash
conda create -n sweetpotato python=3.10 -y
conda activate sweetpotato
pip install -r requirements.txt
```

## Quick Start

### Step 1 — Audit the Data
```bash
python scripts/audit_dataset.py
```

### Step 2 — Prepare Multispectral Dataset
```bash
python scripts/prepare_dataset.py
```

### Step 3 — Train Models (run one at a time)
```bash
python models/train.py --config configs/model1/rgb_baseline.yaml --name model1_rgb
python models/train.py --config configs/model2/nir_baseline.yaml --name model2_nir
python models/train.py --config configs/model3/nir_diff_fusion.yaml --name model3_nirfusion
python models/train.py --config configs/model4/spd_only.yaml --name model4_spd
python models/train.py --config configs/model5/sweetpotato_yolo.yaml --name model5_full
```

### Step 4 — Compare Models
```bash
python models/evaluate.py \
  --checkpoints runs/model1_rgb/weights/best.pt runs/model2_nir/weights/best.pt \
  --data processed_data/RGB/data.yaml \
  --output results/comparison
```

### Step 5 — Sample-Level Grading Evaluation
```bash
python grading/evaluate_grading.py --csv results/grading_predictions.csv --output results/grading_eval
```

## Ablation Study Design

| Model | Band Input | Architecture | Purpose |
|-------|-----------|--------------|---------|
| Model 1 | RGB | yolo26m-seg baseline | Visible-light baseline |
| Model 2 | R/NIR1/NIR2 | yolo26m-seg standard | NIR domain benefit |
| Model 3 | R/NIR1/NIR2/Diff | + NIRDiffFusion | Spectral diff benefit |
| Model 4 | R/NIR1/NIR2 | + SPD-Conv stem | Spatial preservation |
| **Model 5** | **R/NIR1/NIR2/Diff** | **Full SweetPotatoYOLO** | **Proposed system** |

All trained **from scratch** on `yolo26m-seg.pt` (ImageNet weights only).
