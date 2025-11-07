# ViT Deepfake Detection

This project trains a Vision Transformer (ViT) binary classifier to distinguish real vs fake (deepfake/AI-generated) images. It includes scripts to prepare dataset manifests, train the model, evaluate on test sets, visualize training metrics, and run single-image inference.

## Installation

### Clone the Repository

```bash
git clone https://github.com/phantom-712/Deepfake_Detector.git
cd Deepfake_Detector
```

## Directory Structure

```
ViT_Deepfake/
├── data/
│   ├── real_cifake_images/          # Training images (real)
│   ├── fake_cifake_images/           # Training images (fake)
│   ├── test/                         # Optional test images (unlabeled)
│   ├── train_manifest.csv            # Created by prepare_data.py
│   ├── test_manifest.csv             # Created by prepare_data.py
│   └── dataset_index.csv             # Created by prepare_data.py
├── models/
│   ├── best_vit_base_patch16_224.pth # Best model (tracked in git)
│   └── epoch_*.pth                    # Per-epoch checkpoints (ignored in git)
├── results/
│   └── training_log_vit_base_patch16_224.csv  # Training metrics
├── plots/
│   └── training_metrics_vit_base_patch16_224.png  # Training visualization
├── json/
│   └── preds_vit_base_patch16_224.json  # Test predictions
├── scripts/
│   ├── prepare_data.py               # Dataset preparation
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Test evaluation
│   ├── visualize_training.py         # Metrics visualization
│   ├── inference.py                  # Single image inference
│   └── model.py                      # ViT model definition
├── Dockerfile                         # Docker container definition
├── docker-compose.yml                 # Docker Compose configuration
├── docker-run.sh                      # Docker helper script (Linux/Mac)
├── docker-run.ps1                     # Docker helper script (Windows)
├── .dockerignore                      # Docker ignore rules
├── .gitignore                         # Git ignore rules
├── README.md                           # This file
└── requirements.txt                   # Python dependencies
```

## Quick Start

### Option 1: Local Installation

#### Step 1: Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:** torch, torchvision, timm, numpy, pandas, scikit-learn, pillow, tqdm, imbalanced-learn, matplotlib

#### Step 3: Prepare Data

```bash
python scripts/prepare_data.py
```

Or with custom train/validation ratio:
```bash
python scripts/prepare_data.py --train_ratio 0.9
```

This creates train/validation splits and generates manifest CSV files. Default train_ratio is 0.9 (90% train, 10% validation).

#### Step 4: Train Model

```bash
python scripts/train.py
```

Or with custom parameters:
```bash
python scripts/train.py --pretrained --epochs 15 --batch_size 16 --lr 3e-5
```

**All parameters are optional** - defaults will be used if not specified:
- `--model_name`: ViT model name (default: `vit_base_patch16_224`)
- `--pretrained`: Use ImageNet pretrained weights (optional, recommended)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 3e-5)
- `--epochs`: Number of epochs (default: 15)
- `--img_size`: Input image size (default: 224)

**Outputs:**
- `models/epoch_{N}_{model_name}.pth`: Per-epoch checkpoints
- `models/best_{model_name}.pth`: Best model (highest validation AUC)
- `results/training_log_{model_name}.csv`: Metrics per epoch

#### Step 5: Visualize Training Metrics

```bash
python scripts/visualize_training.py
```

Or with custom model name:
```bash
python scripts/visualize_training.py --model_name vit_base_patch16_224
```

This generates training plots and prints summary statistics. Default model_name is `vit_base_patch16_224`.

**Outputs:**
- `plots/training_metrics_{model_name}.png`: Training metrics visualization

#### Step 6: Evaluate on Test Set

```bash
python scripts/evaluate.py
```

Or with custom parameters:
```bash
python scripts/evaluate.py --model_name vit_base_patch16_224 --batch_size 16 --threshold 0.5
```

**All parameters are optional** - defaults will be used if not specified:
- `--model_name`: Model name (default: `vit_base_patch16_224`)
- `--batch_size`: Batch size (default: 16)
- `--img_size`: Input image size (default: 224)
- `--threshold`: Classification threshold (default: 0.5)

**Outputs:**
- `json/preds_{model_name}.json`: Predictions in JSON format

#### Step 7: Run Inference on Single Image

```bash
python scripts/inference.py --image path/to/image.jpg
```

Or with custom parameters:
```bash
python scripts/inference.py --image path/to/image.jpg --model_name vit_base_patch16_224 --img_size 224 --threshold 0.5
```

**Required parameter:**
- `--image`: Path to image file (required)

**Optional parameters** - defaults will be used if not specified:
- `--model_name`: Model name (default: `vit_base_patch16_224`)
- `--img_size`: Input image size (default: 224)
- `--threshold`: Classification threshold (default: 0.5)

### Option 2: Docker

#### Prerequisites

- Docker installed
- Docker Compose (optional)
- NVIDIA Docker runtime (for GPU support)

#### Step 1: Build Docker Image

```bash
docker build -t vit-deepfake:latest .
```

#### Step 2: Prepare Data

**Using Docker Compose:**
```bash
docker-compose run vit-deepfake python scripts/prepare_data.py
```

**Using Docker directly:**
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  vit-deepfake:latest \
  python scripts/prepare_data.py
```

Or with custom train_ratio:
```bash
docker-compose run vit-deepfake python scripts/prepare_data.py --train_ratio 0.9
```

#### Step 3: Train Model

**Using Docker Compose:**
```bash
docker-compose run vit-deepfake python scripts/train.py
```

**Using Docker directly (with GPU):**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/results:/app/results" \
  vit-deepfake:latest \
  python scripts/train.py
```

Or with custom parameters:
```bash
docker-compose run vit-deepfake python scripts/train.py --pretrained --epochs 15 --batch_size 16 --lr 3e-5
```

#### Step 4: Visualize Training Metrics

**Using Docker Compose:**
```bash
docker-compose run vit-deepfake python scripts/visualize_training.py
```

**Using Docker directly:**
```bash
docker run --rm \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/plots:/app/plots" \
  vit-deepfake:latest \
  python scripts/visualize_training.py
```

Or with custom model_name:
```bash
docker-compose run vit-deepfake python scripts/visualize_training.py --model_name vit_base_patch16_224
```

#### Step 5: Evaluate on Test Set

**Using Docker Compose:**
```bash
docker-compose run vit-deepfake python scripts/evaluate.py
```

**Using Docker directly (with GPU):**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/json:/app/json" \
  vit-deepfake:latest \
  python scripts/evaluate.py
```

Or with custom parameters:
```bash
docker-compose run vit-deepfake python scripts/evaluate.py --model_name vit_base_patch16_224 --batch_size 16
```

#### Step 6: Run Inference

**Using Docker Compose:**
```bash
docker-compose run vit-deepfake python scripts/inference.py --image path/to/image.jpg
```

**Using Docker directly (with GPU):**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  vit-deepfake:latest \
  python scripts/inference.py --image path/to/image.jpg
```

Or with custom parameters:
```bash
docker-compose run vit-deepfake python scripts/inference.py --image path/to/image.jpg --model_name vit_base_patch16_224 --threshold 0.5
```

#### Using Helper Scripts

**Linux/Mac (Bash):**
```bash
chmod +x docker-run.sh
./docker-run.sh build
./docker-run.sh prepare
./docker-run.sh train
./docker-run.sh visualize
./docker-run.sh evaluate
./docker-run.sh inference --image path/to/image.jpg
```

Or with custom parameters:
```bash
./docker-run.sh train --pretrained --epochs 15 --batch_size 16
./docker-run.sh visualize --model_name vit_base_patch16_224
./docker-run.sh evaluate --model_name vit_base_patch16_224
```

**Windows (PowerShell):**
```powershell
.\docker-run.ps1 build
.\docker-run.ps1 prepare
.\docker-run.ps1 train
.\docker-run.ps1 visualize
.\docker-run.ps1 evaluate
.\docker-run.ps1 inference --image path/to/image.jpg
```

Or with custom parameters:
```powershell
.\docker-run.ps1 train --pretrained --epochs 15 --batch_size 16
.\docker-run.ps1 visualize --model_name vit_base_patch16_224
.\docker-run.ps1 evaluate --model_name vit_base_patch16_224
```

## Project Summary

### Key Features

- **Class Imbalance Handling:** Automatically oversamples minority class using RandomOverSampler
- **Data Augmentation:** Comprehensive augmentation for training (random crops, flips, rotations, color jitter, blur, affine transforms)
- **Robust Error Handling:** Handles NaN/Inf values and edge cases during validation
- **Checkpointing:** Saves per-epoch checkpoints and best model based on validation AUC
- **Metrics Logging:** Logs train/val loss, AUC, F1, accuracy, and average probability per epoch
- **Visualization:** Generates training curves and summary statistics

### Scripts Overview

1. **prepare_data.py**: Creates dataset manifests for training/validation and test
2. **train.py**: Trains a ViT binary classifier for deepfake detection
3. **visualize_training.py**: Visualizes training metrics and generates plots
4. **evaluate.py**: Evaluates trained model on test manifest and saves predictions
5. **inference.py**: Runs inference on a single image

### Notes

1. The classifier outputs a single logit; use sigmoid to get probability
2. Threshold 0.5: `prob >= 0.5` → "fake", else → "real"
3. CUDA is automatically detected; falls back to CPU if unavailable
4. Best model is saved based on highest validation AUC
5. Epoch checkpoints are ignored in git (only best model is tracked)

### Troubleshooting

1. **Manifests are empty:** Ensure image folders exist and contain supported image formats (.jpg, .jpeg, .png, .bmp)
2. **Validation data is empty:** Run `prepare_data.py` with `--train_ratio < 1.0` (default is 0.9)
3. **Checkpoint loading fails:** Make sure `models/best_{model_name}.pth` exists (created during training)
4. **Out-of-memory on GPU:** Reduce `--batch_size` or `--img_size`
5. **NaN/Inf in validation:** Check data quality and model initialization

## Training Results

**Model:** `vit_base_patch16_224` (15 epochs)

### Best Metrics (Epoch 12):
- **AUC:** 0.8003
- **F1 Score:** 0.9333
- **Accuracy:** 0.8814 (88.14%)
- **Validation Loss:** 0.3423

### Final Metrics (Epoch 15):
- **Train Loss:** 0.2826
- **Val Loss:** 0.4684
- **AUC:** 0.7123
- **F1 Score:** 0.8780
- **Accuracy:** 0.7881 (78.81%)

### Average Metrics (across all epochs):
- **Avg Train Loss:** 0.5117
- **Avg Val Loss:** 0.4400
- **Avg AUC:** 0.7003
- **Avg F1 Score:** 0.8889
- **Avg Accuracy:** 0.8062 (80.62%)

Training plots are available in `plots/training_metrics_vit_base_patch16_224.png`

## License

This project is provided as-is for educational and research purposes and submitted to IIIT Bangalore's Synergy,Deepfake Detector Hackathon
