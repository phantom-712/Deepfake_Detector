"""
train.py

Main training script for ViT-based deepfake detection.

Trains a Vision Transformer (ViT) for binary classification (real vs fake).
Handles class imbalance via RandomOverSampler, saves checkpoints and logs metrics.

Usage:
    python scripts/train.py [--pretrained] [--batch_size 16] [--lr 3e-5] [--epochs 15]

Prerequisites: Run prepare_data.py first to generate train_manifest.csv
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler   # <-- added import

from model import ViTBinaryClassifier


# 1. Directories
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST = DATA_DIR / "train_manifest.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# 2. Dataset
class ImageDataset(Dataset):
    """PyTorch Dataset for loading images and binary labels from a DataFrame.
    
    Expects columns: "filepath" (str), "label" in {"real","fake"}.
    Labels are converted: "real" -> 0.0, "fake" -> 1.0
    """
    def __init__(self, df, transform=None):
        """Initialize with DataFrame and optional transform."""
        self.df = df
        self.transform = transform

    def __len__(self):
        """Return dataset length."""
        return len(self.df)

    def __getitem__(self, idx):
        """Load image and map label to 0.0 (real) or 1.0 (fake)."""
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        label = 1.0 if row["label"] == "fake" else 0.0
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# 3. Data transforms
def get_transforms(img_size=224, train=True):
    """Return torchvision transforms for train/val with ImageNet normalization.
    
    Training: aggressive augmentation (crops, flips, rotations, color jitter, blur).
    Validation: resizing and center cropping only.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


# 4. Helper functions
def sigmoid(x):
    """Numerically stable sigmoid for numpy arrays (clips to [-20, 20])."""
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))


def train_one_epoch(model, loader, optimizer, device, criterion):
    """Run one training epoch and return average loss."""
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc="Train", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(imgs).view(-1)  # Flatten logits for BCE compatibility
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, device, criterion):
    """Evaluate model and return (avg_loss, auc, f1, acc, avg_prob).
    
    Handles NaN/Inf values and edge cases. Returns (inf, 0, 0, 0, 0) if dataset is empty.
    """
    model.eval()
    running_loss = 0.0
    all_logits, all_targets = [], []

    if len(loader.dataset) == 0:
        print("WARNING: Validation dataset is empty!")
        return float("inf"), 0, 0, 0, 0

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Validate", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs).view(-1)  # <-- Flatten logits
            
            # Check for NaN or Inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("WARNING: NaN or Inf detected in model logits!")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            loss = criterion(logits, targets)
            
            # Check for NaN or Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Invalid loss detected: {loss.item()}")
                loss = torch.tensor(1e6, device=device)  # Use a large finite value
            
            running_loss += loss.item() * imgs.size(0)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if len(all_logits) == 0:
        print("WARNING: No validation batches processed!")
        return float("inf"), 0, 0, 0, 0

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    
    # Check for NaN/Inf in concatenated arrays
    if np.isnan(logits).any() or np.isinf(logits).any():
        print("WARNING: NaN or Inf in logits after concatenation, replacing...")
        logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
    
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)

    avg_loss = running_loss / len(loader.dataset)
    
    # Check if avg_loss is valid
    if np.isnan(avg_loss) or np.isinf(avg_loss):
        print(f"WARNING: Invalid average loss: {avg_loss}, using fallback value")
        avg_loss = 1e6

    # Compute metrics safely
    try:
        auc = roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.0
    except ValueError:
        auc = 0.0

    f1 = f1_score(targets, preds) if len(np.unique(targets)) > 1 else 0.0
    acc = accuracy_score(targets, preds)
    return avg_loss, auc, f1, acc, probs.mean()


def main(args):
    """Train a ViT binary classifier and save logs/checkpoints/best model."""
    # 1. Load manifest
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    df = pd.read_csv(MANIFEST)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    if train_df.empty:
        raise RuntimeError("Training data is empty. Check prepare_data.py output.")
    if val_df.empty:
        raise RuntimeError("Validation data is empty. Check prepare_data.py output. Ensure some samples are marked as 'val' split.")
    
    print(f"Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Train labels - Real: {len(train_df[train_df['label']=='real'])}, Fake: {len(train_df[train_df['label']=='fake'])}")
    print(f"Val labels - Real: {len(val_df[val_df['label']=='real'])}, Fake: {len(val_df[val_df['label']=='fake'])}")

    # 2. Handle class imbalance via RandomOverSampler
    print("Applying RandomOverSampler to handle class imbalance...")
    label_map = train_df["label"].map({"real": 0, "fake": 1}).values
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(train_df[["filepath"]], label_map)
    train_df = pd.DataFrame({
        "filepath": X_resampled["filepath"],
        "label": ["fake" if y == 1 else "real" for y in y_resampled],
        "split": "train"
    })
    print(f"Resampled dataset: {len(train_df)} samples (balanced real/fake)")

    # 3. Data loaders
    train_loader = DataLoader(
        ImageDataset(train_df, transform=get_transforms(args.img_size, train=True)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        ImageDataset(val_df, transform=get_transforms(args.img_size, train=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 4. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTBinaryClassifier(model_name=args.model_name, pretrained=args.pretrained)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    log_file = RESULTS_DIR / f"training_log_{args.model_name}.csv"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,auc,f1,acc,avg_prob\n")

    print(f"\n=== Training {args.model_name} for {args.epochs} epochs ===")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, auc, f1, acc, avg_prob = validate(model, val_loader, device, criterion)

        # 5. Logging
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | ACC: {acc:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{auc:.4f},{f1:.4f},{acc:.4f},{avg_prob:.4f}\n")

        # 6. Save per-epoch checkpoint
        ckpt_path = MODELS_DIR / f"epoch_{epoch}_{args.model_name}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "auc": auc,
            "f1": f1,
            "acc": acc
        }, ckpt_path)

        # 7. Save best model
        if auc > best_auc:
            best_auc = auc
            best_path = MODELS_DIR / f"best_{args.model_name}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved: {best_path} (AUC={auc:.4f})")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    print(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    main(args)
