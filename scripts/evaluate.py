"""
evaluate.py

Evaluate a trained ViT model on the test manifest and save predictions.
1) Loads best checkpoint from models/
2) Runs batched inference on data/test_manifest.csv
3) Writes JSON: [{"index": int, "prediction": "real|fake"}, ...]
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEST_MANIFEST = DATA_DIR / "test_manifest.csv"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from model import ViTBinaryClassifier

class TestDataset(Dataset):
    """Dataset for unlabeled test images referenced by a manifest DataFrame."""

    def __init__(self, rows, transform=None):
        """Store rows and an optional torchvision transform."""
        self.rows = rows
        self.transform = transform

    def __len__(self):
        """Return number of test items."""
        return len(self.rows)

    def __getitem__(self, idx):
        """Load image and return (tensor, filepath)."""
        row = self.rows.iloc[idx]
        path = row["filepath"]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

def get_transform(img_size=224):
    """Resize and normalize images to ImageNet stats for ViT input."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def sigmoid(x):
    """Sigmoid for numpy arrays/scalars."""
    return 1 / (1 + np.exp(-x))

def main(args):
    """Load best model, run batched inference over test manifest, save JSON."""
    # 1) Load test manifest
    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(f"Test manifest not found: {TEST_MANIFEST}")
    df = pd.read_csv(TEST_MANIFEST)
    if df.empty:
        raise RuntimeError("Test manifest is empty.")

    # 2) Build dataloader
    test_transform = get_transform(img_size=args.img_size)
    ds = TestDataset(df, transform=test_transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3) Load model and checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTBinaryClassifier(model_name=args.model_name, pretrained=False)
    ckpt = MODELS_DIR / f"best_{args.model_name}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    # Accept both state_dict and dict with key "model_state_dict"
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device).eval()

    # 4) Batched inference and thresholding
    results = []
    index = 1  # 1-based indexing
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs, paths = batch
            imgs = imgs.to(device)
            logits = model(imgs).detach().cpu().numpy()
            probs = sigmoid(logits)  # probability of 'fake' (1)
            for prob in probs:
                pred_label = "fake" if float(prob) >= args.threshold else "real"
                results.append({"index": index, "prediction": pred_label})
                index += 1

    # 5) Save predictions JSON
    out_path = OUT_DIR / f"preds_{args.model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions: {out_path}  (total: {len(results)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="vit_base_patch16_224")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
