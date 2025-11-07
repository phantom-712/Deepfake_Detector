"""
prepare_data.py

Create dataset manifests for training/validation and test.

Scans data/real_cifake_images/, data/fake_cifake_images/, data/test/ and writes
train_manifest.csv, test_manifest.csv, and dataset_index.csv.

Usage:
    python scripts/prepare_data.py [--train_ratio 0.9]  # Default: 90% train, 10% val
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = ROOT / "data"
REAL_DIR = DATA_DIR / "real_cifake_images"
FAKE_DIR = DATA_DIR / "fake_cifake_images"
TEST_DIR = DATA_DIR / "test"
MANIFEST_DIR = DATA_DIR  # will write train_manifest.csv and test_manifest.csv


def list_images(folder: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    """Return sorted list of image paths in folder."""
    if not folder.exists():
        return []
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    return sorted(imgs)


def write_manifest(rows: List[dict], out_path: Path):
    """Write rows with fields filepath,label,split to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # write an empty CSV with headers to keep things predictable
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split"])
            writer.writeheader()
        print(f"Warning: wrote empty manifest to {out_path}")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Manifest saved: {out_path} ({len(rows)} rows)")


def create_dataset_manifests(train_ratio=0.9, seed=42):
    """Create train/val split and write manifest CSVs."""
    real = list_images(REAL_DIR)
    fake = list_images(FAKE_DIR)
    test = list_images(TEST_DIR)

    print(f"Found {len(real)} real, {len(fake)} fake, {len(test)} test images.")
    rows = []
    # simple split: put most in train, small in val (if desired)
    import random
    random.seed(seed)

    for p in real:
        split = "train" if random.random() < train_ratio else "val"
        rows.append({"filepath": str(p), "label": "real", "split": split})

    for p in fake:
        split = "train" if random.random() < train_ratio else "val"
        rows.append({"filepath": str(p), "label": "fake", "split": split})

    # Shuffle rows to mix labels
    random.shuffle(rows)

    train_manifest = MANIFEST_DIR / "train_manifest.csv"
    write_manifest(rows, train_manifest)

    # test manifest: if test images exist, write them with label 'unknown'
    test_rows = []
    for p in test:
        test_rows.append({"filepath": str(p), "label": "unknown", "split": "test"})
    test_manifest = MANIFEST_DIR / "test_manifest.csv"
    write_manifest(test_rows, test_manifest)

    # Optionally write combined dataset_index.csv
    dataset_index = MANIFEST_DIR / "dataset_index.csv"
    combined = rows + test_rows
    if combined:
        df = pd.DataFrame(combined)
        df.to_csv(dataset_index, index=False)
        print(f"Combined dataset_index saved: {dataset_index}")
    else:
        # write empty file
        pd.DataFrame(columns=["filepath", "label", "split"]).to_csv(dataset_index, index=False)
        print(f"Wrote empty dataset_index to {dataset_index}")

    return train_manifest, test_manifest, dataset_index


def load_dataframe(manifest_path: Path = None) -> pd.DataFrame:
    """Load manifest CSV into DataFrame."""
    if manifest_path is None:
        manifest_path = MANIFEST_DIR / "train_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    # Defensive checks
    required = {"filepath", "label", "split"}
    if df.empty or not required.issubset(df.columns):
        print("Warning: DataFrame is empty or missing required columns.")
        # attempt to return a DataFrame with expected columns
        return pd.DataFrame(columns=list(required))
    print(f"Loaded DataFrame: {len(df)} rows, columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset manifests")
    parser.add_argument("--train_ratio", type=float, default=0.9, 
                        help="Ratio of data to use for training (rest goes to validation). Default: 0.9")
    args = parser.parse_args()
    
    # Validate train_ratio
    if args.train_ratio <= 0 or args.train_ratio >= 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1 (exclusive), got {args.train_ratio}")
    if args.train_ratio > 0.95:
        print(f"Warning: train_ratio={args.train_ratio} is very high. Validation set will be very small.")

    create_dataset_manifests(train_ratio=args.train_ratio)
    try:
        df = load_dataframe()
        # show distribution if possible
        if not df.empty:
            dist = df.groupby(["split", "label"]).size().reset_index(name="count")
            print("Distribution:\n", dist)
        else:
            print("No training data found.")
    except Exception as e:
        print("Error loading dataframe:", e)
