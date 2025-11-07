"""
inference.py

Predict deepfake probability for a single image using a trained ViT model.
1) Loads best checkpoint from models/
2) Applies standard preprocessing
3) Prints JSON with filepath, score, and predicted label
"""

import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from model import ViTBinaryClassifier

def get_transform(img_size=224):
    """Resize and normalize to ImageNet statistics."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def sigmoid(x):
    """Sigmoid for numpy scalars/arrays."""
    return 1 / (1 + np.exp(-x))

def infer_single(model, img_path, transform, device):
    """Return deepfake probability for one image path."""
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).cpu().numpy().ravel()[0]
        prob = float(sigmoid(logits))
    return prob

def main(args):
    """Load model checkpoint and run inference on a single image."""
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

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

    transform = get_transform(img_size=args.img_size)
    prob = infer_single(model, img_path, transform, device)
    pred_label = "fake" if prob >= args.threshold else "real"
    result = {"filepath": str(img_path), "score": prob, "pred_label": pred_label}
    print(json.dumps(result, indent=2))

    if args.save:
        out = OUT_DIR / f"infer_{img_path.stem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("Saved inference JSON:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model_name", default="vit_base_patch16_224")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    main(args)
