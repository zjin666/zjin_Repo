#!/usr/bin/env python3
"""Pre-load YOLO detector + EfficientNet-B3 classifier into CUDA cache."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO

MODEL_DIR = Path("/app/models")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load YOLO
    yolo_path = MODEL_DIR / "yolo_detector.pt"
    print(f"Loading YOLO from {yolo_path} ...")
    detector = YOLO(str(yolo_path))
    print("YOLO ready.")

    # Load classifier backbone + head from checkpoint (same as run_inference.py)
    from run_inference import build_classifier

    ckpt = torch.load(MODEL_DIR / "checkpoint.pth", map_location="cpu")
    backbone_sd = ckpt["backbone_state_dict"]
    head_weight = ckpt["head_state_dict"]["weight"]
    print(f"Classifier head: {head_weight.shape[0]} classes, {head_weight.shape[1]} dims")

    feat_extractor, head_w = build_classifier(backbone_sd, head_weight, device)

    # Warmup: one forward pass with dummy data
    import torchvision.transforms as T
    from PIL import Image
    import numpy as np

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dummy = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img_t = transform(dummy).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = feat_extractor(img_t)
        feat = F.normalize(feat)
        w_norm = F.normalize(head_w)
        logits = feat @ w_norm.T
        _ = logits[0].topk(1)

    print("Classifier warmup complete. All models ready.")


if __name__ == "__main__":
    main()
