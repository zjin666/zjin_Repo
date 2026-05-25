#!/usr/bin/env python3
"""
Inference pipeline: YOLOv8 detection + EfficientNet-B3 / ArcFace classification.
"""
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO


INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata/13/eval/images"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))

YOLO_CONF = float(os.getenv("YOLO_CONF", "0.1"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.7"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))

CLASSIFY_IMGSZ = 160
SKIP_PREPROC = os.getenv("SKIP_PREPROC", "0") == "1"
CLASSIFIER_THRESH = float(os.getenv("CLASSIFIER_THRESH", "0.3"))
MARGIN_THRESH = float(os.getenv("MARGIN_THRESH", "0.1"))
PREPROC_MODE = os.getenv("PREPROC_MODE", "gentle")  # gentle | original | clean | skip
BBOX_EXPAND = int(os.getenv("BBOX_EXPAND", "10"))  # pixels to expand YOLO bboxes for context
CROP_PAD_SQUARE = os.getenv("CROP_PAD_SQUARE", "1") == "1"  # pad to square before resize
FEAT_DIM = 512


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------
def build_classifier(backbone_state_dict, head_weight, device):
    """Build EfficientNet-B3 feature extractor + load ArcFace head weights."""
    import torchvision.models as models

    enet = models.efficientnet_b3(weights=None)
    backbone_out = enet.classifier[-1].in_features  # 1536

    backbone = nn.Sequential()
    backbone.add_module("features", enet.features)
    backbone.add_module("avgpool", enet.avgpool)
    backbone.add_module("flatten", nn.Flatten())

    feat_extractor = nn.Sequential()
    feat_extractor.add_module("backbone", backbone)
    feat_extractor.add_module("bn1", nn.BatchNorm1d(backbone_out))
    feat_extractor.add_module("dropout", nn.Dropout(0.3))
    feat_extractor.add_module("fc", nn.Linear(backbone_out, FEAT_DIM))
    feat_extractor.add_module("bn2", nn.BatchNorm1d(FEAT_DIM))

    feat_extractor.load_state_dict(backbone_state_dict)
    feat_extractor.to(device)
    feat_extractor.eval()

    head_w = head_weight.to(device)  # (num_classes, feat_dim)
    return feat_extractor, head_w


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------
def find_images():
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if INPUT_DIR.exists():
        return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in suffixes)
    fallback = Path("/saisdata")
    if fallback.exists():
        return sorted(p for p in fallback.rglob("*") if p.suffix.lower() in suffixes)
    return []


classify_transform = T.Compose([
    T.Resize((CLASSIFY_IMGSZ, CLASSIFY_IMGSZ)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def preprocess_domain(crop_pil):
    """Preprocess character crops for classification.

    Modes (set via PREPROC_MODE env var):
      gentle (default) — mild CLAHE only, preserves grayscale detail
      original         — aggressive CLAHE + adaptive threshold + binary blend
      skip             — no preprocessing at all
    """
    import cv2
    import numpy as np

    if PREPROC_MODE == "skip":
        return crop_pil.convert("RGB")
    if PREPROC_MODE == "original":
        img = np.array(crop_pil.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5,
        )
        result = cv2.addWeighted(binary, 0.85, enhanced, 0.15, 0)
        return Image.fromarray(result).convert("RGB")

    if PREPROC_MODE == "clean":
        # Bilateral filter smooths stone texture while preserving stroke edges
        img = np.array(crop_pil.convert("L"))
        filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        # Slight blur before threshold to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Clean up: close small gaps, remove isolated noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        return Image.fromarray(cleaned).convert("RGB")

    # gentle (default) — CLAHE only, preserves grayscale detail
    img = np.array(crop_pil.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(img)
    # Mild gamma to keep strokes crisp
    enhanced = (enhanced / 255.0) ** 0.8 * 255
    result = np.clip(enhanced, 0, 255).astype(np.uint8)
    return Image.fromarray(result).convert("RGB")


def pad_to_square(pil_img):
    """Pad to square with white background, preserving aspect ratio."""
    w, h = pil_img.size
    if w == h:
        return pil_img
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(pil_img, ((side - w) // 2, (side - h) // 2))
    return padded


def classify_crop(crop_pil, feat_extractor, head_w, idx_to_char):
    """Classify a single character crop. Returns (character, confidence) or (None, 0) if ambiguous."""
    if not SKIP_PREPROC:
        crop_pil = preprocess_domain(crop_pil)
    if CROP_PAD_SQUARE:
        crop_pil = pad_to_square(crop_pil)
    img_t = classify_transform(crop_pil).unsqueeze(0)
    img_t = img_t.to(next(feat_extractor.parameters()).device)

    with torch.no_grad():
        feat = feat_extractor(img_t)
        feat = F.normalize(feat)
        w_norm = F.normalize(head_w)
        logits = feat @ w_norm.T
        scores, preds = logits[0].topk(2)

    top1, top2 = float(scores[0]), float(scores[1])
    margin = top1 - top2
    if MARGIN_THRESH > 0 and margin < MARGIN_THRESH:
        return None, 0.0

    class_idx = preds[0].item()
    char = idx_to_char.get(str(class_idx), str(class_idx))
    return char, top1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Label mappings
    # ------------------------------------------------------------------
    with open(MODEL_DIR / "label_mapping.json") as f:
        label_mapping = json.load(f)
    with open(MODEL_DIR / "ID_to_chinese.json") as f:
        id_to_chinese = json.load(f)

    idx_to_char = {}
    for idx_str, folder_id in label_mapping.items():
        idx_to_char[idx_str] = id_to_chinese.get(folder_id, folder_id)
    print(f"Loaded mapping: {len(idx_to_char)} classes")

    # ------------------------------------------------------------------
    # YOLO detector
    # ------------------------------------------------------------------
    yolo_path = MODEL_DIR / "yolo_detector.pt"
    print(f"Loading YOLO from {yolo_path} ...")
    detector = YOLO(str(yolo_path))

    # ------------------------------------------------------------------
    # Classifier
    # ------------------------------------------------------------------
    ckpt = torch.load(MODEL_DIR / "checkpoint.pth", map_location="cpu")
    backbone_sd = ckpt["backbone_state_dict"]
    head_weight = ckpt["head_state_dict"]["weight"]
    print(f"Classes: {head_weight.shape[0]}  |  Feat dim: {head_weight.shape[1]}")

    feat_extractor, head_w = build_classifier(backbone_sd, head_weight, device)
    print("Classifier loaded.")

    # ------------------------------------------------------------------
    # Find images
    # ------------------------------------------------------------------
    image_paths = find_images()
    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Images found: {len(image_paths)}")

    if not image_paths:
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        print(f"No images. Saved empty: {OUTPUT_FILE}")
        return

    # ------------------------------------------------------------------
    # Process each image
    # ------------------------------------------------------------------
    all_results = {}
    for img_idx, image_path in enumerate(image_paths, start=1):
        if img_idx % 25 == 0 or img_idx == 1:
            print(f"[{img_idx}/{len(image_paths)}] {image_path.name}", flush=True)

        image_id = image_path.stem
        try:
            results = detector(str(image_path), conf=YOLO_CONF,
                               iou=YOLO_IOU, imgsz=YOLO_IMGSZ, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                all_results[image_id] = []
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            detections = []

            # Open image once for all crops
            with Image.open(image_path) as img:
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    # Expand bbox for context, clamped to image boundaries
                    ex1 = max(0, int(round(x1)) - BBOX_EXPAND)
                    ey1 = max(0, int(round(y1)) - BBOX_EXPAND)
                    ex2 = min(img.width, int(round(x2)) + BBOX_EXPAND)
                    ey2 = min(img.height, int(round(y2)) + BBOX_EXPAND)
                    x, y, w, h = int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))
                    if w <= 0 or h <= 0:
                        continue

                    crop = img.crop((ex1, ey1, ex2, ey2))
                    char, score = classify_crop(crop, feat_extractor, head_w, idx_to_char)
                    if char is None:
                        continue
                    if CLASSIFIER_THRESH > 0 and score < CLASSIFIER_THRESH:
                        continue
                    detections.append({
                    "bbox": [x, y, w, h],
                    "text": char,
                    "_y": y,
                    "_x": x,
                })

            detections.sort(key=lambda d: (d["_y"], d["_x"]))
            all_results[image_id] = [
                {"bbox": d["bbox"], "text": d["text"]} for d in detections
            ]

        except Exception as exc:
            print(f"Warning: failed to process {image_path}: {exc}")
            traceback.print_exc()
            all_results[image_id] = []

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total_chars = sum(len(v) for v in all_results.values())
    print(f"\nSaved: {OUTPUT_FILE}  ({len(all_results)} images, {total_chars} chars)")


if __name__ == "__main__":
    main()
