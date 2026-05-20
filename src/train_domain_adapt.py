"""
Fine-tune EfficientNet-B3 + ArcFace classifier on combined HUST-OBC + OOD data.

Bridges the domain gap between clean HUST-OBC character crops and noisy
out-of-domain rubbing images by mixing both datasets with stronger augmentation.

Usage:
    python train_domain_adapt.py \
        --obc_dir /mnt/workspace/zjin_Repo_V2/ancient_char/hust_obc_temp/HUST-OBC/deciphered \
        --ood_dir /mnt/workspace/eval/recognition_train_data \
        --checkpoint /mnt/workspace/zjin_Repo/models/checkpoint.pth \
        --label_mapping /mnt/workspace/zjin_Repo/models/label_mapping.json \
        --output_dir ./output_ft --epochs 30
"""
import argparse, os, sys, math, random, shutil, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
from collections import Counter


# ---------------------------------------------------------------------------
# 1. ArcFace
# ---------------------------------------------------------------------------
class ArcFace(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine ** 2).clamp(1e-9, 1.0)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


# ---------------------------------------------------------------------------
# 2. Dataset — mixes OBC + OOD
# ---------------------------------------------------------------------------
class MixedDataset(Dataset):
    """Holds samples from both HUST-OBC and OOD with domain-aware sampling."""

    def __init__(self, obc_samples, ood_samples, transform=None, ood_weight=0.5):
        """
        obc_samples: list of (path, class_id) from HUST-OBC
        ood_samples: list of (path, class_id) from OOD (can be empty)
        ood_weight:  probability of picking from OOD during training
        """
        self.obc_samples = obc_samples
        self.ood_samples = ood_samples
        self.transform = transform
        self.ood_weight = ood_weight

        # Upsample OOD so we don't loop over tiny dataset
        n_obc = len(obc_samples)
        n_ood = len(ood_samples)
        if n_ood > 0 and n_ood < n_obc:
            repeat = max(1, n_obc // n_ood)
            self.ood_samples = ood_samples * repeat
        self.use_ood = len(ood_samples) > 0

    def __len__(self):
        return len(self.obc_samples) + (len(self.ood_samples) if self.use_ood else 0)

    def __getitem__(self, idx):
        if self.use_ood and random.random() < self.ood_weight:
            oid = random.randrange(len(self.ood_samples))
            path, label = self.ood_samples[oid]
        else:
            oid = random.randrange(len(self.obc_samples))
            path, label = self.obc_samples[oid]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
def build_model(num_classes, feat_dim=512):
    enet = models.efficientnet_b3(weights=None)
    backbone_out = enet.classifier[-1].in_features

    backbone = nn.Sequential()
    backbone.add_module("features", enet.features)
    backbone.add_module("avgpool", enet.avgpool)
    backbone.add_module("flatten", nn.Flatten())

    model = nn.Sequential()
    model.add_module("backbone", backbone)
    model.add_module("bn1", nn.BatchNorm1d(backbone_out))
    model.add_module("dropout", nn.Dropout(0.3))
    model.add_module("fc", nn.Linear(backbone_out, feat_dim))
    model.add_module("bn2", nn.BatchNorm1d(feat_dim))

    head = ArcFace(feat_dim, num_classes, s=30.0, m=0.50)
    return model, head


# ---------------------------------------------------------------------------
# 4. Data loading
# ---------------------------------------------------------------------------
def load_obc_samples(data_dir, label_to_idx):
    """Load HUST-OBC from class-subdir layout: data_dir/class_id/images."""
    data_dir = Path(data_dir)
    samples = []
    for p in data_dir.rglob("*.png"):
        parent_name = p.parent.name
        # parent_name is like "0001", look up in label_to_idx
        if parent_name in label_to_idx:
            samples.append((str(p), label_to_idx[parent_name]))
    return samples


def load_ood_samples(data_dir, label_mapping):
    """Load OOD crops from class_idx-subdir layout: data_dir/idx/images."""
    data_dir = Path(data_dir)
    label_to_idx = {v: int(k) for k, v in label_mapping.items()}
    samples = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            class_idx = int(class_dir.name)
        except ValueError:
            continue
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for p in class_dir.glob(ext):
                samples.append((str(p), class_idx))
    return samples


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obc_dir", type=str, required=True)
    parser.add_argument("--ood_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--label_mapping", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=1307)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output_domain_adapt")
    parser.add_argument("--ood_weight", type=float, default=0.5,
                        help="Prob of sampling from OOD during each batch")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Image size: {args.img_size} | Classes: {args.num_classes}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load label mapping
    with open(args.label_mapping) as f:
        label_mapping = json.load(f)
    # label_mapping: {"0": "0001", "1": "0002", ...} (class_idx -> folder_id)
    # Build reverse: folder_id -> class_idx
    folder_to_idx = {v: int(k) for k, v in label_mapping.items()}
    idx_to_folder = {int(k): v for k, v in label_mapping.items()}

    # Load samples
    print("\n=== Loading HUST-OBC samples ===")
    obc_samples = load_obc_samples(args.obc_dir, folder_to_idx)
    print(f"  HUST-OBC samples: {len(obc_samples)}")

    ood_samples = []
    if args.ood_dir and Path(args.ood_dir).exists():
        print("\n=== Loading OOD samples ===")
        ood_samples = load_ood_samples(args.ood_dir, label_mapping)
        print(f"  OOD samples: {len(ood_samples)}")

    all_samples = obc_samples + ood_samples
    labels = [c for _, c in all_samples]
    print(f"  Total samples: {len(all_samples)}")

    if len(all_samples) == 0:
        print("ERROR: No images found.")
        sys.exit(1)

    # Shuffle and simple 90/10 split (stratified split is too slow with 1290 classes)
    import random
    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)}")

    # Separate OOD vs OBC in training set for domain-aware sampling
    obc_set = set(obc_samples)
    train_obc = [(p, l) for p, l in train_samples if (p, l) in obc_set]
    train_ood = [(p, l) for p, l in train_samples if (p, l) not in obc_set]
    print(f"  Train: OBC={len(train_obc)}, OOD={len(train_ood)}")
    print(f"  Val classes: {len(set(l for _,l in val_samples))}")

    # -----------------------------------------------------------------------
    # Transforms — stronger augmentation for domain adaptation
    # -----------------------------------------------------------------------
    train_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.RandomRotation(15),
        T.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(3, sigma=(0.1, 1.0)),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        # Simulate low-resolution (like HUST-OBC small crops)
        T.RandomApply([T.Compose([T.Resize((32, 32)), T.Resize((args.img_size, args.img_size))])], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = MixedDataset(train_obc, train_ood, transform=train_transform,
                            ood_weight=args.ood_weight)
    class ValDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, i):
            img = Image.open(self.samples[i][0]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, self.samples[i][1]

    val_ds = ValDataset(val_samples, val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=4, pin_memory=True)

    # -----------------------------------------------------------------------
    # Load checkpoint and build model
    # -----------------------------------------------------------------------
    print("\n=== Loading checkpoint ===")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if ckpt.get("num_classes") != args.num_classes:
        print(f"WARNING: checkpoint has {ckpt.get('num_classes')} classes, requested {args.num_classes}")

    backbone, head = build_model(args.num_classes, feat_dim=args.feat_dim)

    # Load pretrained weights
    try:
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        print("  Backbone loaded (keys match directly).")
    except Exception as e:
        print(f"  Backbone load issue: {e}")
        # Fallback: strip or add "backbone." prefix as needed
        backbone_sd = {}
        for k, v in ckpt["backbone_state_dict"].items():
            # Both checkpoint and model use "backbone." prefix, but handle edge cases
            if k.startswith("backbone.") and not list(backbone.state_dict().keys())[0].startswith("backbone."):
                backbone_sd[k.replace("backbone.", "", 1)] = v
            elif not k.startswith("backbone.") and list(backbone.state_dict().keys())[0].startswith("backbone."):
                backbone_sd["backbone." + k] = v
            else:
                backbone_sd[k] = v
        backbone.load_state_dict(backbone_sd, strict=False)
        print("  Backbone loaded with key remapping.")

    # Load head weights if shape matches
    if ckpt["head_state_dict"]["weight"].shape == head.weight.shape:
        head.load_state_dict(ckpt["head_state_dict"])
        print("  Loaded head weights.")
    else:
        print(f"  Head shape mismatch: ckpt={ckpt['head_state_dict']['weight'].shape}, "
              f"model={head.weight.shape}. Reinitializing head.")
    train_acc_ckpt = ckpt.get("train_acc", ckpt.get("val_acc", 0))
    print(f"  Checkpoint loaded (train_acc={train_acc_ckpt:.4f})")

    backbone.to(device)
    head.to(device)

    # Freeze early layers (first 3 MBConv)
    freeze_until = 3
    for i, (name, child) in enumerate(backbone[0].features.named_children()):
        if i < freeze_until:
            for param in child.parameters():
                param.requires_grad = False

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss_fn = nn.CrossEntropyLoss()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print("\n=== Fine-tuning ===")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        backbone.train()
        head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            feats = backbone(images)
            logits = head(feats, labels)
            loss = ce_loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = logits.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "acc": f"{preds.eq(labels).sum().item()/images.size(0):.4f}"})

        train_acc = train_correct / train_total
        train_loss_avg = train_loss / train_total

        # Validation
        backbone.eval()
        head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                feats = backbone(images)
                logits = head(feats, labels)
                loss = ce_loss_fn(logits, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = logits.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / val_total
        val_loss_avg = val_loss / val_total

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch+1:3d}/{args.epochs}  "
              f"Train Loss: {train_loss_avg:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss_avg:.4f}  Val Acc: {val_acc:.4f}  "
              f"LR: {current_lr:.2e}")

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint = {
            "epoch": epoch,
            "backbone_state_dict": backbone.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "num_classes": args.num_classes,
            "feat_dim": args.feat_dim,
            "img_size": args.img_size,
        }
        ckpt_path = out_dir / "checkpoint.pth"
        torch.save(checkpoint, ckpt_path)
        if is_best:
            best_path = out_dir / "model_best.pth"
            shutil.copyfile(ckpt_path, best_path)
            print(f"  >>> New best val_acc: {val_acc:.4f}")

        if patience_counter >= args.patience:
            print(f"\n  >>> Early stopping at epoch {epoch+1}")
            break

    # Save final model in deployable format
    final_ckpt = torch.load(out_dir / "model_best.pth")
    # run_inference.py loads:
    #   checkpoint.pth -> backbone_state_dict, head_state_dict
    #   backbone.pth   -> backbone_state_dict (standalone)
    torch.save(final_ckpt, out_dir / "checkpoint.pth")
    torch.save(final_ckpt["backbone_state_dict"], out_dir / "backbone.pth")

    # Copy label mappings
    shutil.copy2(args.label_mapping, out_dir / "label_mapping.json")
    id_to_chinese = Path(args.obc_dir) / "ID_to_chinese.json"
    if id_to_chinese.exists():
        shutil.copy2(id_to_chinese, out_dir / "ID_to_chinese.json")

    print(f"\n===== Fine-tuning complete! =====")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Output: {out_dir.resolve()}")
    print(f"To deploy: cp {out_dir/'checkpoint.pth'} /app/models/checkpoint.pth")
    print(f"          cp {out_dir/'backbone.pth'} /app/models/backbone.pth")


if __name__ == "__main__":
    main()
