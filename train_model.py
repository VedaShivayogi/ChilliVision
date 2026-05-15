"""
ChilliNet Training Script
=========================
Trains the ChilliNet model from scratch using:
  1. Transfer learning from MobileNetV2 (ImageNet weights)
  2. Synthetic augmented dataset if real data not available
  3. Saves weights to backend/chilli_net.pth

Usage:
    python train_model.py                    # Train with synthetic data
    python train_model.py --data ./my_data   # Train with real image folder
    python train_model.py --epochs 30        # Custom epochs

Folder structure for real data:
    my_data/
        byadagi/     *.jpg, *.png ...
        kashmiri/    ...
        guntur/      ...
        kanthari/    ...
        dried/       ...
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.chilli_model import ChilliNet

# ─────────────────────────────────────────
# Synthetic Chilli Image Generator
# ─────────────────────────────────────────

CLASS_PROFILES = {
    0: {"h_range": (0, 15),   "s_range": (180, 255), "v_range": (100, 200), "shape": "elongated"},   # Byadagi
    1: {"h_range": (0, 10),   "s_range": (210, 255), "v_range": (150, 230), "shape": "straight"},    # Kashmiri
    2: {"h_range": (0, 12),   "s_range": (170, 240), "v_range": (80,  170), "shape": "curved"},      # Guntur
    3: {"h_range": (0, 8),    "s_range": (200, 255), "v_range": (130, 200), "shape": "small"},       # Kanthari
    4: {"h_range": (5, 20),   "s_range": (100, 180), "v_range": (60,  130), "shape": "wrinkled"},    # Dried
}


def make_synthetic_chilli(class_id, quality=None, size=(224, 224)):
    """Generate a synthetic chilli image with HSV profile per class."""
    profile = CLASS_PROFILES[class_id]
    img = Image.new("RGB", size, (20, 20, 20))
    draw = ImageDraw.Draw(img)
    
    h = np.random.randint(*profile["h_range"])
    s = np.random.randint(*profile["s_range"])
    v = np.random.randint(*profile["v_range"])
    
    # Convert HSV to RGB
    h_f, s_f, v_f = h / 179.0, s / 255.0, v / 255.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h_f, s_f, v_f)
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    
    cx, cy = size[0] // 2, size[1] // 2
    
    # Draw chilli body based on shape
    shape = profile["shape"]
    if shape == "elongated":
        draw.ellipse([cx - 15, cy - 70, cx + 15, cy + 70], fill=(r, g, b))
    elif shape == "straight":
        draw.ellipse([cx - 12, cy - 65, cx + 12, cy + 65], fill=(r, g, b))
    elif shape == "curved":
        for offset in range(-60, 60, 2):
            curve_x = int(cx + 20 * np.sin(offset / 30.0))
            draw.ellipse([curve_x - 10, cy + offset - 5, curve_x + 10, cy + offset + 5], fill=(r, g, b))
    elif shape == "small":
        draw.ellipse([cx - 6, cy - 30, cx + 6, cy + 30], fill=(r, g, b))
    elif shape == "wrinkled":
        draw.ellipse([cx - 18, cy - 55, cx + 18, cy + 55], fill=(r, g, b))
        # Add wrinkle lines
        for i in range(-50, 50, 10):
            draw.line([(cx - 15, cy + i), (cx + 15, cy + i + 3)], fill=(max(0, r - 40), max(0, g - 10), max(0, b - 10)), width=1)
    
    # Stem
    draw.rectangle([cx - 3, cy - 75, cx + 3, cy - 60], fill=(34, 85, 34))
    
    # Apply quality degradation
    if quality is not None and quality < 60:
        # Add dark spots
        n_spots = int((60 - quality) / 5)
        for _ in range(n_spots):
            sx = np.random.randint(cx - 20, cx + 20)
            sy = np.random.randint(cy - 60, cy + 60)
            spot_size = np.random.randint(3, 10)
            draw.ellipse([sx, sy, sx + spot_size, sy + spot_size], fill=(30, 15, 10))
    
    # Random background noise
    bg_noise = np.random.randint(10, 40, (*size, 3), dtype=np.uint8)
    img_arr = np.array(img)
    mask = img_arr[:, :, 0] < 25
    img_arr[mask] = bg_noise[mask]
    img = Image.fromarray(img_arr)
    
    # Random augmentation
    img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.8, 1.2))
    if np.random.rand() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0, 1.2)))
    
    return img


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class SyntheticChilliDataset(Dataset):
    def __init__(self, samples_per_class=200, transform=None):
        self.transform = transform
        self.data = []
        for class_id in range(5):
            for _ in range(samples_per_class):
                quality = np.random.randint(20, 100)
                self.data.append((class_id, quality))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        class_id, quality = self.data[idx]
        img = make_synthetic_chilli(class_id, quality=quality)
        if self.transform:
            img = self.transform(img)
        return img, class_id, quality / 100.0


class RealChilliDataset(Dataset):
    CLASS_MAP = {"byadagi": 0, "kashmiri": 1, "guntur": 2, "kanthari": 3, "dried": 4}

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        for folder, label in self.CLASS_MAP.items():
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.samples.append((os.path.join(folder_path, fname), label))
        print(f"  Found {len(self.samples)} real images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, 0.8  # assume good quality for labelled real data


# ─────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────

def train(data_dir=None, epochs=20, batch_size=32, lr=1e-3, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if data_dir and os.path.isdir(data_dir):
        print(f"  Using real data from: {data_dir}")
        dataset = RealChilliDataset(data_dir, transform=transform_train)
    else:
        print("  Using synthetic dataset (200 images/class × 5 classes = 1000 total)")
        dataset = SyntheticChilliDataset(samples_per_class=200, transform=transform_train)

    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = ChilliNet(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    best_val_acc = 0.0
    save_path = output_path or os.path.join(os.path.dirname(__file__), "chilli_net.pth")

    print(f"\n  Training for {epochs} epochs...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels, qualities in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            qualities = qualities.float().to(device)

            optimizer.zero_grad()
            type_out, qual_out = model(imgs)
            loss_cls = cls_criterion(type_out, labels)
            loss_reg = reg_criterion(qual_out.squeeze(), qualities)
            loss = loss_cls + 0.3 * loss_reg

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = type_out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                type_out, _ = model(imgs)
                preds = type_out.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100 * val_correct / val_total

        print(f"  Epoch [{epoch:02d}/{epochs}]  Loss: {total_loss/len(train_loader):.4f}  "
              f"Train Acc: {train_acc:.1f}%  Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    ✅ Saved best model → {save_path}")

    print(f"\n  Training complete. Best Val Acc: {best_val_acc:.1f}%")
    print(f"  Model saved to: {save_path}")
    return save_path


# ─────────────────────────────────────────
# CLI Entry-point
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChilliNet model")
    parser.add_argument("--data",   type=str, default=None, help="Path to real image folder")
    parser.add_argument("--epochs", type=int, default=20,   help="Number of training epochs")
    parser.add_argument("--batch",  type=int, default=32,   help="Batch size")
    parser.add_argument("--lr",     type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    args = parser.parse_args()

    print("\n🌶  ChilliNet Training")
    print("=" * 50)
    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        output_path=args.output
    )
