"""
Replication of Thomas et al. 2020:
"Automated Classification of Radiographic Knee Osteoarthritis Severity
Using Deep Neural Networks" — Radiology: AI

Key paper details replicated:
- DenseNet-169 backbone, ImageNet pretrained
- Full image input (no manual joint cropping)
- 299x299 input size
- Stochastic augmentation (80% prob): flip, rotate, zoom, contrast, noise
- Standard CrossEntropyLoss, no class weighting
- Adam optimizer lr=1e-4
- End-to-end training, no frozen layers
- Evaluated with Quadratic Weighted Kappa

Google Colab T4 GPU optimized.
"""

# ── 0. Install extra deps ────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "-q"])

# ── 1. Imports ───────────────────────────────────────────────────────────────
import os, random, time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ── 2. Config ────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("/content/drive/MyDrive/KL_Sorted_Split")
NUM_CLASSES = 5
IMG_SIZE    = 299          # exact Thomas et al. input size
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-4
VAL_FRAC    = 0.2
SEED        = 42
AUG_PROB    = 0.80         # Thomas: "80% probability" stochastic aug
PATIENCE    = 10           # early-stop patience (QWK-based)
BEST_PATH   = "best_model_thomas_colab.pth"
CM_PATH     = "confusion_matrix_thomas_colab.png"
CLASS_NAMES = [f"KL{i}" for i in range(NUM_CLASSES)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(SEED)

# ── 3. Dataset ───────────────────────────────────────────────────────────────
class KneeDataset(Dataset):
    """
    Loads knee X-rays with Thomas et al. stochastic augmentation.
    Each transform is applied independently with probability AUG_PROB.
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment
        self.to_tensor = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    # ── augmentation helpers ─────────────────────────────────────────────────
    @staticmethod
    def _random_flip(img: Image.Image) -> Image.Image:
        return transforms.functional.hflip(img)

    @staticmethod
    def _random_rotation(img: Image.Image) -> Image.Image:
        angle = random.uniform(-20, 20)
        return transforms.functional.rotate(img, angle, fill=0)

    @staticmethod
    def _random_zoom(img: Image.Image) -> Image.Image:
        """Random crop simulating zoom: scale 0.85–1.0 of original."""
        w, h   = img.size
        scale  = random.uniform(0.85, 1.0)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        left   = random.randint(0, w - new_w)
        top    = random.randint(0, h - new_h)
        img    = img.crop((left, top, left + new_w, top + new_h))
        return img.resize((w, h), Image.BILINEAR)

    @staticmethod
    def _random_contrast(img: Image.Image) -> Image.Image:
        factor = random.uniform(0.8, 1.2)
        return transforms.functional.adjust_contrast(img, factor)

    @staticmethod
    def _gaussian_noise(img: Image.Image) -> Image.Image:
        arr  = np.array(img, dtype=np.float32) / 255.0
        noise = np.random.normal(0, 0.05, arr.shape).astype(np.float32)
        arr  = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _apply_aug(self, img: Image.Image) -> Image.Image:
        """Apply each augmentation independently with probability AUG_PROB."""
        aug_fns = [
            self._random_flip,
            self._random_rotation,
            self._random_zoom,
            self._random_contrast,
            self._gaussian_noise,
        ]
        for fn in aug_fns:
            if random.random() < AUG_PROB:
                img = fn(img)
        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = Image.open(row["path"]).convert("RGB")   # X-rays → RGB
        if self.augment:
            img = self._apply_aug(img)
        tensor = self.to_tensor(img)
        return tensor, int(row["label"])


# ── 4. Collect file paths ─────────────────────────────────────────────────────
rows = []
for c in range(NUM_CLASSES):
    folder = DATA_ROOT / f"KL{c}"
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing: {folder}")
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            rows.append({"path": str(p), "label": c})

df = pd.DataFrame(rows)
print(f"\nTotal images: {len(df)}")
print(df["label"].value_counts().sort_index().rename(
    {i: f"KL{i}" for i in range(NUM_CLASSES)}
))

train_df, val_df = train_test_split(
    df, test_size=VAL_FRAC, stratify=df["label"], random_state=SEED
)
print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")

# ── 5. DataLoaders ────────────────────────────────────────────────────────────
train_ds = KneeDataset(train_df, augment=True)
val_ds   = KneeDataset(val_df,   augment=False)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True,
)

# ── 6. Model — DenseNet-169 exactly as Thomas et al. ─────────────────────────
model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)
print(f"\nModel: DenseNet-169 | Classifier input features: {in_features}")

# ── 7. Loss, optimizer, scheduler ────────────────────────────────────────────
# Thomas: standard CrossEntropyLoss, no class weighting
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# Mixed precision scaler for T4 GPU
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# ── 8. Training loop ──────────────────────────────────────────────────────────
best_qwk        = -1.0
patience_counter = 0
history          = []

print("\n" + "="*65)
print("Training — Thomas et al. 2020 replication")
print("="*65)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    train_loss = train_correct = train_total = 0

    for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} [train]", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss    += loss.item() * labels.size(0)
        train_correct += (logits.argmax(1) == labels).sum().item()
        train_total   += labels.size(0)

    # ── Validate ─────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Ep {epoch}/{EPOCHS} [val]  ", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            val_loss    += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ── Metrics ──────────────────────────────────────────────────────────────
    t_loss = train_loss / train_total
    t_acc  = train_correct / train_total
    v_loss = val_loss / len(val_ds)
    v_acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    qwk    = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    # Per-class accuracy
    per_class = {}
    for c in range(NUM_CLASSES):
        idx = [i for i, l in enumerate(all_labels) if l == c]
        if idx:
            per_class[f"KL{c}"] = sum(all_preds[i] == c for i in idx) / len(idx)

    scheduler.step(v_loss)
    elapsed = time.time() - t0

    print(
        f"Ep {epoch:02d} | "
        f"TrLoss={t_loss:.4f} TrAcc={t_acc:.4f} | "
        f"ValLoss={v_loss:.4f} ValAcc={v_acc:.4f} | "
        f"QWK={qwk:.4f} | "
        f"PerClass={per_class} | "
        f"{elapsed:.0f}s"
    )

    history.append(dict(epoch=epoch, t_loss=t_loss, t_acc=t_acc,
                        v_loss=v_loss, v_acc=v_acc, qwk=qwk))

    # ── Checkpoint ───────────────────────────────────────────────────────────
    if qwk > best_qwk:
        best_qwk = qwk
        torch.save(model.state_dict(), BEST_PATH)
        print(f"  ✔ Saved best model  QWK={best_qwk:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no QWK improvement for {PATIENCE} epochs)")
            break

print(f"\nBest QWK: {best_qwk:.4f}")

# ── 9. Final evaluation ───────────────────────────────────────────────────────
model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

final_qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
print(f"\nFinal QWK (best checkpoint): {final_qwk:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

# ── 10. Confusion matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")
fig.colorbar(im, ax=ax)
ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
       xlabel="Predicted", ylabel="True",
       title=f"Confusion Matrix — Thomas et al. replication\nQWK={final_qwk:.4f}")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
fig.savefig(CM_PATH, dpi=150)
plt.show()
print(f"Confusion matrix saved to {CM_PATH}")

# ── 11. Training curves ───────────────────────────────────────────────────────
hist_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(hist_df["epoch"], hist_df["t_loss"], label="Train")
axes[0].plot(hist_df["epoch"], hist_df["v_loss"], label="Val")
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(hist_df["epoch"], hist_df["t_acc"], label="Train")
axes[1].plot(hist_df["epoch"], hist_df["v_acc"], label="Val")
axes[1].set_title("Accuracy"); axes[1].legend()
axes[2].plot(hist_df["epoch"], hist_df["qwk"], color="green")
axes[2].set_title("Quadratic Weighted Kappa")
fig.tight_layout()
plt.savefig("training_curves_thomas_colab.png", dpi=150)
plt.show()
print("Training curves saved.")

# ── 12. Download model ────────────────────────────────────────────────────────
try:
    from google.colab import files
    files.download(BEST_PATH)
    print(f"Downloaded {BEST_PATH}")
except Exception:
    print(f"Model saved at {BEST_PATH} — download manually if needed.")