"""
EfficientNet-V2-S Training Script
Pi et al. 2023 — second best model in ensemble
Kaggle GPU optimized
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "timm", "-q"])

import os, random, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Config
DATA_ROOT   = Path("/kaggle/input/datasets/abrahamkallivayalil/some-knees/KL_Sorted_Split")
BEST_PATH   = "/kaggle/working/best_efficientnet_v2s.pth"
NUM_CLASSES = 5
IMG_SIZE    = 384
BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 1e-4
VAL_FRAC    = 0.2
SEED        = 42
AUG_PROB    = 0.80
PATIENCE    = 10
CLASS_NAMES = [f"KL{i}" for i in range(NUM_CLASSES)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(SEED)

class KneeDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.to_tensor = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    @staticmethod
    def _flip(img): return transforms.functional.hflip(img)

    @staticmethod
    def _rotate(img):
        return transforms.functional.rotate(img, random.uniform(-20, 20), fill=0)

    @staticmethod
    def _zoom(img):
        w, h = img.size
        s = random.uniform(0.85, 1.0)
        nw, nh = int(w*s), int(h*s)
        l, t = random.randint(0, w-nw), random.randint(0, h-nh)
        return img.crop((l, t, l+nw, t+nh)).resize((w, h), Image.BILINEAR)

    @staticmethod
    def _contrast(img):
        return transforms.functional.adjust_contrast(img, random.uniform(0.8, 1.2))

    @staticmethod
    def _noise(img):
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.clip(arr + np.random.normal(0, 0.05, arr.shape), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _aug(self, img):
        for fn in [self._flip, self._rotate, self._zoom, self._contrast, self._noise]:
            if random.random() < AUG_PROB:
                img = fn(img)
        return img

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.augment: img = self._aug(img)
        return self.to_tensor(img), int(row["label"])

# Collect images
rows = []
for c in range(NUM_CLASSES):
    folder = DATA_ROOT / f"KL{c}"
    if not folder.is_dir(): raise FileNotFoundError(f"Missing: {folder}")
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg",".jpeg",".png"}:
            rows.append({"path": str(p), "label": c})

df = pd.DataFrame(rows)
print(f"\nTotal images: {len(df)}")
print(df["label"].value_counts().sort_index().rename({i: f"KL{i}" for i in range(NUM_CLASSES)}))

train_df, val_df = train_test_split(df, test_size=VAL_FRAC, stratify=df["label"], random_state=SEED)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

train_loader = DataLoader(KneeDataset(train_df, augment=True), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(KneeDataset(val_df, augment=False), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

# Model — correct timm name for pretrained EfficientNet-V2-S
model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)
print(f"\nModel: EfficientNet-V2-S | Input: {IMG_SIZE}x{IMG_SIZE}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

best_qwk = -1.0
patience_counter = 0
history = []

print("\n" + "="*60)
print("Training EfficientNet-V2-S")
print("="*60)

for epoch in range(1, EPOCHS+1):
    t0 = time.time()

    # Train
    model.train()
    tr_loss = tr_correct = tr_total = 0
    for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} [train]", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss    += loss.item() * labels.size(0)
        tr_correct += (logits.argmax(1) == labels).sum().item()
        tr_total   += labels.size(0)

    # Validate
    model.eval()
    val_loss = 0
    preds, labs = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Ep {epoch}/{EPOCHS} [val]  ", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            val_loss += loss.item() * labels.size(0)
            preds.extend(logits.argmax(1).cpu().numpy())
            labs.extend(labels.cpu().numpy())

    scheduler.step(val_loss / len(val_df))

    t_loss = tr_loss / tr_total
    t_acc  = tr_correct / tr_total
    v_loss = val_loss / len(val_df)
    v_acc  = sum(p==l for p,l in zip(preds,labs)) / len(labs)
    qwk    = cohen_kappa_score(labs, preds, weights="quadratic")

    per_class = {f"KL{c}": round(sum(preds[i]==c for i,l in enumerate(labs) if l==c) /
                 max(1, sum(1 for l in labs if l==c)), 3) for c in range(NUM_CLASSES)}

    print(f"Ep {epoch:02d} | TrLoss={t_loss:.4f} TrAcc={t_acc:.4f} | "
          f"ValLoss={v_loss:.4f} ValAcc={v_acc:.4f} | QWK={qwk:.4f} | "
          f"PerClass={per_class} | {int(time.time()-t0)}s")

    history.append(dict(epoch=epoch, t_loss=t_loss, t_acc=t_acc,
                        v_loss=v_loss, v_acc=v_acc, qwk=qwk))

    if qwk > best_qwk:
        best_qwk = qwk
        torch.save(model.state_dict(), BEST_PATH)
        print(f"  ✔ Saved QWK={best_qwk:.4f} → {BEST_PATH}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Summary every 5 epochs
    if epoch % 5 == 0:
        print(f"\n{'='*40}")
        print(f"SUMMARY after epoch {epoch}")
        print(f"Best QWK so far: {best_qwk:.4f}")
        best_ep = max(history, key=lambda x: x['qwk'])
        print(f"Best epoch: {best_ep['epoch']} | QWK={best_ep['qwk']:.4f}")
        print(f"{'='*40}\n")

print(f"\nBest QWK: {best_qwk:.4f}")
print(f"Model saved at: {BEST_PATH}")

# Verify file exists
if os.path.exists(BEST_PATH):
    size_mb = os.path.getsize(BEST_PATH) / 1024 / 1024
    print(f"File confirmed at {BEST_PATH} ({size_mb:.1f} MB)")
else:
    print("WARNING: File not found!")

# Final evaluation
model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
model.eval()
preds, labs = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        preds.extend(model(imgs).argmax(1).cpu().numpy())
        labs.extend(labels.numpy())

final_qwk = cohen_kappa_score(labs, preds, weights="quadratic")
print(f"\nFinal QWK: {final_qwk:.4f}")
print("\nClassification Report:")
print(classification_report(labs, preds, target_names=CLASS_NAMES, digits=4))

# Confusion matrix
cm = confusion_matrix(labs, preds)
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(cm, cmap="Blues")
fig.colorbar(im, ax=ax)
ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
       xlabel="Predicted", ylabel="True",
       title=f"EfficientNet-V2-S | QWK={final_qwk:.4f}")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                color="white" if cm[i,j] > thresh else "black")
fig.tight_layout()
fig.savefig("/kaggle/working/confusion_matrix_efficientnet_v2s.png", dpi=150)
plt.show()
print("Done! Download model from Kaggle Output panel.")