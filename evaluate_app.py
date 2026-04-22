"""
Knee OA Classifier — Evaluation Script
Evaluates all three models individually and as ensembles, with and without TTA.
Runs on CPU.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torchvision import models, transforms


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = Path(r"C:\Users\jkall\Downloads\KL_Sorted_Split")

NUM_CLASSES = 5
SEED = 42
VAL_SPLIT = 0.20

DENSENET_INPUT_SIZE = 299
EFFB5_INPUT_SIZE = 456
EFFV2S_INPUT_SIZE = 384

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cpu")


def build_validation_set() -> tuple[list[Path], list[int]]:
    """
    Collect all images from KL_Sorted_Split/{KL0..KL4}, perform an 80/20
    stratified split (SEED=42), and return the validation paths + labels.
    """
    all_paths: list[Path] = []
    all_labels: list[int] = []
    for grade in range(NUM_CLASSES):
        class_dir = DATA_DIR / f"KL{grade}"
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected directory: {class_dir}")
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                all_paths.append(img_file)
                all_labels.append(grade)

    _, val_paths, _, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=VAL_SPLIT,
        stratify=all_labels,
        random_state=SEED,
    )
    print(f"Validation set: {len(val_paths)} images")
    for g in range(NUM_CLASSES):
        n = val_labels.count(g)
        print(f"  KL{g}: {n}")
    return val_paths, val_labels



def make_preprocess(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def pil_to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def tta_views(pil_rgb: Image.Image) -> list[Image.Image]:
    """Return 5 augmented views of the image for TTA."""
    w, h = pil_rgb.size
    crop_w = max(1, int(round(w * 0.9)))
    crop_h = max(1, int(round(h * 0.9)))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    center_crop = pil_rgb.crop(
        (left, top, left + crop_w, top + crop_h)
    ).resize((w, h), Image.BILINEAR)
    fill = (0, 0, 0)
    return [
        pil_rgb,
        pil_rgb.transpose(Image.FLIP_LEFT_RIGHT),
        pil_rgb.rotate(10, fillcolor=fill, expand=False),
        pil_rgb.rotate(-10, fillcolor=fill, expand=False),
        center_crop,
    ]



def _load_state_dict_flexible(model: nn.Module, ckpt_path: Path) -> None:
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)


def _resolve(names: list[str]) -> Path:
    for name in names:
        p = SCRIPT_DIR / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"None of the checkpoint names found: {names}")


def build_densenet169() -> nn.Module:
    m = models.densenet169(weights=None)
    m.classifier = nn.Linear(1664, NUM_CLASSES)
    _load_state_dict_flexible(
        m,
        _resolve(
            [
                "best_model_densenet_colab.pth",
                "best_model_densenet_colab (3).pth",
                "best_model_densenet_colab (1).pth",
            ]
        ),
    )
    m.eval()
    return m


def build_efficientnet_b5() -> nn.Module:
    m = timm.create_model("efficientnet_b5", pretrained=False, num_classes=NUM_CLASSES)
    _load_state_dict_flexible(
        m, _resolve(["best_efficientnet_b5.pth", "best_efficientnet_b5 (1).pth"])
    )
    m.eval()
    return m


def build_efficientnetv2_s() -> nn.Module:
    m = timm.create_model(
        "tf_efficientnetv2_s", pretrained=False, num_classes=NUM_CLASSES
    )
    _load_state_dict_flexible(m, _resolve(["best_efficientnet_v2s.pth"]))
    m.eval()
    return m


@torch.no_grad()
def infer_single(model: nn.Module, preprocess: transforms.Compose, pil_rgb: Image.Image) -> np.ndarray:
    """Return softmax probabilities (shape 5,) for one image, no TTA."""
    t = preprocess(pil_rgb).unsqueeze(0)
    logits = model(t)
    return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()


@torch.no_grad()
def infer_tta(model: nn.Module, preprocess: transforms.Compose, pil_rgb: Image.Image) -> np.ndarray:
    """Return mean softmax probabilities (shape 5,) over 5 TTA views."""
    views = tta_views(pil_rgb)
    x = torch.stack([preprocess(v) for v in views], dim=0)
    logits = model(x)
    return torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()



def run_inference(
    model: nn.Module,
    preprocess: transforms.Compose,
    paths: list[Path],
    use_tta: bool,
    label: str,
) -> np.ndarray:
    """
    Run model on every validation image. Returns ndarray of shape (N, 5).
    """
    n = len(paths)
    probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    fn = infer_tta if use_tta else infer_single
    t0 = time.time()
    for i, p in enumerate(paths):
        if (i + 1) % 200 == 0 or i == n - 1:
            elapsed = time.time() - t0
            pct = 100.0 * (i + 1) / n
            print(f"  [{label}] {i+1}/{n} ({pct:.1f}%) — {elapsed:.0f}s elapsed")
        img = pil_to_rgb(Image.open(p))
        probs[i] = fn(model, preprocess, img)
    return probs



def compute_metrics(true_labels: list[int], preds: list[int]) -> dict:
    acc = accuracy_score(true_labels, preds)
    qwk = cohen_kappa_score(true_labels, preds, weights="quadratic")
    return {"accuracy": acc, "qwk": qwk}


def print_report(name: str, true_labels: list[int], preds: list[int]) -> dict:
    m = compute_metrics(true_labels, preds)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  QWK      : {m['qwk']:.4f}")
    print(f"  Accuracy : {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)")
    print()
    print(
        classification_report(
            true_labels,
            preds,
            target_names=[f"KL{g}" for g in range(NUM_CLASSES)],
            digits=4,
        )
    )
    return m


def save_confusion_matrix(name: str, true_labels: list[int], preds: list[int]) -> None:
    cm = confusion_matrix(true_labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"KL{g}" for g in range(NUM_CLASSES)])
    ax.set_yticklabels([f"KL{g}" for g in range(NUM_CLASSES)])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    safe_name = name.replace(" ", "_").replace("/", "_").replace("—", "-")
    ax.set_title(f"{name}\nQWK={cohen_kappa_score(true_labels, preds, weights='quadratic'):.4f}  "
                 f"Acc={accuracy_score(true_labels, preds)*100:.2f}%",
                 fontsize=11)
    fig.tight_layout()
    out = SCRIPT_DIR / f"cm_{safe_name}.png"
    fig.savefig(str(out), dpi=120)
    plt.close(fig)
    print(f"  Saved confusion matrix → {out.name}")



def main() -> None:
    print("=" * 60)
    print("  Knee OA Classifier — Full Evaluation")
    print("=" * 60)

    
    val_paths, val_labels = build_validation_set()

    # ---- Load ensemble weights ----
    weights_path = SCRIPT_DIR / "optimized_ensemble_weights_tta.npy"
    raw_weights = np.load(str(weights_path)).astype(np.float64)  # (3, 5)
    if raw_weights.shape != (3, NUM_CLASSES):
        raise RuntimeError(f"Expected weights shape (3,5), got {raw_weights.shape}")
    # Clip negatives, column-normalise so each class sums to 1 across models
    raw_weights = np.clip(raw_weights, 0.0, None)
    CLASS_WEIGHT_MATRIX = raw_weights / (
        raw_weights.sum(axis=0, keepdims=True) + 1e-12
    )
    print(f"Loaded optimized weights:\n{CLASS_WEIGHT_MATRIX}\n")

    # ---- Load models ----
    print("Loading models…")
    densenet = build_densenet169()
    print("  DenseNet-169 loaded")
    effb5 = build_efficientnet_b5()
    print("  EfficientNet-B5 loaded")
    effv2s = build_efficientnetv2_s()
    print("  EfficientNet-V2-S loaded")
    print()

    pre_dn = make_preprocess(DENSENET_INPUT_SIZE)
    pre_b5 = make_preprocess(EFFB5_INPUT_SIZE)
    pre_v2 = make_preprocess(EFFV2S_INPUT_SIZE)

    print("Running inference (this may take a while on CPU)…\n")

    print(">>> DenseNet-169, no TTA")
    dn_noTTA  = run_inference(densenet, pre_dn, val_paths, False, "DenseNet-169 no-TTA")
    print(">>> DenseNet-169, TTA")
    dn_TTA    = run_inference(densenet, pre_dn, val_paths, True,  "DenseNet-169 TTA")

    print(">>> EfficientNet-B5, no TTA")
    b5_noTTA  = run_inference(effb5,   pre_b5, val_paths, False, "EffB5 no-TTA")
    print(">>> EfficientNet-B5, TTA")
    b5_TTA    = run_inference(effb5,   pre_b5, val_paths, True,  "EffB5 TTA")

    print(">>> EfficientNet-V2-S, no TTA")
    v2_noTTA  = run_inference(effv2s,  pre_v2, val_paths, False, "EffV2S no-TTA")
    print(">>> EfficientNet-V2-S, TTA")
    v2_TTA    = run_inference(effv2s,  pre_v2, val_paths, True,  "EffV2S TTA")

    print("\nInference complete.\n")

    avg_noTTA = (dn_noTTA + b5_noTTA + v2_noTTA) / 3.0
    avg_TTA   = (dn_TTA   + b5_TTA   + v2_TTA)   / 3.0

    # Optimized weighted ensemble — CLASS_WEIGHT_MATRIX is (3,5)
    # Each model's probs (N,5) is scaled by its per-class weight, then summed
    def weighted_ensemble(m0, m1, m2):
        # CLASS_WEIGHT_MATRIX[i] is shape (5,) — broadcast over N samples
        combined = (
            m0 * CLASS_WEIGHT_MATRIX[0]
            + m1 * CLASS_WEIGHT_MATRIX[1]
            + m2 * CLASS_WEIGHT_MATRIX[2]
        )
        # row-normalise
        combined /= combined.sum(axis=1, keepdims=True) + 1e-12
        return combined

    opt_noTTA = weighted_ensemble(dn_noTTA, b5_noTTA, v2_noTTA)
    opt_TTA   = weighted_ensemble(dn_TTA,   b5_TTA,   v2_TTA)

    # ---- All 10 methods ----
    methods: list[tuple[str, np.ndarray]] = [
        ("DenseNet-169 — no TTA",                     dn_noTTA),
        ("EfficientNet-B5 — no TTA",                  b5_noTTA),
        ("EfficientNet-V2-S — no TTA",                v2_noTTA),
        ("DenseNet-169 — TTA",                        dn_TTA),
        ("EfficientNet-B5 — TTA",                     b5_TTA),
        ("EfficientNet-V2-S — TTA",                   v2_TTA),
        ("Simple Average Ensemble — no TTA",          avg_noTTA),
        ("Simple Average Ensemble — TTA",             avg_TTA),
        ("Optimized Weighted Ensemble — no TTA",      opt_noTTA),
        ("Optimized Weighted Ensemble — TTA (FINAL)", opt_TTA),
    ]

    summary: list[dict] = []
    for name, probs in methods:
        preds = np.argmax(probs, axis=1).tolist()
        m = print_report(name, val_labels, preds)
        save_confusion_matrix(name, val_labels, preds)
        summary.append({"name": name, **m})

    # ---- Final summary table ----
    print("\n")
    print("=" * 72)
    print("  FINAL SUMMARY — All 10 Methods")
    print("=" * 72)
    header = f"{'#':>2}  {'Method':<46}  {'QWK':>7}  {'Accuracy':>9}"
    print(header)
    print("-" * 72)
    for i, row in enumerate(summary, 1):
        marker = " *" if i == len(summary) else "  "
        print(
            f"{i:>2}{marker}{'':0}  {row['name']:<46}  "
            f"{row['qwk']:>7.4f}  {row['accuracy']*100:>8.2f}%"
        )
    print("-" * 72)
    print("  * = FINAL SYSTEM")
    print()

    best = max(summary, key=lambda r: r["qwk"])
    print(f"  Best QWK      : {best['name']}")
    print(f"    QWK      = {best['qwk']:.4f}")
    print(f"    Accuracy = {best['accuracy']*100:.2f}%")
    final = summary[-1]
    print()
    print(f"  Final System  : {final['name']}")
    print(f"    QWK      = {final['qwk']:.4f}")
    print(f"    Accuracy = {final['accuracy']*100:.2f}%")
    print()


if __name__ == "__main__":
    main()
