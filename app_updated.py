from __future__ import annotations

import gdown
from flask import Flask, render_template, request, jsonify
import base64
import io
import os

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)

NUM_CLASSES = 5
DENSENET_INPUT_SIZE = 299
EFFB5_INPUT_SIZE = 456
EFFV2S_INPUT_SIZE = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_LABELS = {
    0: {
        "grade": "Grade 0",
        "severity": "Normal",
        "description": "No radiographic features of osteoarthritis. Normal joint space and bone structure.",
        "color": "#27ae60",
    },
    1: {
        "grade": "Grade 1",
        "severity": "Doubtful",
        "description": "Questionable joint space narrowing. Possible osteophytic lipping. Minimal changes.",
        "color": "#2ecc71",
    },
    2: {
        "grade": "Grade 2",
        "severity": "Mild",
        "description": "Definite osteophytes. Possible joint space narrowing on anteroposterior weight-bearing radiograph.",
        "color": "#f39c12",
    },
    3: {
        "grade": "Grade 3",
        "severity": "Moderate",
        "description": "Multiple osteophytes. Definite joint space narrowing. Sclerosis. Possible attrition of bone.",
        "color": "#e67e22",
    },
    4: {
        "grade": "Grade 4",
        "severity": "Severe",
        "description": "Large osteophytes. Marked joint space narrowing. Severe sclerosis. Definite bony deformity.",
        "color": "#c0392b",
    },
}


def _make_preprocess(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _pil_to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode != "RGB":
        return pil_image.convert("RGB")
    return pil_image


def _tta_views(pil_rgb: Image.Image) -> list[Image.Image]:
    """original, hflip, rot+10, rot-10, center-crop 90% then resize back."""
    w, h = pil_rgb.size
    crop_w = max(1, int(round(w * 0.9)))
    crop_h = max(1, int(round(h * 0.9)))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    center_crop = pil_rgb.crop((left, top, left + crop_w, top + crop_h)).resize(
        (w, h), Image.BILINEAR
    )
    fill = (0, 0, 0)
    return [
        pil_rgb,
        pil_rgb.transpose(Image.FLIP_LEFT_RIGHT),
        pil_rgb.rotate(10, fillcolor=fill, expand=False),
        pil_rgb.rotate(-10, fillcolor=fill, expand=False),
        center_crop,
    ]


def generate_gradcam(
    target_layer: nn.Module,
    model: nn.Module,
    tensor: torch.Tensor,
    predicted_class: int,
) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inp, out):
        activations.append(out)

    def backward_hook(_module, _grad_in, grad_out):
        if grad_out[0] is not None:
            gradients.append(grad_out[0])

    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    tensor = tensor.detach().clone()
    tensor.requires_grad_(True)

    try:
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        score = output[0, predicted_class]
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("GradCAM hooks did not capture activations or gradients.")

        act = activations[0][0]
        grad = gradients[0][0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * act).sum(dim=0))

        cam_np = cam.detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np
    finally:
        h_fwd.remove()
        h_bwd.remove()


def overlay_cam_on_image(cam_np: np.ndarray, original_image: Image.Image) -> str:
    orig = original_image.convert("RGB")
    w, h = orig.size
    heatmap_u8 = (cv2.resize(cam_np, (w, h)) * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    orig_np = np.asarray(orig, dtype=np.float32)
    overlay = 0.4 * heatmap_rgb.astype(np.float32) + 0.6 * orig_np
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode GradCAM overlay as JPEG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


_model_dir = os.path.dirname(os.path.abspath(__file__))

# Google Drive model download setup
MODEL_FILES = {
    "best_efficientnet_v2s.pth": "1iJN0__jm1heUERZty8nHqnCuzuxvQHsg",
    "best_efficientnet_b5.pth": "1XQfVD5PT2f9YY5pm6FRl_BBI5FtA74TC",
    "best_model_thomas_colab.pth": "1SB1xVOwjlQGYqdCqYK3fuvRHNfMLGtZX",
}

def download_models():
    for filename, file_id in MODEL_FILES.items():
        path = os.path.join(_model_dir, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)

# DOWNLOAD MODELS BEFORE LOADING
download_models()


def _resolve_checkpoint(names: str | list[str]) -> str:
    if isinstance(names, str):
        names = [names]
    for name in names:
        path = os.path.join(_model_dir, name)
        if os.path.isfile(path):
            return path
    print("ERROR: No checkpoint found. Tried:")
    for name in names:
        print(f"  {os.path.join(_model_dir, name)}")
    raise SystemExit(1)


def _load_state_dict_flexible(model: nn.Module, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format for {ckpt_path}")
    state = {
        (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)


def _build_densenet169() -> nn.Module:
    m = models.densenet169(weights=None)
    m.classifier = nn.Linear(1664, NUM_CLASSES)
    return m


def _build_efficientnet_b5() -> nn.Module:
    return timm.create_model("efficientnet_b5", pretrained=False, num_classes=NUM_CLASSES)


def _build_tf_efficientnetv2_s() -> nn.Module:
    return timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=NUM_CLASSES)


MODEL_ORDER = ["densenet169", "efficientnet_b5", "efficientnet_v2_s"]
MODEL_SPECS = {
    "densenet169": {
        "builder": _build_densenet169,
        "checkpoint_names": ["best_model_thomas_colab.pth"],
        "input_size": DENSENET_INPUT_SIZE,
    },
    "efficientnet_b5": {
        "builder": _build_efficientnet_b5,
        "checkpoint_names": ["best_efficientnet_b5.pth"],
        "input_size": EFFB5_INPUT_SIZE,
    },
    "efficientnet_v2_s": {
        "builder": _build_tf_efficientnetv2_s,
        "checkpoint_names": ["best_efficientnet_v2s.pth"],
        "input_size": EFFV2S_INPUT_SIZE,
    },
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, "optimized_ensemble_weights_tta.npy")
if not os.path.isfile(weights_path):
    print(f"ERROR: Optimized weights file not found: {weights_path}")
    raise SystemExit(1)
CLASS_WEIGHT_MATRIX = np.load(weights_path).astype(np.float64)
if CLASS_WEIGHT_MATRIX.shape != (3, NUM_CLASSES):
    raise RuntimeError(
        f"optimized_ensemble_weights_tta.npy must have shape (3, {NUM_CLASSES}), got {CLASS_WEIGHT_MATRIX.shape}"
    )
CLASS_WEIGHT_MATRIX = np.clip(CLASS_WEIGHT_MATRIX, 0.0, None)
CLASS_WEIGHT_MATRIX = CLASS_WEIGHT_MATRIX / (CLASS_WEIGHT_MATRIX.sum(axis=0, keepdims=True) + 1e-12)
print(f"Loaded optimized TTA weights: {weights_path}")

MODELS: dict[str, nn.Module] = {}
GRADCAM_TARGET_DENSENET: nn.Module

print("Loading ensemble models...")
for model_name in MODEL_ORDER:
    spec = MODEL_SPECS[model_name]
    model_path = _resolve_checkpoint(spec["checkpoint_names"])
    model_instance = spec["builder"]()
    _load_state_dict_flexible(model_instance, model_path)
    model_instance.eval()
    MODELS[model_name] = model_instance
    if model_name == "densenet169":
        GRADCAM_TARGET_DENSENET = model_instance.features.denseblock4
    print(f"Loaded {model_name} from {model_path}")
print("Ensemble ready: 3-model class-weighted TTA ensemble\n")


@app.route("/")
def home():
    if os.path.exists("templates/index_professional.html"):
        return render_template("index_professional.html")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed_extensions = {"png", "jpg", "jpeg"}
    if not (
        "." in file.filename
        and file.filename.rsplit(".", 1)[1].lower() in allowed_extensions
    ):
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, or JPEG"}), 400

    try:
        raw = file.read()
        image = Image.open(io.BytesIO(raw))
        pil_rgb = _pil_to_rgb(image)
        tta_imgs = _tta_views(pil_rgb)

        probs_by_model: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for model_idx, model_name in enumerate(MODEL_ORDER):
                spec = MODEL_SPECS[model_name]
                preprocess = _make_preprocess(spec["input_size"])
                x_tta = torch.stack([preprocess(v) for v in tta_imgs], dim=0)
                logits = MODELS[model_name](x_tta)
                probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
                probs_by_model[model_name] = probs

        stacked = np.stack([probs_by_model[m] for m in MODEL_ORDER], axis=0)  # (3,5)
        weighted_probs = (CLASS_WEIGHT_MATRIX * stacked).sum(axis=0)
        weighted_probs = weighted_probs / (float(weighted_probs.sum()) + 1e-8)

        predicted_class = int(np.argmax(weighted_probs))
        confidence = float(weighted_probs[predicted_class]) * 100.0
        all_probabilities = {
            i: float(weighted_probs[i]) * 100.0 for i in range(NUM_CLASSES)
        }

        # DenseNet GradCAM on original image only (no TTA).
        x_for_cam = _make_preprocess(DENSENET_INPUT_SIZE)(pil_rgb).unsqueeze(0)
        cam_np = generate_gradcam(
            GRADCAM_TARGET_DENSENET,
            MODELS["densenet169"],
            x_for_cam,
            predicted_class,
        )
        gradcam_b64 = overlay_cam_on_image(cam_np, image)

        result = {
            "success": True,
            "predicted_class": predicted_class,
            "grade": CLASS_LABELS[predicted_class]["grade"],
            "severity": CLASS_LABELS[predicted_class]["severity"],
            "description": CLASS_LABELS[predicted_class]["description"],
            "confidence": round(confidence, 2),
            "color": CLASS_LABELS[predicted_class]["color"],
            "all_probabilities": {
                f"Grade {i}": round(all_probabilities[i], 2) for i in range(NUM_CLASSES)
            },
            "ensemble": True,
            "ensemble_models": list(MODEL_ORDER),
            "gradcam_fusion_weights": {"densenet169": 1.0},
            "model_probabilities": {
                name: {
                    f"Grade {i}": round(float(prob[i]) * 100.0, 2)
                    for i in range(NUM_CLASSES)
                }
                for name, prob in probs_by_model.items()
            },
            "gradcam_image": "data:image/jpeg;base64," + gradcam_b64,
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    print("Knee Osteoarthritis Severity Assessment System")
    print("Models: DenseNet-169 + EfficientNet-B5 + EfficientNet-V2-S (TTA + optimized weights)")
    port = int(os.environ.get("PORT", 10000))
    print(f"http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port)
