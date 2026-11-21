"""
Grad-CAM heatmaps (Fig. 8): which parts of the SEM image drove the prediction?

Works for both strategies:
  - FS: one model  -> one heatmap
  - CV: 4 fold models -> the 4 heatmaps (all for the SAME ensemble-predicted
        class) are averaged into one heatmap

Two modes:
  1. --images_dir: a folder of hand-picked images (Fig. 8 "easy" and "hard"
     cases), one sub-folder per true class:
         gradcam_images/
         ├── 0/  ├── 1/  ├── 2/  ├── 3/  └── 4/
  2. --data_dir: all 42 held-out TEST images (used for Tables IV and V).
     Heatmaps are saved as test_<index>_class_<true class>.npy

Output:
    figures/gradcam/<Label>/class_<c>/<image>_overlay.png/.svg   (image + heatmap)
    figures/gradcam/<Label>/class_<c>/<image>_heatmap.png/.svg   (heatmap only)
    figures/gradcam/<Label>/summary.csv                          (prediction + confidence)
    outputs/<Model>_<cv|fs>/gradcam/<image>.npy                  (raw heatmaps, for analysis)

Usage:
    python -m src.gradcam --model EfficientNetB0 --strategy cv --images_dir gradcam_images
    python -m src.gradcam --model ResNet152V2   --strategy fs --data_dir data
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.config import CV_FOLDS, IMG_EXTENSIONS, IMG_SIZE
from src.models import SHORT_NAMES, build_model, get_preprocess_fn


# ------------------------------------------------------------ model setup ---

def load_trained_models(model_name, strategy, out_dir, num_classes=5):
    """Rebuild the architecture and load the saved weights (4 folds or 1 model)."""
    run_dir = Path(out_dir) / f"{model_name}_{strategy}"
    if strategy == "cv":
        weight_files = [run_dir / f"fold_{k}.weights.h5" for k in range(1, CV_FOLDS + 1)]
    else:
        weight_files = [run_dir / "model.weights.h5"]

    models = []
    for wf in weight_files:
        model = build_model(model_name, num_classes)
        model.load_weights(wf)
        models.append(model)
        print(f"✅ Loaded {wf}")
    return models


def find_last_conv_layer(model):
    """Grad-CAM uses the LAST convolutional layer (most semantic, still spatial)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_model(model):
    """Return a model that outputs (last conv feature maps, class logits).

    We take gradients of the LOGITS (scores before softmax), because softmax
    saturates and can make the gradients vanishingly small.
    Note: this switches the model's final activation to linear.
    """
    model.layers[-1].activation = tf.keras.activations.linear
    # Each fold model has its own layers, so find the conv layer in THIS model
    conv_layer = find_last_conv_layer(model)
    return tf.keras.Model(model.inputs, [conv_layer.output, model.output])


# --------------------------------------------------------------- Grad-CAM ---

def gradcam_heatmap(grad_model, x, class_idx):
    """Standard Grad-CAM for one image and one target class.

    1. Gradient of the class score w.r.t. each feature map.
    2. Average gradient per channel = importance weight of that channel.
    3. Weighted sum of the feature maps -> keep positive values (ReLU).
    """
    with tf.GradientTape() as tape:
        conv_maps, logits = grad_model(x, training=False)
        class_score = logits[:, class_idx]
    grads = tape.gradient(class_score, conv_maps)

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))              # (channels,)
    heatmap = tf.reduce_sum(conv_maps[0] * weights, axis=-1)     # (h, w)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)          # scale to [0, 1]

    # Upsample from the small conv grid (e.g. 6x6) to the image size
    return tf.image.resize(heatmap[..., None], IMG_SIZE).numpy().squeeze()


def explain_image(img, grad_models, preprocess_fn):
    """Predict the class (ensemble if several models) and return its Grad-CAM.

    Returns: heatmap (H, W) in [0, 1], predicted class, confidence.
    """
    x = preprocess_fn(img.astype("float32").copy())[None, ...]   # (1, H, W, 3)

    # Ensemble prediction: average the softmax probabilities of all models
    probs = np.mean([tf.nn.softmax(gm(x, training=False)[1]).numpy()[0] for gm in grad_models], axis=0)
    pred = int(np.argmax(probs))

    # Every model explains the SAME (ensemble) predicted class, then average
    heatmap = np.mean([gradcam_heatmap(gm, x, pred) for gm in grad_models], axis=0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap, pred, float(probs[pred])


# ----------------------------------------------------------------- saving ---

def save_figures(img, heatmap, save_dir, name, dpi=600):
    """Save the overlay (SEM image + 40% heatmap) and the heatmap alone."""
    save_dir.mkdir(parents=True, exist_ok=True)
    figsize = (img.shape[1] / 100, img.shape[0] / 100)

    for suffix, show_img, alpha in [("overlay", True, 0.4), ("heatmap", False, 1.0)]:
        fig, ax = plt.subplots(figsize=figsize)
        if show_img:
            ax.imshow(img.astype("uint8"))
        ax.imshow(heatmap, cmap="jet", alpha=alpha)
        ax.axis("off")
        for ext in ("png", "svg"):
            fig.savefig(save_dir / f"{name}_{suffix}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


# ------------------------------------------------------------------- main ---

def iter_images(images_dir=None, data_dir=None):
    """Yield (name, true_class, image) from a hand-picked folder OR the test set."""
    if images_dir:
        for class_dir in sorted(p for p in Path(images_dir).iterdir() if p.is_dir()):
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in IMG_EXTENSIONS:
                    img = np.array(Image.open(img_path).convert("RGB").resize(IMG_SIZE))
                    yield img_path.stem, class_dir.name, img
    else:
        from src.data import load_images_and_labels, split_test_set
        images, labels, _ = load_images_and_labels(data_dir)
        _, x_test, _, y_test = split_test_set(images, labels)   # same 42 test images as always
        for i, (img, y) in enumerate(zip(x_test, y_test)):
            yield f"test_{i:03d}", str(y), img


def main(model_name, strategy, images_dir, data_dir, out_dir, fig_dir):
    models = load_trained_models(model_name, strategy, out_dir)
    grad_models = [make_gradcam_model(m) for m in models]
    preprocess_fn = get_preprocess_fn(model_name)

    label = f"{SHORT_NAMES.get(model_name, model_name)}-{strategy.upper()}"   # e.g. "EffB0-CV"
    fig_root = Path(fig_dir) / "gradcam" / label
    npy_root = Path(out_dir) / f"{model_name}_{strategy}" / "gradcam"

    test_mode = images_dir is None
    if test_mode:                       # keep test-set results apart from Fig. 8 cases
        fig_root, npy_root = fig_root / "test_set", npy_root / "test_set"
    npy_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, true_class, img in iter_images(images_dir, data_dir):
        heatmap, pred, conf = explain_image(img, grad_models, preprocess_fn)

        np.save(npy_root / f"{name}_class_{true_class}.npy", heatmap)
        if not test_mode:               # figures only for the hand-picked cases
            save_figures(img, heatmap, fig_root / f"class_{true_class}", name)
        rows.append({"image": name, "true_class": true_class,
                     "predicted_class": pred, "confidence": round(conf, 4)})
        print(f"{name}: true {true_class} | predicted {pred} ({conf:.2f})")

    fig_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(fig_root / "summary.csv", index=False)
    print(f"\n✅ Grad-CAM figures saved to {fig_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM for a trained model")
    parser.add_argument("--model", required=True, help="e.g. EfficientNetB0")
    parser.add_argument("--strategy", choices=["cv", "fs"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images_dir", help="hand-picked images in class sub-folders 0–4 (Fig. 8)")
    source.add_argument("--data_dir", help="full dataset -> explain the 42 test images (Tables IV–V)")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--fig_dir", default="figures")
    args = parser.parse_args()
    main(args.model, args.strategy, args.images_dir, args.data_dir, args.out_dir, args.fig_dir)
