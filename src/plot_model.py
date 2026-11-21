"""
Per-model figures (Fig. 4 training curves, Fig. 6 confusion matrices, plus ROC
and confidence histograms) for one trained model.

Reads what train_cv.py / train_fs.py saved in outputs/<Model>_<cv|fs>/:
    test_probs.npy, y_test.npy, histories.json (CV) or history.json (FS)

Usage:
    python -m src.plot_model --model EfficientNetB0 --strategy cv
    python -m src.plot_model --model ResNet152V2   --strategy fs
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from src.models import SHORT_NAMES


# ----------------------------------------------------------------- style ---

def set_paper_style():
    """Same journal-style fonts for every figure (editable text in SVG/PDF)."""
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "serif",
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })


def save(fig, save_dir, name, dpi=600):
    """Save as high-resolution PNG + vector SVG."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(save_dir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"✅ Saved {save_dir / name}.png/.svg")


# ------------------------------------------------------------ the plots ---

def plot_training_history(histories, title):
    """Fig. 4: accuracy (top row) and loss (bottom row).

    `histories` is a list of Keras history dicts: 4 folds for CV, 1 for FS.
    """
    n = len(histories)
    fig, axs = plt.subplots(2, n, figsize=(3 * n if n > 1 else 4, 5),
                            squeeze=False, constrained_layout=True)

    for i, h in enumerate(histories):
        epochs = range(1, len(h["loss"]) + 1)

        # Top: accuracy in %
        axs[0, i].plot(epochs, np.array(h["accuracy"]) * 100, "-", label="Train")
        axs[0, i].plot(epochs, np.array(h["val_accuracy"]) * 100, "--", label="Validation")
        # Bottom: loss
        axs[1, i].plot(epochs, h["loss"], "-", label="Train")
        axs[1, i].plot(epochs, h["val_loss"], "--", label="Validation")

        if n > 1:
            axs[0, i].set_title(f"Fold {i + 1}")
        axs[1, i].set_xlabel("Epoch")
        for ax in axs[:, i]:
            ax.set_xlim(1, len(h["loss"]))

    axs[0, 0].set_ylabel("Accuracy (%)")
    axs[1, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    fig.suptitle(title)
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """Fig. 6: row-normalised confusion matrix (% of each true class)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Percentage (%)"}, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    return fig


def plot_roc(y_true, y_probs, class_names, title):
    """One-vs-rest ROC curve per class + macro-average."""
    n_classes = len(class_names)
    y_onehot = label_binarize(y_true, classes=range(n_classes))
    colors = sns.color_palette("colorblind", n_classes + 1)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    grid = np.linspace(0, 1, 200)   # common FPR grid for the macro-average
    mean_tpr = np.zeros_like(grid)

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs[:, i])
        mean_tpr += np.interp(grid, fpr, tpr) / n_classes
        ax.plot(fpr, tpr, color=colors[i], lw=1.5, label=f"Class {name} (AUC={auc(fpr, tpr):.2f})")

    ax.plot(grid, mean_tpr, ":", color=colors[-1], lw=2,
            label=f"Macro-average (AUC={auc(grid, mean_tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1, label="Random")

    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01),
           xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    return fig


def plot_confidence(y_true, y_probs, title, bins=40):
    """How confident is the model when it is right vs. wrong?

    Confidence = highest predicted probability for each test image.
    Dashed lines show the median confidence of each group.
    """
    df = pd.DataFrame({
        "confidence": y_probs.max(axis=1),
        "correct": np.argmax(y_probs, axis=1) == y_true,
    })
    colors = sns.color_palette("colorblind", 2)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for is_correct, color, label in [(True, colors[0], "Correct"), (False, colors[1], "Incorrect")]:
        group = df[df["correct"] == is_correct]["confidence"]
        if group.empty:
            continue
        sns.histplot(group, bins=np.linspace(0, 1, bins + 1), color=color, alpha=0.5,
                     label=f"{label} (n={len(group)})", ax=ax)
        if len(group) > 1:
            sns.kdeplot(group, color=color, fill=True, alpha=0.25, ax=ax, warn_singular=False)
        ax.axvline(group.median(), color=color, linestyle="--", lw=1)

    # Zoom in on the range where predictions actually fall
    lo, hi = df["confidence"].min(), df["confidence"].max()
    ax.set_xlim(max(0, lo - 0.05), min(1, hi + 0.05))
    ax.set(xlabel="Confidence", ylabel="Frequency", title=title)
    ax.legend(loc="upper right")
    return fig


# ------------------------------------------------------------------ main ---

def main(model_name, strategy, out_dir, fig_dir):
    run_dir = Path(out_dir) / f"{model_name}_{strategy}"
    y_test = np.load(run_dir / "y_test.npy")
    y_probs = np.load(run_dir / "test_probs.npy")
    y_pred = np.argmax(y_probs, axis=1)
    class_names = [str(c) for c in range(y_probs.shape[1])]

    # CV saves a list of 4 fold histories, FS saves a single dict
    if strategy == "cv":
        histories = json.loads((run_dir / "histories.json").read_text())
    else:
        histories = [json.loads((run_dir / "history.json").read_text())]

    label = f"{SHORT_NAMES.get(model_name, model_name)}-{strategy.upper()}"   # e.g. "EffB0-CV"
    save_dir = Path(fig_dir) / label

    set_paper_style()
    save(plot_training_history(histories, f"Training curves — {label}"), save_dir, "training_curves")
    save(plot_confusion_matrix(y_test, y_pred, class_names, f"Confusion matrix — {label}"), save_dir, "confusion_matrix")
    save(plot_roc(y_test, y_probs, class_names, f"ROC — {label}"), save_dir, "roc")
    save(plot_confidence(y_test, y_probs, f"Confidence — {label}"), save_dir, "confidence_hist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figures for one trained model")
    parser.add_argument("--model", required=True, help="e.g. EfficientNetB0")
    parser.add_argument("--strategy", choices=["cv", "fs"], required=True)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--fig_dir", default="figures")
    args = parser.parse_args()
    main(args.model, args.strategy, args.out_dir, args.fig_dir)
