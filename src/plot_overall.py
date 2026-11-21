"""
Fig. 3: overall performance of all CNNs under CV and fixed split (FS).

Reads outputs/<Model>_cv/metrics.json and outputs/<Model>_fs/metrics.json
(created by train_cv.py / train_fs.py) and draws two side-by-side heatmaps.
The best model under CV is outlined.

Usage:
    python -m src.plot_overall --out_dir outputs --save figures/fig3_overall.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from src.models import MODEL_REGISTRY, SHORT_NAMES

METRICS = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1"}


def collect_results(out_dir, strategy):
    """Build a table: rows = models (short names), columns = 3 metrics."""
    rows = {}
    for model_name in MODEL_REGISTRY:
        path = Path(out_dir) / f"{model_name}_{strategy}" / "metrics.json"
        if not path.exists():
            print(f"Missing: {path} (skipped)")
            continue
        test = json.loads(path.read_text())["test"]
        rows[SHORT_NAMES[model_name]] = {label: test[key] for key, label in METRICS.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def plot(cv, fs, save_path):
    vmin = min(cv.values.min(), fs.values.min())
    vmax = max(cv.values.max(), fs.values.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, df, title, show_cbar in [(axes[0], cv, "CV", False), (axes[1], fs, "FS", True)]:
        sns.heatmap(df, ax=ax, annot=True, fmt=".2f", cmap="viridis",
                    vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white",
                    cbar=show_cbar, cbar_kws={"label": "Performance"})
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Outline the best CV model (highest macro F1)
    best_row = cv.index.get_loc(cv["Macro-F1"].idxmax())
    axes[0].add_patch(Rectangle((0, best_row), cv.shape[1], 1,
                                fill=False, edgecolor="black", lw=2))

    axes[0].set_ylabel("CNN Architecture", fontsize=12, fontweight="bold")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--save", default="figures/fig3_overall.png")
    args = parser.parse_args()
    plot(collect_results(args.out_dir, "cv"), collect_results(args.out_dir, "fs"), args.save)
