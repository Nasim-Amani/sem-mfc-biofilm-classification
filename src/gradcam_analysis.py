"""
Quantitative Grad-CAM comparison between architectures (Tables IV and V).

Input: the test-set heatmaps saved by
    python -m src.gradcam --model <Model> --strategy <cv|fs> --data_dir data
for every model you want to compare (the same 42 test images for each).

Computed:
  Table IV  spatial entropy of each heatmap: how spread out is the attention?
            (higher = attention spread over more of the image)
  Table V   for every pair of models, on the same image:
            - SSIM: how similar are the two heatmaps overall?
            - IoU of the top-20% regions: do they highlight the same spots?
  Tests     paired Wilcoxon signed-rank tests (same 42 images for each model)

Usage:
    python -m src.gradcam_analysis --runs EfficientNetB0_cv EfficientNetV2B1_cv ResNet152V2_fs
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity

from src.models import SHORT_NAMES


# --------------------------------------------------------------- metrics ---

def spatial_entropy(cam):
    """Shannon entropy of the heatmap treated as a probability map."""
    p = cam.astype(np.float64).ravel()
    if p.sum() == 0:
        return 0.0
    p = p / p.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def top_region_iou(cam1, cam2, top_percent=20):
    """IoU of the most activated `top_percent` % of pixels in each heatmap."""
    mask1 = cam1 >= np.percentile(cam1, 100 - top_percent)
    mask2 = cam2 >= np.percentile(cam2, 100 - top_percent)
    union = np.logical_or(mask1, mask2).sum()
    return float(np.logical_and(mask1, mask2).sum() / union) if union else 0.0


# ----------------------------------------------------------------- loading ---

def load_heatmaps(out_dir, run):
    """Return {image_name: heatmap} for one run, e.g. 'EfficientNetB0_cv'."""
    folder = Path(out_dir) / run / "gradcam" / "test_set"
    cams = {f.stem: np.load(f) for f in sorted(folder.glob("*.npy"))}
    if not cams:
        raise FileNotFoundError(f"No heatmaps in {folder}. Run src.gradcam with --data_dir first.")
    return cams


def run_label(run):
    """'EfficientNetB0_cv' -> 'EffB0-CV' (names used in the paper)."""
    model, strategy = run.rsplit("_", 1)
    return f"{SHORT_NAMES.get(model, model)}-{strategy.upper()}"


def mean_sd(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}"


# ------------------------------------------------------------------- main ---

def main(runs, out_dir, save_dir):
    cams = {run_label(r): load_heatmaps(out_dir, r) for r in runs}
    labels = list(cams)

    # Every model must have explained exactly the same test images
    images = sorted(cams[labels[0]])
    for lab in labels:
        assert sorted(cams[lab]) == images, f"{lab} has different test images"
    print(f"Comparing {len(labels)} models on {len(images)} test images\n")

    # ---- Table IV: entropy per model ------------------------------------
    entropy = pd.DataFrame({lab: [spatial_entropy(cams[lab][img]) for img in images] for lab in labels},
                           index=images)
    table4 = pd.DataFrame({"Mean entropy ± SD": {lab: mean_sd(entropy[lab]) for lab in labels}})
    print("TABLE IV: Grad-CAM spatial entropy\n", table4, "\n")

    # Paired test: is model A's entropy different from model B's on the same images?
    entropy_tests = pd.DataFrame([
        {"Comparison": f"{a} vs {b}", "p_value": wilcoxon(entropy[a], entropy[b]).pvalue}
        for a, b in combinations(labels, 2)
    ])
    print("Entropy, paired Wilcoxon between models\n", entropy_tests, "\n")

    # ---- Table V: SSIM and IoU per pair of models ------------------------
    pairs = [f"{a} vs {b}" for a, b in combinations(labels, 2)]
    ssim = pd.DataFrame({f"{a} vs {b}": [structural_similarity(cams[a][i], cams[b][i], data_range=1.0)
                                         for i in images]
                         for a, b in combinations(labels, 2)}, index=images)
    iou = pd.DataFrame({f"{a} vs {b}": [top_region_iou(cams[a][i], cams[b][i]) for i in images]
                        for a, b in combinations(labels, 2)}, index=images)

    table5 = pd.DataFrame({
        "Mean SSIM ± SD": {p: mean_sd(ssim[p]) for p in pairs},
        "Mean Top-20% IoU ± SD": {p: mean_sd(iou[p]) for p in pairs},
    })
    print("TABLE V: pairwise similarity and overlap\n", table5, "\n")

    # Paired test BETWEEN model pairs: e.g. is SSIM(EffB0, EffV2B1) higher than
    # SSIM(EffB0, Res152V2) on the same images?
    pair_tests = pd.DataFrame([
        {"Comparison": f"[{p1}] vs [{p2}]",
         "SSIM p_value": wilcoxon(ssim[p1], ssim[p2]).pvalue,
         "IoU p_value": wilcoxon(iou[p1], iou[p2]).pvalue}
        for p1, p2 in combinations(pairs, 2)
    ])
    print("SSIM / IoU, paired Wilcoxon between model pairs\n", pair_tests)

    # ---- Save ------------------------------------------------------------
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    table4.to_csv(save_dir / "table4_entropy.csv")
    table5.to_csv(save_dir / "table5_ssim_iou.csv")
    entropy_tests.to_csv(save_dir / "wilcoxon_entropy.csv", index=False)
    pair_tests.to_csv(save_dir / "wilcoxon_ssim_iou.csv", index=False)
    print(f"\n✅ Tables saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entropy, SSIM and IoU of Grad-CAM maps")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run folders in outputs/, e.g. EfficientNetB0_cv ResNet152V2_fs")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--save_dir", default="results/gradcam_analysis")
    args = parser.parse_args()
    main(args.runs, args.out_dir, args.save_dir)
