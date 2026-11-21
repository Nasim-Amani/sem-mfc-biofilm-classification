"""
Traditional image-analysis baselines (Section 4.3, Table II).

Each SEM image is described by 8 handcrafted numbers, then classified by an
SVM or a Random Forest using the SAME test set and 4-fold CV as the CNNs.

Handcrafted features (computed on the grayscale image):
  - intensity:  mean, standard deviation, histogram entropy
  - structure:  edge density (Canny edges)
  - texture:    GLCM contrast, correlation, energy, homogeneity

Usage:
    python -m src.baselines --data_dir data
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import entropy
from skimage.feature import canny, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import CV_FOLDS, SEED
from src.data import load_images_and_labels, split_test_set
from src.evaluate import evaluate
from src.utils import save_json


def extract_features(rgb_images):
    """Turn each image into a vector of 8 handcrafted features."""
    features = []
    for rgb in rgb_images:
        gray = np.asarray(Image.fromarray(rgb).convert("L"))      # 8-bit grayscale
        g = gray.astype(np.float32)

        # Intensity statistics
        hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
        intensity = [g.mean(), g.std(), entropy(hist + 1e-10)]

        # Edge density: fraction of pixels that lie on an edge
        edge_density = canny(g / 255.0).mean()

        # GLCM texture (gray levels reduced from 256 to 8 for a stable matrix)
        glcm = graycomatrix((gray // 32).astype(np.uint8), distances=[1], angles=[0],
                            levels=8, symmetric=True, normed=True)
        texture = [graycoprops(glcm, p)[0, 0] for p in ("contrast", "correlation", "energy", "homogeneity")]

        features.append(intensity + [edge_density] + texture)
    return np.asarray(features, dtype=np.float32)


BASELINES = {
    # SVM needs scaled features; class_weight balances the minority classes
    "SVM": lambda: make_pipeline(StandardScaler(),
                                 SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                                     probability=True, random_state=SEED)),
    "RF":  lambda: RandomForestClassifier(class_weight="balanced", random_state=SEED),
}


def main(data_dir, out_dir):
    images, labels, class_names = load_images_and_labels(data_dir)
    x_train_val, x_test, y_train_val, y_test = split_test_set(images, labels)

    X_train_val, X_test = extract_features(x_train_val), extract_features(x_test)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    for name, make_model in BASELINES.items():
        fold_test_probs = []
        for train_idx, _ in skf.split(X_train_val, y_train_val):
            model = make_model().fit(X_train_val[train_idx], y_train_val[train_idx])
            fold_test_probs.append(model.predict_proba(X_test))

        # Same as the CNNs: average the 4 fold models on the test set
        test_probs = np.mean(fold_test_probs, axis=0)
        metrics = evaluate(y_test, test_probs, class_names, title=f"Handcrafted features + {name}")

        run_dir = Path(out_dir) / f"{name}_cv"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "test_probs.npy", test_probs)
        np.save(run_dir / "y_test.npy", y_test)
        save_json({"test": metrics}, run_dir / "metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM and Random Forest baselines")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
