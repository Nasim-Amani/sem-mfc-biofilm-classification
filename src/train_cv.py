"""
Train one architecture with 4-fold stratified cross-validation (CV).

Steps:
  1. Hold out the 20% test set.
  2. Train one model per fold on the remaining 80%.
  3. Test set prediction = average of the 4 fold models (ensemble).

Usage:
    python -m src.train_cv --model EfficientNetB0 --data_dir data
"""

import argparse
import math
from pathlib import Path

import numpy as np

from src.config import (BATCH_SIZE, CV_FOLDS, EARLY_STOP_PATIENCE, EPOCHS,
                        LEARNING_RATE, SEED)
from src.utils import save_json, set_seed


def main(model_name, data_dir, out_dir):
    set_seed(SEED)  # must run before TensorFlow builds anything

    # Imported after set_seed so the seeds/determinism settings apply
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    from src.data import load_images_and_labels, make_generators, split_test_set
    from src.evaluate import evaluate
    from src.models import build_model, get_preprocess_fn

    out_dir = Path(out_dir) / f"{model_name}_cv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Data ---
    images, labels, class_names = load_images_and_labels(data_dir)
    x_train_val, x_test, y_train_val, y_test = split_test_set(images, labels)
    print(f"Train+val: {x_train_val.shape} | Test: {x_test.shape}")

    train_datagen, eval_datagen = make_generators(get_preprocess_fn(model_name))
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    fold_models, histories, fold_val_metrics = [], [], []

    # --- 2. One model per fold ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_val, y_train_val), start=1):
        print(f"\n📁 Fold {fold}/{CV_FOLDS}")
        x_tr, y_tr = x_train_val[train_idx], y_train_val[train_idx]
        x_val, y_val = x_train_val[val_idx], y_train_val[val_idx]

        # Class weights balance the minority colonisation classes
        classes = np.unique(y_tr)
        weights = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weight = dict(zip(classes, weights))

        model = build_model(model_name, num_classes=len(class_names))
        model.compile(optimizer=Adam(LEARNING_RATE),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(
            train_datagen.flow(x_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True, seed=SEED),
            validation_data=eval_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False),
            steps_per_epoch=math.ceil(len(x_tr) / BATCH_SIZE),
            validation_steps=math.ceil(len(x_val) / BATCH_SIZE),
            epochs=EPOCHS,
            class_weight=class_weight,
            callbacks=[EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE,
                                     restore_best_weights=True, verbose=1)],
            verbose=2,
        )

        # Validation score of this fold (same preprocessing as training)
        val_probs = model.predict(eval_datagen.flow(x_val, batch_size=BATCH_SIZE, shuffle=False), verbose=0)
        fold_val_metrics.append(evaluate(y_val, val_probs, class_names, title=f"Fold {fold} validation"))

        model.save_weights(out_dir / f"fold_{fold}.weights.h5")
        fold_models.append(model)
        histories.append(history.history)

    # --- 3. Ensemble on the held-out test set ---
    test_flow = lambda: eval_datagen.flow(x_test, batch_size=BATCH_SIZE, shuffle=False)
    test_probs = np.mean([m.predict(test_flow(), verbose=0) for m in fold_models], axis=0)
    test_metrics = evaluate(y_test, test_probs, class_names, title=f"{model_name}-CV ensemble, test set")

    # --- Save everything needed for the figures/tables ---
    np.save(out_dir / "test_probs.npy", test_probs)
    np.save(out_dir / "y_test.npy", y_test)  # true labels, needed by plot_model.py
    save_json(histories, out_dir / "histories.json")
    save_json({"test": test_metrics, "folds_val": fold_val_metrics}, out_dir / "metrics.json")
    print(f"\n✅ Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validation training")
    parser.add_argument("--model", required=True, help="e.g. EfficientNetB0")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    main(args.model, args.data_dir, args.out_dir)
