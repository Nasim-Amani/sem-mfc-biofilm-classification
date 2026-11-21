"""
Train one architecture with a single fixed split (FS).

Steps:
  1. Hold out the same 20% test set as CV.
  2. Split the remaining 80% once: 75% train / 25% validation.
  3. Train one model and evaluate it on the test set.

Usage:
    python -m src.train_fs --model ResNet152V2 --data_dir data
"""

import argparse
import math
from pathlib import Path

import numpy as np

from src.config import (BATCH_SIZE, EARLY_STOP_PATIENCE, EPOCHS, FS_VAL_SIZE,
                        LEARNING_RATE, SEED)
from src.utils import save_json, set_seed


def main(model_name, data_dir, out_dir):
    set_seed(SEED)  # must run before TensorFlow builds anything

    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    from src.data import load_images_and_labels, make_generators, split_test_set
    from src.evaluate import evaluate
    from src.models import build_model, get_preprocess_fn

    out_dir = Path(out_dir) / f"{model_name}_fs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1–2. Data: test set, then one fixed train/validation split ---
    images, labels, class_names = load_images_and_labels(data_dir)
    x_train_val, x_test, y_train_val, y_test = split_test_set(images, labels)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train_val, y_train_val, test_size=FS_VAL_SIZE,
        stratify=y_train_val, random_state=SEED,
    )
    print(f"Train: {x_tr.shape} | Val: {x_val.shape} | Test: {x_test.shape}")

    train_datagen, eval_datagen = make_generators(get_preprocess_fn(model_name))

    classes = np.unique(y_tr)
    class_weight = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))

    # --- 3. Train ---
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

    # --- Evaluate (generators apply the same preprocessing as training) ---
    predict = lambda x: model.predict(eval_datagen.flow(x, batch_size=BATCH_SIZE, shuffle=False), verbose=0)
    val_metrics = evaluate(y_val, predict(x_val), class_names, title="Validation")
    test_probs = predict(x_test)
    test_metrics = evaluate(y_test, test_probs, class_names, title=f"{model_name}-FS, test set")

    # --- Save ---
    model.save_weights(out_dir / "model.weights.h5")
    np.save(out_dir / "test_probs.npy", test_probs)
    np.save(out_dir / "y_test.npy", y_test)  # true labels, needed by plot_model.py
    save_json(history.history, out_dir / "history.json")
    save_json({"test": test_metrics, "val": val_metrics}, out_dir / "metrics.json")
    print(f"\n✅ Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed-split training")
    parser.add_argument("--model", required=True, help="e.g. ResNet152V2")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    main(args.model, args.data_dir, args.out_dir)
