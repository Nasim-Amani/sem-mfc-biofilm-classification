"""Small helpers: reproducibility and saving results."""

import json
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Fix all random seeds so runs are as repeatable as possible.

    Must be called BEFORE building or training any model.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    import tensorflow as tf  # imported here so the env vars above apply first

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Full GPU determinism (not available on every TF version / device)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("Warning: determinism not fully enabled:", e)


def save_json(obj, path) -> None:
    """Save a dict/list as a readable JSON file (creates folders if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
