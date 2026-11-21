"""
Loading the SEM patches and preparing data for training.

Expected folder layout (one sub-folder per colonisation class):

    data/
    ├── 0/   minimal attachment
    ├── 1/   low attachment
    ├── 2/   intermediate attachment
    ├── 3/   moderate-to-high attachment
    └── 4/   dense colonisation
"""

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import AUGMENTATION, IMG_EXTENSIONS, SEED, TEST_SIZE


def load_images_and_labels(data_dir):
    """Read all images from class folders.

    Returns:
        images:      array of shape (N, H, W, 3)
        labels:      array of shape (N,) with class indices
        class_names: sorted list of folder names, e.g. ['0', '1', '2', '3', '4']
    """
    data_dir = Path(data_dir)
    class_names = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    images, labels = [], []
    for cls in class_names:
        for image_path in sorted((data_dir / cls).glob("*")):
            if image_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            try:
                with Image.open(image_path) as img:
                    images.append(np.array(img.convert("RGB")))
                    labels.append(class_to_idx[cls])
            except Exception as e:
                print(f"Skipping {image_path}: {e}")

    return np.array(images), np.array(labels), class_names


def split_test_set(images, labels):
    """Hold out a stratified test set (20%).

    The SAME test set is used for every model and both validation
    strategies, so all results in the paper are directly comparable.
    """
    return train_test_split(
        images, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=SEED,
    )


def make_generators(preprocess_fn):
    """Create image generators.

    - train: augmentation + the model's own preprocessing
    - eval:  preprocessing only (used for validation AND test data)

    Each pretrained model expects its inputs scaled in a specific way,
    so `preprocess_fn` comes from models.py (e.g. ResNet vs EfficientNet).
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn, **AUGMENTATION)
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    return train_datagen, eval_datagen
