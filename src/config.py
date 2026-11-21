"""
Shared settings for all experiments.

Every model in the paper (CV and fixed split) used these same values,
so they live in one place instead of being repeated in each notebook.
"""

# --- Reproducibility ---
SEED = 42

# --- Images ---
IMG_SIZE = (175, 175)          # SEM patches are 175 x 175 RGB
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# --- Data splits ---
TEST_SIZE = 0.20               # 20% held-out test set (same for CV and FS)
FS_VAL_SIZE = 0.25             # fixed split: 25% of train_val -> validation
CV_FOLDS = 4                   # cross-validation: 4 stratified folds

# --- Training ---
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
EPOCHS = 200
EARLY_STOP_PATIENCE = 20
DROPOUT = 0.45

# --- Data augmentation (training data only) ---
AUGMENTATION = dict(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    shear_range=5,
    brightness_range=[0.9, 1.1],
    fill_mode="nearest",
)
