"""
One function to build any of the pretrained CNNs used in the paper.

All models share the same design:
    frozen ImageNet backbone -> GlobalAveragePooling -> BatchNorm -> Dropout -> Softmax
Only the backbone (and its matching input preprocessing) changes.
"""

import tensorflow as tf
from tensorflow.keras import applications as A
from tensorflow.keras import layers

from src.config import DROPOUT, IMG_SIZE

# name -> (backbone constructor, matching preprocessing function)
# NOTE: EfficientNet / EfficientNetV2 include their own rescaling,
#       so their preprocess_input simply passes the image through.
MODEL_REGISTRY = {
    "EfficientNetB0":   (A.EfficientNetB0,   A.efficientnet.preprocess_input),
    "EfficientNetB1":   (A.EfficientNetB1,   A.efficientnet.preprocess_input),
    "EfficientNetB2":   (A.EfficientNetB2,   A.efficientnet.preprocess_input),
    "EfficientNetB3":   (A.EfficientNetB3,   A.efficientnet.preprocess_input),
    "EfficientNetV2B0": (A.EfficientNetV2B0, A.efficientnet_v2.preprocess_input),
    "EfficientNetV2B1": (A.EfficientNetV2B1, A.efficientnet_v2.preprocess_input),
    "EfficientNetV2B2": (A.EfficientNetV2B2, A.efficientnet_v2.preprocess_input),
    "EfficientNetV2B3": (A.EfficientNetV2B3, A.efficientnet_v2.preprocess_input),
    "MobileNet":        (A.MobileNet,        A.mobilenet.preprocess_input),
    "MobileNetV2":      (A.MobileNetV2,      A.mobilenet_v2.preprocess_input),
    "NASNetMobile":     (A.NASNetMobile,     A.nasnet.preprocess_input),
    "DenseNet201":      (A.DenseNet201,      A.densenet.preprocess_input),
    "DenseNet121":      (A.DenseNet121,      A.densenet.preprocess_input),
    "VGG19":            (A.VGG19,            A.vgg19.preprocess_input),
    "ResNet152V2":      (A.ResNet152V2,      A.resnet_v2.preprocess_input),
}

# Short names used in the paper's figures and tables
SHORT_NAMES = {
    "EfficientNetB0": "EffB0", "EfficientNetB1": "EffB1",
    "EfficientNetB2": "EffB2", "EfficientNetB3": "EffB3",
    "EfficientNetV2B0": "EffV2B0", "EfficientNetV2B1": "EffV2B1",
    "EfficientNetV2B2": "EffV2B2", "EfficientNetV2B3": "EffV2B3",
    "MobileNet": "MobNet", "MobileNetV2": "MobNetV2", "NASNetMobile": "NASNet",
    "DenseNet201": "Dense201", "DenseNet121": "Dense121",
    "VGG19": "VGG19", "ResNet152V2": "Res152V2",
}


def get_preprocess_fn(name):
    """Return the input preprocessing function that matches the backbone."""
    return MODEL_REGISTRY[name][1]


def build_model(name, num_classes):
    """Build a transfer-learning classifier.

    Args:
        name:        a key of MODEL_REGISTRY, e.g. "EfficientNetB0"
        num_classes: number of colonisation classes (5)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")

    backbone_fn, _ = MODEL_REGISTRY[name]
    input_shape = (*IMG_SIZE, 3)

    inputs = layers.Input(shape=input_shape)
    backbone = backbone_fn(weights="imagenet", include_top=False, input_tensor=inputs)
    backbone.trainable = False  # feature extractor only (small dataset)

    # Classification head (identical for every architecture)
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name=f"{name}_classifier")
