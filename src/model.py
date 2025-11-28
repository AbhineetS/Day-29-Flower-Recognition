# src/model.py
import tensorflow as tf
from tensorflow.keras import layers

def build_model(input_shape=(224,224,3), num_classes=2, backbone="efficientnetb0", base_trainable=False):
    """Simple transfer-learning classifier using EfficientNetB0 as default."""
    # use Keras application by name
    backbone = backbone.lower()
    if backbone.startswith("efficientnetb0"):
        base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights="imagenet")
    else:
        # fallback: EfficientNetB0
        base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights="imagenet")

    base.trainable = bool(base_trainable)

    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    if num_classes == 1:
        out = layers.Dense(1, activation="sigmoid", name="pred")(x)
    else:
        out = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name=f"{backbone}_classifier")
    return model