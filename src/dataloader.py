# src/dataloader.py
from pathlib import Path
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def get_image_datasets(data_dir, img_size=224, batch_size=16, seed=123):
    """
    Accepts either:
      - data_dir/train/<class> and data_dir/val/<class>
    OR
      - data_dir/<class> (auto-splits 80/20)
    Returns: (train_ds, val_ds, class_names)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if train_dir.exists() and val_dir.exists():
        # already split
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(train_dir),
            labels="inferred",
            label_mode="int",
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(val_dir),
            labels="inferred",
            label_mode="int",
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        class_names = train_ds.class_names
    else:
        # assume data_dir/class_name/*.jpg and do an automatic split
        ds = tf.keras.utils.image_dataset_from_directory(
            str(data_dir),
            labels="inferred",
            label_mode="int",
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            validation_split=0.2,
            subset="training",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(data_dir),
            labels="inferred",
            label_mode="int",
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            validation_split=0.2,
            subset="validation",
        )
        train_ds = ds
        class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names