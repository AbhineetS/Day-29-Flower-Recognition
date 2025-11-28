# src/train.py
import argparse
from pathlib import Path
import sys

import tensorflow as tf

# package-relative imports (required when running `python -m src.train`)
from .dataloader import get_image_datasets
from .model import build_model
from .utils import ensure_dir, save_model_and_history, plot_history

def safe_set_optimizer_lr(optimizer, lr):
    """Robustly set optimizer learning rate from string/float/schedule."""
    try:
        # allow strings like "1e-4"
        lr_val = float(lr) if isinstance(lr, (str, bytes)) else lr
    except Exception:
        lr_val = lr
    try:
        if hasattr(optimizer.learning_rate, "assign"):
            optimizer.learning_rate.assign(lr_val)
        else:
            tf.keras.backend.set_value(optimizer.learning_rate, float(lr_val))
    except Exception as e:
        # last resort: replace attribute
        try:
            optimizer.learning_rate = lr_val
        except Exception:
            print("Warning: could not set optimizer.learning_rate:", e, file=sys.stderr)

def make_callbacks(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "ckpt_epoch-{epoch:02d}.h5"),
        save_best_only=False,
        verbose=1,
    )
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    return [ckpt, early, reduce_lr]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="./artifacts")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--learning-rate", default="1e-4")
    p.add_argument("--backbone", default="efficientnetb0")
    p.add_argument("--base-trainable", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_dir)

    print("📚 Loading datasets...")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    train_ds, val_ds, class_names = get_image_datasets(
        str(data_dir),
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=123,
    )
    num_classes = len(class_names)
    print(f"Detected classes: {num_classes} -> {class_names}")

    print("🧠 Building model...")
    model = build_model(input_shape=(args.img_size, args.img_size, 3),
                        num_classes=num_classes,
                        backbone=args.backbone,
                        base_trainable=args.base_trainable)

    print("⚙️ Compiling model...")
    opt = tf.keras.optimizers.Adam()
    # Use sparse_categorical_crossentropy because image_dataset_from_directory returns integer labels
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print(f"🔧 Setting optimizer learning rate to {args.learning_rate}")
    safe_set_optimizer_lr(model.optimizer, args.learning_rate)

    callbacks = make_callbacks(out_dir)

    print("🚀 Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    print("✅ Saving artifacts...")
    save_model_and_history(model, history, str(out_dir))
    try:
        plot_history(history, str(out_dir / "training_history.png"))
    except Exception as e:
        print("Warning: plot failed:", e)

    print("🎉 Done.")

if __name__ == "__main__":
    main()