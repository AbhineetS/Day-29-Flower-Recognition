# src/utils.py
from pathlib import Path
import json
import matplotlib.pyplot as plt

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_model_and_history(model, history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "final_model.keras"
    model.save(str(model_path))
    # save history
    hist_path = out_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history.history, f)
    return str(model_path)

def plot_history(history, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist = history.history
    plt.figure(figsize=(6,4))
    if "loss" in hist:
        plt.plot(hist["loss"], label="loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Training Loss")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()