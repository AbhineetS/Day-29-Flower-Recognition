# create_dummy_data.py
# Quick script to create a tiny dummy dataset for testing (3 classes)
import os
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("data")
classes = ["rose", "tulip", "daisy"]
sizes = [(224,224)]

def make_image(path):
    arr = (np.random.rand(224,224,3) * 255).astype("uint8")
    img = Image.fromarray(arr)
    img.save(path, format="JPEG", quality=80)

# create directories train/val with small images
for split, count in [("train", 48), ("val", 12)]:
    for cls in classes:
        d = ROOT / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count // len(classes)):
            make_image(d / f"{cls}_{i}.jpg")

print("Dummy dataset created at ./data with classes:", classes)