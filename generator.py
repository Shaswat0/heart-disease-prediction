import os
import math
import numpy as np
import pandas as pd
from PIL import Image

# Paths
csv_path = "data/patient_data.csv"
output_dir = "data/ECG_images"

os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    # Select numeric features only (skip target)
    features = row.drop('target').values.astype(float)

    # Scale features to 0-255
    features = np.clip(features * 10, 0, 255)

    # Determine square image size
    size = math.ceil(math.sqrt(len(features)))
    pixels = np.zeros((size, size), dtype=np.uint8)

    # Fill pixels
    for i, val in enumerate(features):
        x = i // size
        y = i % size
        pixels[x, y] = int(val)

    # Add optional noise
    noise = np.random.normal(0, 5, (size, size))  # small Gaussian noise
    noisy_pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)

    # Save image
    img = Image.fromarray(noisy_pixels, mode='L')
    img.save(os.path.join(output_dir, f"patient_{idx}.png"))

print(f"Generated {len(df)} images in {output_dir}")
