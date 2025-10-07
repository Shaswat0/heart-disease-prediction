# utils/generator.py

"""
Generates a synthetic patient_data.csv and random ECG images for quick demo/testing.
"""

import numpy as np
import pandas as pd
import os
from PIL import Image

ROOT_DATA = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "patient_data.csv")
ECG_DIR = os.path.join(os.path.dirname(__file__), "data", "ECG_images")

def generate_synthetic(num_samples=150):
    os.makedirs(ECG_DIR, exist_ok=True)
    np.random.seed(42)
    data = {
        "age": np.random.randint(29, 77, size=num_samples),
        "sex": np.random.randint(0, 2, size=num_samples),
        "cp": np.random.randint(0, 4, size=num_samples),
        "trestbps": np.random.randint(94, 200, size=num_samples),
        "chol": np.random.randint(126, 564, size=num_samples),
        "fbs": np.random.randint(0, 2, size=num_samples),
        "restecg": np.random.randint(0, 2, size=num_samples),
        "thalach": np.random.randint(71, 202, size=num_samples),
        "exang": np.random.randint(0, 2, size=num_samples),
        "oldpeak": np.round(np.random.uniform(0.0, 6.0, size=num_samples), 1),
        "slope": np.random.randint(0, 3, size=num_samples),
        "ca": np.random.randint(0, 4, size=num_samples),
        "thal": np.random.randint(0, 3, size=num_samples),
        "target": np.random.randint(0, 2, size=num_samples)
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    df.to_csv(os.path.join(os.path.dirname(__file__), "data", "patient_data.csv"), index=False)
    # create random images
    for i in range(num_samples):
        arr = (np.clip(np.random.normal(loc=128, scale=60, size=(64, 64, 3)), 0, 255)).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(ECG_DIR, f"{i}.png"))
    print(f"Generated synthetic CSV and {num_samples} ECG images in data/")

if __name__ == "__main__":
    generate_synthetic(150)
