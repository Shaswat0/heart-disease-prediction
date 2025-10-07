# utils/data_preprocessing.py

"""
Loads CSV, matches ECG images (if present), performs normalization/transforms,
and returns PyTorch DataLoaders for clients.
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
from torchvision import transforms
from PIL import Image
from .config import DATA_PATH, ECG_IMAGE_PATH, BATCH_SIZE, IMAGE_SIZE, INPUT_DIM, DEVICE

NUMERIC_COLUMNS = None  # will be determined after reading CSV

class TabularImageDataset(Dataset):
    def __init__(self, df, image_dir=None, image_col=None, transform=None):
        global NUMERIC_COLUMNS
        self.df = df.reset_index(drop=True)
        if NUMERIC_COLUMNS is None:
            # All columns except 'target' and possible 'ecg_image' are numeric features
            NUMERIC_COLUMNS = [c for c in df.columns if c not in ("target", "ecg_image")]
        self.X_tab = self.df[NUMERIC_COLUMNS].values.astype(np.float32)
        self.y = self.df["target"].values.astype(np.int64)
        self.image_dir = image_dir
        self.image_col = image_col
        self.transform = transform

        # If images are not provided, set placeholder zeros
        self.has_images = (image_dir is not None) or (image_col in self.df.columns)

    def __len__(self):
        return len(self.df)

    def _load_image_by_index(self, idx):
        # If ecg_image column exists, use it
        if self.image_col in self.df.columns:
            img_name = self.df.loc[idx, self.image_col]
            img_path = os.path.join(self.image_dir, img_name)
        else:
            # fallback: use sorted file list by index mapping
            files = sorted(os.listdir(self.image_dir))
            if idx < len(files):
                img_path = os.path.join(self.image_dir, files[idx])
            else:
                return None
        if not os.path.exists(img_path):
            return None
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        tab = torch.tensor(self.X_tab[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)

        img = None
        if self.has_images and self.image_dir and os.path.exists(self.image_dir):
            img = self._load_image_by_index(idx)
        if img is None:
            # return zeros if no image
            img = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)

        return {"tabular": tab, "image": img, "label": label}


def get_full_dataframe():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Please provide dataset or run utils/generator.py")
    df = pd.read_csv(DATA_PATH)
    # Ensure 'target' exists
    if "target" not in df.columns:
        raise ValueError("CSV must contain 'target' column.")
    return df

def get_client_dataloaders(num_clients=3, batch_size=BATCH_SIZE, image_dir=ECG_IMAGE_PATH):
    df = get_full_dataframe()
    # Shuffle once
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    splits = np.array_split(df, num_clients)
    loaders = []
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    for i, subdf in enumerate(splits):
        ds = TabularImageDataset(subdf, image_dir=image_dir if os.path.exists(image_dir) else None, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    return loaders

def load_client_data(client_id, num_clients=3, batch_size=BATCH_SIZE, image_dir=ECG_IMAGE_PATH):
    loaders = get_client_dataloaders(num_clients=num_clients, batch_size=batch_size, image_dir=image_dir)
    # client_id is 1-indexed in our design
    if client_id < 1 or client_id > len(loaders):
        raise ValueError("client_id out of range")
    return loaders[client_id - 1]

def preprocess_input(user_input, image_path=None):
    """
    user_input: list or 1D array of numeric features (len = INPUT_DIM)
    image_path: optional path to ECG image to include
    Returns dict: {'tabular': tensor(1,INPUT_DIM), 'image': tensor(1,3,H,W)}
    """
    from torchvision import transforms
    import torch
    # Normalize / convert to tensor
    tab = torch.tensor(user_input, dtype=torch.float32).unsqueeze(0)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    if image_path is not None and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0)
    else:
        img = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    return {"tabular": tab.to(DEVICE), "image": img.to(DEVICE)}
