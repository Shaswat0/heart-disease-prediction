# utils/config.py

import os

# ====== PATHS ======
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "patient_data.csv")
ECG_IMAGE_PATH = os.path.join(ROOT, "data", "ECG_images")
MODEL_PATH = os.path.join(ROOT, "global_model.pth")  # saved global model (state_dict)

# ====== FEDERATED SETTINGS ======
NUM_CLIENTS = 3
ROUNDS = 1    # number of federated rounds (for simulation)
AGGREGATION_METHOD = "FedAvg"

# ====== MODEL SETTINGS ======
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 3     # keep small by default for demo; increase as needed
INPUT_DIM = 13  # number of tabular features (adjust if different)
IMAGE_SIZE = 64  # ECG image will be resized to (64,64)

# ====== QUANTUM SETTINGS ======
USE_QUANTUM_OPTIMIZATION = True

# ====== DEVICE ======
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
