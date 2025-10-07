# server/federated_server.py

"""
Coordinator: loads local models, averages them, applies optional quantum smoothing,
and writes a global model file.
"""

import os
import torch
from server.model import get_global_model
from server.aggregate import average_state_dicts
from utils.config import NUM_CLIENTS, MODEL_PATH, DEVICE, INPUT_DIM
from utils.quantum_module import quantum_optimize_model_weights

def start_server(num_clients=NUM_CLIENTS):
    print("Starting federated aggregation server...")
    state_dicts = []
    for i in range(num_clients):
        # expect local model saved in clients/client_i/client_i_models/local_model.pth
        candidate_dirs = [
            os.path.join("clients", f"client_{i+1}", f"client_{i+1}_models", "local_model.pth"),
            os.path.join("clients", f"client_{i+1}", "local_model.pth"),
        ]
        found = False
        for p in candidate_dirs:
            if os.path.exists(p):
                sd = torch.load(p, map_location="cpu")
                state_dicts.append(sd)
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Local model for client {i+1} not found. Looked at: {candidate_dirs}")
    avg_sd = average_state_dicts(state_dicts)

    # optional quantum-inspired smoothing of weights
    smoothed = quantum_optimize_model_weights(avg_sd)
    # create model and load state
    model = get_global_model(tab_in=INPUT_DIM)
    model.load_state_dict(smoothed)
    # save full model state_dict
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved global model state_dict at {MODEL_PATH}")

if __name__ == "__main__":
    start_server()
