import torch
import pandas as pd
from client_1.model import GNNModel
from utils.data_preprocessing import create_graph, load_patient_data
from utils.quantum_module import quantum_inference

# Load global model
global_weights_path = "global_model.pth"
features, labels = load_patient_data("data/patient_data.csv")

input_dim = features.shape[1]
global_model = GNNModel(input_dim)
global_model.load_state_dict(torch.load(global_weights_path))
global_model.eval()

data = create_graph(features)

with torch.no_grad():
    pred = global_model(data)
    q_out = quantum_inference(features[0])
    final_pred = torch.clamp(pred + 0.1 * q_out, 0, 1)

print("Predictions (probability of heart disease):")
print(final_pred.numpy())
