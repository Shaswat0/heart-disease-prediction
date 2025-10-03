from client_1.train import train_client as train_client1
from client_2.train import train_client as train_client2
from client_3.train import train_client as train_client3
from server.aggregate import aggregate_models
from utils.data_preprocessing import load_patient_data

# Load data
features, labels = load_patient_data()
n = len(features)
f1, l1 = features[:n//3], labels[:n//3]
f2, l2 = features[n//3:2*n//3], labels[n//3:2*n//3]
f3, l3 = features[2*n//3:], labels[2*n//3:]

# Train clients
sd1 = train_client1(1, f1, l1)
sd2 = train_client2(2, f2, l2)
sd3 = train_client3(3, f3, l3)

# Aggregate
global_weights = aggregate_models([sd1, sd2, sd3])

# Save global model
import torch
torch.save(global_weights, "global_model.pth")

print("Federated learning complete! Global model ready.")
