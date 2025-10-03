import torch
from .model import GNNModel
from utils.data_preprocessing import load_patient_data, create_graph
from utils.quantum_module import quantum_inference

def train_client(client_id, features, labels, epochs=5):
    input_dim = features.shape[1]
    data = create_graph(features)
    model = GNNModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data).view(-1)
        q_out = quantum_inference(features[0])
        out = out + 0.1 * q_out
        loss = criterion(out, torch.tensor(labels[:1]))
        loss.backward()
        optimizer.step()
    return model.state_dict()
