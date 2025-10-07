# clients/client_1/train.py

"""
Train local client model on client's split and save local state_dict.
Usage:
    python clients/client_1/train.py --client_id 1
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from clients.client_1.model import CombinedModel
from utils.data_preprocessing import load_client_data
from utils.config import EPOCHS, LEARNING_RATE, DEVICE
import os

def train_client(client_id, num_clients=3):
    print(f"Starting training for client {client_id}")
    loader = load_client_data(client_id, num_clients=num_clients)
    model = CombinedModel(tab_in=loader.dataset.X_tab.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in loader:
            tab = batch["tabular"].to(DEVICE)
            img = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(tab, img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * tab.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        acc = correct / total
        print(f"[Client {client_id}] Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f} Acc: {acc:.4f}")

    # Save local state_dict
    out_dir = os.path.join(os.path.dirname(__file__), f"client_{client_id}_models")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "local_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved local model for client {client_id} at {save_path}")
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()
    train_client(args.client_id, args.num_clients)
