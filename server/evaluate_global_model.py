# server/evaluate_global_model.py

import torch
from server.model import get_global_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
data = pd.read_csv('data/heart.csv')

X = data.drop('target', axis=1)
y = data['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale tabular features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Convert to tensors
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# ✅ Dummy image input — since this model expects both (tabular + image)
#    We'll use zeros with the same shape it expects
#    You can adjust (3, 224, 224) if your model uses a different image size
dummy_images = torch.zeros((len(X_test_tensor), 3, 224, 224))

# Load global model
model = get_global_model(tab_in=X.shape[1])
model.load_state_dict(torch.load("global_model.pth"))
model.eval()

# Evaluate
with torch.no_grad():
    outputs = model(X_test_tensor, dummy_images)  # ✅ now both inputs
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"✅ Global Model Accuracy: {acc * 100:.2f}%")
