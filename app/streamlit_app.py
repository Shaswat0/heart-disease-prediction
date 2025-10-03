import streamlit as st
import torch
from client_1.model import GNNModel
from utils.data_preprocessing import create_graph, load_patient_data
from utils.quantum_module import quantum_inference
from PIL import Image
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Load global model ---
global_weights_path = "../global_model.pth"
features, labels = load_patient_data("../data/patient_data.csv")
input_dim = features.shape[1]

global_model = GNNModel(input_dim)
global_model.load_state_dict(torch.load(global_weights_path))
global_model.eval()

st.title("Heart Disease Prediction (GNN + Quantum)")

# Upload image or enter CSV data
uploaded_file = st.file_uploader("Upload ECG Image (optional)", type=["png", "jpg"])
manual_input = st.text_area("Or enter patient data manually (comma-separated)")

def preprocess_input(uploaded_file=None, manual_input=None):
    if uploaded_file:
        # For demo, just convert image to grayscale mean as a single feature vector
        image = Image.open(uploaded_file).convert("L").resize((32,32))
        arr = np.array(image)/255.0
        feature_vector = arr.mean(axis=0)  # simple feature vector
        features = np.tile(feature_vector, (1, input_dim))[:,:input_dim]  # match dimension
    elif manual_input:
        vals = np.array([float(x) for x in manual_input.strip().split(",")], dtype=np.float32)
        features = np.expand_dims(vals, axis=0)
    else:
        st.error("Please upload an image or enter data!")
        return None
    return features

features_input = preprocess_input(uploaded_file, manual_input)
if features_input is not None:
    # Create graph for GNN
    data = create_graph(features_input)
    
    with torch.no_grad():
        pred = global_model(data)
        q_out = quantum_inference(features_input[0])
        final_pred = torch.clamp(pred + 0.1 * q_out, 0, 1)
    
    st.write("Predicted probability of heart disease:", final_pred.numpy())
