# app/streamlit_app.py

"""
Streamlit UI to input tabular features and an ECG image and get prediction.
"""

import streamlit as st
import torch
from utils.config import MODEL_PATH, INPUT_DIM, DEVICE
from utils.data_preprocessing import preprocess_input
from utils.quantum_module import quantum_optimize_features
from server.model import get_global_model
import os

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Global model not found. Train clients and run server to generate model.")
        return None
    model = get_global_model(tab_in=INPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def main():
    st.title("❤️ Privacy-Preserving Heart Disease Prediction")
    st.write("Enter patient features and optionally upload an ECG image.")

    # Example inputs - adapt to your CSV columns order
    st.subheader("Patient Features (tabular)")
    age = st.number_input("age", min_value=1, max_value=120, value=55)
    sex = st.selectbox("sex (0=female,1=male)", [0,1], index=1)
    cp = st.number_input("cp (chest pain type 0-3)", min_value=0, max_value=3, value=1)
    trestbps = st.number_input("trestbps", min_value=50, max_value=250, value=130)
    chol = st.number_input("chol", min_value=100, max_value=600, value=230)
    fbs = st.selectbox("fbs (fasting blood sugar > 120 mg/dl)", [0,1], index=0)
    restecg = st.number_input("restecg (0-2)", min_value=0, max_value=2, value=1)
    thalach = st.number_input("thalach", min_value=50, max_value=250, value=150)
    exang = st.selectbox("exang (exercise induced angina 0/1)", [0,1], index=0)
    oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    slope = st.number_input("slope", min_value=0, max_value=2, value=1)
    ca = st.number_input("ca", min_value=0, max_value=3, value=0)
    thal = st.number_input("thal", min_value=0, max_value=3, value=1)

    file = st.file_uploader("Upload ECG image (optional)", type=["png","jpg","jpeg"])

    if st.button("Predict"):
        model = load_model()
        if model is None:
            return
        # build user vector with same order as CSV
        user_vec = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        image_path = None
        if file is not None:
            temp_path = os.path.join(".", "temp_ecg_upload.png")
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            image_path = temp_path
        data = preprocess_input(user_vec, image_path=image_path)
        tab = data["tabular"]
        img = data["image"]
        # optional quantum optimize features
        tab = quantum_optimize_features(tab)

        with torch.no_grad():
            out = model(tab.to(DEVICE), img.to(DEVICE))
            pred = torch.argmax(out, dim=1).item()
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

        st.success(f"Prediction: {'Heart Disease' if pred==1 else 'Healthy'}")
        st.write(f"Confidence: Healthy={probs[0]:.3f}, HeartDisease={probs[1]:.3f}")

if __name__ == "__main__":
    main()
