# ❤️ Privacy-Preserving Heart Disease Prediction

### Using Federated Learning, Graph Neural Networks, and Quantum Optimization

---

## 🚀 Overview
This project builds a privacy-preserving system to predict heart disease risk using **federated learning**, **graph neural networks**, and **quantum-inspired optimization**.

---

## 🧩 Architecture
- **Clients:** Train models locally on private patient data.
- **Server:** Aggregates updates using Federated Averaging.
- **Quantum Optimization:** Enhances feature representation or training weights.
- **Frontend:** Streamlit app for easy user interaction.

---

## ⚙️ Run Instructions
```bash
# Generate synthetic dataset
python utils/generator.py

# Train each client
python clients/client_1/train.py
python clients/client_2/train.py
python clients/client_3/train.py

# Aggregate models on server
python server/federated_server.py

# Run Streamlit app
streamlit run app/streamlit_app.py
