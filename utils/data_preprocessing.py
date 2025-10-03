import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

def load_patient_data(csv_path="data/patient_data.csv"):
    df = pd.read_csv(csv_path)
    features = df.drop("target", axis=1).values.astype(np.float32)
    labels = df["target"].values.astype(np.float32)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    return features, labels

def create_graph(features, threshold=0.5):
    sim_matrix = cosine_similarity(features)
    edge_index = np.array(np.where(sim_matrix > threshold))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)
