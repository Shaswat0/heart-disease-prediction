# utils/quantum_module.py

"""
Simulated quantum optimization. Keeps lightweight, deterministic.
"""

import torch
from .config import USE_QUANTUM_OPTIMIZATION

def quantum_optimize_features(tab_tensor):
    """
    Simulate a small transformation on features.
    Input: tensor shape (batch, features)
    """
    if not USE_QUANTUM_OPTIMIZATION:
        return tab_tensor
    # simple deterministic transform: scaled sinusoidal mapping
    return tab_tensor * torch.sin(tab_tensor + 0.5)

def quantum_optimize_model_weights(state_dict):
    """
    Optional: slight smoothing of weights (simulated). Accepts and returns state_dict.
    """
    if not USE_QUANTUM_OPTIMIZATION:
        return state_dict
    new = {}
    for k, v in state_dict.items():
        tensor = v.float()
        new[k] = (tensor * 0.98) + (0.02 * torch.tanh(tensor))
    return new
