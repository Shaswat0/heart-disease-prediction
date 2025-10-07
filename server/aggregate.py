# server/aggregate.py

"""
Federated averaging over state_dicts.
"""

import torch
from copy import deepcopy

def average_state_dicts(state_dicts):
    """
    state_dicts: list of state_dict (tensors)
    returns averaged state_dict
    """
    if not state_dicts:
        raise ValueError("No state_dicts provided for averaging.")
    avg = deepcopy(state_dicts[0])
    for key in avg.keys():
        # sum then divide
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg[key] = torch.mean(stacked, dim=0)
    return avg
