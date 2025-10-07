# server/model.py

"""
Factory to produce a model instance matching client models.
"""

from clients.client_1.model import CombinedModel

def get_global_model(tab_in=13):
    return CombinedModel(tab_in=tab_in)
