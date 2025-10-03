def aggregate_models(client_state_dicts):
    agg_state_dict = {}
    for key in client_state_dicts[0].keys():
        agg_state_dict[key] = sum([client[key] for client in client_state_dicts]) / len(client_state_dicts)
    return agg_state_dict
