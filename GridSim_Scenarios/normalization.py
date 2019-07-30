def normalize_value(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)


def normalize_simplified_observations(data, min_val, max_val):
    obs_n = list()
    for obs_seq in data:
        obs_seq_list = list()
        for obs in obs_seq:
            obs_seq_list.append((normalize_value(obs[0], min_val, max_val), obs[1]))
        obs_n.append(obs_seq_list)
    return obs_n


def normalize_simplified_predictions(data, min_val, max_val):
    pred_n = list()
    for pred_seq in data:
        pred_seq_list = list()
        for p in pred_seq:
            pred_seq_list.append(normalize_value(p, min_val, max_val))
        pred_n.append(pred_seq_list)
    return pred_n


def normalize_sensor_array_observations(data, min_val, max_val):
    h_n = list()
    for h in data:
        h_n_seq = list()
        for h_seq in h:
            ray_list = list()
            for ray in h_seq:
                ray_list.append(normalize_value(ray, min_val, max_val))
            h_n_seq.append(ray_list)
        h_n.append(h_n_seq)
    return h_n


def normalize_sensor_array_predictions(data, min_val, max_val):
    return normalize_sensor_array_observations(data, min_val, max_val)


def normalize_actions(data, min_val, max_val):
    a_n = list()
    for a in data:
        a_seq_list = list()
        for a_seq in a:
            a_seq_list.append(normalize_value(a_seq, min_val, max_val))
        a_n.append(a_seq_list)
    return a_n


def normalize_previous_actions(data, min_val, max_val):
    p_a_n = list()
    for p_a in data:
        p_a_seq_list = list()
        for p in p_a:
            p_a_seq_list.append(normalize_value(p, min_val, max_val))
        p_a_n.append(p_a_seq_list)
    return p_a_n
