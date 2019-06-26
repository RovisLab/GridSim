import os
import random


def has_non_zero(arr):
    for delta, in_fov, action in arr:
        if in_fov != 0:
            return True
    return False


def variable_sequence_length_preprocessing(tmp_fp, pred_size, min_seq_len=10, max_seq_len=100, padding=True):
    """
    Return an increasing sequence of training data
    :param tmp_fp: temporary data file recorded with GridSim
    :param min_seq_len: minimum length of generated sequences
    :param max_seq_len: maximum length of generated sequences
    :param pred_size: prediction horizon size
    :param padding: if True, sequences will be zero-padded to max_seq_len size
    :return: history, prev_actions, actions, predictions
    """
    with open(tmp_fp, "r") as tmp_f:
        lines = tmp_f.readlines()
    elements = list()
    for line in lines:
        delta, in_fov, action = line.split(",")
        elements.append((float(delta), float(in_fov), float(action)))
    elem_idx = 0
    history = list()
    actions = list()
    prev_actions = list()
    predictions = list()
    while elem_idx < len(elements) - pred_size:
        if min_seq_len < 0:
            min_seq_len = 0
        if max_seq_len + elem_idx > len(elements) - pred_size:
            max_seq_len = len(elements) - (pred_size + elem_idx)
        sequence_len = random.randrange(min_seq_len, max_seq_len)
        h = [[delta, in_fov] for delta, in_fov, _ in elements[elem_idx:elem_idx+sequence_len]]
        prev_a = elements[elem_idx+sequence_len][2]
        a = [action for _, _, action in elements[elem_idx + sequence_len: elem_idx + sequence_len + pred_size]]
        p = [delta for delta, _, _ in elements[elem_idx + sequence_len: elem_idx + sequence_len + pred_size]]
        if padding:
            h.extend((max_seq_len - len(h)) * [0.0, 0.0])
        history.append(h)
        actions.append(a)
        prev_actions.append(prev_a)
        predictions.append(p)
        elem_idx += 1
    return history, prev_actions, actions, predictions


def preprocess_temp_file(tmp_fp, h_size, pred_size):
    with open(tmp_fp, "r") as tmp_f:
        lines = tmp_f.readlines()
    elements = list()
    for line in lines:
        delta, in_fov, action = line.split(",")
        elements.append((float(delta), float(in_fov), float(action)))
    elem_idx = 0
    history_elements = list()
    previous_actions = list()
    actions_elements = list()
    prediction_elements = list()
    while elem_idx < len(elements) - (h_size + pred_size):
        history = elements[elem_idx:elem_idx + h_size]
        if has_non_zero(history):
            predictions = elements[elem_idx + h_size: elem_idx + h_size + pred_size]
            h_elems = list()
            a_elems = list()
            p_elems = list()
            if has_non_zero(predictions):
                for delta, in_fov, _ in history:
                    h_elems.append([delta, in_fov])
                history_elements.append(h_elems)
                previous_actions.append(h_size * [history[-1][2]])
                for delta, _, action in predictions:
                    a_elems.append(action)
                    p_elems.append(delta)
                actions_elements.append(a_elems)
                prediction_elements.append(p_elems)
            else:
                elem_idx += 1
                continue
        elem_idx += 1
    return history_elements, previous_actions, actions_elements, prediction_elements


def preprocess_all_data(base_path, h_size, pred_size):
    files2parse = [os.path.join(base_path, f) for f in os.listdir(base_path) if "tmp" in f and ".npy" in f]
    actions_f = os.path.join(base_path, "actions.npy")
    observations_f = os.path.join(base_path, "observations.npy")
    predictions_f = os.path.join(base_path, "predictions.npy")
    prev_act_f = os.path.join(base_path, "prev_actions.npy")
    for f in files2parse:
        write_output_files(actions_f, observations_f, predictions_f, prev_act_f, f, h_size, pred_size)


def write_output_files(action_fp, obs_fp, pred_fp, prev_action_fp, tmp_fp, h_size, pred_size):
    history, prev_act, actions, predictions = preprocess_temp_file(tmp_fp, h_size, pred_size)
    with open(action_fp, "a") as act_f:
        with open(obs_fp, "a") as obs_f:
            with open(pred_fp, "a") as pred_f:
                with open(prev_action_fp, "a") as prev_f:
                    for h in history:
                        for idx in range(len(h)):
                            obs_f.write("{0},{1},".format(h[idx][0], h[idx][1]))
                        obs_f.write("\n")
                    for p_act in prev_act:
                        for idx in range(len(p_act)):
                            prev_f.write("{0},".format(p_act[idx]))
                        prev_f.write("\n")
                    for pred in predictions:
                        for idx in range(len(pred)):
                            pred_f.write("{0},".format(pred[idx]))
                        pred_f.write("\n")
                    for act in actions:
                        for idx in range(len(act)):
                            act_f.write("{0},".format(act[idx]))
                        act_f.write("\n")


def create_validation_data(base_path, temp_fn, h_size, pred_size):
    action_val_fn = os.path.join(base_path, "actions_val.npy")
    obs_val_fn = os.path.join(base_path, "observations_val.npy")
    pred_val_fn = os.path.join(base_path, "predictions_val.npy")
    prev_act_val_fn = os.path.join(base_path, "prev_actions_val.npy")
    write_output_files(action_val_fn, obs_val_fn, pred_val_fn, prev_act_val_fn, temp_fn, h_size, pred_size)


if __name__ == "__main__":
    preprocess_all_data(base_path=os.path.join(os.path.dirname(__file__),
                                               "resources", "traffic_cars_data", "state_estimation_data"), h_size=10,
                        pred_size=10)

    '''base_path = os.path.join(os.path.dirname(__file__), "resources", "traffic_cars_data", "state_estimation_data")
    temp_f = os.path.join(base_path, "tmp.npy")
    create_validation_data(base_path, temp_f, 10, 10)'''
