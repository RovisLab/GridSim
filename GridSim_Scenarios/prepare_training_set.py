import os
import random
import math


RAY_MAX_LEN = 150.0


def has_non_zero(arr):
    for delta, in_fov, action in arr:
        if in_fov != 0:
            return True
    return False


def has_non_zero_arr(arr):
    for front, rear, _ in arr:
        for f, r in zip(front, rear):
            if not math.isclose(f, RAY_MAX_LEN) or not math.isclose(r, RAY_MAX_LEN):
                return True
    return False


def get_elements_from_gridsim_record_file(tmp_fp):
    with open(tmp_fp, "r") as tmp_f:
        lines = tmp_f.readlines()
    elements = list()
    for line in lines:
        delta, in_fov, action = line.split(",")
        elements.append((float(delta), float(in_fov), float(action)))
    return elements


def get_elements_sensor_array_gridsim_record_file(tmp_fp):
    with open(tmp_fp, "r") as tmp_f:
        lines = tmp_f.readlines()
    elements = list()
    for line in lines:
        elem = list()
        for e in line.split(","):
            try:
                elem.append(float(e))
            except ValueError:
                pass
        elem.pop(0)  # remove size of record
        action = elem.pop(len(elem) - 1)
        elements.append((elem[:len(elem) // 2], elem[len(elem) // 2:], action))
    return elements


def sensor_array_fixed_sequence_preprocessing(tmp_fp,
                                              h_size,
                                              pred_size,
                                              min_seq_len=10,
                                              max_seq_len=100,
                                              full_sequence=False):
    elements = get_elements_sensor_array_gridsim_record_file(tmp_fp)
    history = list()
    actions = list()
    prev_actions = list()
    predictions = list()
    idx = 0
    while idx < len(elements) - (h_size + pred_size):
        h = elements[idx:idx + h_size]
        if has_non_zero_arr(h):
            p = elements[idx + h_size:idx+h_size+pred_size]
            if has_non_zero_arr(p):
                h_elems = list()
                a_elems = list()
                p_elems = list()
                for front, rear, a in h:
                    h_elems.append((front, rear))
                history.append(h_elems)
                prev_actions.append(h_size * [h[-1][2]])
                for front, rear, a in p:
                    p_elems.append((front, rear))
                    a_elems.append(a)
                actions.append(a_elems)
                predictions.append(p_elems)
        idx += 1
    return history, prev_actions, actions, predictions


def variable_sequence_length_preprocessing(tmp_fp,
                                           h_size,
                                           pred_size,
                                           min_seq_len=10,
                                           max_seq_len=100,
                                           full_sequence=False):
    """
    Return an increasing sequence of training data
    :param tmp_fp: temporary data file recorded with GridSim
    :param min_seq_len: minimum length of generated sequences
    :param max_seq_len: maximum length of generated sequences
    :param pred_size: prediction horizon size
    :param full_sequence: if True, return the full sequence - pred_size as history + pred_size predictions
    :param normalize: normalize data in [0-1] range
    :return: history, prev_actions, actions, predictions
    """
    elements = get_elements_from_gridsim_record_file(tmp_fp)
    elem_idx = 0
    history = list()
    actions = list()
    prev_actions = list()
    predictions = list()
    if full_sequence:
        seq_end = len(elements) - pred_size
        h = [[delta, in_fov] for delta, in_fov, _ in elements[:seq_end]]
        prev_a = len(h) * [elements[seq_end - 1][2]]
        a = [action for _, _, action in elements[seq_end:len(elements)]]
        p = [delta for delta, _, _ in elements[seq_end:len(elements)]]
        return [h], [prev_a], [a], [p]

    while True:
        if min_seq_len < 0:
            min_seq_len = 0
        if max_seq_len + elem_idx > len(elements) - pred_size:
            max_seq_len = len(elements) - (pred_size + elem_idx)
        if min_seq_len < max_seq_len:
            sequence_len = random.randrange(min_seq_len, max_seq_len)
        elif min_seq_len >= max_seq_len:
            sequence_len = min_seq_len

        if elem_idx + sequence_len + pred_size > len(elements):
            break

        h = [[delta, in_fov] for delta, in_fov, _ in elements[elem_idx:elem_idx+sequence_len]]
        prev_a = len(h) * [elements[elem_idx+sequence_len][2]]
        a = [action for _, _, action in elements[elem_idx + sequence_len: elem_idx + sequence_len + pred_size]]
        p = [delta for delta, _, _ in elements[elem_idx + sequence_len: elem_idx + sequence_len + pred_size]]
        history.append(h)
        actions.append(a)
        prev_actions.append(prev_a)
        predictions.append(p)
        elem_idx += 1
    return history, prev_actions, actions, predictions


def preprocess_temp_file(tmp_fp, h_size, pred_size, min_seq_len, max_seq_len, full_sequence):
    elements = get_elements_from_gridsim_record_file(tmp_fp)
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
        elem_idx += 1
    return history_elements, previous_actions, actions_elements, prediction_elements


def get_min_max_obs():
    return -16.0, 0.0


def get_min_max_actions():
    return -25.0, 25.0


def get_min_max_predictions():
    return -16.0, 0.0


def get_min_max_prev_actions():
    return -25.0, 25.0


def __normalize(x, min_val, max_val):
    return (x + abs(min_val)) / (max_val - min_val) if x != 0.0 else 0.0


def normalize_observations(observations):
    min_val, max_val = get_min_max_obs()
    for idx in range(len(observations)):
        for idx2 in range(len(observations[idx])):
            observations[idx][idx2][0] = __normalize(observations[idx][idx2][0], min_val, max_val)
    return observations


def normalize_actions(actions):
    min_val, max_val = get_min_max_actions()
    for idx in range(len(actions)):
        for idx2 in range(len(actions[idx])):
            actions[idx][idx2] = __normalize(actions[idx][idx2], min_val, max_val)
    return actions


def normalize_predictions(predictions):
    min_val, max_val = get_min_max_predictions()
    for idx in range(len(predictions)):
        for idx2 in range(len(predictions[idx])):
            predictions[idx][idx2] = __normalize(predictions[idx][idx2], min_val, max_val)
    return predictions


def normalize_prev_actions(prev_actions):
    min_val, max_val = get_min_max_prev_actions()
    for idx in range(len(prev_actions)):
        p_a_n = __normalize(prev_actions[idx][0], min_val, max_val)
        for idx2 in range(len(prev_actions[idx])):
            prev_actions[idx][idx2] = p_a_n
    return prev_actions


def writer_sensor_array(base_path, history, prev_act, actions, predictions, val=False):
    action_fp = os.path.join(base_path, "actions.npy") if val is False \
        else os.path.join(base_path, "actions_val.npy")
    obs_fp = os.path.join(base_path, "observations.npy") if val is False \
        else os.path.join(base_path, "observations_val.npy")
    pred_fp = os.path.join(base_path, "predictions.npy") if val is False \
        else os.path.join(base_path, "predictions_val.npy")
    prev_action_fp = os.path.join(base_path, "prev_actions.npy") if val is False \
        else os.path.join(base_path, "prev_actions_val.npy")
    with open(action_fp, "a") as act_f:
        with open(obs_fp, "a") as obs_f:
            with open(pred_fp, "a") as pred_f:
                with open(prev_action_fp, "a") as prev_f:
                    for obs in history:
                        obs_f.write("{0},{1},".format(len(obs), len(obs[0][0])))
                        for idx in range(len(obs)):
                            for f in obs[idx][0]:
                                obs_f.write("{0},".format(f))
                            for r in obs[idx][1]:
                                obs_f.write("{0},".format(r))
                        obs_f.write("\n")
                    for p_act in prev_act:
                        for idx in range(len(p_act)):
                            prev_f.write("{0},".format(p_act[idx]))
                        prev_f.write("\n")
                    for act in actions:
                        for idx in range(len(act)):
                            act_f.write("{0},".format(act[idx]))
                        act_f.write("\n")
                    for p in predictions:
                        for idx in range(len(p)):
                            for f in p[idx][0]:
                                pred_f.write("{0},".format(f))
                            for r in p[idx][1]:
                                pred_f.write("{0},".format(r))
                        pred_f.write("\n")


def writer_simplified(base_path, history, prev_act, actions, predictions, val=False):
    action_fp = os.path.join(base_path, "actions.npy") if val is False \
        else os.path.join(base_path, "actions_val.npy")
    obs_fp = os.path.join(base_path, "observations.npy") if val is False \
        else os.path.join(base_path, "observations_val.npy")
    pred_fp = os.path.join(base_path, "predictions.npy") if val is False \
        else os.path.join(base_path, "predictions_val.npy")
    prev_action_fp = os.path.join(base_path, "prev_actions.npy") if val is False \
        else os.path.join(base_path, "prev_actions_val.npy")
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


class SequenceProcessor(object):
    def __init__(self,
                 base_path=os.path.join(os.path.dirname(__file__),
                                        "resources",
                                        "traffic_cars_data",
                                        "state_estimation_data"),
                 h_size=10,
                 pred_size=10,
                 min_seq=10,
                 max_seq=100,
                 full_seq=False,
                 constant_seq=True,
                 normalize=False,
                 preprocessor=preprocess_temp_file,
                 writer=writer_simplified):
        self.base_path = base_path
        self.h_size = h_size
        self.pred_size = pred_size
        self.min_seq = min_seq
        self.max_seq = max_seq
        self.full_seq = full_seq
        self.constant_seq = constant_seq
        self.normalize = normalize
        self.preprocessor = preprocessor
        self.writer = writer
        self.validation = False

    def __normalize(self, history, prev_actions, actions, predictions):
        return normalize_observations(history), \
               normalize_prev_actions(prev_actions), \
               normalize_actions(actions), \
               normalize_predictions(predictions)

    def _create_validation_data(self, val_data_fn):
        history, prev_actions, actions, predictions = self.preprocessor(val_data_fn,
                                                                        self.h_size,
                                                                        self.pred_size,
                                                                        self.min_seq,
                                                                        self.max_seq,
                                                                        self.full_seq)
        if self.normalize:
            history, prev_actions, actions, predictions = self.__normalize(history,
                                                                           prev_actions,
                                                                           actions,
                                                                           predictions)
        self.writer(self.base_path, history, prev_actions, actions, predictions, True)

    def process_all_data(self):
        files = [os.path.join(self.base_path, f) for f in os.listdir(self.base_path) if "tmp" in f and f.endswith(".npy")]
        val_file = None
        for f in files:
            if "_val" in f:
                val_file = f
                files.remove(f)
        if val_file is not None:
            self.validation = True

        for f in files:
            history, prev_actions, actions, predictions = self.preprocessor(f,
                                                                            self.h_size,
                                                                            self.pred_size,
                                                                            self.min_seq,
                                                                            self.max_seq,
                                                                            self.full_seq)
            if self.normalize:
                history, prev_actions, actions, predictions = self.__normalize(history,
                                                                               prev_actions,
                                                                               actions,
                                                                               predictions)
            self.writer(self.base_path, history, prev_actions, actions, predictions)

        if self.validation:
            self._create_validation_data(val_file)


if __name__ == "__main__":
    sp = SequenceProcessor(normalize=False,
                           h_size=10,
                           preprocessor=sensor_array_fixed_sequence_preprocessing,
                           writer=writer_sensor_array)
    sp.process_all_data()
