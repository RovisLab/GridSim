import os
import random
import math


RAY_MAX_LEN = 150.0

SENSOR_ARRAY_MIN_VAL = 0.0
SENSOR_ARRAY_MAX_VAL = 150.0


class ModelTypes(object):
    SIMPLIFIED = 0
    SENSOR_ARRAY = 1


class PreprocessorTypes(object):
    FIXED_LENGTH = 0
    VARIABLE_LENGTH = 1
    FULL_SEQUENCE = 2


class DataPreprocessor(object):
    def __init__(self, model_type, preprocessor_type, normalize):
        self.model_type = model_type
        self.preprocessor_type = preprocessor_type
        if self.model_type == ModelTypes.SIMPLIFIED:
            if self.preprocessor_type == PreprocessorTypes.FIXED_LENGTH:
                self.preprocessor = preprocess_temp_file
            else:
                self.preprocessor = variable_sequence_length_preprocessing
            self.writer = writer_simplified
            self.training_files = get_training_files_simplified
            if normalize is True:
                self.normalizer = normalize_simplified
        else:
            self.preprocessor = sensor_array_fixed_sequence_preprocessing
            self.writer = writer_sensor_array
            self.training_files = get_training_files_sensor_array
            if normalize is True:
                self.normalizer = normalize_sensor_array


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


def get_elements_sensor_array_gridsim_record_file(fp):
    if not isinstance(fp, tuple):
        raise Exception("Wrong data type. Needs tuple")
    dist_front = fp[0]
    dist_rear = fp[1]
    action_file = fp[2]
    with open(dist_front, "r") as tmp_ff:
        lines_front = tmp_ff.readlines()
    with open(dist_rear, "r") as tmp_fr:
        lines_rear = tmp_fr.readlines()
    with open(action_file, "r") as tmp_a:
        lines_actions = tmp_a.readlines()
    elements = list()
    for line_front, line_rear, action in zip(lines_front, lines_rear, lines_actions):
        front_elem = list()
        rear_elem = list()
        actions = list()
        for f in line_front.split(","):
            try:
                front_elem.append(float(f))
            except ValueError:
                pass
        for f in line_rear.split(","):
            try:
                rear_elem.append(float(f))
            except ValueError:
                pass
        for a in action.split(","):
            try:
                actions.append(float(a))
            except ValueError:
                pass
        elements.append((front_elem, rear_elem, actions))
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
                for elem_idx in range(len(h)):
                    h_elems.append((h[elem_idx][0][:], h[elem_idx][1][:]))
                history.append(h_elems)
                prev_actions.append(h_size * [h[-1][2][0]])
                for elem_idx in range(len(p)):
                    p_elems.append((p[elem_idx][0][:], p[elem_idx][1][:]))
                    a_elems.append(p[elem_idx][2][0])
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
                for idx in range(len(history)):
                    h_elems.append([history[0], history[1]])
                history_elements.append(h_elems)
                previous_actions.append(h_size * [history[-1][2]])
                for idx in range(len(predictions)):
                    a_elems.append(predictions[idx][2])
                    p_elems.append(predictions[idx][0])
                actions_elements.append(a_elems)
                prediction_elements.append(p_elems)
        elem_idx += 1
    return history_elements, previous_actions, actions_elements, prediction_elements


def get_min_max_obs():
    return -16.0, 0.0


def get_min_max_actions():
    return 0.0, 25.0


def get_min_max_predictions():
    return -16.0, 0.0


def get_min_max_prev_actions():
    return 0.0, 25.0


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


def normalize_simplified(observations, prev_actions, actions, predictions):
    return normalize_observations(observations), \
           normalize_prev_actions(prev_actions), \
           normalize_actions(actions), \
           normalize_predictions(predictions)


def normalize_sensor_array(observations, prev_actions, actions, predictions):
    prev_actions = normalize_prev_actions(prev_actions)
    actions = normalize_actions(actions)
    observations = normalize_observations_sensor_array(observations)
    predictions = normalize_predictions_sensor_array(predictions)

    return observations, prev_actions, actions, predictions


def normalize_observations_sensor_array(observations):
    for idx in range(len(observations)):
        for idx2 in range(len(observations[idx])):
            for idx3 in range(len(observations[idx][idx2][0])):
                observations[idx][idx2][0][idx3] = __normalize(observations[idx][idx2][0][idx3],
                                                               SENSOR_ARRAY_MIN_VAL,
                                                               SENSOR_ARRAY_MAX_VAL)
                observations[idx][idx2][1][idx3] = __normalize(observations[idx][idx2][1][idx3],
                                                               SENSOR_ARRAY_MIN_VAL,
                                                               SENSOR_ARRAY_MAX_VAL)
    return observations


def normalize_predictions_sensor_array(predictions):
    for idx in range(len(predictions)):
        for idx2 in range(len(predictions[idx])):
            for idx3 in range(len(predictions[idx][idx2][0])):
                predictions[idx][idx2][0][idx3] = __normalize(predictions[idx][idx2][0][idx3],
                                                              SENSOR_ARRAY_MIN_VAL,
                                                              SENSOR_ARRAY_MAX_VAL)
                predictions[idx][idx2][1][idx3] = __normalize(predictions[idx][idx2][1][idx3],
                                                              SENSOR_ARRAY_MIN_VAL,
                                                              SENSOR_ARRAY_MAX_VAL)
    return predictions


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


def get_training_files_simplified(base_path):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if "tmp" in f and f.endswith(".npy")]
    val_files = list()
    for f in files:
        if "_val" in f:
            val_files.append(f)
            files.remove(f)
    return [files, val_files]


def get_training_files_sensor_array(base_path):
    front_filenames = [os.path.join(base_path, f) for f in os.listdir(base_path) if "front_sensor_distances" in f]
    rear_filenames = [os.path.join(base_path, f) for f in os.listdir(base_path) if "rear_sensor_distances" in f]
    vel_filenames = [os.path.join(base_path, f) for f in os.listdir(base_path) if "velocity" in f]
    groups = list()
    val_groups = list()
    for ff, fr, fv in zip(front_filenames, rear_filenames, vel_filenames):
        if "_val" in ff and "_val" in fr and "_val" in fv:
            val_groups.append((ff, fr, fv))
        else:
            groups.append((ff, fr, fv))

    return [groups, val_groups]


class SequenceProcessor(object):
    def __init__(self,
                 data_processor,
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
                 normalize=False):
        self.data_processor = data_processor
        self.base_path = base_path
        self.h_size = h_size
        self.pred_size = pred_size
        self.min_seq = min_seq
        self.max_seq = max_seq
        self.full_seq = full_seq
        self.constant_seq = constant_seq
        self.normalize = normalize
        self.preprocessor = self.data_processor.preprocessor
        self.writer = self.data_processor.writer
        self.training_files = self.data_processor.training_files
        self.normalizer = self.data_processor.normalizer
        self.validation = False

    def __normalize(self, history, prev_actions, actions, predictions):
        return self.normalizer(observations=history, prev_actions=prev_actions,
                               actions=actions, predictions=predictions)

    def _create_validation_data(self, val_data_fn):
        history, prev_actions, actions, predictions = self.preprocessor(val_data_fn,
                                                                        self.h_size,
                                                                        self.pred_size,
                                                                        self.min_seq,
                                                                        self.max_seq,
                                                                        self.full_seq)
        if self.normalize:
            history, prev_actions, actions, predictions = self.__normalize(history=history,
                                                                           prev_actions=prev_actions,
                                                                           actions=actions,
                                                                           predictions=predictions)
        self.writer(self.base_path, history, prev_actions, actions, predictions, True)

    def process_all_data(self):
        files, val_files = self.training_files(self.base_path)
        if len(val_files) > 0:
            self.validation = True

        for f in files:
            history, prev_actions, actions, predictions = self.preprocessor(f,
                                                                            self.h_size,
                                                                            self.pred_size,
                                                                            self.min_seq,
                                                                            self.max_seq,
                                                                            self.full_seq)
            if self.normalize:
                history, prev_actions, actions, predictions = self.__normalize(history=history,
                                                                               prev_actions=prev_actions,
                                                                               actions=actions,
                                                                               predictions=predictions)
            self.writer(self.base_path, history, prev_actions, actions, predictions)

        if self.validation:
            for f in val_files:
                self._create_validation_data(f)


if __name__ == "__main__":
    m_type = ModelTypes.SENSOR_ARRAY
    p_type = PreprocessorTypes.FIXED_LENGTH
    norm = True
    dp = DataPreprocessor(model_type=m_type, preprocessor_type=p_type, normalize=norm)
    sp = SequenceProcessor(data_processor=dp,
                           normalize=norm,
                           h_size=10)
    sp.process_all_data()
