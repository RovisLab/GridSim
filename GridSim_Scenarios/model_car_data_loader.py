import os
import cv2
import numpy as np
from keras.utils import Sequence


def get_model_car_observations(base_path, obs_str):
    raw_elements = obs_str.split(",")
    processed_elements = list()
    if len(raw_elements[-1]) == 0:
        raw_elements.pop()
    for idx in range(0, len(raw_elements), 7):
        us_fl = float(raw_elements[idx])
        us_fcl = float(raw_elements[idx+1])
        us_fc = float(raw_elements[idx+2])
        us_fcr = float(raw_elements[idx+3])
        us_fr = float(raw_elements[idx+4])
        img_rgb = cv2.imread(os.path.join(base_path, raw_elements[idx+5]))
        img_d = cv2.imread(os.path.join(base_path, raw_elements[idx+6]))
        processed_elements.append((us_fl, us_fcl, us_fc, us_fcr, us_fr, img_rgb, img_d))
    return processed_elements


def get_model_car_predictions(pred_str):
    raw_elements = pred_str.split(",")
    processed_elements = list()
    if len(raw_elements[-1]) == 0:
        raw_elements.pop()
    for idx in range(0, len(raw_elements), 5):
        us_fl = float(raw_elements[idx])
        us_fcl = float(raw_elements[idx + 1])
        us_fc = float(raw_elements[idx + 2])
        us_fcr = float(raw_elements[idx + 3])
        us_fr = float(raw_elements[idx + 4])
        processed_elements.append((us_fl, us_fcl, us_fc, us_fcr, us_fr))
    return processed_elements


def get_model_car_actions(action_str):
    raw_elements = action_str.split(",")
    processed_elements = list()
    if len(raw_elements[-1]) == 0:
        raw_elements.pop()
    for idx in range(0, len(raw_elements), 2):
        vel = float(raw_elements[idx])
        steering_angle = float(raw_elements[idx+1])
        processed_elements.append((vel, steering_angle))
    return processed_elements


def get_model_car_prev_actions(prev_action_str):
    processed_elements = list()
    raw_elements = prev_action_str.split(",")
    if len(raw_elements[-1]) == 0:
        raw_elements.pop()
    for x in raw_elements:
        mini_list = list()
        for xx in x:
            mini_list.append(float(xx))
        processed_elements.append(mini_list)
    return processed_elements


class StateEstimationModelCarDataGenerator(Sequence):
    """
    Observation File
    (us_fl, us_fcl, us_fc, us_fcr, us_fr, img_rgb, img_d), ...
    ...

    Action File
    (velocity, steering_angle, ), ...
    ...

    Previous Action File
    (velocity, steering_angle)
    ...

    Prediction File
    (us_fl, us_fcl, us_fc, us_fcr, us_fr), ...
    ...
    """
    def __init__(self, input_file_path, batch_size, prediction_horizon_size, shuffle=True, validation=False):
        self.input_file_path = input_file_path
        self.rovis_desc_file = os.path.join(self.input_file_path, "desc_file.csv")
        self.sensor_desc_file = os.path.join(self.input_file_path, "sensor_desc.csv")
        self.batch_size = batch_size
        self.prediction_horizon_size = prediction_horizon_size
        self.shuffle = shuffle
        self.validation = validation
        self.action_file = os.path.join(input_file_path, "actions.npy") \
            if not self.validation else os.path.join(input_file_path, "actions_val.npy")

        self.observation_file = os.path.join(input_file_path, "observations.npy") \
            if not self.validation else os.path.join(input_file_path, "observations_val.npy")

        self.prediction_file = os.path.join(input_file_path, "predictions.npy") \
            if not self.validation else os.path.join(input_file_path, "predictions_val.npy")

        self.prev_action_file = os.path.join(input_file_path, "prev_actions.npy") \
            if not self.validation else os.path.join(input_file_path, "prev_actions_val.npy")

        self.num_samples = self.__get_num_samples()
        self.num_samples = self.num_samples if self.num_samples % batch_size == 0 \
            else self.num_samples - (self.num_samples % batch_size)
        self.prediction_horizon_size = prediction_horizon_size
        self.print_generator_details = True
        self.file_markers = list()
        self.file_markers.append((0, 0, 0, 0))
        self.cache_file_markers = list()
        self.__get_file_markers()

    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size))

    def __get_num_samples(self):
        with open(self.action_file, "r") as f:
            num_lines = sum(1 for line in f if len(line) > 1)
        return num_lines

    def __get_file_markers(self):
        with open(self.action_file, "r") as act_f:
            with open(self.prediction_file, "r") as pred_f:
                with open(self.prev_action_file, "r") as prev_act_f:
                    with open(self.observation_file, "r") as obs_f:
                        global_idx = 0
                        data_len = self.__len__()
                        while global_idx < data_len:
                            idx = 0
                            while idx < self.batch_size:
                                act_f.readline()
                                pred_f.readline()
                                prev_act_f.readline()
                                obs_f.readline()
                                idx += 1
                            # (obs, act, prev_act, pred)
                            self.file_markers.append((obs_f.tell(), act_f.tell(), prev_act_f.tell(), pred_f.tell()))
                            global_idx += 1
        self.file_markers.pop()
        for i in range(len(self.file_markers)):
            self.cache_file_markers.append(self.file_markers[i])

    def read_observations(self, obs_str):
        return get_model_car_observations(base_path=self.input_file_path, obs_str=obs_str)

    def read_predictions(self, pred_str):
        return get_model_car_predictions(pred_str=pred_str)

    def read_actions(self, action_str):
        return get_model_car_actions(action_str=action_str)

    def read_prev_actions(self, prev_act_str):
        return get_model_car_prev_actions(prev_action_str=prev_act_str)

    def get_batch(self):
        actions = list()
        observations = list()
        prev_actions = list()
        predictions = list()
        with open(self.action_file, "r") as act_f:
            with open(self.prediction_file, "r") as pred_f:
                with open(self.prev_action_file, "r") as prev_act_f:
                    with open(self.observation_file, "r") as obs_f:
                        obs_f.seek(self.file_markers[0][0])
                        act_f.seek(self.file_markers[0][1])
                        prev_act_f.seek(self.file_markers[0][2])
                        pred_f.seek(self.file_markers[0][3])

                        idx = 0
                        while idx < self.batch_size:
                            crt_actions = self.read_actions(act_f.readline())
                            crt_prev_actions = self.read_prev_actions(prev_act_f.readline())
                            crt_predictions = self.read_predictions(pred_f.readline())
                            crt_observations = self.read_observations(obs_f.readline())
                            if len(crt_actions) > 0:
                                actions.append(crt_actions)
                            if len(crt_observations) > 0:
                                observations.append(crt_observations)
                            # if len(crt_prev_actions) > 0:
                            prev_actions.append(crt_prev_actions)
                            if len(crt_predictions) > 0:
                                predictions.append(crt_predictions)
                            idx += 1
                        self.file_markers.pop(0)
        return actions, observations, prev_actions, predictions

    def __getitem__(self, item):
        actions, observations, prev_actions, predictions = self.get_batch()
        return self._format_data(actions, observations, prev_actions, predictions)

    def _format_data(self, observations, actions, prev_actions, predictions):
        for idx in range(len(prev_actions)):
            for idx2 in range(len(prev_actions[idx])):
                prev_actions[idx][idx2] = [prev_actions[idx][idx2]]
        observations = np.array(observations)
        actions = np.array(actions)
        prev_actions = np.array(prev_actions)

        p = list()
        if len(predictions):
            for idx in range(len(predictions[0])):
                pp = list()
                for idx2 in range(len(predictions)):
                    pp.append(predictions[idx2][idx])
                p.append(pp)
        return [observations, actions, prev_actions], p

    def on_epoch_end(self):
        pass
