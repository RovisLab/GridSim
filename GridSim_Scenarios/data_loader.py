from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import random
import cv2


class StateEstimationDataGeneratorImpl(object):
    def __init__(self, input_file_path, batch_size, prediction_horizon_size, shuffle=True, validation=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation = validation
        if self.validation:
            self.shuffle = False
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
        self.file_markers = list()  # (obs, act, prev_act, pred)
        self.file_markers.append((0, 0, 0, 0))
        self.cache_file_markers = list()
        self.__get_file_markers()

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
        elements = list()
        for elem in obs_str.split(","):
            try:
                elements.append(float(elem))
            except ValueError:
                pass
        observations = list()
        in_fovs = list()
        for h_idx in range(0, len(elements), 2):
            observations.append(elements[h_idx])
            in_fovs.append(elements[h_idx + 1])
        return observations, in_fovs

    def read_actions(self, action_str):
        elements = list()
        for elem in action_str.split(","):
            try:
                elements.append(float(elem))
            except ValueError:
                pass
        actions = list()
        if len(elements) == self.prediction_horizon_size:
            for act_idx in range(self.prediction_horizon_size):
                actions.append(elements[act_idx])
        return actions

    def read_predictions(self, pred_str):
        elements = list()
        for elem in pred_str.split(","):
            try:
                elements.append(float(elem))
            except ValueError:
                pass
        predictions = list()
        if len(elements) == self.prediction_horizon_size:
            for pred_idx in range(self.prediction_horizon_size):
                predictions.append(elements[pred_idx])
        return predictions

    def read_prev_actions(self, prev_action_str):
        elements = list()
        for elem in prev_action_str.split(","):
            try:
                elements.append(float(elem))
            except ValueError:
                pass
        return elements

    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size))

    def get_batch(self):
        actions = list()
        observations = list()
        in_fovs = list()
        prev_actions = list()
        predictions = list()
        with open(self.action_file, "r") as act_f:
            with open(self.prediction_file, "r") as pred_f:
                with open(self.prev_action_file, "r") as prev_act_f:
                    with open(self.observation_file, "r") as obs_f:
                        # (obs, act, prev_act, pred)
                        obs_f.seek(self.file_markers[0][0])
                        act_f.seek(self.file_markers[0][1])
                        prev_act_f.seek(self.file_markers[0][2])
                        pred_f.seek(self.file_markers[0][3])

                        idx = 0
                        while idx < self.batch_size:
                            crt_actions = self.read_actions(act_f.readline())
                            crt_prev_actions = self.read_prev_actions(prev_act_f.readline())
                            crt_predictions = self.read_predictions(pred_f.readline())
                            crt_observations, crt_in_fovs = self.read_observations(obs_f.readline())
                            if len(crt_actions) > 0:
                                actions.append(crt_actions)
                            if len(crt_observations) > 0:
                                observations.append(crt_observations)
                            if len(crt_in_fovs) > 0:
                                in_fovs.append(crt_in_fovs)
                            if len(crt_prev_actions) > 0:
                                prev_actions.append(crt_prev_actions)
                            if len(crt_predictions) > 0:
                                predictions.append(crt_predictions)
                            idx += 1
                        self.file_markers.pop(0)
        return actions, observations, in_fovs, prev_actions, predictions


class StateEstimationSensorArrayDataGeneratorImpl(StateEstimationDataGeneratorImpl):
    def __init__(self, input_file_path, batch_size, prediction_horizon_size, shuffle=True, validation=False):
        super(StateEstimationSensorArrayDataGeneratorImpl, self).__init__(input_file_path=input_file_path,
                                                                          batch_size=batch_size,
                                                                          prediction_horizon_size=prediction_horizon_size,
                                                                          shuffle=shuffle,
                                                                          validation=validation)

    def read_observations(self, obs_str):
        elements = list()
        for elem in obs_str.split(","):
            try:
                elements.append(float(elem))
            except ValueError:
                pass
        observations = list()
        if len(elements) > 2:
            seq_num = elements[0]
            seq_size = elements[1]
            for h_idx in range(2, len(elements)):
                observations.append(elements[h_idx])
            observations = np.array(observations).reshape((int(seq_num), 2 * int(seq_size))).tolist()
        return observations

    def read_predictions(self, pred_str):
        elements = list()
        for elem in pred_str.split(","):
            try:
                elements.append(float(elem))
            except ValueError:
                pass
        predictions = list()
        for h_idx in range(len(elements)):
            predictions.append(elements[h_idx])
        predictions = np.array(predictions).reshape((self.prediction_horizon_size, -1)).tolist()
        return predictions

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


class StateEstimationSensorArrayDataGenerator(Sequence):
    def __init__(self, input_file_path, batch_size, prediction_horizon_size, shuffle=True, validation=False):
        self.__impl__ = StateEstimationSensorArrayDataGeneratorImpl(input_file_path=input_file_path,
                                                                    batch_size=batch_size,
                                                                    prediction_horizon_size=prediction_horizon_size,
                                                                    shuffle=shuffle,
                                                                    validation=validation)

    def __len__(self):
        return self.__impl__.__len__()

    def __getitem__(self, item):
        actions, observations, prev_actions, predictions = self.__impl__.get_batch()
        return self._format_data(observations, actions, prev_actions, predictions)

    def _format_data(self, observations, actions, prev_actions, predictions):
        for idx in range(len(prev_actions)):
            for idx2 in range(len(prev_actions[idx])):
                prev_actions[idx][idx2] = [prev_actions[idx][idx2]]
        observations = np.array(observations)
        actions = np.array(actions)
        #prev_actions = np.array(prev_actions)
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
        self.__impl__.file_markers = list()
        for i in range(len(self.__impl__.cache_file_markers)):
            self.__impl__.file_markers.append(self.__impl__.cache_file_markers[i])
        if self.__impl__.shuffle is True:
            random.shuffle(self.__impl__.file_markers)

    def reset_file_markers(self):
        for i in range(len(self.__impl__.cache_file_markers)):
            self.__impl__.file_markers.append(self.__impl__.cache_file_markers[i])


class StateEstimationDataGenerator(Sequence):
    def __init__(self, input_file_path, batch_size, prediction_horizon_size, shuffle=True, validation=False):
        self.__impl__ = StateEstimationDataGeneratorImpl(input_file_path=input_file_path,
                                                         batch_size=batch_size,
                                                         prediction_horizon_size=prediction_horizon_size,
                                                         shuffle=shuffle,
                                                         validation=validation)

    def __len__(self):
        return self.__impl__.__len__()

    def __getitem__(self, item):
        actions, observations, in_fovs, prev_actions, predictions = self.__impl__.get_batch()
        return self._format_data(observations, in_fovs, actions, prev_actions, predictions)

    def _format_data(self, observations, in_fovs, actions, prev_actions, predictions):
        for idx in range(len(observations)):
            for idx2 in range(len(observations[idx])):
                observations[idx][idx2] = [observations[idx][idx2]]
        for idx in range(len(in_fovs)):
            for idx2 in range(len(in_fovs[idx])):
                in_fovs[idx][idx2] = [in_fovs[idx][idx2]]
        for idx in range(len(prev_actions)):
            for idx2 in range(len(prev_actions[idx])):
                prev_actions[idx][idx2] = [prev_actions[idx][idx2]]
        observations = pad_sequences(observations)
        in_fovs = pad_sequences(in_fovs)
        actions = np.array(actions)
        prev_actions = pad_sequences(prev_actions)

        p = list()
        if len(predictions):
            for idx in range(len(predictions[0])):
                pp = list()
                for idx2 in range(len(predictions)):
                    pp.append(predictions[idx2][idx])
                p.append(pp)

        return [observations, in_fovs, actions, prev_actions], p

    def reset_file_markers(self):
        for i in range(len(self.__impl__.cache_file_markers)):
            self.__impl__.file_markers.append(self.__impl__.cache_file_markers[i])

    def on_epoch_end(self):
        self.__impl__.file_markers = list()
        for i in range(len(self.__impl__.cache_file_markers)):
            self.__impl__.file_markers.append(self.__impl__.cache_file_markers[i])
        if self.__impl__.shuffle is True:
            random.shuffle(self.__impl__.file_markers)


class StateEstimationModelCar(Sequence):
    """
    rovis_data_descriptor_file:
    timestamp, delta_ts, path2image_rgb (e.g 1234, 12, ./samples/1234RGB.png)
    timestamp, delta_ts, path2image_d
    ...

    sensor_data_descriptor_file:
    timestamp, velocity, steering_angle, x, y, pitch, us_fl, us_fcl, us_fc, us_fcr, us_fr
    ...
    """
    def __init__(self, input_file_path, batch_size, prediction_horizon_size, shuffle=True, validation=False):
        self.input_file_path = input_file_path
        self.batch_size = batch_size
        self.prediction_horizon_size = prediction_horizon_size
        self.shuffle = shuffle
        self.validation = validation

    def parse_rovis_descriptor_file(self):
        pass

    def get_batch(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def on_epoch_end(self):
        pass
