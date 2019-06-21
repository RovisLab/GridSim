from keras.utils import Sequence
import os
import numpy as np
import random


class StateEstimationDataGenerator(Sequence):
    def __init__(self, input_file_path, batch_size, history_size, prediction_horizon_size, shuffle=True, validation=False, normalize=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation = validation
        if self.validation:
            self.shuffle = False
        self.normalize = normalize
        self.action_file = os.path.join(input_file_path, "actions.npy") \
            if not self.validation else os.path.join(input_file_path, "actions_val.npy")
        self.observation_file = os.path.join(input_file_path, "observations.npy") \
            if not self.validation else os.path.join(input_file_path, "observations_val.npy")
        self.prediction_file = os.path.join(input_file_path, "predictions.npy") \
            if not self.validation else os.path.join(input_file_path, "predictions_val.npy")
        self.prev_action_file = os.path.join(input_file_path, "prev_actions.npy") \
            if not self.validation else os.path.join(input_file_path, "prev_actions_val.npy")
        self.observation_file_n = os.path.join(input_file_path, "observations_n.npy")
        self.action_file_n = os.path.join(input_file_path, "actions_n.npy")
        self.prediction_file_n = os.path.join(input_file_path, "predictions_n.npy")
        self.prev_action_file_n = os.path.join(input_file_path, "prev_actions_n.npy")

        self.num_samples = self.__get_num_samples()
        self.num_samples = self.num_samples if self.num_samples % batch_size == 0 else self.num_samples - (self.num_samples % batch_size)
        self.history_size = history_size
        self.prediction_horizon_size = prediction_horizon_size
        self.last_fp_actions = 0
        self.last_fp_predictions = 0
        self.last_fp_observations = 0
        self.last_fp_prev_actions = 0
        self.print_generator_details = True
        self.file_markers = list()  # (obs, act, prev_act, pred)
        self.file_markers.append((0, 0, 0, 0))
        self.cache_file_markers = list()
        self.__get_file_markers()
        if self.print_generator_details:
            print("State Estimation Generator: number of samples: {0}, batch_size: {1}, num_steps: {2}".format(
                self.num_samples, self.batch_size, self.__len__()
            ))
            print("Data Generator Length: {0}".format(self.__len__()))

        if self.normalize:
            self.__normalize_actions()
            self.__normalize_observations()
            self.__normalize_predictions()
            self.__normalize_prev_actions()

            self.action_file = self.action_file_n
            self.observation_file = self.observation_file_n
            self.prediction_file = self.prediction_file_n
            self.prev_action_file = self.prev_action_file_n

    def __get_min_max_obs(self):
        with open(self.observation_file, "r") as obs_f:
            min_val = 1000
            max_val = -1000
            while True:
                observations = self.read_observations(obs_f.readline())
                if len(observations) == 0:
                    break
                observations = [observations[idx][0] for idx in range(len(observations))]
                min_obs = min(observations)
                if min_obs < min_val:
                    min_val = min_obs
                max_obs = max(observations)
                if max_obs > max_val:
                    max_val = max_obs
        return min_val, max_val

    def __get_min_max_actions(self):
        with open(self.action_file, "r") as act_f:
            min_val = 1000
            max_val = -1000
            while True:
                actions = self.read_actions(act_f.readline())
                if len(actions) == 0:
                    break
                min_act = min(actions)
                max_act = max(actions)
                if min_act < min_val:
                    min_val = min_act
                if max_act > max_val:
                    max_val = max_act
        return min_val, max_val

    def __get_min_max_predictions(self):
        with open(self.prediction_file, "r") as pred_f:
            min_val = 1000
            max_val = -1000
            while True:
                predictions = self.read_predictions(pred_f.readline())
                if len(predictions) == 0:
                    break
                min_pred = min(predictions)
                max_pred = max(predictions)
                if min_pred < min_val:
                    min_val = min_pred
                if max_pred > max_val:
                    max_val = max_pred
        return min_val, max_val

    def __get_min_max_prev_actions(self):
        with open(self.prev_action_file, "r") as prev_f:
            min_val = 1000
            max_val = -1000
            while True:
                prev_actions = self.read_prev_actions(prev_f.readline())
                if len(prev_actions) == 0:
                    break
                min_prev = min(prev_actions)
                max_prev = max(prev_actions)
                if min_prev < min_val:
                    min_val = min_prev
                if max_prev > max_val:
                    max_val = max_prev
        return min_val, max_val

    def __normalize(self, x, min_val, max_val):
        return (x + abs(min_val)) / (max_val - min_val)

    def __normalize_observations(self):
        min_val, max_val = self.__get_min_max_obs()
        with open(self.observation_file, "r") as obs_f:
            with open(self.observation_file_n, "w") as obs_fn:
                while True:
                    observations = self.read_observations(obs_f.readline())
                    if len(observations) == 0:
                        break
                    for idx in range(0, len(observations)):
                        observations[idx][0] = self.__normalize(observations[idx][0], min_val, max_val)
                    for idx in range(len(observations)):
                        obs_fn.write("{0},{1},".format(observations[idx][0], observations[idx][1]))
                    obs_fn.write("\n")

    def __normalize_actions(self):
        min_val, max_val = self.__get_min_max_actions()
        with open(self.action_file, "r") as act_f:
            with open(self.action_file_n, "w") as act_fn:
                while True:
                    actions = self.read_actions(act_f.readline())
                    if len(actions) == 0:
                        break
                    for idx in range(len(actions)):
                        actions[idx] = self.__normalize(actions[idx], min_val, max_val)
                    for act in actions:
                        act_fn.write("{0},".format(act))
                    act_fn.write("\n")

    def __normalize_predictions(self):
        min_val, max_val = self.__get_min_max_predictions()
        with open(self.prediction_file, "r") as pred_f:
            with open(self.prediction_file_n, "w") as pred_fn:
                while True:
                    predictions = self.read_predictions(pred_f.readline())
                    if len(predictions) == 0:
                        break
                    for idx in range(len(predictions)):
                        predictions[idx] = self.__normalize(predictions[idx], min_val, max_val)
                    for pred in predictions:
                        pred_fn.write("{0},".format(pred))
                    pred_fn.write("\n")

    def __normalize_prev_actions(self):
        min_val, max_val = self.__get_min_max_prev_actions()
        with open(self.prev_action_file, "r") as prev_f:
            with open(self.prev_action_file_n, "w") as prev_fn:
                while True:
                    prev_actions = self.read_prev_actions(prev_f.readline())
                    if len(prev_actions) == 0:
                        break
                    for idx in range(len(prev_actions)):
                        prev_actions[idx] = self.__normalize(prev_actions[idx], min_val, max_val)
                    for pred in prev_actions:
                        prev_fn.write("{0},".format(pred))
                    prev_fn.write("\n")

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
        for i in range(len(self.file_markers)):
            self.cache_file_markers.append(self.file_markers[i])

    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size) - self.history_size)

    def read_observations(self, obs_str):
        elements = list()
        for elem in obs_str.split(","):
            try:
                elements.append(float(elem))
            except ValueError:
                pass
        observations = list()
        if len(elements) == 2 * self.history_size:
            for h_idx in range(0, 2 * self.history_size, 2):
                observations.append([elements[h_idx], elements[h_idx + 1]])
        return observations

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

    def __getitem__(self, item):
        actions = list()
        observations = list()
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
                            crt_observations = self.read_observations(obs_f.readline())
                            if len(crt_actions) > 0:
                                actions.append(crt_actions)
                            if len(crt_observations) > 0:
                                observations.append(crt_observations)
                            if len(crt_prev_actions) > 0:
                                prev_actions.append(crt_prev_actions)
                            if len(crt_predictions) > 0:
                                predictions.append(crt_predictions)
                            idx += 1
                        self.file_markers.pop(0)

        p = list()
        if len(predictions):
            for idx in range(len(predictions[0])):
                pp = list()
                for idx2 in range(len(predictions)):
                    pp.append(predictions[idx2][idx])
                p.append(pp)

        return [np.array(observations), np.array(actions),
                np.array(prev_actions).reshape((len(prev_actions), self.history_size, 1))], p

    def on_epoch_end(self):
        self.file_markers = list()
        for i in range(len(self.cache_file_markers)):
            self.file_markers.append(self.cache_file_markers[i])
        if self.shuffle is True:
            random.shuffle(self.file_markers)
