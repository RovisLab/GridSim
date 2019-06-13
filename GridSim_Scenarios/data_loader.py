from keras.utils import Sequence
import numpy as np


class StateEstimationDataGenerator(Sequence):
    def __init__(self, actions_file, observations_file, predictions_file, batch_size, history_size, prediction_horizon_size):
        self.batch_size = batch_size
        self.action_file = actions_file
        self.observation_file = observations_file
        self.prediction_file = predictions_file
        self.num_samples = self.__get_num_samples()
        self.history_size = history_size
        self.prediction_horizon_size = prediction_horizon_size
        self.last_fp_actions = 0
        self.last_fp_predictions = 0
        self.last_fp_observations = 0
        self.print_generator_details = True
        if self.print_generator_details:
            print("State Estimation Generator: number of samples: {0}, batch_size: {1}, num_steps: {2}".format(
                self.num_samples, self.batch_size, self.__len__()
            ))

        self.observations = list()
        self.actions = list()
        self.prediction = list()
        self.history = list()
        self.prev_actions = list()
        self.pred_actions = list()
        self.final_predictions = list()

    def __get_num_samples(self):
        with open(self.action_file, "r") as f:
            num_lines = sum(1 for line in f if len(line) > 1)
        return num_lines

    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size) - self.history_size)

    def _process_obs(self, observation_str):
        f_elems = list()
        elements = observation_str.split(",")
        try:
            for elem in elements:
                f_elems.append(float(elem))
        except ValueError:
            pass
        return f_elems

    def _process_action(self, action_str):
        f_elems = list()
        elements = action_str.split(",")
        try:
            for elem in elements:
                f_elems.append(float(elem))
        except ValueError:
            pass
        return f_elems

    def _process_prediction(self, pred_str):
        f_elems = list()
        elements = pred_str.split(",")
        try:
            for elem in elements:
                f_elems.append(float(elem))
        except ValueError:
            pass
        return f_elems

    def buffer_data(self):
        with open(self.observation_file, "r") as obs_f:
            with open(self.prediction_file, "r") as pred_f:
                with open(self.action_file, "r") as action_f:
                    obs_f.seek(self.last_fp_observations)
                    pred_f.seek(self.last_fp_predictions)
                    action_f.seek(self.last_fp_actions)
                    idx = 0
                    while idx < self.batch_size * self.history_size:
                        obs = self._process_obs(obs_f.readline())
                        if len(obs) == 0:
                            break
                        self.observations.append(obs)
                        pred = self._process_prediction(pred_f.readline())
                        if len(pred) == 0:
                            break
                        self.prediction.append(pred)
                        act = self._process_action(action_f.readline())
                        if len(act) == 0:
                            break
                        self.actions.append(act)
                        idx += 1
                    self.last_fp_actions = action_f.tell()
                    self.last_fp_predictions = pred_f.tell()
                    self.last_fp_observations = obs_f.tell()

    def process_data(self):
        if len(self.observations) <= self.batch_size * self.history_size:
            self.buffer_data()
        self.history = list()
        self.prev_actions = list()
        self.pred_actions = list()
        self.final_predictions = list()
        batch_idx = 0
        while batch_idx < self.batch_size:
            seq_idx = 0
            crt_h_seq = list()
            while seq_idx < self.history_size:
                crt_h_seq.append(self.observations[seq_idx])
                seq_idx += 1
            self.prev_actions.append(self.actions[seq_idx - 1])
            self.history.append(crt_h_seq)
            self.pred_actions.append(
                self.actions[seq_idx + self.prediction_horizon_size]
            )
            self.final_predictions.append(
                self.prediction[seq_idx + self.prediction_horizon_size]
            )
            # Remove consumed observations, actions and predictions
            self.observations.pop(0)
            self.actions.pop(0)
            self.prediction.pop(0)

            batch_idx += 1
        p = list()
        for idx in range(len(self.final_predictions[0])):
            pp = list()
            for idx2 in range(len(self.final_predictions)):
                pp.append(self.final_predictions[idx2][idx])
            p.append(pp)
        return [np.array(self.history), np.array(self.pred_actions),
                np.array(self.prev_actions).reshape((self.batch_size, self.history_size, 1))], p

    def __getitem__(self, item):
        """
        observations = list()
        actions = list()
        predictions = list()
        prev_actions = list()
        prev_actions.append(0.0)
        with open(self.observation_file, "r") as obs_f:
            with open(self.prediction_file, "r") as pred_f:
                with open(self.action_file, "r") as action_f:
                    obs_f.seek(self.last_fp_observations)
                    pred_f.seek(self.last_fp_predictions)
                    action_f.seek(self.last_fp_actions)
                    idx = 0
                    while idx < self.batch_size:
                        obs = self._process_obs(obs_f.readline())
                        observations.append(obs)
                        actions.append(self._process_action(action_f.readline()))
                        predictions.append(self._process_prediction(pred_f.readline()))
                        idx += 1
                    for idx in range(1, len(actions)):
                        prev_actions.append(actions[idx][-1])
                    self.last_fp_predictions = pred_f.tell()
                    self.last_fp_actions = action_f.tell()
                    self.last_fp_observations = obs_f.tell()

        p = list()
        for idx in range(len(predictions[0])):
            pp = list()
            for idx2 in range(len(predictions)):
                pp.append(predictions[idx2][idx])
            p.append(np.array(pp).reshape((len(predictions), 1)))

        return [np.array(observations), np.array(actions), np.array(prev_actions)], p
        """
        return self.process_data()

    def on_epoch_end(self):
        self.last_fp_observations = 0
        self.last_fp_actions = 0
        self.last_fp_predictions = 0
