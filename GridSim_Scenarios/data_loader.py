from keras.utils import Sequence
import numpy as np


class StateEstimationDataGenerator(Sequence):
    def __init__(self, actions_file, observations_file, predictions_file, batch_size):
        self.batch_size = batch_size
        self.action_file = actions_file
        self.observation_file = observations_file
        self.prediction_file = predictions_file
        self.num_samples = self.__get_num_samples()
        self.last_fp_actions = 0
        self.last_fp_predictions = 0
        self.last_fp_observations = 0
        self.prev_action = 0.0

    def __get_num_samples(self):
        with open(self.action_file, "r") as f:
            num_lines = sum(1 for line in f if len(line) > 1)
        return num_lines

    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size))

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

    def __getitem__(self, item):
        observations = list()
        actions = list()
        predictions = list()
        with open(self.observation_file, "r") as obs_f:
            with open(self.prediction_file, "r") as pred_f:
                with open(self.action_file, "r") as action_f:
                    obs_f.seek(offset=self.last_fp_observations)
                    pred_f.seek(offset=self.last_fp_predictions)
                    action_f.seek(offset=self.last_fp_actions)
                    idx = 0
                    while idx < self.batch_size:
                        observations.append(self._process_obs(obs_f.readline()))
                        actions.append(self._process_action(action_f.readline()))
                        predictions.append(self._process_prediction(pred_f.readline()))
                        self.prev_action = actions[0]
                    self.last_fp_predictions = pred_f.tell()
                    self.last_fp_actions = action_f.tell()
                    self.last_fp_observations = obs_f.tell()
        return [observations, actions, self.prev_action], predictions


class DataGenerator(Sequence):
    def __init__(self, list_ids, labels, batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #  Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        # Generates data containing batch_size samples: (n_samples, *dim, n_channels)
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=float)

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # Store sample
            x[i, ] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]
        return x, y
