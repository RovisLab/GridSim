import os
import csv
from keras.models import Model, load_model
from keras.layers import Input, Dense, GRU, Concatenate, concatenate, Reshape, Lambda
from keras.optimizers import Adam
from data_loader import StateEstimationDataGenerator
import keras.backend as K


def find_num(n, k):
    rem = n % k
    if rem == 0:
        return n
    else:
        return n - rem


def merge_training_files(base_path, fieldnames, prediction_horizon_size, history_size,
                         output_path_observations, output_path_actions, output_path_predictions):
    h_t, a_t = prepare_data(base_path, fieldnames, prediction_horizon_size, history_size)
    with open(output_path_observations, "w") as obs_f:
        with open(output_path_actions, "w") as action_f:
            with open(output_path_predictions, "w") as pred_f:
                for idx in range(len(h_t) - prediction_horizon_size):
                    obs_f.write("{0},{1}\n".format(h_t[idx][0], h_t[idx][1]))
                    for i in range(0, prediction_horizon_size):
                        action_f.write("{0},".format(a_t[idx + i + 1]))
                        pred_f.write("{0},".format(h_t[idx + i + 1][0]))
                    action_f.write("\n")
                    pred_f.write("\n")


def prepare_data(data_path, fieldnames, prediction_horizon_size, history_size):
    """
    Parse all csv files located in data_path folder
    :param data_path:
    :param fieldnames:
    :param prediction_horizon_size:
    :param history_size:
    :return:
    """
    if os.path.isdir(data_path):
        files = os.listdir(data_path)
        files = [f for f in files if "state_estimation" in f and f.endswith(".csv")]
        h_t = list()
        a_t = list()
        for f in files:
            with open(os.path.join(data_path, f), "r") as csv_file:
                reader = csv.DictReader(csv_file, fieldnames=fieldnames)
                len_f = len(list(reader))
                max_idx = find_num(len_f, prediction_horizon_size + history_size)
                idx = 0
                csv_file.seek(0)
                next(reader, None)  # skip header
                for row in reader:
                    if idx >= max_idx:
                        break
                    h_t.append((float(row["car_0_d_y"]), float(row["car_0_in_fov"])))
                    a_t.append(float(row["ego_vel"]))
                    idx += 1
        return h_t, a_t
    return [], []


def split_data_train_labels(h_t, a_t, prediction_horizon_size):
    observations, actions, results = list(), list(), list()
    for obs_idx in range(len(h_t) - prediction_horizon_size):
        observations.append((h_t[obs_idx][0], h_t[obs_idx][1]))
        for a_idx in range(prediction_horizon_size):
            actions.append(a_t[:obs_idx + a_idx + 1])
            results.append(h_t[obs_idx + a_idx + 1][0])
    return observations, actions, results


class WorldModel(object):
    def __init__(self, input_shape, prediction_horizon_size):
        self.input_shape = input_shape
        self.input_layer_num_units = 8
        self.mlp_layer_num_units = 8
        self.gru_layer_num_units = 128
        self.mlp_hidden_layer_size = prediction_horizon_size
        self.mlp_output_layer_size = 1
        self.model = None
        self.action_shape = (prediction_horizon_size,)
        self.batch_size = 32
        self._build_architecture()

    def _build_architecture(self):
        input_shape = Input(shape=self.input_shape)
        action_layer = Input(shape=self.action_shape)
        prev_action = Input(shape=(1,))
        gru_input = Concatenate()([input_shape, prev_action])
        input_gru = Reshape(target_shape=(1, int(gru_input.shape[1])))(gru_input)
        gru = GRU(units=self.gru_layer_num_units)(input_gru)
        mlp_outputs = list()
        for idx in range(self.mlp_hidden_layer_size):
            #mlp_inputs = action_layer[:idx + 1]
            mlp_inputs = Lambda(lambda x, i: x[:i])
            mlp_inputs.arguments = {'i': idx}
            mlp_inputs = mlp_inputs(action_layer)
            mlp_in = Concatenate()([gru, mlp_inputs])
            mlp = Dense(units=self.mlp_layer_num_units, activation="relu")(mlp_in)
            mlp_output = Dense(units=self.mlp_output_layer_size, activation="relu")(mlp)
            mlp_outputs.append(mlp_output)

        self.model = Model([input_shape, action_layer, prev_action], mlp_outputs)
        self.model.compile(optimizer=Adam(lr=0.00005), loss="categorical_crossentropy", metrics=["accuracy"])

    def train_network(self, epochs=10, batch_size=256):
        generator = StateEstimationDataGenerator(actions_file=os.path.join(os.path.dirname(__file__),
                                                                           "resources",
                                                                           "traffic_cars_data",
                                                                           "state_estimation_data",
                                                                           "actions.npy"),
                                                 observations_file=os.path.join(os.path.dirname(__file__),
                                                                                "resources",
                                                                                "traffic_cars_data",
                                                                                "state_estimation_data",
                                                                                "observations.npy"),
                                                 predictions_file=os.path.join(os.path.dirname(__file__),
                                                                               "resources",
                                                                               "traffic_cars_data",
                                                                               "state_estimation_data",
                                                                               "predictions.npy"),
                                                 batch_size=batch_size)
        self.model.fit_generator(generator=generator, epochs=epochs)

    def evaluate_network(self, x_test, y_test, batch_size=128):
        self.model.evaluate(x_test, y_test, batch_size=batch_size)

    def predict(self, data_frame):
        return self.model.predict(x=data_frame)

    def save_model(self, model_path):
        return self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)


if __name__ == "__main__":
    model = WorldModel(input_shape=(2,), prediction_horizon_size=10)
    model.train_network(epochs=10, batch_size=32)

