import os
from keras.models import Model, load_model
from keras.layers import Input, Dense, GRU, Concatenate, concatenate, Reshape, Lambda
from keras.optimizers import Adam
from data_loader import StateEstimationDataGenerator


def find_num(n, k):
    rem = n % k
    if rem == 0:
        return n
    else:
        return n - rem


def extract_observations(tmp_fp, h_size):
    h_seq_idx = 0
    obs_batch = list()
    prev_action = None
    remember_pos = 0
    while h_seq_idx < h_size:
        elements = list()
        if h_seq_idx == 1:
            remember_pos = tmp_fp.tell()  # remember file cursor position for next iteration
        line = tmp_fp.readline().split(",")
        if len(line) == 0:
            break
        for e in line:
            try:
                elements.append(float(e))
            except ValueError:
                pass
        obs_batch.append(elements[:2])
        prev_action = [elements[2] for _ in range(h_size)]
        h_seq_idx += 1
    return remember_pos, prev_action, obs_batch


def extract_predictions_and_actions(tmp_fp, pred_h_size):
    act_seq_idx = 0
    act_batch = list()
    pred_batch = list()
    while act_seq_idx < pred_h_size:
        elements = list()
        line = tmp_fp.readline().split(",")
        if len(line) < 3:
            break
        for e in line:
            try:
                elements.append(float(e))
            except ValueError:
                pass
        act_batch.append(elements[2])
        pred_batch.append(elements[0])
        act_seq_idx += 1
    return pred_batch, act_batch


def convert_gridsim_output(fp_in, fp_out, h_size, pred_h_size):
    with open(fp_in, "r") as f:
        num_lines = sum(1 for line in f if len(line) > 1)
    with open(fp_in, "r") as tmp_f:
        with open(os.path.join(fp_out, "actions.npy"), "a") as act_f:
            with open(os.path.join(fp_out, "predictions.npy"), "a") as pred_f:
                with open(os.path.join(fp_out, "observations.npy"), "a") as obs_f:
                    with open(os.path.join(fp_out, "prev_actions.npy"), "a") as prev_act_f:
                        idx = 0
                        remember_cursor = 0
                        while idx < num_lines - h_size - (pred_h_size - 1):
                            # Extract training data from temp file
                            tmp_f.seek(remember_cursor)
                            remember_cursor, prev_action, obs_batch = extract_observations(tmp_f, h_size)
                            pred_batch, act_batch = extract_predictions_and_actions(tmp_f, pred_h_size)

                            # Write to output files
                            for i in range(len(prev_action)):
                                prev_act_f.write("{0},".format(prev_action[i]))
                            prev_act_f.write("\n")
                            for i in range(len(obs_batch)):
                                obs_f.write("{0},{1},".format(obs_batch[i][0], obs_batch[i][1]))
                            obs_f.write("\n")
                            for i in range(len(pred_batch)):
                                pred_f.write("{0},".format(pred_batch[i]))
                            pred_f.write("\n")
                            for i in range(len(act_batch)):
                                act_f.write("{0},".format(act_batch[i]))
                            act_f.write("\n")

                            # Update while index
                            idx += 1


def preprocess_all_training_data(base_path, h_size, pred_h_size):
    files_to_parse = [os.path.join(base_path, x) for x in os.listdir(base_path) if "tmp" in x and ".npy" in x]
    for f in files_to_parse:
        convert_gridsim_output(f, base_path, h_size, pred_h_size)


class WorldModel(object):
    def __init__(self, prediction_horizon_size, history_size):
        self.input_shape = (history_size, 2)  # 2 - observation size (pos_y_dif, in_fov)
        self.input_layer_num_units = 8
        self.mlp_layer_num_units = 8
        self.gru_layer_num_units = 128
        self.mlp_hidden_layer_size = prediction_horizon_size
        self.history_size = history_size
        self.mlp_output_layer_size = 1
        self.model = None
        self.action_shape = (prediction_horizon_size,)
        self.batch_size = 32
        self.print_summary = True
        self._build_architecture()

    def _build_architecture(self):
        input_layer = Input(shape=self.input_shape)
        action_layer = Input(shape=self.action_shape)
        prev_action = Input(shape=(self.history_size, 1))
        gru_input = Concatenate()([input_layer, prev_action])
        # input_gru = Reshape(target_shape=(self.history_size, int(gru_input.shape[1])))(gru_input)
        gru = GRU(units=self.gru_layer_num_units)(gru_input)
        mlp_outputs = list()
        for idx in range(self.mlp_hidden_layer_size):
            mlp_inputs = Lambda(lambda x: x[:, :idx+1])(action_layer)
            mlp_in = Concatenate()([gru, mlp_inputs])
            mlp = Dense(units=self.mlp_layer_num_units, activation="relu")(mlp_in)
            mlp_output = Dense(units=self.mlp_output_layer_size, activation="relu")(mlp)
            mlp_outputs.append(mlp_output)

        self.model = Model([input_layer, action_layer, prev_action], mlp_outputs)
        self.model.compile(optimizer=Adam(lr=0.00005), loss="mean_squared_error", metrics=["accuracy"])

    def train_network(self, epochs=10, batch_size=256):
        if self.print_summary:
            self.model.summary()
        generator = StateEstimationDataGenerator(input_file_path=os.path.join(os.path.dirname(__file__),
                                                                              "resources",
                                                                              "traffic_cars_data",
                                                                              "state_estimation_data"),
                                                 batch_size=batch_size,
                                                 history_size=self.history_size,
                                                 prediction_horizon_size=self.mlp_hidden_layer_size)
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
    '''model = WorldModel(prediction_horizon_size=10, history_size=10)
    model.train_network(epochs=30, batch_size=32)'''
    preprocess_all_training_data(base_path=os.path.join(os.path.dirname(__file__),
                                                        "resources", "traffic_cars_data", "state_estimation_data"),
                                 h_size=10,
                                 pred_h_size=10)
