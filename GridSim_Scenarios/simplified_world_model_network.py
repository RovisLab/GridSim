from keras.models import Model
from keras.layers import Input, Dense, GRU


class WorldModel(object):
    def __init__(self, input_shape, prediction_horizon_size):
        self.input_shape = input_shape
        self.input_layer_num_units = 8
        self.mlp_layer_num_units = 8
        self.gru_layer_num_units = 128
        self.mlp_layer_size = prediction_horizon_size
        self.model = None

    def _build_architecture(self):
        input_shape = Input(shape=self.input_shape)
        input_layer = Dense(units=self.input_layer_num_units, activation="relu")(input_shape)
        gru = GRU(units=self.gru_layer_num_units)(input_layer)
        for idx in range(self.mlp_layer_size):
            mlp = Dense(units=self.mlp_layer_num_units)
            # TODO Activation size here
