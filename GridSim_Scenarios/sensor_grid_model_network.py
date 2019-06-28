import os
from keras.layers import Conv2D, Dense, GRU, Concatenate, Lambda, Masking, Input
import matplotlib.pyplot as plt


class WorldModel(object):
    def __init__(self, prediction_horizon_size, validation=False):
        self.prediction_horizon_size = prediction_horizon_size
        self.validation = validation
        self.input_shape =

    def _build_architecture(self):
        input_layer = Input(shape=self.input_shape)

    def train_model(self, epochs=100, batch_size=32):
        pass


if __name__ == "__main__":
    pass
