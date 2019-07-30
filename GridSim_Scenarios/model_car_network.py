import os
import numpy as np
from keras.layers import Conv2D, Dense, GRU, Concatenate, Lambda, Masking, Input, Reshape, Flatten, Conv1D
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
import matplotlib.pyplot as plt
from model_car_data_loader import StateEstimationModelCarDataGenerator


class WorldModel(object):
    def __init__(self, h_size, pred_horizon_size, num_rays, action_shape, prev_action_shape, rgb_shape, d_shape):
        self.h_size = h_size
        self.pred_horizon_size = pred_horizon_size
        self.num_rays = num_rays
        self.action_shape = action_shape
        self.prev_action_shape = prev_action_shape
        self.rgb_shape = (None, rgb_shape[0], rgb_shape[1], rgb_shape[2])
        self.d_shape = (None, d_shape[0], d_shape[1])

    def _build_architecture(self):
        sensor_input_layer = Input(shape=(None, self.num_rays))
        sensor_dense = Dense(units=100, activation="relu")(sensor_input_layer)

        rgb_input_layer = Input(shape=self.rgb_shape)
        rgb_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(rgb_input_layer)
        rgb_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(rgb_conv)
        rgb_dense = Dense(units=256, activation="relu")

        d_input_layer = Input(shape=self.d_shape)
        d_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(d_input_layer)
        d_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(d_conv)
        d_dense = Dense(units=256, activation="relu")

        action_input_layer = Input(shape=self.action_shape)
        action_dense = Dense(units=100, activation="relu")(action_input_layer)

        prev_action_input_layer = Input(shape=self.prev_action_shape)
        prev_action_dense = Dense(units=100, activation="relu")(prev_action_input_layer)

