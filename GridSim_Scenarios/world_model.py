from keras.layers import Dense, Conv2D, GRU, Input, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import numpy as np


class WorldModel(object):
    def __init__(self, prediction_horizon_size, history_dimension):
        self.prediction_horizon_size = prediction_horizon_size
        self.model = None
        self.cnn_output_size = 256
        self.gru_num_units = 128
        self.cnn_l1_kernel_size = (3, 3)
        self.cnn_l2_kernel_size = (3, 3)
        self.cnn_l1_strides = (1, 1)
        self.cnn_l2_strides = (2, 2)
        self.cnn_l1_num_filters = 16
        self.cnn_l2_num_filters = 16
        self.mlp_hidden_size = 64
        self.mlp_output_size = 25
        self.lr = 0.0005
        self.input_shape = (5, 5, 1)
        self.optimizer = Adam(lr=self.lr)
        self.loss = categorical_crossentropy
        self.batch_size = 256
        self.num_epochs = 100
        self.h_observations = list()
        self.h_actions = list()
        self.history_dimension = history_dimension

        self.conv_net = self._build_conv_net()
        self.gru_net = self._build_gru_net()
        self.mlp_net = [self._build_mlp_net() for _ in range(self.prediction_horizon_size)]

    def _build_conv_net(self):
        cnn_model = Sequential()
        cnn_model.add(Conv2D(filters=self.cnn_l1_num_filters,
                             kernel_size=self.cnn_l1_kernel_size,
                             strides=self.cnn_l1_strides,
                             padding='same'))
        cnn_model.add(Conv2D(filters=self.cnn_l2_num_filters,
                             kernel_size=self.cnn_l2_kernel_size,
                             strides=self.cnn_l2_strides,
                             padding='same'))
        cnn_model.add(Dense(self.cnn_output_size, activation='relu'))
        cnn_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return cnn_model

    def _build_gru_net(self):
        gru_model = Sequential()
        gru_model.add(GRU(units=self.gru_num_units))
        gru_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return gru_model

    def _build_mlp_net(self):
        mlp_model = Sequential()
        mlp_model.add(Dense(units=self.mlp_hidden_size, activation='relu'))
        mlp_model.add(Dense(units=self.mlp_output_size, activation='softmax'))
        mlp_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return mlp_model

    def f_fi(self, o_t):
        """
        Perform CNN predict on input o_t
        :param o_t: input of CNN (observation at time step t)
        :return:
        """
        return self.conv_net.predict(o_t)

    def f_theta(self, z_t, a_t_1, b_t_1):
        """
        Perform GRU predict
        :param z_t: observations transformed by the CNN
        :param a_t_1: previous action
        :param b_t_1: previous belief
        :return:
        """
        return self.gru_net.predict(np.concatenate((z_t, a_t_1, b_t_1)))

    def f_xi(self, k, b, a):
        """
        Perform MLPs predict
        :param k: index of the MLP
        :param b: belief
        :param a: list of actions
        :return:
        """
        return self.mlp_net[k].predict(np.concatenate((b, a)))

    def train(self):
        """
        Train World Model network.
        :return:
        """
        pass

    def summary(self):
        self.model.summary()

    def predict(self, actions_sequence):
        """
        Predict K frames, given sequence of actions
        :param actions_sequence: a list of actions
        :return: probability of future observations, given actions
        """
        pass

