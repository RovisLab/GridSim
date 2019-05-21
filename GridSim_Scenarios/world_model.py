from keras.layers import Dense, Conv2D, GRU, Input, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import numpy as np


class WorldModel(object):
    def __init__(self, prediction_horizon_size):
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

        self.conv_net = self.build_conv_net()
        self.gru_net = self.build_gru_net()
        self.mlp_net = [self.build_mlp_net() for i in range(self.prediction_horizon_size)]

    def build_conv_net(self):
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

    def build_gru_net(self):
        gru_model = Sequential()
        gru_model.add(GRU(units=self.gru_num_units))
        gru_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return gru_model

    def build_mlp_net(self):
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
        pass

    def f_theta(self, z_t, a_t_1, b_t_1):
        """
        Perform GRU predict
        :param z_t: observations passed through input CNN
        :param a_t_1: previous action
        :param b_t_1: previous belief
        :return:
        """
        pass

    def f_xi(self, k, b, a):
        """
        Perform MLPs predict
        :param b: belief
        :param a: list of actions
        :return:
        """
        pass

    def calc_loss(self, p, o):
        """
        Calculate loss function
        :param p: probability
        :param o: observation
        :return:
        """
        pass

    def train(self, h_t, K=10, T=100):
        """
        Train World Model network.
        :param h_t: History: h_t = (o_0, a_0, o_1, a_1, ..., a_t-1, o_t)
        :param K: number of frame predictions
        :param T: number of training steps
        :return:
        """
        b = 0
        o = [h_t[idx] for idx in range(0, len(h_t), 2)]
        a = [h_t[idx] for idx in range(1, len(h_t), 2)]
        for t in range(1, T - K):
            z_t = self.f_fi(o[t])
            b_t = self.f_theta(z_t=z_t, a_t_1=a[t - 1], b_t_1=b)
            b = b_t
            A = [a[t - 1]]
            P = list()
            L = list()
            for k in range(1, K):
                p = self.f_xi(k=k, b=b, a=A)
                A.append(a[t + k - 1])
                P.append(p)
                L.append(self.calc_loss(p, o[t + k]))
            self.update_reward()
        self.update_weights()
        self.update_policy()

    def summary(self):
        self.model.summary()


class EvaluationNetwork(object):
    def __init__(self):
        pass


class RewardNetwork(object):
    def __init__(self):
        pass


class AgentObservation(object):
    def __init__(self, max_num_obstacles, obstacles, walls):
        self.max_num_obstacles = max_num_obstacles
        self.walls = np.zeros(shape=(5, 5), dtype=np.float32)
        self.obstacles = np.zeros(shape=(self.max_num_obstacles, 5, 5))
        self._add_obstacles(obstacles)
        self._add_walls(walls)

    def _add_obstacles(self, pos_2d):
        obstacles = np.zeros(shape=(self.max_num_obstacles, 5, 5))
        idx = 0
        for pos in pos_2d:
            obs = np.zeros(shape=(5, 5))
            obstacles[pos[0]][pos[1]] = 1
            if idx < self.max_num_obstacles:
                obstacles[idx] = obs
        self.obstacles = obstacles

    def _add_walls(self, pos_2d):
        """
        Add observed walls (positions in 2D as list)
        :param pos_2d: list of walls as 2D coordinates
        :return: None
        """
        walls = np.zeros(shape=(5, 5))
        for pos in pos_2d:
            walls[pos[0]][pos[1]] = 1
        self.walls = walls


class AgentAction(object):
    STAY = np.array([1, 0, 0, 0, 0])
    UP = np.array([0, 1, 0, 0, 0])
    DOWN = np.array([0, 0, 1, 0, 0])
    LEFT = np.array([0, 0, 0, 1, 0])
    RIGHT = np.array([0, 0, 0, 0, 1])

    def __init__(self, action=STAY):
        self.action = action


if __name__ == "__main__":
    model = WorldModel(prediction_horizon_size=10)
    conv = model.build_conv_net()
    gru = model.build_gru_net()
    mlp = model.build_mlp_net()
