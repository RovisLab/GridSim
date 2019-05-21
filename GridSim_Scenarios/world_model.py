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

    def calc_loss(self, p, o):
        """
        Calculate loss function
        :param p: probability
        :param o: observation
        :return:
        """
        pass

    def train(self, h_t, K=10):
        """
        Train World Model network.
        :param h_t: History: h_t = (o_0, a_0, o_1, a_1, ..., a_t-1, o_t)
        :param K: number of frame predictions
        :return:
        """
        T = len(h_t)
        b = 0
        o = [h_t[idx] for idx in range(0, len(h_t), 2)]
        a = [h_t[idx] for idx in range(1, len(h_t), 2)]
        for t in range(1, T - K):
            A = [a[t - 1]]
            P = list()
            L = list()
            z_t = self.f_fi(o[t])
            b_t = self.f_theta(z_t=z_t, a_t_1=a[t - 1], b_t_1=b)
            b = b_t
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


def get_fov(car_pos):
    """
    Returns an array of point from the field of view of the agent
    :param car_pos: global position of the agent
    :return: FOV
    """
    arr = list()
    for i in range(car_pos[1] - 2, car_pos[1] + 3):
        for j in range(car_pos[0] - 2, car_pos[0] + 3):
            arr.append((i, j))
    return arr


def intersects(car_pos, obstacle_pos):
    fov = get_fov(car_pos)
    if obstacle_pos in fov:
        return True
    return False


class Obstacle(object):
    def __init__(self, onehot_encoding, global_position):
        self.encoding = onehot_encoding
        self.global_position = global_position

    def is_obstacle_in_agent_fov(self, car_pos):
        """
        Find if the obstacle is in an agents' field of view
        :param car_pos: position of the agent
        :return: True if obstacle is in FOV, False otherwise
        """
        if intersects(car_pos, self.global_position) is True:
            return True
        return False


class AgentObservation(object):
    def __init__(self, num_obstacles, obstacles_in_fov, walls_in_fov):
        self.walls = list()
        self.obstacles = list()
        for ob in obstacles_in_fov:
            self.obstacles.append(ob)
        for wall in walls_in_fov:
            self.walls.append(wall)
        self.num_obstacles = num_obstacles


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
