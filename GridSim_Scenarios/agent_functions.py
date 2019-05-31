import numpy as np


def get_sine_wave_vector(num_samples, amplitude, phi, num_periods):
    t = np.linspace(0, 2 * np.pi * num_periods, num_samples)
    return t, amplitude * np.sin(t + phi)


class AgentAccelerationPattern(object):
    ACCELERATE_UNTIL_MAX_SPEED = 0
    SINUSOIDAL = 1

    def __init__(self, mode, ns=100, num_periods=1):
        self.mode = mode
        self.ns = ns
        self.acc_vec = np.zeros(self.ns)
        self.num_periods = num_periods
        self.crt_speed_idx = 0
        if self.mode == AgentAccelerationPattern.SINUSOIDAL:
            self._generate_acceleration_vector()

    def get_num_samples(self):
        return self.ns

    def _generate_acceleration_vector(self):
        _, acc_vect = get_sine_wave_vector(num_samples=self.ns, amplitude=1, phi=0, num_periods=self.num_periods)
        self.acc_vec = acc_vect

    def get_current_acc(self):
        acc = self.acc_vec[self.crt_speed_idx % len(self.acc_vec)]
        self.crt_speed_idx += 1
        return acc


class GridSimScenario(object):
    USER_CONTROL_NORMAL = 0
    FOLLOW_LEFT_BEHIND_CATCH_UP = 1
    TRAIL_OVERTAKE_STOP = 2
    OVERTAKE_LEFT_BEHIND_OVERTAKE = 3

    def __init__(self, num_cars, scenario_type):
        self.num_cars = num_cars
        self.scenario_type = scenario_type


class AgentAction(object):
    STAY = np.array([1, 0, 0, 0, 0])
    UP = np.array([0, 1, 0, 0, 0])
    DOWN = np.array([0, 0, 1, 0, 0])
    LEFT = np.array([0, 0, 0, 1, 0])
    RIGHT = np.array([0, 0, 0, 0, 1])
    ACTION_DELTA = 0.5


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
    def __init__(self, car_pos, walls, cars):
        self.car_pos = car_pos
        self.walls = walls
        self.cars = cars
        self.obstacle_array = self._build_observation()

    def _build_observation(self):
        obstacle_arr = list()
        obstacle_arr.append(self.walls)
        for car in self.cars:
            if car.is_obstacle_in_agent_fov(self.car_pos) is True:
                obstacle_arr.append(car.encoding)
            else:
                obstacle_arr.append([0, 0, 0, 0, 0])
        return obstacle_arr
