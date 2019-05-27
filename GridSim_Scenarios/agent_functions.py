import numpy as np


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
