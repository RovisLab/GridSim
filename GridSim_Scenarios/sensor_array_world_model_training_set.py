from math import isclose
MAX_SENSOR_DISTANCE = 150.0


def object_in_sensor_fov(sensor_array):
    for s in sensor_array:
        if not isclose(s, MAX_SENSOR_DISTANCE):
            return True
    return False


class FrontSensorArrayTrainingSet(object):
    def __init__(self, base_path, strict=False, h_size=10, pred_size=10):
        self.base_path = base_path
        self.strict = strict
        self.h_size = h_size
        self.pred_size = pred_size

