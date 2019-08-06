import os


class ModelCarTrainingSet(object):
    """
    rovis_data_descriptor_file:
    timestamp, delta_ts, path2image_rgb (e.g 1234, 12, ./samples/1234RGB.png)
    timestamp, delta_ts, path2image_d
    ...

    sensor_data_descriptor_file:
    timestamp, velocity, steering_angle, x, y, yaw, us_fl, us_fcl, us_fc, us_fcr, us_fr
    ...
    """
    def __init__(self, base_path, strict=True, h_size=10, pred_size=10):
        self.base_path = base_path
        self.strict = strict
        self.h_size = h_size
        self.pred_size = pred_size
        self.rovis_data_descriptor = os.path.join(self.base_path, "rovis_data_description.csv")
        self.sensor_data_descriptor = os.path.join(self.base_path, "sensor_data_description.csv")
