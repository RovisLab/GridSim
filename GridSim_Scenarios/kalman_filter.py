from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np


class KalmanFilter(object):
    def __init__(self, dim_x, dim_z, dim_u, prediction_horizon=10):
        """
        Extended Kalman Filter. Performs predict / update over the prior database (150 historic points for GridSim,
        50 points for ModelCar) and then predicts over the 10 future frames
        :param dim_x: State vector size (30 for GridSim, 5 for ModelCar)
        :param dim_z: Measurement vector size (30 for GridSim, 5 for ModelCar)
        :param dim_u: Control vector size (1 for both GridSim and ModelCar)  u = [ego_car_velocity]
        :param prediction_horizon: Length of frames to be predicted (10)
        """
        self.dataset = list()
        self.predictions = list()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.prediction_horizon = prediction_horizon
        self.ekf = ExtendedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z, dim_u=self.dim_u)
        self.results = list()

    def initialize_filter(self, x0, P, R, Q=0):
        self.ekf.x = x0
        self.ekf.F = np.eye(self.dim_x)
        if np.isscalar(P):
            self.ekf.P *= P
        else:
            self.ekf.P[:] = P
        self.ekf.R *= R
        if np.isscalar(Q):
            self.ekf.Q = Q_discrete_white_noise(dim=self.dim_x)
        else:
            self.ekf.Q[:] = Q

    def _jacobian_h(self):
        return np.eye(self.dim_x)

    def _h_x(self):
        return np.eye(self.dim_x)

    def run(self):
        #  Run over the stored dataset, to get prior
        for v_cmd, rays in self.dataset:
            self.ekf.predict(v_cmd)
            self.ekf.update(z=rays, HJacobian=self._jacobian_h, Hx=self._h_x)

        #  Predict for "prediction_horizon" frames
        for v_cmd in self.predictions:
            self.ekf.predict(v_cmd)
            self.results.append(self.ekf.x)
