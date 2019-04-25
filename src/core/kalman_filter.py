from __future__ import division, print_function

import numpy as np


class KalmanFilter(object):
    def __init__(self, rate=10, ra=1.5, sv=3.0):
        self.dt = 1/rate  # delta time
        self.m = np.zeros((3,1))
        # initial state
        self.x = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        # initial uncertainty
        self.P = 100.0*np.eye(6)
        # dynamic matrix
        self.A = np.matrix([[1.0, 0.0, 0.0, self.dt, 0.0,     0.0],
                            [0.0, 1.0, 0.0, 0.0,     self.dt, 0.0],
                            [0.0, 0.0, 1.0, 0.0,     0.0,     self.dt],
                            [0.0, 0.0, 0.0, 1.0,     0.0,     0.0],
                            [0.0, 0.0, 0.0, 0.0,     1.0,     0.0],
                            [0.0, 0.0, 0.0, 0.0,     0.0,     1.0]])
        # measurement matrix
        self.H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        # measurement noise covariance
        self.ra = ra
        # self.R = np.matrix([[ra, 0.0, 0.0],
        #                     [0.0, ra, 0.0],
        #                     [0.0, 0.0, ra]])

        self.R = np.matrix([[ra, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, ra, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, ra, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, ra, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, ra, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, ra]])
        # process noise covariance
        self.sv = sv
        self.G = np.matrix([[1/2.0*self.dt**2],
                            [1/2.0*self.dt**2],
                            [1/2.0*self.dt**2],
                            [self.dt],
                            [self.dt],
                            [self.dt]])
        self.Q = self.G*self.G.T*self.sv
        # identity matrix
        self.I = np.eye(6)
        self.last_result = np.zeros((6,1))

    def predict(self):
        # Project the state ahead
        self.x = self.A*self.x 
        # Project the error covariance ahead
        self.P = self.A*self.P*self.A.T + self.Q
        self.last_result = self.x  # same last predicted result
        return self.x

    def correct(self, m, flag):
        if not flag:  # update using prediction
            self.m = self.last_result[:3]
        else:  # update using detection
            self.m = m
        # Compute the Kalman Gain
        vel = (self.m - self.last_result[:3]) / self.dt
        self.S =  self.H*self.P*self.H.T + self.R
        self.K = (self.P*self.H.T) * np.linalg.pinv(self.S)
        # Update the estimate via z
        self.Z = np.append(self.m, vel, axis=0).reshape(self.H.shape[0],1)
        self.y = self.Z - (self.H*self.x)  # Innovation or Residual
        self.x = self.x + (self.K*self.y)
        # Update the error covariance
        self.P = (self.I - (self.K*self.H))*self.P
        self.last_result = self.x
        return self.x
