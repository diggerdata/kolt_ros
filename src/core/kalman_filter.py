from __future__ import division, print_function

import numpy as np


class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self, rate=10, ra=1.5, sv=3.0):
        """Initialize variable used by Kalman Filter class
        Args:
            None
        Return:
            None
        """
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
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Project the state ahead
        self.x = self.A*self.x 
        # Project the error covariance ahead
        self.P = self.A*self.P*self.A.T + self.Q
        self.last_result = self.x  # same last predicted result
        return self.x

    def correct(self, m, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """
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
