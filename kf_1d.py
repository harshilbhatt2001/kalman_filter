# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:15:22 2020

@author: Harshil
"""

# KALMAN FILTER FOR 1-D ACCELERATED MOTION

import numpy as np

# offset of each variable in state vector
iX = 0
iVx = 1
NUMVARS = 2


class KF:
    def __init__(self, initial_x, 
                       initial_vx,
                       accel_variance):

        # Mean of state GRV
        self._x = np.zeros(NUMVARS)

        self._x[iX] = initial_x
        self._x[iVx] = initial_vx
        
        self._accel_variance = accel_variance

        # Covariance of state GRV
        self._P = np.eye(NUMVARS)

    def predict(self, dt):
        # x = F x
        # P = F P Ft + G Gt a
        F = np.eye(NUMVARS)
        F[iX, iVx] = dt
        self._x = F.dot(self._x)

        G = np.zeros((2, 1))
        G[iX] = 0.5 * dt**2
        G[iVx] = dt
        
        self._P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance


    def update(self, meas_value, meas_variance):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.array([1, 0]).reshape((1, 2))

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y_matrix = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        self._x = self._x + K.dot(y_matrix)
        self._P = (np.eye(2) - K.dot(H)).dot(self._P)


    @property
    def cov(self):
        return self._P

    @property
    def mean(self):
        return self._x

    @property
    def pos(self):
        return self._x[iX]
    @property
    def vel(self):
        return self._x[iVx]    
