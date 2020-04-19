# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:52:22 2020

@author: Harshil
"""

# KALMAN FILTER FOR 3-D ACCELERATED MOTION

import numpy as np

# offset of each variable in state vector
iX = 0
iVx = 1
iY = 0
iVy = 1
iZ = 0
iVz = 1
NUMVARS = 6


class KF:
    def __init__(self, initial_x, 
                       initial_vx,
                       initial_y,
                       initial_vy,
                       initial_z,
                       initial_vz,
                       accel_variance):

        # Mean of state GRV
        self._x = np.zeros(NUMVARS)

        self._x[0] = initial_x
        self._x[1] = initial_y
        self._x[2] = initial_z
        self._x[3] = initial_vx
        self._x[4] = initial_vy
        self._x[5] = initial_vz
        self._x = self._x.reshape((6, 1))
        self._accel_variance = accel_variance

        # Covariance of state GRV
        self._P = np.eye(NUMVARS)

    def predict(self, dt):
        # x = F x
        # P = F P Ft + G Gt a
        F = np.eye(NUMVARS)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self._x = F.dot(self._x)

        G = np.zeros((6, 3))
        G[0, 0] = 0.5 * dt**2
        G[1, 1] = 0.5 * dt**2
        G[2, 2] = 0.5 * dt**2
        G[3, 0] = dt
        G[4, 1] = dt
        G[5, 2] = dt
        
        self._P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance


    def update(self, meas_value_x,
                     meas_value_y,
                     meas_value_z,
                     meas_variance):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]).reshape((3, 6))

        z = np.array([[meas_value_x], [meas_value_y],[meas_value_z]])

        tmp = np.array([meas_variance, meas_variance, meas_variance])
        R = np.array([tmp.T.dot(tmp)])
        
        y_matrix = z - H.dot(self._x).reshape((3, 1))
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        self._x = self._x + K.dot(y_matrix)
        self._P = (np.eye(6) - K.dot(H)).dot(self._P)



    @property
    def cov(self):
        return self._P

    @property
    def mean(self):
        return self._x

    @property
    def pos_x(self):
        return self._x[0]
    
    @property
    def pos_y(self):
        return self._x[1]

    @property
    def pos_z(self):
        return self._x[2]

    @property
    def vel_x(self):
        return self._x[3] 

    @property
    def vel_y(self):
        return self._x[4] 

    @property
    def vel_z(self):
        return self._x[5]    

        