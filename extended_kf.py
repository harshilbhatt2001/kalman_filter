# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:22:22 2020

@author: Harshil
"""

# Used when sensor data is Non-Linear
# Sensor data considered is RADAR

import numpy as np

iRho = 0
iRho_dot = 1
iPhi = 5.49779 # 315°
NUMVARS = 3

class ExtKF:
    def __init__(self, initial_rho,
                       initial_phi,
                       initial_rhodot):

        self._x = np.zeros(NUMVARS)
        self._x[0] = initial_rho
        self._x[1] = initial_phi
        self._x[2] = initial_rhodot

        self._P = np.eye(NUMVARS)

    def predict(self, dt):
        # x = F x
        # P = F P Ft + Q

        

    @property
    def cov(self):
        return self._P

    @property
    def mean(self):
        return self._x

    @property
    def dist(self):
        return self._x[0]

    @property
    def angvel(self):
        return self._x[1]

    @property
    def vel(self):
        return self._x[2]
