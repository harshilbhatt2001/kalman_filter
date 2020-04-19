# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:14:22 2020

@author: Harshil
"""


import numpy as np
from kf_3d import KF
import unittest 

        
class TestKF(unittest.TestCase):

    def test_can_construct_with_x_and_v(self):
        x = 0.2
        vx = 2.3
        y = 0.2
        vy = 2.3
        z = 0.2
        vz = 2.3

        kf = KF(initial_x=x, initial_y=y, initial_z=z, 
                initial_vx=vx, initial_vy=vy, initial_vz=vz,
                accel_variance=1.2)
        self.assertAlmostEqual(kf.pos_x, x)
        self.assertAlmostEqual(kf.pos_y, y)
        self.assertAlmostEqual(kf.pos_z, z)
        self.assertAlmostEqual(kf.vel_x, vx)
        self.assertAlmostEqual(kf.vel_y, vy)
        self.assertAlmostEqual(kf.vel_z, vz)


    def test_after_calling_predict_mean_and_covariance_are_of_right_shape(self):
        x = 0.2
        vx = 2.3
        y = 0.2
        vy = 2.3
        z = 0.2
        vz = 2.3

        kf = KF(initial_x=x, initial_y=y, initial_z=z, 
                initial_vx=vx, initial_vy=vy, initial_vz=vz,
                accel_variance=1.2)
        
        kf.predict(dt=0.1)

        self.assertEqual(kf.cov.shape, (6, 6))
        self.assertEqual(kf.mean.shape, (6, ))

    def test_calling_predict_increases_state_uncertainty(self):
        x = 0.2
        vx = 2.3
        y = 0.2
        vy = 2.3
        z = 0.2
        vz = 2.3

        kf = KF(initial_x=x, initial_y=y, initial_z=z, 
                initial_vx=vx, initial_vy=vy, initial_vz=vz,
                accel_variance=1.2)
        
        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt=0.1)
            det_after = np.linalg.det(kf.cov)

            self.assertGreater(det_after, det_before)
            print(det_before, det_after)

    def test_calling_predict_decreases_state_uncertainty(self):
        x = 0.2
        vx = 2.3
        y = 0.2
        vy = 2.3
        z = 0.2
        vz = 2.3

        kf = KF(initial_x=x, initial_y=y, initial_z=z, 
                initial_vx=vx, initial_vy=vy, initial_vz=vz,
                accel_variance=1.2)

        det_before = np.linalg.det(kf.cov)
        kf.update(meas_value_x=0.1,
                  meas_value_y=0.1,
                  meas_value_z=0.1,
                  meas_variance=0.01)
        det_after = np.linalg.det(kf.cov)

        self.assertLess(det_after, det_before)