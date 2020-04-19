# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:34:22 2020

@author: Harshil
"""

import numpy as np
from extended_kf import ExtKF
import unittest


class TestKF(unittest.TestCase):

    def test_can_construct_with_rho_phi_and_rhodot(self):
        rho = 1
        phi = 5.5
        rho_dot = 0.2

        kf = ExtKF(initial_rho=rho, initial_phi=phi, initial_rhodot=rho_dot)
        self.assertAlmostEqual(kf.dist, rho)
        self.assertAlmostEqual(kf.angvel, phi)
        self.assertAlmostEqual(kf.vel, rho_dot)
