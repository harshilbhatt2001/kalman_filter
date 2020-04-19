# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:36:22 2020

@author: Harshil
"""

from kf_3d import KF

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



plt.ion()
plt.figure()

real_x = 0.0
real_y = 0.0
real_z = 0.0
real_vx = 0.9
real_vy = 2.1
real_vz = 2.7
meas_variance = 0.1 ** 2


kf = KF(initial_x=0.0, initial_vx=1.0,
        initial_y=0.0, initial_vy=2.0,
        initial_z=0.0, initial_vz=3.0,
        accel_variance=0.1)

DT = 0.1
NUMSTEPS = 1000
MEAS_EVERY_STEPS = 20

mus = []
covs = []
real_xs = []
real_ys = []
real_zs = []
real_vxs = []
real_vys = []
real_vzs = []

for step in range(NUMSTEPS):
    if step > 500:
        real_vx *= 0.9
        real_vy *= 0.9
        real_vz *= 0.9

    covs.append(kf.cov)
    mus.append(kf.mean)

    real_x = real_x + DT * real_vx
    real_y = real_y + DT * real_vy
    real_z = real_z + DT * real_vz

    kf.predict(dt=DT)
    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        kf.update(meas_value_x = real_x + np.random.randn() * np.sqrt(meas_variance),
                  meas_value_y = real_y + np.random.randn() * np.sqrt(meas_variance),
                  meas_value_z = real_z + np.random.randn() * np.sqrt(meas_variance), 
                  meas_variance = meas_variance)

    real_xs.append(real_x)
    real_ys.append(real_y)
    real_zs.append(real_z)
    real_vxs.append(real_vx)
    real_vys.append(real_vy)
    real_vzs.append(real_vz)

plt.subplot(2, 1, 1)
plt.title('Position X')
plt.plot([mu[0] for mu in mus], 'r')
plt.plot(real_xs, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')

plt.subplot(2, 1, 2)
plt.title('Velocity X')
plt.plot([mu[3] for mu in mus], 'r')
plt.plot(real_vxs, 'b')
plt.plot([mu[3] - 2*np.sqrt(cov[3, 3]) for mu, cov in zip(mus, covs)], 'r--')
plt.plot([mu[3] + 2*np.sqrt(cov[3, 3]) for mu, cov in zip(mus, covs)], 'r--')

plt.show()
plt.ginput(1)

