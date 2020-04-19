# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:03:22 2020

@author: Harshil
"""

from kf_1d import KF

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



plt.ion()
plt.figure()

real_x = 0.0
meas_variance = 0.1 ** 2
real_vx = 0.9

kf = KF(initial_x=0.0, initial_vx=1.0, accel_variance=0.1)

DT = 0.1
NUMSTEPS = 1000
MEAS_EVERY_STEPS = 20

mus = []
covs = []
real_xs = []
real_vxs = []

for step in range(NUMSTEPS):
    if step > 500:
        real_vx *= 0.9

    covs.append(kf.cov)
    mus.append(kf.mean)

    real_x = real_x + DT * real_vx

    kf.predict(dt=DT)
    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        kf.update(meas_value = real_x + np.random.randn() * np.sqrt(meas_variance), 
                  meas_variance = meas_variance)

    real_xs.append(real_x)
    real_vxs.append(real_vx)

plt.subplot(2, 1, 1)
plt.title('Position')
plt.plot([mu[0] for mu in mus], 'r')
plt.plot(real_xs, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')


plt.subplot(2, 1, 2)
plt.title('Velocity')
plt.plot([mu[1] for mu in mus], 'r')
plt.plot(real_vxs, 'b')
plt.plot([mu[1] - 2*np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
plt.plot([mu[1] + 2*np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')


plt.show()
plt.ginput(1)

