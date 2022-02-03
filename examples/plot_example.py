#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
===========================
Kernel Estimation
===========================

How to run kernel estimation
"""

import numpy as np
import matplotlib.pyplot as plt
import VolterraBasis as vb
import VolterraBasis.basis as bf

trj = np.loadtxt("example_lj.trj")
xva_list = []
print(trj.shape)
for i in range(1, trj.shape[1]):
    xf = vb.xframe(trj[:, i], trj[:, 0])
    xvaf = vb.compute_va(xf)
    xva_list.append(xvaf)

Nsplines = 10
mymem = vb.Pos_gle(xva_list, bf.BSplineFeatures(Nsplines), Nsplines, trunc=10, kT=1.0, with_const=True, saveall=False)
# mymem = vb.Pos_gle(xva_list, bf.LinearFeatures(), 1, trunc=10, kT=1.0, with_const=False, saveall=False)
print("Dimension of observable", mymem.dim_x)
mymem.compute_mean_force()
print(mymem.force_coeff)
mymem.compute_corrs()
mymem.compute_kernel(method="trapz")
time, kernel = mymem.kernel_eval([1.5, 2.0, 2.5])
print(time.shape, kernel.shape)
# To find a correct parametrization of the space
bins = np.histogram_bin_edges(xvaf["x"], bins=15)
xfa = (bins[1:] + bins[:-1]) / 2.0
force = mymem.dU(xfa)


# Compute noise
time_noise, noise_reconstructed, _, _, _ = mymem.compute_noise(xva_list[0], trunc_kernel=200)


fig_kernel, axs = plt.subplots(1, 3)
# Force plot
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$-dU(x)/dx$")
axs[0].grid()
axs[0].plot(xfa, force)
# Kernel plot
axs[1].set_title("Memory kernel")
axs[1].set_xscale("log")
axs[1].set_xlabel("$t$")
axs[1].set_ylabel("$\\Gamma$")
axs[1].grid()
axs[1].plot(time, kernel[:, :, 0], "-x")

# Noise plot
axs[2].set_title("Noise")
axs[2].set_xlabel("$t$")
axs[2].set_ylabel("$\\xi_t$")
axs[2].grid()
axs[2].plot(time_noise, noise_reconstructed, "-")
plt.show()
