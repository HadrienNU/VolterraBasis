#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
===========================
Prony Series Estimation
===========================

Memory kernel fitted by a prony series
"""

import numpy as np
import matplotlib.pyplot as plt

import VolterraBasis as vb
import VolterraBasis.basis as bf

trj = np.loadtxt("example_lj.trj")
xva_list = []
print(trj.shape)
for i in range(1, trj.shape[1]):
    xf = vb.xframe(trj[:, i], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    xva_list.append(xvaf)


Nsplines = 10

estimator = vb.Estimator_gle(xva_list, vb.Pos_gle_const_kernel, bf.BSplineFeatures(Nsplines), trunc=10, saveall=False)
# mymem = vb.Pos_gle_overdamped(xva_list, bf.BSplineFeatures(Nsplines, remove_const=False), trunc=10, kT=1.0, saveall=False)
estimator.compute_mean_force()
estimator.compute_corrs()
model = estimator.compute_kernel(method="trapz")
time_ker, kernel = model.kernel["time"], model.kernel
print("Prony")
A_prony = vb.prony_fit_kernel(time_ker, kernel, thres=None, N_keep=150)
kernel_filtered = vb.prony_inspect_data(kernel[:, 0, 0], thres=None, N_keep=150)
print("Actual number of terms in the series: ", A_prony[0][0][1].shape[0])
fig_kernel, axs = plt.subplots(1, 1)
# # # Kernel plot
axs.set_title("Memory kernel")
axs.set_xscale("log")
axs.set_xlabel("$t$")
axs.set_ylabel("$K(x=2.0,t)$")
axs.grid()


axs.plot(time_ker, kernel[:, 0, 0], "-", label="Memory Kernel")
axs.plot(time_ker, kernel_filtered, "-o", label="Data Filtered")
axs.plot(time_ker, vb.prony_series_kernel_eval(time_ker, A_prony)[:, 0, 0], "-x", label="Prony fit")
axs.legend(loc="best")

plt.show()
