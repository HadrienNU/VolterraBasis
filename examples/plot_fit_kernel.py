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
mymem = vb.Pos_gle_const_kernel(xva_list, bf.BSplineFeatures(Nsplines), trunc=10, kT=1.0, saveall=False)
# mymem = vb.Pos_gle_overdamped(xva_list, bf.BSplineFeatures(Nsplines, remove_const=False), trunc=10, kT=1.0, saveall=False)
mymem.compute_mean_force()
harmonic_coeffs = -1 * mymem.force_coeff[0]
# print(mymem.force_coeff)
mymem.compute_corrs()
kernel = mymem.compute_kernel(method="trapz")


fig_kernel, axs = plt.subplots(1, 1)
# Kernel plot
axs.set_title("Memory kernel")
# axs.set_xscale("log")
axs.set_xlabel("$t$")
axs.set_ylabel("$K(x=2.0,t)$")
axs.grid()
axs.plot(mymem.time, kernel[:, 0, 0], "-", label="Memory Kernel")
for type in ["exp", "sech", "gaussian"]:
    print("Fit: " + str(type))
    params = vb.memory_fit(mymem.time.ravel(), kernel[:, 0, 0], type=type)
    print(params)
    axs.plot(mymem.time, vb.memory_fit_eval(mymem.time, params), "-x", label="Fit " + str(type))
axs.legend(loc="best")

plt.show()
