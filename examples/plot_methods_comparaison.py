#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
===========================
Method comparaison
===========================

Comparaison of the various algorithm for inversion of the Volterra Integral equation
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
estimator = vb.Estimator_gle(xva_list, vb.Pos_gle, bf.BSplineFeatures(Nsplines), trunc=10, saveall=False)
# mymem = vb.Pos_gle_overdamped(xva_list, bf.BSplineFeatures(Nsplines, remove_const=False), trunc=10, kT=1.0, saveall=False)
estimator.compute_mean_force()
# print(mymem.force_coeff)
estimator.compute_corrs()

fig_kernel, axs = plt.subplots(1, 1)
# # # Kernel plot
axs.set_title("Memory kernel")
axs.set_xscale("log")
axs.set_xlabel("$t$")
axs.set_ylabel("$K(x=2.0,t)$")
axs.set_ylim([-500, 2000])
axs.grid()
# Iterate over method for comparaison
for method in ["rectangular", "midpoint", "midpoint_w_richardson", "trapz", "second_kind_rect", "second_kind_trapz"]:
    model = estimator.compute_kernel(method=method)
    time_ker, kernel_vb = model.kernel_eval([2.0])
    axs.plot(time_ker, kernel_vb[:, :, 0, 0], "-o", label=method)
axs.legend(loc="best")

plt.show()
