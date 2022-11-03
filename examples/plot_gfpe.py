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
    xf = vb.xframe(trj[:, i], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    xva_list.append(xvaf)

mymem = vb.Pos_gfpe(xva_list, bf.SmoothIndicatorFeatures([[1.4, 1.5]], "quartic"), trunc=10, saveall=False)
print("Dimension of observable", mymem.dim_x)
mymem.compute_mean_force()
print(mymem.force_coeff)
print(mymem.N_basis_elt, mymem.N_basis_elt_force, mymem.N_basis_elt_kernel)
# print(mymem.basis.b1.n_output_features_, mymem.basis.b2.n_output_features_)
mymem.compute_corrs()
mymem.compute_kernel(method="trapz")
print(mymem.time.shape, mymem.kernel.shape)
# To find a correct parametrization of the space


fig_kernel, axs = plt.subplots(1, 1)
# Kernel plot
axs.set_title("Memory kernel")
axs.set_xscale("log")
axs.set_xlabel("$t$")
axs.set_ylabel("$\\Gamma$")
axs.grid()
axs.plot(mymem.time, mymem.kernel[:, 0, 0], "-x")
axs.plot(mymem.time, mymem.kernel[:, 0, 1], "-x")
axs.plot(mymem.time, mymem.kernel[:, 1, 0], "-x")
axs.plot(mymem.time, mymem.kernel[:, 1, 1], "-x")

plt.show()
