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

# import sys
#
# sys.path.insert(0, "../")  # To use local version of the library, remove when installed

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
mymem = vb.Pos_gle(xva_list, bf.BSplineFeatures(Nsplines), trunc=10, kT=1.0, with_const=True, saveall=False)
mymem.compute_mean_force()
harmonic_coeffs = -1 * mymem.force_coeff[0]
# print(mymem.force_coeff)
mymem.compute_corrs()
mymem.compute_kernel()


fig_kernel, axs = plt.subplots(1, 1)

# #
# # # Kernel plot
axs.set_title("Memory kernel")
axs.set_xscale("log")
axs.set_xlabel("$t$")
axs.set_ylabel("$K(x=2.0,t)$")
axs.set_ylim([-500, 2000])
axs.grid()
# Iterate over method for comparaison
for method in ["rectangular", "midpoint", "midpoint_w_richardson", "trapz", "second_kind"]:
    mymem.compute_kernel(method=method)
    time_ker, kernel_vb = mymem.kernel_eval([2.0])
    axs.plot(time_ker, kernel_vb[:, :, 0, 0], "-o", label=method)
axs.legend(loc="best")

plt.show()
