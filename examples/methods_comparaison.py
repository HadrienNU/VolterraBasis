#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import glob
import matplotlib.pyplot as plt

import sys

sys.path.append("../")  # To add location of the library, remove when installed

# The memory tools import
import VolterraBasis as vb
import VolterraBasis.basis as bf

trj = np.loadtxt("example_gle.trj")
xva_list = []
print(trj.shape)
for i in range(1, trj.shape[1]):
    xf = vb.xframe(trj[:, i], trj[:, 0])
    xvaf = vb.compute_va(xf, correct_jumps=True)
    xva_list.append(xvaf)


Nsplines = 10
mymem = vb.Pos_gle(xva_list, bf.BSplineFeatures(Nsplines), Nsplines, trunc=10, kT=1.0, with_const=True, saveall=False)
mymem.compute_mean_force()
harmonic_coeffs = -1 * mymem.force_coeff[0]
# print(mymem.force_coeff)
mymem.compute_corrs()
mymem.compute_kernel()


fig_kernel, axs = plt.subplots(1, 1)

# #
# # # Kernel plot
axs[0].set_title("Memory kernel")
axs[0].set_xscale("log")
axs[0].set_xlabel("$t$")
axs[0].set_ylabel("$K(x=2.0,t)$")
axs[0].grid()
# Iterate over method for comparaison
for method in ["rectangular", "midpoint", "midpoint_w_richardson", "trapz", "second_kind"]:
    print(method)
    mymem.compute_kernel(method=method)
    time_ker, kernel_vb = mymem.kernel_eval([2.0])
    axs[0].plot(time_ker, kernel_vb, "-o", label=method)
axs[0].legend(loc="best")

plt.show()
