#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
==========================================
Checking solution of volterra equation
==========================================

How to run memory kernel estimation
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
# mymem = vb.Pos_gle(xva_list, bf.PolynomialFeatures(deg=1), trunc=10, kT=1.0, saveall=False)
# mymem = vb.Pos_gle(xva_list, bf.LinearFeatures(), trunc=10, kT=1.0, saveall=False)
print("Dimension of observable", estimator.model.dim_x, estimator.model.rank_projection)
estimator.compute_mean_force()
estimator.compute_corrs()
model = estimator.compute_kernel(method="trapz")

res_diff = estimator.check_volterra_inversion(return_diff=False)

fig_kernel, axs = plt.subplots(1, 1)
# Kernel plot
axs.set_title("Correlation diff")
axs.set_xscale("log")
axs.grid()
estimator.bkdxcorrw.sel(dim_basis=0).squeeze().plot.line("-", x="time_trunc", ax=axs)
axs.plot(np.arange(res_diff.shape[-1]), res_diff[0, 0, :].T, "x")
axs.set_xlabel("$t$")

plt.show()
