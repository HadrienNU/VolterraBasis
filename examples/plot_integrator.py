#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
===========================
Memory Kernel Estimation
===========================

How to run memory kernel estimation
"""

import numpy as np
import matplotlib.pyplot as plt

import VolterraBasis as vb
import VolterraBasis.basis as bf

trj = np.loadtxt("example_lj.trj")
xva_list = []
print(trj.shape)
corrs_vv_md = 0.0
for i in range(1, trj.shape[1]):
    xf = vb.xframe(trj[:, i], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    xva_list.append(xvaf)
    # corrs_vv_md += vb.correlation(xvaf["v"]) / trj.shape[1]

Nsplines = 5
ntrajs = 1
mymem = vb.Pos_gle(xva_list, bf.BSplineFeatures(Nsplines), trunc=10, kT=1.0, saveall=False)
# mymem = vb.Pos_gle(xva_list, bf.PolynomialFeatures(deg=1), trunc=10, kT=1.0, saveall=False)
# mymem = vb.Pos_gle(xva_list, bf.LinearFeatures(), trunc=10, kT=1.0, saveall=False)
print("Dimension of observable", mymem.dim_x)
mymem.compute_mean_force()
print(mymem.force_coeff)
mymem.compute_corrs()
mymem.compute_kernel(method="trapz")


integrator = vb.Integrator_gle(mymem, np.ones(Nsplines - 1))
start = integrator.initial_conditions(xvaf)
print(start)
xva_new = []
# corrs_vv_cg = 0.0
for n in range(ntrajs):
    xva, noise = integrator.run(3000, start)
    xva_new.append(xva)
    # corrs_vv_cg += vb.correlation(xva["v"]) / ntrajs

# analyse = vb.Pos_gle(xva_new, bf.LinearFeatures(), trunc=10, kT=1.0, saveall=False)
# analyse.compute_mean_force()
# analyse.compute_corrs()
# analyse.compute_kernel(method="trapz")
fig_integration, axs = plt.subplots(1, 2)
# New traj plot
axs[0].set_title("Traj")
axs[0].set_xlabel("$t$")
axs[0].set_ylabel("$x$")
axs[0].grid()
axs[0].plot(xva["time"], xva["x"])
# Correlation plot
# axs[1].set_title("Corrs")
# axs[1].set_xscale("log")
# axs[1].set_xlabel("$t$")
# axs[1].set_ylabel("$\\langle v,v \\rangle$")
# axs[1].grid()
# axs[1].plot(analyse.time, corrs_vv_cg[:, 0, 0], "-x")


plt.show()
