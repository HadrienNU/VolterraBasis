#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
===========================
Memory Kernel Integration
===========================

How to run integration of the GLE once estimated.
Note: Due to time limit on readthedocs, the trajectories here are too short for convergence and figure are quite noisy.
"""

import numpy as np
import matplotlib.pyplot as plt

import VolterraBasis as vb
import VolterraBasis.basis as bf


def compute_1d_fe(xva_list):
    """
    Computes the free energy from the trajectoy using a cubic spline
    interpolation.

    Parameters
    ----------

    bins : str, or int, default="auto"
        The number of bins. It is passed to the numpy.histogram routine,
        see its documentation for details.
    hist: bool, default=False
        If False return the free energy else return the histogram
    """

    # # D'abord on obtient les bins
    min_x = np.min([xva["x"].min("time") for xva in xva_list])
    max_x = np.max([xva["x"].max("time") for xva in xva_list])

    n_bins = 50
    x_bins = np.linspace(min_x, max_x, n_bins)
    mean_val = 0
    count_bins = 0
    for xva in xva_list:
        # add v^2 to the list
        ds_groups = xva.assign({"v2": xva["v"] * xva["v"]}).groupby_bins("x", x_bins)
        # print(ds_groups)
        mean_val += ds_groups.sum().fillna(0)
        count_bins += ds_groups.count().fillna(0)
    fehist = (count_bins / count_bins.sum())["x"]
    mean_val = mean_val / count_bins
    pf = fehist.to_numpy()

    xfa = (x_bins[1:] + x_bins[:-1]) / 2.0

    xf = xfa[np.nonzero(pf)]
    fe = -np.log(pf[np.nonzero(pf)])
    fe -= np.min(fe)
    mean_a = mean_val["a"].to_numpy()[np.nonzero(pf)]

    return xf, fe, mean_a


trj = np.loadtxt("example_lj.trj")
xva_list = []
print(trj.shape)
corrs_vv_md = 0.0
for i in range(1, trj.shape[1]):
    xf = vb.xframe(trj[:, i], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    xva_list.append(xvaf)
    corrs_vv_md += vb.correlation(xvaf["v"]) / (trj.shape[1] - 1)
Nsplines = 10

ntrajs = trj.shape[1] - 1

xf_md, fe_md, mean_a_md = compute_1d_fe(xva_list)


mymem = vb.Pos_gle(xva_list, bf.BSplineFeatures(Nsplines), trunc=0.1, kT=1.0, saveall=False)
# mymem = vb.Pos_gle(xva_list, bf.PolynomialFeatures(deg=3), trunc=10, kT=1.0, saveall=False)
# mymem = vb.Pos_gle_const_kernel(xva_list, bf.LinearFeatures(), trunc=10, kT=1.0, saveall=False)
print("Dimension of observable", mymem.dim_x)
mymem.compute_mean_force()

mymem.compute_corrs()
mymem.compute_kernel(method="trapz")

force_md = mymem.force_eval(xf_md)


integrator = vb.Integrator_gle(mymem)  # np.ones(Nsplines - 1)

xva_new = []
corrs_vv_cg = 0.0
for n in range(ntrajs):
    start = integrator.initial_conditions(xva_list[n])
    xva = integrator.run(40000, start)
    xva = vb.compute_a(xva)
    xva_new.append(xva)
    corrs_vv_cg += vb.correlation(xva["v"]) / ntrajs

xf_cg, fe_cg, mean_a_cg = compute_1d_fe(xva_new)


fig_integration, axs = plt.subplots(2, 2)
# New traj plot
axs[0, 0].set_title("Traj")
axs[0, 0].set_xlabel("$t$")
axs[0, 0].set_ylabel("$r(t)$")
axs[0, 0].grid()
for n in range(ntrajs):
    axs[0, 0].plot(xva_new[n]["time"], xva_new[n]["x"], "-")

# Density Plot
axs[0, 1].set_title("Density")
axs[0, 1].set_xlabel("$r$")
axs[0, 1].set_ylabel("PMF")
axs[0, 1].grid()
axs[0, 1].plot(xf_md, fe_md, "-", label="MD")
axs[0, 1].plot(xf_cg, fe_cg, "-", label="CG")

# Force Plot
axs[1, 1].set_title("Force")
axs[1, 1].set_xlabel("$r$")
axs[1, 1].set_ylabel("f(r)")
axs[1, 1].grid()
axs[1, 1].plot(xf_md, force_md, "-", label="MD mean force")
axs[1, 1].plot(xf_cg, mean_a_cg, "-", label="CG")

# Correlation plot
axs[1, 0].set_title("Corrs")
axs[1, 0].set_xscale("log")
axs[1, 0].set_xlabel("$t$")
axs[1, 0].set_ylabel("$\\langle v,v \\rangle$")
axs[1, 0].grid()
axs[1, 0].plot(corrs_vv_md[:1000, 0, 0], "-", label="MD")
axs[1, 0].plot(corrs_vv_cg[:1000, 0, 0], "-", label="CG")


plt.show()
