#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
==========================================
Generalized Fokker Planck equation
==========================================

How to run GFPE estimation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import VolterraBasis as vb
import VolterraBasis.basis as bf

trj = np.loadtxt("example_lj.trj")
xva_list = []
print(trj.shape)
for i in range(1, trj.shape[1]):
    xf = vb.xframe(trj[:, i], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    xva_list.append(xvaf)
basis_indicator = bf.SmoothIndicatorFeatures([[1.4, 1.5]], "tricube")
basis_indicator = bf.SmoothIndicatorFeatures([[1.0, 1.1], [1.4, 1.5], [1.6, 1.7], [2.0, 2.1]], "tricube")
basis_splines = bf.BSplineFeatures(10, remove_const=False)

estimator = vb.Estimator_gle(xva_list, vb.Pos_gle_overdamped, basis_indicator, trunc=10, saveall=False)
estimator.to_gfpe()

estimator.compute_mean_force()
estimator.compute_corrs()
model = estimator.compute_kernel(method="trapz")


fig_kernel, axs = plt.subplots(1, 1)
# Kernel plot
axs.set_title("Memory kernel")
axs.set_xscale("log")
axs.set_xlabel("$t$")
axs.set_ylabel("$\\Gamma$")
axs.grid()
# axs.plot(model.time, model.kernel[:, 0, 0], "-x")
# axs.plot(model.time, model.kernel[:, 0, 1], "-x")
print(model.kernel.shape, model.kernel.dims)
axs.plot(model.kernel["time_kernel"], model.kernel[:, 2, :], "-x")
axs.plot(model.kernel["time_kernel"], model.kernel[:, :, 2], "-x")


occ = estimator.compute_basis_mean()
print(occ)

time, bkbk = model.evolve_volterra(estimator.bkbkcorrw.isel(time_trunc=0), 500, method="rect")
print(bkbk.shape)
time, flux = model.flux_from_volterra(bkbk)
#
# fig_pt = plt.figure("Probability of time")
# plt.grid()
# plt.plot(t_new, p_t[:, :], "-x")
# plt.scatter(t_new[-1] * np.ones(model.dim_obs), occ)  # Plot occupations that should be long time limit
#
# t_num = np.arange(model.trunc_ind) * (t_new[1] - t_new[0])
# p_t_num = np.einsum("ikj, kl, l->ij", estimator.bkbkcorrw, np.diag(1.0 / occ), p0)
#
# plt.plot(t_num, p_t_num, "--")
#
#
# plt.plot(t_new, np.sum(p_t, axis=1), "-o")
# plt.plot(t_num, np.sum(p_t_num, axis=1), "--o")
#
# fig, ax_anim = plt.subplots()
# ax_anim.grid()
# time_text = ax_anim.text(0.85, 0.95, "0.0", horizontalalignment="left", verticalalignment="top", transform=ax_anim.transAxes)
#
# xrange = np.linspace(0.8, 3.0, 150)
# E_eval_unnorm = model.basis_vector(vb.models._convert_input_array_for_evaluation(xrange, 1), compute_for="force")
# norm_E = np.trapz(E_eval_unnorm, x=xrange, axis=0)
# E_eval = E_eval_unnorm @ np.diag(norm_E)
#
#
# proba_val = E_eval @ np.max(p_t[:, :], axis=0)
# dt = t_new[1] - t_new[0]
# print(E_eval.shape, norm_E)
# (ln,) = ax_anim.plot(xrange, proba_val, "-")
#
#
# def update(frame):
#     proba_val = E_eval @ p_t[frame, :]
#     ln.set_data(xrange, proba_val)
#     time_text.set_text("%.3f" % (frame * dt))
#     return (ln, time_text)
#
#
# ani = animation.FuncAnimation(fig, update, frames=np.arange(p_t.shape[0]), blit=True, interval=10)
#
#
plt.show()
