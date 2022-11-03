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
mymem = vb.Pos_gfpe(xva_list, basis_indicator, trunc=10, saveall=False)
print("Dimension of observable", mymem.dim_x)
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

mymem.kernel[:, :, -1] = 0
p0 = np.zeros(mymem.dim_obs)
p0[0] = 1
t_new, p_t = mymem.solve_gfpe(5000, method="trapz", p0=p0)
t_new, p_t_bis = mymem.solve_gfpe(5000, method="python", p0=p0)

fig_pt = plt.figure("Probability of time")
plt.grid()
plt.plot(t_new, p_t[:, 0, 0], "-")
plt.plot(t_new, p_t_bis[:, 2], "+")


fig, ax_anim = plt.subplots()
ax_anim.grid()
time_text = ax_anim.text(0.85, 0.95, "0.0", horizontalalignment="left", verticalalignment="top", transform=ax_anim.transAxes)

xrange = np.linspace(0.8, 3.0, 150)
E_eval = mymem.basis_vector(vb.pos_gle_base._convert_input_array_for_evaluation(xrange, 1), compute_for="force")
proba_val = E_eval @ p_t_bis[0, :]
dt = t_new[1] - t_new[0]

(ln,) = ax_anim.plot(xrange, proba_val, "-")


def update(frame):
    proba_val = E_eval @ p_t_bis[frame, :]
    ln.set_data(xrange, proba_val)
    time_text.set_text("%.3f" % (frame * dt))
    return (ln, time_text)


ani = animation.FuncAnimation(fig, update, frames=np.arange(p_t.shape[0]), blit=True, interval=10)


plt.show()
