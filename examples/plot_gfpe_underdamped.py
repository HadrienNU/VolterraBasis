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
import matplotlib.animation as animation

import VolterraBasis as vb
import VolterraBasis.basis as bf

trj = np.loadtxt("example_lj.trj")
xva_list = []
print(trj.shape)
for i in range(1, trj.shape[1]):
    xf = vb.xframe(trj[:, i], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    xva_under = vb.concat_underdamped(xvaf)
    xva_list.append(xva_under)
# print(xva_under.head())
basis_x = bf.SmoothIndicatorFeatures([[1.4, 1.5]], "quartic")
basis_v = bf.SmoothIndicatorFeatures([[-1.1, -1.0], [1.0, 1.1]], "tricube", periodic=True)
basis_comb = bf.TensorialBasis2D(basis_x, basis_v)
mymem = vb.Pos_gfpe(xva_list, basis_comb, trunc=10, saveall=False)
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
# axs.set_xscale("log")
axs.set_xlabel("$t$")
axs.set_ylabel("$\\Gamma$")
axs.grid()
axs.plot(mymem.time, mymem.kernel[:, basis_comb.comb_indices(0, 0), basis_comb.comb_indices(0, 0)], "-x")
axs.plot(mymem.time, mymem.kernel[:, basis_comb.comb_indices(1, 0), basis_comb.comb_indices(0, 0)], "-x")
axs.plot(mymem.time, mymem.kernel[:, basis_comb.comb_indices(0, 0), basis_comb.comb_indices(1, 0)], "-x")
axs.plot(mymem.time, mymem.kernel[:, basis_comb.comb_indices(1, 0), basis_comb.comb_indices(1, 0)], "-x")


# Survival problem
sink_index = basis_comb.comb_indices(1, 1)
mymem.kernel[:, :, sink_index] = 0
p0 = np.zeros(mymem.dim_obs)
p0[0] = 1
t_new, p_t = mymem.solve_gfpe(5000, method="trapz", p0=p0)
p_t = p_t[:, :, 0]
fig_pt = plt.figure("Probability of time")
plt.grid()
plt.plot(t_new, p_t[:, 0], "-")


fig, ax_anim = plt.subplots()
ax_anim.grid()
dt = t_new[1] - t_new[0]
time_text = ax_anim.text(0.85, 0.95, "0.0", horizontalalignment="left", verticalalignment="top", transform=ax_anim.transAxes)

xrange = np.linspace(0.8, 3.0, 50)
yrange = np.linspace(-2.0, 2.0, 50)
# Do mesh
xx, yy = np.meshgrid(xrange, yrange)
E_eval = mymem.basis_vector(vb.pos_gle_base._convert_input_array_for_evaluation(np.column_stack((xx, yy)), 2), compute_for="force")
proba_val = E_eval @ p_t[0, :]
print(E_eval.shape, proba_val.shape)
ln = ax_anim.pcolormesh(xx, yy, proba_val.reshape(50, 50).T, shading="gouraud")


def update(frame):
    proba_val = E_eval @ p_t[frame, :]
    ln.set_array(proba_val.reshape(50, 50).T)
    time_text.set_text("%.3f" % (frame * dt))
    return (ln, time_text)


ani = animation.FuncAnimation(fig, update, frames=np.arange(p_t.shape[0]), blit=True, interval=10)


plt.show()
