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
basis_v = bf.SmoothIndicatorFeatures([[-1.1, -1.0], [1.0, 1.1]], "tricube", periodic=False)
basis_comb = bf.TensorialBasis2D(basis_x, basis_v)
mymem = vb.Pos_gfpe(xva_list, basis_comb, trunc=0.01, saveall=False)
print("Dimension of observable", mymem.dim_obs)
mymem.compute_mean_force()

print(mymem.force_coeff)

mymem.compute_corrs()
mymem.compute_kernel(method="rect")
print(mymem.time.shape, mymem.kernel.shape)


fig_kernel, axs = plt.subplots(1, 1)
# Kernel plot
axs.set_title("Memory kernel")
# axs.set_xscale("log")
axs.set_xlabel("$t$")
axs.set_ylabel("$\\Gamma$")
axs.grid()
axs.plot(mymem.time, mymem.kernel[:, basis_comb.comb_indices(0, 0), :], "-x")
# axs.plot(mymem.time, mymem.kernel[:, basis_comb.comb_indices(0, 0), basis_comb.comb_indices(0, 0)], "-x")
# axs.plot(mymem.time, mymem.kernel[:, basis_comb.comb_indices(1, 0), basis_comb.comb_indices(0, 0)], "-x")
# axs.plot(mymem.time, mymem.kernel[:, basis_comb.comb_indices(0, 0), basis_comb.comb_indices(1, 0)], "-x")
# axs.plot(mymem.time, mymem.kernel[:, basis_comb.comb_indices(1, 0), basis_comb.comb_indices(1, 0)], "-x")


# Survival problem
# sink_index = basis_comb.comb_indices(1, 1)
p0 = np.zeros(mymem.dim_obs)
p0[basis_comb.comb_indices(0, 1)] = 1
t_new, p_t = mymem.solve_gfpe(5000, method="trapz", p0=p0)
fig_pt = plt.figure("Probability of time")
plt.grid()
plt.plot(t_new, p_t, "-")

plt.plot(t_new, np.sum(p_t, axis=1), "-o")

fig, ax_anim = plt.subplots()
ax_anim.grid()
dt = t_new[1] - t_new[0]
time_text = ax_anim.text(0.05, 1.05, "0.0", horizontalalignment="left", verticalalignment="top", transform=ax_anim.transAxes)

xrange = np.linspace(0.8, 3.0, 50)
yrange = np.linspace(-2.0, 2.0, 50)
# Do mesh
xx, yy = np.meshgrid(xrange, yrange)
E_eval = basis_comb.basis(np.column_stack((xx.flatten(), yy.flatten())))
proba_val = E_eval @ p_t[0, :]
print(E_eval.shape, proba_val.shape, xx.shape, yy.shape, np.column_stack((xx.flatten(), yy.flatten())).shape)
quad = ax_anim.pcolormesh(xx, yy, proba_val.reshape(50, 50), shading="gouraud", cmap="viridis")
fig.colorbar(quad)


def update(frame):
    proba_val = E_eval @ p_t[frame, :]
    quad.set_array(proba_val.reshape(50, 50))
    time_text.set_text("%.3f" % (frame * dt))
    return (quad, time_text)


ani = animation.FuncAnimation(fig, update, frames=np.arange(p_t.shape[0]), blit=True, interval=10)
#
#
# fig_basis, axis_basis = plt.subplots(2, 1)
#
# Ex_basis = basis_x.basis(xrange.reshape(-1, 1))
# print(Ex_basis.shape)
#
# axis_basis[0].plot(xrange, Ex_basis)
#
# Ev_basis = basis_v.basis(yrange.reshape(-1, 1))
# print(Ev_basis.shape)
#
# axis_basis[1].plot(yrange, Ev_basis)

plt.show()
