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
import skfem
from skfem.visuals.matplotlib import draw

import sys

sys.path.insert(0, "../")  # To use local version of the library, remove when installed

import VolterraBasis as vb
import VolterraBasis.basis as bf

trj = np.loadtxt("example_2d.trj")
print(trj.shape)

xf = vb.xframe(trj[:, (1, 3)], trj[:, 0] - trj[0, 0])
xvaf = vb.compute_va(xf)

vertices, tri = vb.centroid_driven_mesh(xvaf["x"].data, 30)
# print(vertices.shape, tri.shape)
# plt.plot(xvaf["x"].data[:, 0], xvaf["x"].data[:, 1], "x")
# plt.triplot(vertices[:, 0], vertices[:, 1], tri)
# plt.plot(vertices[:, 0], vertices[:, 1], "o")
# plt.show()
m = skfem.MeshTri(vertices.T, tri.T)
e = skfem.ElementVector(skfem.ElementTriP1())  # skfem.ElementTriRT0()  #
basis = skfem.CellBasis(m, e)
print(xvaf)
print("Init")
mymem = vb.Pos_gle_fem([xvaf], basis, vb.ElementFinder(m), trunc=1, kT=1.0, saveall=False)
print("Dimension of observable", mymem.dim_x)
mymem.compute_mean_force()
# print(mymem.force_coeff)
print("Force coeff shape", mymem.force_coeff.shape, mymem.force_coeff[basis.nodal_dofs].T.shape)
force = mymem.force_coeff[basis.nodal_dofs].T
# Plot force
# m.save("force_field.vtk", {"force": force})
# ax = draw(m)
# ax.quiver(*m.p, *force.T, m.p[0])
# plt.show()
# print(mymem.N_basis_elt, mymem.N_basis_elt_force, mymem.N_basis_elt_kernel)
# # print(mymem.basis.b1.n_output_features_, mymem.basis.b2.n_output_features_)
mymem.compute_corrs()
mymem.compute_kernel(method="rectangular")  # Do matrix inversion in the python code
time, kernel = mymem.kernel_eval([[1.5, 1.0], [2.0, 1.5], [2.5, 1.0]])
print(time.shape, kernel.shape)
# # To find a correct parametrization of the space
# bins = np.histogram_bin_edges(xvaf["x"], bins=15)
# xfa = (bins[1:] + bins[:-1]) / 2.0
# x, y = np.meshgrid(xfa, xfa)
# X = np.vstack((x.flatten(), y.flatten())).T
# force = mymem.dU(X)
#
# # Compute noise
# time_noise, noise_reconstructed, _, _, _ = mymem.compute_noise(xvaf, trunc_kernel=200)
#
#
fig_kernel, axs = plt.subplots(1, 3)
# Force plot
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$y$")
# axs[0].grid()
draw(m, ax=axs[0])
axs[0].quiver(*m.p, *force.T, m.p[0])
# Kernel plot
axs[1].set_title("Memory kernel")
axs[1].set_xscale("log")
axs[1].set_xlabel("$t$")
axs[1].set_ylabel("$\\Gamma$")
axs[1].grid()
axs[1].plot(time, kernel[:, :, 0, 0], "-x")
axs[1].plot(time, kernel[:, :, 0, 1], "-x")
axs[1].plot(time, kernel[:, :, 1, 0], "-x")
axs[1].plot(time, kernel[:, :, 1, 1], "-x")

# # Noise plot
# axs[2].set_title("Noise")
# axs[2].set_xlabel("$t$")
# axs[2].set_ylabel("$\\xi_t$")
# axs[2].grid()
# axs[2].plot(time_noise, noise_reconstructed, "-")
plt.show()
