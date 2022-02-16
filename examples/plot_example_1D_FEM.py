#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
=====================================
Kernel Estimation with Finite element
=====================================

How to run kernel estimation
"""
import numpy as np
import matplotlib.pyplot as plt
import skfem
from skfem.visuals.matplotlib import draw, plot
import scipy.interpolate

import sys

sys.path.insert(0, "../")  # To use local version of the library, remove when installed

import VolterraBasis as vb
import VolterraBasis.basis as bf


def compute_1d_fe(xva_list, bins="auto", hist=False):
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

    fehist = np.histogram(np.concatenate([xva["x"].data for xva in xva_list]), bins=bins, density=True)
    xfa = (fehist[1][1:] + fehist[1][:-1]) / 2.0
    pf = fehist[0]
    if hist:
        return xfa, pf
    xf = xfa[np.nonzero(pf)]
    fe = -np.log(pf[np.nonzero(pf)])
    fe -= np.min(fe)

    fe_spline = scipy.interpolate.splrep(xf, fe, s=0)
    dxf = xf[1] - xf[0]
    xfine = np.arange(xf[0], xf[-1], dxf / 10.0)
    yi_t = scipy.interpolate.splev(xfine, fe_spline)
    return xfine, yi_t


trj = np.loadtxt("example_lj.trj")
print(trj.shape)

xva_list = []
print(trj.shape)
for i in range(1, trj.shape[1]):
    xf = vb.xframe(trj[:, i], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    xva_list.append(xvaf)

# xf = vb.xframe(trj[:, 1:2], trj[:, 0] - trj[0, 0])
# xvaf = vb.compute_va(xf)
x_min = xf.min()["x"].data
x_max = xf.max()["x"].data
m = skfem.MeshLine(vb.uniform_line(x_min, x_max, 30))
base_elem = skfem.ElementLineP2()  # skfem.ElementTriRT0()  #
e = skfem.ElementVector(base_elem)
basis = skfem.CellBasis(m, e)

mymem = vb.Pos_gle_fem([xvaf], basis, vb.ElementFinder(m), trunc=1, kT=1.0, saveall=False)
print("Dimension of observable", mymem.dim_x)
mymem.compute_mean_force()
# print("Force coeff shape", mymem.force_coeff.shape, mymem.force_coeff[basis.nodal_dofs].T.shape)
force = mymem.force_coeff[basis.nodal_dofs].T

basis_pot = skfem.CellBasis(m, skfem.ElementLineHermite())
# potential = mymem.integrate_vector_field(basis_pot)
mymem.compute_pmf(basis_pot)

mymem.compute_corrs()
mymem.compute_kernel(method="midpoint")
time, kernel = mymem.kernel_eval([1.5, 2.0, 2.5])
print(time.shape, kernel.shape)

# # Compute noise
# time_noise, noise_reconstructed, _, _, _ = mymem.compute_noise(xvaf, trunc_kernel=200)

fig_kernel, axs = plt.subplots(1, 3)
# Force plot
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F$")
axs[0].grid()
plot(m, force, ax=axs[0])

# Potential Plot
axs[1].set_title("Potential")
# plot(basis_pot, potential, shading="gouraud", colorbar=True, ax=axs[1])

plot(basis_pot, mymem.hist_coeff, shading="gouraud", colorbar=True, ax=axs[1], color="-x")
axs[1].set_title("Potential")
axs[1].grid()
axs[1].plot(*compute_1d_fe([xvaf], bins=30, hist=True))

# Kernel plot
axs[2].set_title("Memory kernel")
axs[2].set_xscale("log")
axs[2].set_xlabel("$t$")
axs[2].set_ylabel("$\\Gamma$")
axs[2].grid()
axs[2].plot(time, kernel[:, :, 0, 0], "-x")
# # Noise plot
# axs[2].set_title("Noise")
# axs[2].set_xlabel("$t$")
# axs[2].set_ylabel("$\\xi_t$")
# axs[2].grid()
# axs[2].plot(time_noise, noise_reconstructed, "-")
plt.show()
