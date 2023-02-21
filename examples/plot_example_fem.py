#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
===============================================
Memory Kernel Estimation with the usual GLE
===============================================

How to run memory kernel estimation
"""

import numpy as np
import dask.array as da

#
import VolterraBasis as vb
import VolterraBasis.basis as bf

import skfem

trj = np.loadtxt("example_lj.trj")
vertices, tri = bf.centroid_driven_mesh(trj[:, 1:3], bins=25)

m = skfem.MeshTri(vertices.T, tri.T)
e = skfem.ElementTriP1()  # skfem.ElementTriRT0()  #
basis_fem = skfem.CellBasis(m, e)

xva_list = []
# trj = da.from_array(trj, chunks=(100, 2))
xf = vb.xframe(trj[:, 1:3], trj[:, 0] - trj[0, 0])
xvaf = vb.compute_va(xf)
xva_list.append(xvaf)

print("Set up traj")

estimator = vb.Estimator_gle(xva_list, vb.Pos_gle_overdamped, bf.FEMScalarFeatures(basis_fem), trunc=1, saveall=False, verbose=False)
model = estimator.compute_mean_force()

xfa = trj[:10, 1:3]

force = model.force_eval(xfa)

#
# model.inv_mass_eval(xfa)
#
estimator.compute_corrs(second_order_method=False)
model = estimator.compute_kernel(method="rect")

time, noise, a, force, mem = model.compute_noise(xvaf)


kernel = model.kernel_eval(xfa)

coeffs = model.save_model()
print(coeffs)
new_model = model.load_model(model.basis, coeffs)

new_kernel = new_model.kernel_eval(xfa)
