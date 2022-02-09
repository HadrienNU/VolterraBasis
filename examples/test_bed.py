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

trj = np.loadtxt("example_2d.trj", max_rows=200)
print(trj.shape)

xf = vb.xframe(trj[:, (1, 3)], trj[:, 0] - trj[0, 0])
xva = vb.compute_va(xf)

vertices, tri = vb.centroid_driven_mesh(xva["x"].data, 10)
# print(vertices.shape, tri.shape)
# plt.plot(xvaf["x"].data[:, 0], xvaf["x"].data[:, 1], "x")
# plt.triplot(vertices[:, 0], vertices[:, 1], tri)
# plt.plot(vertices[:, 0], vertices[:, 1], "o")
# plt.show()
m = skfem.MeshTri(vertices.T, tri.T)
e = skfem.ElementVector(skfem.ElementTriP1())  # skfem.ElementTriRT0()  #
basis = skfem.CellBasis(m, e)
xvaf = vb.compute_element_location(xva, vb.ElementFinder(m))

grouped_xva = list(xvaf.groupby("elem"))[2][1]
