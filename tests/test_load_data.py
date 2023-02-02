import pytest
import numpy as np
import VolterraBasis as vb


"""
Pour les tests: on peut tester:
- Si les trajectoires chargées ont les bonnes propriètés
- Si les fonctions compute_... donnent des résultast avec la bonne shape
- On peut faire les tests avec tous les models
- On peut faire un ensemble de tests pour toutes les fonctions de bases (vérification de la shape de sortie)
"""


def test_load_traj():
    trj = np.loadtxt("../examples/example_lj.trj")
    xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    assert "a" in xvaf.keys()

    assert "dt" in xvaf.attrs


def test_velocity_from_file():
    trj = np.loadtxt("../examples/example_lj.trj")
    xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0], v=trj[:, 2])
    assert "v" in xf.keys()


#
# trj = np.loadtxt("../examples/example_lj.trj")
# xva_list = []
# print(trj.shape)
# for i in range(1, trj.shape[1]):
#     xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0])
#     xvaf = vb.compute_va(xf)
#     xva_list.append(xvaf)
#
# estimator = vb.Estimator_gle(xva_list, vb.Pos_gle, bf.BSplineFeatures(10), trunc=10, saveall=False)
# # mymem = vb.Pos_gle(xva_list, bf.PolynomialFeatures(deg=1), trunc=10, kT=1.0, saveall=False)
# # mymem = vb.Pos_gle(xva_list, bf.LinearFeatures(), trunc=10, kT=1.0, saveall=False)
# print("Dimension of observable", estimator.model.dim_x)
# estimator.compute_mean_force()
# estimator.compute_corrs()
# model = estimator.compute_kernel(method="trapz")
# kernel = model.kernel_eval([1.5, 2.0, 2.5])
# print(kernel)
# # To find a correct parametrization of the space
# xfa = np.linspace(1.0, 3.0, 25)
# force = model.force_eval(xfa)
#
#
# # Compute noise
# time_noise, noise_reconstructed, _, _, _ = model.compute_noise(xva_list[0], trunc_kernel=200)
