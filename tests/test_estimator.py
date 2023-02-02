import pytest
import numpy as np
import VolterraBasis as vb
import VolterraBasis.basis as bf


"""
Pour les tests: on peut tester:
- Si les trajectoires chargées ont les bonnes propriètés
- Si les fonctions compute_... donnent des résultast avec la bonne shape
- On peut faire les tests avec tous les models
- On peut faire un ensemble de tests pour toutes les fonctions de bases (vérification de la shape de sortie)
"""


@pytest.fixture
def traj_list():
    trj = np.loadtxt("../examples/example_lj.trj")
    xva_list = []
    print(trj.shape)
    for i in range(1, trj.shape[1]):
        xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0])
        xvaf = vb.compute_va(xf)
        xva_list.append(xvaf)
    return xva_list


# Faire un test parametrisé en fonction du model
def test_estimator(traj_list):
    estimator = vb.Estimator_gle(traj_list, vb.Pos_gle, bf.BSplineFeatures(10), trunc=10, saveall=False, verbose=False)

    assert estimator.model.dim_x == 1
    model = estimator.compute_mean_force()

    assert model.force_coeff.shape == (10, 1)

    model = estimator.compute_corrs()

    assert estimator.bkbkcorrw.shape == (9, 9, 2000)

    model = estimator.compute_kernel(method="trapz")

    assert model.kernel.shape == (1999, 9, 1)


# Parametrize test on invertion method
@pytest.mark.parametrize("method,expected", [("rect", (2000, 9, 1)), ("trapz", (1999, 9, 1))])
def test_kernel_method(traj_list, method, expected):
    estimator = vb.Estimator_gle(traj_list, vb.Pos_gle, bf.BSplineFeatures(10), trunc=10, saveall=False, verbose=False)
    model = estimator.compute_mean_force()
    estimator.compute_corrs()

    model = estimator.compute_kernel(method=method)

    assert model.kernel.shape == expected


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

# A implemter
# def test_estimator_2D(traj_list_2D):
