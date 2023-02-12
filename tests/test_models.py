import pytest
import os
import numpy as np
import VolterraBasis as vb
import VolterraBasis.basis as bf


"""
Pour les tests: on peut tester:
- On peut faire les tests avec tous les models
- Il faut tester ici toutes les fonctions d'évaluation ainsi que la sauvegarde et le chargement des models
"""


@pytest.fixture
def traj_list():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_lj.trj"))
    xva_list = []
    print(trj.shape)
    for i in range(1, trj.shape[1]):
        xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0])
        xvaf = vb.compute_va(xf)
        xva_list.append(xvaf)
    return xva_list


# Idem avec Pos_gle_no_vel_basis Pos_gle_const_kernel Pos_gle_overdamped Pos_gle_overdamped_const_kernel Pos_gle_with_friction Pos_gle_hybrid


@pytest.mark.parametrize(
    "model,basis,parameters",
    [
        (model,) + basis
        for model in [vb.Pos_gle]  # , vb.Pos_gle_no_vel_basis, vb.Pos_gle_const_kernel, vb.Pos_gle_overdamped, vb.Pos_gle_overdamped_const_kernel, vb.Pos_gle_with_friction, vb.Pos_gle_hybrid]
        for basis in [
            (bf.LinearFeatures, {}),
            # (bf.PolynomialFeatures, {"deg": 3, "remove_const": False}),
            # (bf.FourierFeatures, {"order": 3, "freq": 1.0, "remove_const": False}),
            # (bf.SplineFctFeatures, {"knots": np.linspace(-1, 1, 8), "coeffs": np.logspace(1, 2, 8), "k": 3}),
            # (bf.BSplineFeatures, {"n_knots": 8, "k": 3, "remove_const": False}),
            # (bf.PolynomialFeatures, {"deg": 3, "polynom": np.polynomial.Chebyshev}),
            # (bf.FourierFeatures, {"order": 3, "freq": 1.0}),
            # (bf.BSplineFeatures, {"n_knots": 8, "k": 3}),
            # (bf.SmoothIndicatorFeatures, {"states_boundary": [[1.0, 1.5], [2.0, 3.0], [3.7, 4.0]], "periodic": False}),
            # (bf.SmoothIndicatorFeatures, {"states_boundary": [[1.0, 1.5], [2.0, 3.0], [3.7, 4.0]], "boundary_type": "quartic", "periodic": True}),
        ]
    ],
)
def test_pos_gle(traj_list, model, basis, parameters):
    estimator = vb.Estimator_gle(traj_list, model, basis(**parameters), trunc=10, saveall=False, verbose=False)
    model = estimator.compute_mean_force()

    xfa = [1.1, 1.5, 2.0]

    force = model.force_eval(xfa)

    assert force.shape == (len(xfa), 1)
    #
    # pmf = model.pmf_eval(xfa)
    #
    # assert pmf.shape == (len(xfa), 1)
    #
    # model.inv_mass_eval(xfa)
    #
    estimator.compute_corrs()
    model = estimator.compute_kernel(method="trapz")

    # time, noise, a, force, mem = model.compute_noise(traj_list[0])
    #
    # assert noise.shape == traj_list[0].shape

    kernel = model.kernel_eval(xfa)

    assert kernel.shape == (model.trunc_ind - 1, 1, len(xfa), 1)

    coeffs = model.save_model()

    new_model = model.load_model(coeffs, **parameters)

    new_kernel = new_model.kernel_eval(xfa)

    assert kernel == new_kernel


# Pour les 2 suivantes faire des tests en plus Pos_gle_with_friction Pos_gle_hybrid
