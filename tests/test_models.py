import pytest
import os
import numpy as np
import dask.array as da
import VolterraBasis as vb
import VolterraBasis.basis as bf


@pytest.fixture
def traj_list(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_lj.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    xva_list = []
    for i in range(1, trj.shape[1]):
        xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0])
        xvaf = vb.compute_va(xf)
        xva_list.append(xvaf)
    return xva_list


# Idem avec Pos_gle_no_vel_basis Pos_gle_const_kernel Pos_gle_overdamped Pos_gle_overdamped_const_kernel Pos_gle_with_friction Pos_gle_hybrid
@pytest.mark.parametrize("traj_list", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize("model", [vb.Pos_gle, vb.Pos_gle_no_vel_basis, vb.Pos_gle_const_kernel, vb.Pos_gle_overdamped, vb.Pos_gle_hybrid])
@pytest.mark.parametrize(
    "basis,parameters",
    [
        (bf.LinearFeatures, {}),
        (bf.PolynomialFeatures, {"deg": 3, "remove_const": False}),
        (bf.FourierFeatures, {"order": 3, "freq": 1.0, "remove_const": False}),
        (bf.SplineFctFeatures, {"knots": np.linspace(-1, 1, 8), "coeffs": np.logspace(1, 2, 8), "k": 3}),
        (bf.BSplineFeatures, {"n_knots": 8, "k": 3, "remove_const": False}),
        (bf.PolynomialFeatures, {"deg": 3, "polynom": np.polynomial.Chebyshev}),
        (bf.FourierFeatures, {"order": 3, "freq": 1.0, "remove_const": True}),
        (bf.BSplineFeatures, {"n_knots": 8, "k": 3, "remove_const": True}),
        # (bf.SmoothIndicatorFeatures, {"states_boundary": [[1.0, 1.5], [2.0, 3.0], [3.7, 4.0]], "periodic": False}),
        # (bf.SmoothIndicatorFeatures, {"states_boundary": [[1.0, 1.5], [2.0, 3.0], [3.7, 4.0]], "boundary_type": "quartic", "periodic": True}),
    ],
)
def test_pos_gle(traj_list, model, basis, parameters):
    estimator = vb.Estimator_gle(traj_list, model, basis(**parameters), trunc=1, saveall=False, verbose=False)
    model = estimator.compute_mean_force()

    xfa = [1.1, 1.5, 2.0]

    force = model.force_eval(xfa)

    assert force.shape == (len(xfa), 1)

    model = estimator.compute_effective_mass()
    pmf = model.pmf_eval(xfa)

    assert pmf.shape == (len(xfa), 1)

    # model = estimator..compute_pos_effective_mass()
    # mass_pos = model.inv_mass_eval(xfa)
    # assert mass_pos.shape == (len(xfa), 1, 1)

    estimator.compute_corrs()
    model = estimator.compute_kernel(method="trapz")

    time, noise, a, force, mem = model.compute_noise(traj_list[0])

    assert noise.shape == traj_list[0]["a"].shape

    kernel = model.kernel_eval(xfa)

    assert kernel.shape == (model.trunc_ind - 1, 1, len(xfa), 1)

    coeffs = model.save_model()
    print(coeffs)
    new_model = model.load_model(model.basis, coeffs)

    assert (model.force_coeff == new_model.force_coeff).all()

    new_kernel = new_model.kernel_eval(xfa)

    np.testing.assert_allclose(kernel.values, new_kernel.values)


@pytest.mark.parametrize("traj_list", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "basis,parameters",
    [
        (bf.LinearFeatures, {}),
        (bf.SplineFctFeatures, {"knots": np.linspace(-1, 1, 8), "coeffs": np.logspace(1, 2, 8), "k": 3}),
        (bf.PolynomialFeatures, {"deg": 3, "polynom": np.polynomial.Chebyshev, "remove_const": True}),
        (bf.FourierFeatures, {"order": 3, "freq": 1.0, "remove_const": True}),
        (bf.BSplineFeatures, {"n_knots": 8, "k": 3, "remove_const": True}),
        # (bf.SmoothIndicatorFeatures, {"states_boundary": [[1.0, 1.5], [2.0, 3.0], [3.7, 4.0]], "periodic": False}),
        # (bf.SmoothIndicatorFeatures, {"states_boundary": [[1.0, 1.5], [2.0, 3.0], [3.7, 4.0]], "boundary_type": "quartic", "periodic": True}),
    ],
)
def test_pos_gle_w_friction(traj_list, basis, parameters):
    estimator = vb.Estimator_gle(traj_list, vb.Pos_gle_with_friction, basis(**parameters), trunc=1, saveall=False, verbose=False)
    model = estimator.compute_mean_force()

    xfa = [1.1, 1.5, 2.0]

    force = model.force_eval(xfa)

    assert force.shape == (len(xfa), 1)

    friction = model.friction_force_eval(xfa)

    assert friction.shape == (len(xfa), 1, 1)
    #
    pmf = model.pmf_eval(xfa)
    assert pmf.shape == (len(xfa), 1)
    #
    # model.inv_mass_eval(xfa)
    #
    estimator.compute_corrs()
    model = estimator.compute_kernel(method="trapz")

    time, noise, a, force, mem = model.compute_noise(traj_list[0])

    assert noise.shape == traj_list[0]["a"].shape

    kernel = model.kernel_eval(xfa)

    assert kernel.shape == (model.trunc_ind - 1, 1, len(xfa), 1)

    coeffs = model.save_model()
    print(coeffs)
    new_model = model.load_model(model.basis, coeffs)

    assert (model.force_coeff == new_model.force_coeff).all()

    new_kernel = new_model.kernel_eval(xfa)

    np.testing.assert_allclose(kernel.values, new_kernel.values)


@pytest.mark.skip(reason="FEM not working or now")
@pytest.mark.parametrize("model", [vb.Pos_gle, vb.Pos_gle_no_vel_basis, vb.Pos_gle_const_kernel, vb.Pos_gle_overdamped, vb.Pos_gle_hybrid])
def test_fem(model):
    import skfem

    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_lj.trj"))

    vertices, tri = bf.centroid_driven_mesh(trj[:, 1:3], bins=25)

    m = skfem.MeshTri(vertices.T, tri.T)
    e = skfem.ElementTriP1()  # skfem.ElementTriRT0()  #
    basis_fem = skfem.CellBasis(m, e)

    xva_list = []
    trj = da.from_array(trj)
    xf = vb.xframe(trj[:, 1:3], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    xva_list.append(xvaf)
    estimator = vb.Estimator_gle(xva_list, model, bf.FEMScalarFeatures(basis_fem), trunc=1, saveall=False, verbose=False)
    model = estimator.compute_mean_force()

    xfa = trj[:10, 1:3]

    force = model.force_eval(xfa)

    assert force.shape == xfa.shape

    pmf = model.pmf_eval(xfa)

    assert pmf.shape == xfa.shape
    #
    # model.inv_mass_eval(xfa)
    #
    estimator.compute_corrs(second_order_method=False)
    model = estimator.compute_kernel(method="rect")

    time, noise, a, force, mem = model.compute_noise(traj_list[0])

    assert noise.shape == traj_list[0]["a"].shape

    kernel = model.kernel_eval(xfa)

    assert kernel.shape == (model.trunc_ind - 1, 1, xfa.shape[0], 2)

    coeffs = model.save_model()
    print(coeffs)
    new_model = model.load_model(model.basis, coeffs)

    assert (model.force_coeff == new_model.force_coeff).all()

    new_kernel = new_model.kernel_eval(xfa)

    np.testing.assert_allclose(kernel.values, new_kernel.values)
