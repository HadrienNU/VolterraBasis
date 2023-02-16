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
    print(trj.shape)
    for i in range(1, trj.shape[1]):
        xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0])
        xvaf = vb.compute_va(xf)
        xva_list.append(xvaf)
    return xva_list


@pytest.mark.parametrize("traj_list", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_estimator(traj_list, n_jobs, request):

    estimator = vb.Estimator_gle(traj_list, vb.Pos_gle, bf.BSplineFeatures(10), trunc=1, saveall=False, verbose=True, n_jobs=n_jobs)
    assert estimator.model.dim_x == 1

    model = estimator.compute_mean_force()
    assert model.force_coeff.shape == (10, 1)

    model = estimator.compute_gram_force()
    assert model.gram_force.shape == (10, 10)

    model = estimator.compute_gram_kernel()
    assert model.gram_kernel.shape == (9, 9)

    model = estimator.compute_effective_mass()
    assert model.eff_mass.shape == (1, 1)

    # model = estimator.compute_pos_effective_mass()
    # assert model.inv_mass_coeff.shape == (10, 1)

    model = estimator.compute_corrs()
    assert estimator.bkbkcorrw.shape == (9, 9, 200)

    model = estimator.compute_kernel(method="trapz")
    assert model.kernel.shape == (199, 9, 1)

    # volterra_corr = estimator.check_volterra_inversion() # Too long to run
    # np.testing.assert_allclose(volterra_corr, estimator.bkdxcorrw, rtol=1e-2)

    time, corrs_noise = estimator.compute_projected_corrs()
    assert corrs_noise.shape == (198, 1, 1)


@pytest.mark.parametrize("traj_list", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize("method", ["rect", "trapz", "trapz_stab"])
def test_gfpe(traj_list, method):
    estimator = vb.Estimator_gle(traj_list, vb.Pos_gle_overdamped, bf.BSplineFeatures(10, remove_const=False), trunc=1, saveall=False, verbose=True)
    estimator.to_gfpe()
    assert estimator.model.dim_obs == 10

    mean_val = estimator.compute_basis_mean()
    assert mean_val.shape == (10,)

    model = estimator.compute_mean_force()
    assert model.force_coeff.shape == (10, 10)

    model = estimator.compute_corrs()
    assert estimator.bkbkcorrw.shape == (10, 10, 200)

    model = estimator.compute_kernel(method="trapz")
    assert model.kernel.shape == (199, 10, 10)

    # time, bkbk = model.evolve_volterra(estimator.bkbkcorrw.isel(time_trunc=0), 500, method=method)
    #
    # assert bkbk.shape == (10, 10, 500)
    #
    # time, flux = model.flux_from_volterra(bkbk)
    #
    # assert flux.shape == (500, 10)


# Parametrize test on correlation computation method
@pytest.mark.parametrize("traj_list", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize("method,vectorize", [("fft", False), ("fft", True), ("direct", False), ("direct", True)])
def test_corrs_method(traj_list, method, vectorize):
    estimator = vb.Estimator_gle(traj_list, vb.Pos_gle, bf.BSplineFeatures(10, remove_const=False), trunc=1, saveall=False, verbose=False)
    estimator.set_zero_force()
    estimator.compute_corrs(method=method, vectorize=vectorize, second_order_method=True)

    assert estimator.bkbkcorrw.shape == (9, 9, 200)

    estimator.compute_corrs(method=method, vectorize=vectorize, second_order_method=False)

    assert estimator.dotbkbkcorrw[0, 0] == 0


# Parametrize test on invertion method
@pytest.mark.parametrize("traj_list", ["dask"], indirect=True)
@pytest.mark.parametrize("method,expected", [("rect", (2000, 9, 1)), ("midpoint", (1000, 9, 1)), ("midpoint_w_richardson", (333, 9, 1)), ("trapz", (1999, 9, 1)), ("second_kind_rect", (2000, 9, 1)), ("second_kind_trapz", (2000, 9, 1))])
def test_kernel_method(traj_list, method, expected):
    estimator = vb.Estimator_gle(traj_list, vb.Pos_gle, bf.BSplineFeatures(10, remove_const=False), trunc=10, saveall=False, verbose=False)
    model = estimator.compute_mean_force()
    estimator.compute_corrs()

    model = estimator.compute_kernel(method=method)

    assert model.kernel.shape == expected
