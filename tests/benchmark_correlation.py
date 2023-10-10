import os
import pytest
import numpy as np
import xarray as xr
import dask.array as da
import VolterraBasis as vb
import VolterraBasis.basis as bf
import time


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


def _correlation_ufunc(weight, xva, model, method="fft", vectorize=False, second_order_method=True, **kwargs):
    """
    Do the correlation
    Return 4 array with dimensions

    bkbkcorrw :(trunc_ind, N_basis_elt_kernel, N_basis_elt_kernel)
    bkdxcorrw :(trunc_ind, N_basis_elt_kernel, dim_obs)
    dotbkdxcorrw :(trunc_ind, N_basis_elt_kernel, dim_obs)
    dotbkbkcorrw :(trunc_ind, N_basis_elt_kernel, N_basis_elt_kernel)
    """
    if method == "fft" and vectorize:
        func = vb.correlation.correlation_1D
    elif method == "direct" and not vectorize:
        func = vb.correlation.correlation_direct_ND
    elif method == "direct" and vectorize:
        func = vb.correlation.correlation_direct_1D
    else:
        func = vb.correlation.correlation_ND
    _, E, _ = model.basis_vector(xva)

    bkbkcorrw = xr.apply_ufunc(
        func,
        E.rename({"dim_basis": "dim_basis'"}),
        E,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time_trunc"]],
        exclude_dims={"time"},
        kwargs={"trunc": model.trunc_ind},
        dask_gufunc_kwargs={"output_sizes": {"time_trunc": model.trunc_ind}, "allow_rechunk": True},
        vectorize=vectorize,
        dask="parallelized",
    )
    bkbkcorrw.to_netcdf("corr.nc")
    return (bkbkcorrw,)


@pytest.mark.parametrize("traj_list", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize("method,vectorize", [("fft", False), ("fft", True), ("direct", False), ("direct", True)])
def test_corrs_method(traj_list, method, vectorize, benchmark):
    estimator = vb.Estimator_gle(traj_list, vb.Pos_gle, bf.BSplineFeatures(25, remove_const=False), trunc=1, saveall=False, verbose=False)
    res = benchmark(estimator.loop_over_trajs, _correlation_ufunc, estimator.model, method=method, vectorize=vectorize)
    assert res[0].shape == (25, 25, 200)
