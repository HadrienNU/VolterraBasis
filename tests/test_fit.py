import pytest
import os
import numpy as np
import dask.array as da
import VolterraBasis as vb
import VolterraBasis.basis as bf


@pytest.fixture
def kernel():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_lj.trj"))
    trj = da.from_array(trj)
    xva_list = []
    print(trj.shape)
    for i in range(1, trj.shape[1]):
        xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0])
        xvaf = vb.compute_va(xf)
        xva_list.append(xvaf)
    estimator = vb.Estimator_gle(xva_list, vb.Pos_gle_const_kernel, bf.BSplineFeatures(10), trunc=10, saveall=False)

    # mymem = vb.Pos_gle_overdamped(xva_list, bf.BSplineFeatures(Nsplines, remove_const=False), trunc=10, kT=1.0, saveall=False)
    estimator.compute_mean_force()
    estimator.compute_corrs()
    model = estimator.compute_kernel(method="trapz")
    return model.kernel


@pytest.mark.parametrize("type", ["exp", "sech", "gaussian", "sech_one", "sech_two", "prony"])
def test_fit_memory(kernel, type):
    params = vb.memory_fit(kernel["time_kernel"], kernel[:, 0, 0], type=type)
    fitted_mem = vb.memory_fit_eval(kernel["time_kernel"], params)

    assert fitted_mem.shape == kernel.shape
