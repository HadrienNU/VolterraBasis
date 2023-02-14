import pytest
import os
import numpy as np
import VolterraBasis as vb


@pytest.fixture
def lj_path():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "../examples/example_lj.trj")


def test_load_traj(lj_path):
    trj = np.loadtxt(lj_path)
    xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    assert "a" in xvaf.keys()

    assert "dt" in xvaf.attrs


def test_velocity_from_file(lj_path):
    trj = np.loadtxt(lj_path)
    xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0], v=trj[:, 2])
    assert "v" in xf.keys()


# Add test on mesh
