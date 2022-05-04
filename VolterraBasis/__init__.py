import numpy as np
import xarray as xr
from ._version import __version__

from .pos_gle_instance import Pos_gle, Pos_gle_with_friction, Pos_gle_no_vel_basis, Pos_gle_const_kernel, Pos_gle_hybrid, Pos_gle_overdamped, Pos_gle_overdamped_const_kernel, Pos_gle_linear_proj
from .pos_gle_fem import ElementFinder, Pos_gle_fem
from .gle_integrate import Integrator_gle
from .memory_fit import memory_fit, memory_fit_eval, memory_fit_kernel, memory_kernel_eval
from .prony_fit import prony_inspect_data, prony_fit_times_serie, prony_fit_kernel, prony_series_eval, prony_series_kernel_eval
from .correlation import correlation_ND as correlation
from . import basis
from .mesh import *

__all__ = ["Pos_gle", "Pos_gle_with_friction", "Pos_gle_no_vel_basis", "Pos_gle_const_kernel", "Pos_gle_hybrid"]
__all__ += ["Pos_gle_overdamped", "Pos_gle_overdamped_const_kernel"]
__all__ += ["Integrator_gle"]
__all__ += ["correlation"]
__all__ += ["memory_fit", "memory_fit_eval", "memory_fit_kernel", "memory_kernel_eval"]
__all__ += ["prony_fit_times_serie", "prony_series_eval", "prony_fit_kernel", "prony_series_kernel_eval", "prony_inspect_data"]
__all__ += ["ElementFinder", "Pos_gle_fem"]


def xframe(x, time, v=None, a=None, fix_time=False, round_time=1.0e-4, dt=-1):
    """
    Creates a xarray dataset (['t', 'x']) from a trajectory.

    Parameters
    ----------
    x : numpy array
        The time series.
    time : numpy array
        The respective time values.
    fix_time : bool, default=False
        Round first timestep to round_time precision and replace times.
    round_time : float, default=1.e-4
        When fix_time is set times are rounded to this value.
    dt : float, default=-1
        When positive, this value is used for fixing the time instead of
        the first timestep.
    v : numpy array, default=None
        Velocity if computed externally
    a : numpy array, default=None
        Acceleration if computed externally
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    time = np.asarray(time)
    if fix_time:
        if dt < 0:
            dt = np.round((time[1] - time[0]) / round_time) * round_time
        time = np.linspace(0.0, dt * (x.shape[0] - 1), x.shape[0])
        time[1] = dt
    else:
        dt = time[1] - time[0]
    if v is None:
        ds = xr.Dataset({"x": (["time", "dim_x"], x)}, coords={"time": time}, attrs={"dt": dt})
    else:
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        ds = xr.Dataset({"x": (["time", "dim_x"], x), "v": (["time", "dim_x"], v)}, coords={"time": time}, attrs={"dt": dt})
    if a is not None:
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        ds = ds[["x", "v"]].assign({"a": a})
    return ds


def compute_a(xvf):
    """
    Computes the acceleration from a dataset with ['t', 'x', 'v'].

    Parameters
    ----------
    xvf : xarray dataset (['x', 'v'])
    """
    # Compute diff
    diffs = xvf.shift({"time": -1}) - xvf.shift({"time": 1})
    dt = xvf.attrs["dt"]

    xva = xvf[["x", "v"]].assign({"a": diffs["v"] / (2.0 * dt)})
    return xva.dropna("time")


def compute_va(xf, correct_jumps=False):
    """
    Computes velocity and acceleration from a dataset with ['t', 'x'] as
    returned by xframe.

    Parameters
    ----------
    xf : xarray dataframe (['t', 'x'])

    correct_jumps : bool, default=False
        Jumps in the trajectory are removed (relevant for periodic data).
    """
    diffs = xf - xf.shift({"time": 1})
    dt = xf.attrs["dt"]
    if correct_jumps:  # TODO
        raise NotImplementedError("Periodic data are not implemented yet")

    ddiffs = diffs.shift({"time": -1}) - diffs
    sdiffs = diffs.shift({"time": -1}) + diffs

    # xva = pd.DataFrame({"t": xf["t"], "x": xf["x"], "v": sdiffs["x"] / (2.0 * dt), "a": ddiffs["x"] / dt ** 2}, index=xf.index)
    xva = xf[["x"]].assign({"v": sdiffs["x"] / (2.0 * dt), "a": ddiffs["x"] / dt ** 2})

    return xva.dropna("time")


def compute_element_location(xva, element_finder):
    """
    Get element from trajectory
    """
    return xva.update({"elem": (["time"], element_finder.find(xva["x"].data))})
