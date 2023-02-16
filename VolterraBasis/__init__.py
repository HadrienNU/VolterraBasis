import numpy as np
import xarray as xr
import scipy.interpolate
from ._version import __version__

from .models import Pos_gle, Pos_gle_with_friction, Pos_gle_no_vel_basis, Pos_gle_const_kernel, Pos_gle_hybrid, Pos_gle_overdamped  # , Pos_gle_overdamped_const_kernel
from .gle_estimation import Estimator_gle

from .gle_integrate import Integrator_gle, Integrator_gle_const_kernel, KarhunenLoeveNoiseGenerator
from .fit_memory import memory_fit, memory_fit_eval, memory_fit_kernel, memory_kernel_eval
from .fit_prony import prony_inspect_data, prony_fit_times_serie, prony_fit_kernel, prony_series_eval, prony_series_kernel_eval
from .correlation import correlation_ND as correlation_fft
from .correlation import correlation_direct_ND as correlation_direct

__all__ = ["Pos_gle", "Pos_gle_with_friction", "Pos_gle_no_vel_basis", "Pos_gle_const_kernel", "Pos_gle_hybrid"]
__all__ += ["Pos_gle_overdamped", "Pos_gle_overdamped_const_kernel"]
__all__ += ["Trajectories_handler"]
__all__ += ["Estimator_gle", "Integrator_gle"]
__all__ += ["correlation_fft", "correlation_direct"]
__all__ += ["memory_fit", "memory_fit_eval", "memory_fit_kernel", "memory_kernel_eval"]
__all__ += ["prony_fit_times_serie", "prony_series_eval", "prony_fit_kernel", "prony_series_kernel_eval", "prony_inspect_data"]

# TODO:
# Changer le code de gle integrate pour prendre en entrée un model directement


def xframe(x, time, v=None, a=None, fix_time=False, round_time=1.0e-4, dt=-1):
    """
    Creates a xarray dataset (['t', 'x']) from a trajectory.

    Parameters
    ----------
    x : array
        The time series. The array can be in any type as long as xarray can handle it.
        This include numpy array, dask array,...
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
    # Quick check for duck numpy-like array
    if "shape" not in x.__dir__() or "__array__" not in x.__dir__():
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
        ds = ds[["x", "v"]].assign({"a": (["time", "dim_x"], a)})  # {"a": a})
    return ds


def compute_a_from_vel(xvf):
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


def compute_a(xvf):
    """
    Computes the acceleration from a dataset with ['t', 'x', 'v'].

    Parameters
    ----------
    xvf : xarray dataset (['x', 'v'])
    """
    # Compute diff
    diffs = xvf - xvf.shift({"time": 1})
    dt = xvf.attrs["dt"]

    ddiffs = diffs.shift({"time": -1}) - diffs
    xva = xvf[["x", "v"]].assign({"a": ddiffs["x"] / dt ** 2})
    return xva.dropna("time")


def compute_va(xf, correct_jumps=False, jump=2 * np.pi, jump_thr=1.75 * np.pi):
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
        diffs = xr.where(diffs["x"] > -jump_thr, diffs, diffs + jump)
        diffs = xr.where(diffs["x"] < jump_thr, diffs, diffs - jump)
        # raise NotImplementedError("Periodic data are not implemented yet")

    ddiffs = diffs.shift({"time": -1}) - diffs
    sdiffs = diffs.shift({"time": -1}) + diffs

    # xva = pd.DataFrame({"t": xf["t"], "x": xf["x"], "v": sdiffs["x"] / (2.0 * dt), "a": ddiffs["x"] / dt ** 2}, index=xf.index)
    xva = xf[["x"]].assign({"v": sdiffs["x"] / (2.0 * dt), "a": ddiffs["x"] / dt ** 2})

    return xva.dropna("time")


def compute_va_gjf(xf, correct_jumps=False, jump=2 * np.pi, jump_thr=1.75 * np.pi):
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
        diffs = xr.where(diffs["x"] > -jump_thr, diffs, diffs + jump)
        diffs = xr.where(diffs["x"] < jump_thr, diffs, diffs - jump)
        # raise NotImplementedError("Periodic data are not implemented yet")

    ddiffs = diffs.shift({"time": -1}) - diffs

    # xva = pd.DataFrame({"t": xf["t"], "x": xf["x"], "v": sdiffs["x"] / (2.0 * dt), "a": ddiffs["x"] / dt ** 2}, index=xf.index)
    xva = xf[["x"]].assign({"v": diffs["x"] / dt, "a": ddiffs["x"] / dt ** 2})

    return xva.dropna("time")


def concat_underdamped(xva):
    """
    Return the DataSet such that x is now (x,v) and v is now (v,a),
    """
    new_x = xr.concat([xva["x"], xva["v"]], dim="dim_x")
    new_v = xr.concat([xva["v"], xva["a"]], dim="dim_x")
    return xr.Dataset({"x": new_x, "v": new_v}, attrs={"dt": xva.dt})


def compute_1d_fe(xva_list, bins=150, kT=2.494, hist=False):
    """
    Computes the free energy from the trajectoy using a cubic spline
    interpolation.

    Parameters
    ----------

    bins : str, or int, default="auto"
        The number of bins. It is passed to the numpy.histogram routine,
        see its documentation for details.
    hist: bool, default=False
        If False return the free energy else return the histogram
    """
    if isinstance(bins, str):
        x_bins = np.histogram_bin_edges(np.concatenate([xva["x"].data for xva in xva_list], axis=0), bins="auto")
    elif isinstance(bins, int):
        # # D'abord on obtient les bins
        min_x = np.min([xva["x"].min("time") for xva in xva_list])
        max_x = np.max([xva["x"].max("time") for xva in xva_list])
        x_bins = np.linspace(min_x, max_x, bins)
    else:
        raise ValueError("bins should be a str or a int")

    mean_val = 0
    count_bins = 0
    for xva in xva_list:
        # add v^2 to the list
        ds_groups = xva.assign({"v2": xva["v"] * xva["v"]}).groupby_bins("x", x_bins)
        # print(ds_groups)
        mean_val += ds_groups.sum().fillna(0)
        count_bins += ds_groups.count().fillna(0)
    mass_avg = (mean_val["v2"].sum() / count_bins["v2"].sum()).to_numpy()
    fehist = (count_bins / count_bins.sum())["x"]
    mean_val = mean_val / count_bins
    pf = fehist.to_numpy()

    xfa = (x_bins[1:] + x_bins[:-1]) / 2.0
    if hist:
        return xfa, pf

    xf = xfa[np.nonzero(pf)]
    fe = -np.log(pf[np.nonzero(pf)])
    fe -= np.min(fe)
    pf_bis = pf[np.nonzero(pf)]
    mass = mean_val["v2"].to_numpy()[np.nonzero(pf)]
    mean_a = mean_val["a"].to_numpy()[np.nonzero(pf)]

    fe_spline = scipy.interpolate.splrep(xf, fe, s=0)
    force = -1 * scipy.interpolate.splev(xf, fe_spline, der=1) * kT / mass_avg

    mass_rho_spline = scipy.interpolate.splrep(xf, mass * pf_bis, s=0)
    force_tot = scipy.interpolate.splev(xf, mass_rho_spline, der=1) / pf_bis

    return xf, fe, force, mean_a, force_tot, mass
