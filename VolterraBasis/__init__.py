import numpy as np
import pandas as pd

from .pos_gle import Pos_gle, Pos_gle_with_friction, Pos_gle_no_vel_basis
from .pos_gle_overdamped import Pos_gle_overdamped
from .correlation import correlation1D as correlation
from . import basis

__all__ = ["Pos_gle", "Pos_gle_with_friction", "Pos_gle_no_vel_basis", "Pos_gle_overdamped", "correlation"]

__version__ ="1.0"

def xframe(x, time, v=None, fix_time=False, round_time=1.0e-4, dt=-1):
    """
    Creates a pandas dataframe (['t', 'x']) from a trajectory. Currently the time
    is saved twice, as an index and as a separate field.

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
    """
    x = np.asarray(x)
    time = np.asarray(time)
    if fix_time:
        if dt < 0:
            dt = np.round((time[1] - time[0]) / round_time) * round_time
        time = np.linspace(0.0, dt * (x.size - 1), x.size)
        time[1] = dt
    if v is None:
        df = pd.DataFrame({"t": time.ravel(), "x": x.ravel()}, index=time.ravel())
    else:
        df = pd.DataFrame({"t": time.ravel(), "x": x.ravel(), "v": v.ravel()}, index=time.ravel())
    df.index.name = "#t"
    return df


def compute_a(xvf):
    """
    Computes the acceleration from a data frame with ['t', 'x', 'v'].

    Parameters
    ----------
    xvf : pandas dataframe (['t', 'x', 'v'])
    """
    diffs = xvf.shift(-1) - xvf.shift(1)
    dt = xvf.iloc[1]["t"] - xvf.iloc[0]["t"]
    xva = pd.DataFrame({"t": xvf["t"], "x": xvf["x"], "v": xvf["v"], "a": diffs["v"] / (2.0 * dt)}, index=xvf.index)
    xva = xva[["t", "x", "v", "a"]]
    xva.index.name = "#t"

    return xva.dropna()


def compute_va(xf, correct_jumps=False, jump=360, jump_thr=270):
    """
    Computes velocity and acceleration from a data frame with ['t', 'x'] as
    returned by xframe.

    Parameters
    ----------
    xf : pandas dataframe (['t', 'x'])

    correct_jumps : bool, default=False
        Jumps in the trajectory are removed (relevant for periodic data).
    jump : float, default=360
        The size of a jump.
    jump_thr : float, default=270
        Threshold for jump detection.
    """
    diffs = xf - xf.shift(1)
    dt = xf.iloc[1]["t"] - xf.iloc[0]["t"]
    if correct_jumps:
        diffs.loc[diffs["x"] < jump_thr, "x"] += jump
        diffs.loc[diffs["x"] > jump_thr, "x"] -= jump

    ddiffs = diffs.shift(-1) - diffs
    sdiffs = diffs.shift(-1) + diffs

    xva = pd.DataFrame({"t": xf["t"], "x": xf["x"], "v": sdiffs["x"] / (2.0 * dt), "a": ddiffs["x"] / dt ** 2}, index=xf.index)
    xva = xva[["t", "x", "v", "a"]]
    xva.index.name = "#t"

    return xva.dropna()
