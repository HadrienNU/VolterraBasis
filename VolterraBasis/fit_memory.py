#!python3
# -*- coding: utf-8 -*-

"""
Some functionnal fit of the memory kernel
"""

import numpy as np
import scipy.optimize
import scipy.integrate

from .fit_prony import prony_series_eval, prony_fit_times_serie


def asymptotic(x, a, b):
    return a + b / x


def asymptotic_fit_integral(times, data):
    """
    Compute cumulative integral and fit last part of it with a+b/t to get asymptotic
    We only consider last part
    """
    integral = scipy.integrate.cumulative_trapezoid(data, times, initial=0)[len(times) // 2 :]
    popt, pcov = scipy.optimize.curve_fit(asymptotic, times[len(times) // 2 :], integral)
    return popt


def markovian_friction(times, kernel):
    """
    Compute long-time limit of the integral of the memory kernel
    """
    _, dim_basis, dim_x = kernel.shape
    popt = [[0] * dim_basis for i in range(dim_x)]
    for d in range(dim_x):
        for k in range(dim_basis):
            # Idealement il faudrait calculer la valeur de l'intégrale pour
            integral_val = scipy.integrate.simpson(kernel[:, k, d], times.ravel())  # Trouver un moyen d'avoir le comportemement à temps long
            popt[d][k] = integral_val
    return type, popt


def exp_fct(t, a, b):
    return a * np.exp(-b * t)


def sech_fct(t, a, b, nu):
    return a / np.cosh(b * t) ** (nu)


def sech_one_fct(t, a, b):
    return a / np.cosh(b * t)


def sech_two_fct(t, a, b):
    return a / np.cosh(b * t) ** (2)


def gaussian_fct(t, a, b):
    return a * np.exp(-b * t ** 2)


def memory_fit(times, data, type="exp", **kwargs):
    """
    Fit a memory kernel using the type fonction
    """
    times = np.asarray(times).squeeze()
    # Check format of the data
    if times.ndim > 1 or data.ndim > 1:
        raise ValueError("Incorrect shape of the data")
    if type in ["exp", "exponential"]:
        func = exp_fct
        p0 = [data[0], 0.1 / (times[1] - times[0])]
    elif type in ["sech"]:
        func = sech_fct
        p0 = [data[0], 0.1 / np.sqrt((times[1] - times[0])), 1.5]
    elif type in ["sech_one"]:
        func = sech_one_fct
        p0 = [data[0], 0.1 / np.sqrt((times[1] - times[0]))]
    elif type in ["sech_two"]:
        func = sech_two_fct
        p0 = [data[0], 0.1 / np.sqrt((times[1] - times[0]))]
    elif type in ["gaussian"]:
        func = gaussian_fct
        p0 = [data[0], 0.1 / np.sqrt((times[1] - times[0]))]
    elif type in ["prony"]:
        return type, prony_fit_times_serie(data, (times[1] - times[0]).squeeze(), **kwargs)
    else:
        raise ValueError("Not implemented type")
    popt, pcov = scipy.optimize.curve_fit(func, times, data, p0=p0, **kwargs)
    return type, popt


def memory_fit_eval(times, params, type=None):
    """
    Evaluate the fit at given times
    """
    if type is None:
        type, popt = params
    else:
        popt = params
    if type in ["exp", "exponential"]:
        func = exp_fct
    elif type in ["sech"]:
        func = sech_fct
    elif type in ["sech_one"]:
        func = sech_one_fct
    elif type in ["sech_two"]:
        func = sech_two_fct
    elif type in ["gaussian"]:
        func = gaussian_fct
    elif type in ["prony"]:
        func = prony_series_eval
    else:
        raise ValueError("Not implemented type")

    return func(times, *popt)


def memory_fit_kernel(times, kernel, type="exp", **kwargs):
    """
    Fit memory kernel using  type function. This fit one series per components of the memory kernel.
    Parameters
    ----------
        times: numpy array
            The time component of the data
        kernel: numpy array
            The kernel to be fitted
        thres: float, default=None
            A threshold that determined the numerical zero in the filetring of the data.
            If None, it is set to the value of precision of the float on the machine
        N_keep: int, default=None
            Maximum number of terms in the series to keep.
            If None, it is determined from the threshold.

    Note: Smaller N_keep or higher threshold result in faster analysis. Result can also depend strongly of the value of either thres or N_keep
    """
    _, dim_basis, dim_x = kernel.shape
    popt = [[0] * dim_basis for i in range(dim_x)]
    for d in range(dim_x):
        for k in range(dim_basis):
            popt[d][k] = memory_fit(np.asarray(times).squeeze(), kernel[:, k, d], type, **kwargs)[1]
    return type, popt


def memory_kernel_eval(times, params, type=None):
    """
    Get series evaluated at times
    You can then use pos_gle.kernel_eval(x, prony_eval) to get kernel at those points
    Parameters
    ----------
        times: numpy array
            Points at which evaluate the series
        params:
            The result of fit_kernel
        type: str
            The type of function to use
    """
    if type is None:
        type, popt = params
    else:
        popt = params
    nb_times = times.shape[0]
    dim_x = len(popt)
    dim_basis = len(popt[0])
    fit = np.zeros((nb_times, dim_basis, dim_x))
    for d in range(dim_x):
        for k in range(dim_basis):
            fit[:, k, d] = memory_fit_eval(np.asarray(times), popt[d][k], type=type)
    return fit
