#!python3
# -*- coding: utf-8 -*-

"""
Implementation of the prony fit using LDLt decomposition of the Hankel matrix
"""

import numpy as np
import scipy.linalg


def reconstruct_data(H):
    """
    Once the Hankel matrix have been filtered, show the effective resulting time serie.
    """
    y = np.empty(H.shape[0] + H.shape[1] - 1)
    y[: H.shape[0]] = H[:, 0]
    y[H.shape[1] - 1 :] = H[-1, :]
    return y


def phi_bis(u, v, A):
    return np.dot(u, np.dot(A[: len(u), : len(v)], v))


def ldl_hankel(y, n=None, thres=None, N_keep=None):
    """
    Compute the LDL^T decomposition of the Hankel matrix
    """
    if n is None:
        n = (y.shape[0] - 1) // 2
    hankel = scipy.linalg.hankel(y[: n + 1], y[n : 2 * n + 1])
    u_svd, s_svd, vh_svd = np.linalg.svd(hankel, full_matrices=False, hermitian=True)
    if thres is None:
        thres = s_svd.max() * (n + 1) * np.finfo(s_svd.dtype).eps  # Set thresold from numerical precision
    N_rank = np.count_nonzero(s_svd > thres)

    if N_keep is None:
        N = N_rank
    else:
        N = min(N_keep, N_rank)  # Number of singular values to keep, given from rank of Hankel matrix or from input
        # Update the thresold from highest non keeped singular value
        thres = s_svd[N]
    N = min(N, n)  # Limit the number of vector
    # Keep only the N highest singular value
    # print("Prony: Singular values. Value lower than thres are considered zero.")
    # print(N_rank, s_svd[:N])
    H = np.dot(u_svd[:, :N] * s_svd[:N], vh_svd[:N, :])
    u = np.zeros((N, n + 1))  # Matrix for polynomial coefficients
    D = np.zeros(N)
    v = np.identity(n + 1)
    curr_i = 0
    for i in range(0, n + 1):
        # print("----- ", i, curr_i)
        tilteu = v[i, :]
        for j in range(curr_i):
            tilteu = tilteu - phi_bis(tilteu, u[j, :], H) / phi_bis(u[j, :], u[j, :], H) * u[j, :]
        curr_norm = phi_bis(tilteu, tilteu, H)
        if np.abs(curr_norm) >= thres:  # If the vector i is already contained in the basis do not add
            u[curr_i, :] = tilteu
            D[curr_i] = curr_norm
            curr_i += 1
        if curr_i >= N:  # We completed the basis
            break
    if curr_i == 0:
        raise ValueError("Prony: Filtering of the data is too high. Try input higher N_keep or smaller thres.")
    Dhalf = np.diag(1 / np.sqrt(np.abs(D[:curr_i])))
    dlu = Dhalf @ u[:curr_i, :]
    return dlu, np.diag(np.sign(D[:curr_i])), H


def get_jacobi_matrix_reduced(y, n=None, thres=None, N_keep=None, debug=False):
    """
    Compute the Jacobi matrix
    """
    if n is None:  # Then use all data
        n = (y.shape[0] - 1) // 2
    dlu, D, H = ldl_hankel(y, n, thres=thres, N_keep=N_keep)
    shift = scipy.linalg.toeplitz([(0, 1)[i == 1] for i in range(n + 1)], np.zeros(n + 1))
    J_ldl = dlu @ (H @ shift) @ dlu.T @ D
    if debug:
        print("Prony: Result dimension: {}".format(J_ldl.shape[0]))
    return J_ldl, H


def clean_eigenvalues(J, dt, remove=True, debug=False):
    """
    Compute log of J after cleaning for eigenvalues
    Do some cleaning step to be more insensitive to the noise
    If remove is False, do not remove eigenvalues with absolute value higher than 1.
    They correspond to diverging exponentials.
    """
    # print("Cleaning")
    lamb, vect = np.linalg.eig(J)
    if remove:
        vect = vect[: np.sum((np.abs(lamb) <= 1)), (np.abs(lamb) <= 1)]  # remove last rows
        lamb = lamb[np.abs(lamb) <= 1]
        # print(lamb)
    # We need to duplicate real eigenvalue with negative value
    dupl = np.logical_and(np.abs(np.imag(lamb)) < 1e-15, np.real(lamb < 0))
    n_dupl = len(lamb[dupl])
    log_lamp = np.log(lamb) / dt
    # add eigenvector
    new_vect = np.zeros((len(lamb) + n_dupl, len(lamb) + n_dupl), dtype=complex)
    new_vect[: len(lamb), : len(lamb)] = vect
    new_vect[: len(lamb), len(lamb) :] = vect[:, dupl]
    new_vect[len(lamb) :, : len(lamb)][:, dupl] = 1j * np.diag(np.sign(np.imag(log_lamp[dupl])))
    new_vect[len(lamb) :, len(lamb) :] = -1j * np.diag(np.sign(np.imag(log_lamp[dupl])))
    log_lamp = np.concatenate((log_lamp, np.conj(log_lamp[dupl])))
    # reconstruct A
    if debug:
        print("Prony: Cleaning eigenvalues change the number of eigenvalues from {} to {}".format(J.shape[0], log_lamp.shape[0]))
    return np.real(new_vect @ np.diag(log_lamp) @ np.linalg.inv(new_vect))  # A is a real matrix


def prony_inspect_data(data, thres=None, N_keep=None):
    """
    Inspect data when fit fails.
    Comparing the rank number to N_keep indicate whatever thres or N_keep is controlling the filtering of the data.
    Return filtered data, that can be compared to orignal data.
    If there are still issue, try truncating the data to remove more noise.
    """
    y = data.ravel() / data.ravel()[0]
    n = (y.shape[0] - 1) // 2
    hankel = scipy.linalg.hankel(y[: n + 1], y[n : 2 * n + 1])
    u_svd, s_svd, vh_svd = np.linalg.svd(hankel, full_matrices=False, hermitian=True)
    if thres is None:
        thres = s_svd.max() * (n + 1) * np.finfo(s_svd.dtype).eps  # Set thresold from numerical precision
    N_rank = np.count_nonzero(s_svd > thres)

    if N_keep is None:
        N = N_rank
    else:
        N = min(N_keep, N_rank)  # Number of singular values to keep, given from rank of Hankel matrix or from input
        # Update the thresold from highest non keeped singular value
        thres = s_svd[N]
    N = min(N, n)  # Limit the number of vector
    # Keep only the N highest singular value
    print("Prony: Singular values. Value lower than thres are considered zero.")
    print(s_svd[:N])
    print("Prony: Rank number: {} Wanted keeped: {} Keeped singular values: {}".format(N_rank, N_keep, N))
    H = np.dot(u_svd[:, :N] * s_svd[:N], vh_svd[:N, :])
    return data.ravel()[0] * reconstruct_data(H)


def prony_fit_times_serie(data, dt, thres=None, N_keep=None, remove=True):
    """
    Fit one time series.
    Parameters
    ----------
        times: numpy array
            The time component of the data
        kernel: numpy array
            The kernel to be fitted
        thres: float, default=None
            A threshold that determined the numerical zero in the filetring of the data.
            If None, it is set to the value of precision of the float on the machine.
        N_keep: int, default=None
            Maximum number of terms in the series to keep.
            If None, it is determined from the threshold.
        remove: bool, default=True
            If true, remove diverging exponentials.
        debug: bool, default=False
            A flag indicating to output as well the filtered data.
    """
    J, _ = get_jacobi_matrix_reduced(data / data[0], thres=thres, N_keep=N_keep)
    A = clean_eigenvalues(J, dt, remove=remove)
    return data[0], A


def prony_series_eval(times, y0, A):
    """
    Get series evaluated at times
    You can then use pos_gle.kernel_eval(x, prony_eval) to get kernel at those points
    Parameters
    ----------
        times: numpy array
            Points at which evaluate the series
        y0: float
            Initial value of the data
        A :
            Result of the prony fitting.
    """
    nb_times = times.shape[0]
    fit = np.zeros((nb_times,))
    for i, t in enumerate(times.flatten()):  # TODO: Implement with linear system
        fit[i] = y0 * scipy.linalg.expm(t * A)[0, 0]
    return fit


def prony_fit_kernel(times, kernel, thres=None, N_keep=None):
    """
    Fit memory kernel using prony series. This fit one series per components of the memory kernel.
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
    dt = (times[1] - times[0]).values
    list_A = [[0] * dim_basis for i in range(dim_x)]
    for d in range(dim_x):
        for k in range(dim_basis):
            list_A[d][k] = prony_fit_times_serie(kernel[:, k, d].values, dt, thres=thres, N_keep=N_keep)
    return list_A


def prony_series_kernel_eval(times, list_A):
    """
    Get series evaluated at times
    You can then use pos_gle.kernel_eval(x, prony_eval) to get kernel at those points
    Parameters
    ----------
        times: numpy array
            Points at which evaluate the series
        list_A:
            The result of fit_kernel
    """
    nb_times = times.shape[0]
    dim_x = len(list_A)
    dim_basis = len(list_A[0])
    fit = np.zeros((nb_times, dim_basis, dim_x))
    for d in range(dim_x):
        for k in range(dim_basis):
            for i, t in enumerate(times.flatten()):
                fit[i, k, d] = list_A[d][k][0] * scipy.linalg.expm(t * list_A[d][k][1])[0, 0]
    return fit
