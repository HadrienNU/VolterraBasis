import numpy as np
import xarray as xr
from scipy.integrate import trapezoid, simpson

from .fkernel import kernel_first_kind_trapz, kernel_first_kind_rect, kernel_first_kind_midpoint, kernel_second_kind_rect, kernel_second_kind_trapz
from .fkernel import memory_rect, memory_trapz, corrs_rect, corrs_trapz
from .correlation import correlation_1D, correlation_ND


def _convert_input_array_for_evaluation(array, dim_x):
    """
    Take input and return xarray Dataset with correct shape
    """
    if isinstance(array, xr.Dataset):  # TODO add check on dimension of array
        return array
    else:
        x = np.asarray(array).reshape(-1, dim_x)
        return xr.Dataset({"x": (["time", "dim_x"], x)})


class Trajectories_handler(object):
    """
    The main class for the data.
    """

    def __init__(self, xva_arg, trunc=1.0, L_obs="a", verbose=True, **kwargs):
        """
        Create an instance of the Trajectories_handler class.

        Parameters
        ----------
        xva_arg : xarray dataset (['time', 'x', 'v', 'a']) or list of datasets.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a xarray timeseries
            or a listlike collection of them.
        trunc : float, default=1.0
            Truncate all correlation functions and the memory kernel after this
            time value.
        L_obs: str, default= "a"
            Name of the column containing the time derivative of the observable
        verbose : bool, default=True
            Set verbosity.
        """
        self.verbose = verbose
        self.set_of_obs = ["x", "v", L_obs]

        self._do_check(xva_arg)  # Do some check on the trajectories

        # Save trajectory properties
        if self.xva_list is None:
            return

        self.dim_x = self.xva_list[0].dims["dim_x"]
        self.dim_obs = self.xva_list[0][L_obs].shape[1]

        # processing input arguments
        self.weights = np.array([xva["time"].shape[0] for xva in self.xva_list], dtype=int)  # Should be the various lenght of trajectory
        self.weightsum = np.sum(self.weights)
        if self.verbose:
            print("Found trajectories with the following lengths:")
            print(self.weights)

        lastinds = np.array([xva["time"][-1] for xva in self.xva_list])
        smallest = np.min(lastinds)
        if smallest < trunc:
            if self.verbose:
                print("Warning: Found a trajectory shorter than " "the argument trunc. Override.")
            trunc = smallest
        # Find index of the time truncation
        self.trunc_ind = (self.xva_list[0]["time"] <= trunc).sum().data
        if self.verbose:
            print("Trajectories are truncated at lenght {} for dynamic analysis".format(self.trunc_ind))

        self.compute_gram = Trajectories_handler._compute_gram
        self.compute_projection = Trajectories_handler._projection_on_basis

    def _do_check(self, xva_arg):
        if xva_arg is not None:
            if isinstance(xva_arg, xr.Dataset):
                self.xva_list = [xva_arg]
            else:
                self.xva_list = xva_arg
            for xva in self.xva_list:
                for col in self.set_of_obs:
                    if col not in xva.data_vars:
                        raise Exception("Please provide time,{} dataset, " "or an iterable collection (i.e. list) " "of time,{} dataset.".format(self.set_of_obs, self.set_of_obs))
                if "time" not in xva.dims:
                    raise Exception("Time is not a coordinate. Please provide dataset with time, " "or an iterable collection (i.e. list) " "of dataset with time.")
                if "dt" not in xva.attrs:
                    raise Exception("Timestep not in dataset attrs")
        else:
            self.xva_list = None

    def _loop_over_traj_serial(self, func, model, **kwargs):
        """
        A generator for iteration over trajectories
        """
        # Et voir alors pour faire une version parallélisé (en distribué)
        array_res = [func(weight, xva, model, **kwargs) for weight, xva in zip(self.weights, self.xva_list)]
        res = [0.0] * len(array_res[0])
        for weight, single_res in zip(self.weights, array_res):
            for i, arr in enumerate(single_res):
                res[i] += arr.to_numpy() * weight / self.weightsum
        return res

    @classmethod
    def _projection_on_basis(weight, xva, model, gram_type="force"):
        """
        Do the needed scalar product for one traj
        """
        E = model.basis_vector(xva, compute_for=gram_type)
        avg_disp = xr.dot(E, xva[model.L_obs]) / weight
        avg_gram = xr.dot(E, E.rename({"dim_basis": "dim_basis'"})) / weight
        return avg_disp, avg_gram

    @classmethod
    def _compute_gram(weight, xva, model, gram_type="force"):
        """
        Do the needed scalar product for one traj
        """
        E = model.basis_vector(xva, compute_for=gram_type)
        avg_gram = xr.dot(E, E.rename({"dim_basis": "dim_basis'"})) / weight
        return (avg_gram,)

    @classmethod
    def _correlation_all(weight, xva, model, method="fft", vectorize=False, second_order_method=True):
        """
        Do the correlation
        Return 4 array with dimensions

        bkbkcorrw :(trunc_ind, N_basis_elt_kernel, N_basis_elt_kernel)
        bkdxcorrw :(trunc_ind, N_basis_elt_kernel, dim_obs)
        dotbkdxcorrw :(trunc_ind, N_basis_elt_kernel, dim_obs)
        dotbkbkcorrw :(trunc_ind, N_basis_elt_kernel, N_basis_elt_kernel)
        """
        if method == "fft" and vectorize:
            func = correlation_1D
        # elif method == "direct" and not vectorize:
        #     func = correlation_direct_ND
        # elif method == "direct" and vectorize:
        #     func = correlation_direct_1D
        else:
            func = correlation_ND
        E_force, E, dE = model.basis_vector(xva)
        ortho_xva = xva[model.L_obs] - np.matmul(E_force, model.force_coeff)  # TODO

        bkdxcorrw = xr.apply_ufunc(func, E, ortho_xva, input_core_dims=[["time"], ["time"]], output_core_dims=[["time_trunc"]], exclude_dims={"time"}, kwargs={"trunc": model.trunc_ind}, vectorize=vectorize, dask="forbidden")
        bkbkcorrw = xr.apply_ufunc(correlation_ND, E, input_core_dims=[["time"]], output_core_dims=[["dim_basis'", "time_trunc"]], exclude_dims={"time"}, kwargs={"trunc": model.trunc_ind}, vectorize=vectorize, dask="forbidden")
        if second_order_method:
            dotbkdxcorrw = xr.apply_ufunc(func, dE, ortho_xva, input_core_dims=[["time"], ["time"]], output_core_dims=[["time_trunc"]], exclude_dims={"time"}, kwargs={"trunc": model.trunc_ind}, vectorize=vectorize, dask="forbidden")
            dotbkbkcorrw = xr.apply_ufunc(func, dE.rename({"dim_basis": "dim_basis'"}), E, input_core_dims=[["time"], ["time"]], output_core_dims=[["time_trunc"]], exclude_dims={"time"}, kwargs={"trunc": model.trunc_ind}, vectorize=vectorize, dask="forbidden")
        else:
            # We can compute only the first element then, that is faster
            dotbkdxcorrw = xr.dot(dE, xva[model.L_obs], ortho_xva) / weight  # Maybe reshape to add a dimension of size 1 and name time_trunc
            dotbkbkcorrw = 0.0
        return bkdxcorrw, dotbkdxcorrw, bkbkcorrw, dotbkbkcorrw

    # @classmethod
    # def _correlation_vectorize(weight, xva, model, method="fft"):
    #     """
    #     Vectorized version of the correlation
    #     Return 4 array with dimensions
    #
    #     bkbkcorrw :(trunc_ind, N_basis_elt_kernel, N_basis_elt_kernel)
    #     bkdxcorrw :(trunc_ind, N_basis_elt_kernel, dim_obs)
    #     dotbkdxcorrw :(trunc_ind, N_basis_elt_kernel, dim_obs)
    #     dotbkbkcorrw :(trunc_ind, N_basis_elt_kernel, N_basis_elt_kernel)
    #
    #     """
    #     if method == "direct":
    #         func = correlation_direct_1D
    #     else:
    #         func = correlation_1D
    #     E_force, E, dE = model.basis_vector(xva)
    #     force = np.matmul(E_force, model.force_coeff)
    #     dim_obs = force.shape[-1]
    #     N_basis_elt_kernel = E.shape[1]
    #     # On va le faire avec un apply u_func et un vectorize
    #     bkbkcorrw = np.zeros((model.trunc_ind, N_basis_elt_kernel, N_basis_elt_kernel))
    #     bkdxcorrw = np.zeros((model.trunc_ind, N_basis_elt_kernel, dim_obs))
    #
    #     dotbkdxcorrw = np.zeros((model.trunc_ind, N_basis_elt_kernel, dim_obs))
    #     dotbkbkcorrw = np.zeros((model.trunc_ind, N_basis_elt_kernel, N_basis_elt_kernel))
    #
    #     for n in range(N_basis_elt_kernel):
    #         for d in range(model.dim_obs):
    #             bkdxcorrw[:, n, d] += weight * correlation_1D(E[:, n], xva[model.L_obs].data[:, d] - force[:, d], trunc=model.trunc_ind)  # Correlate derivative of observable minus mean value
    #             dotbkdxcorrw[:, n, d] += weight * correlation_1D(dE[:, n], xva[model.L_obs].data[:, d] - force[:, d], trunc=model.trunc_ind)
    #         for m in range(N_basis_elt_kernel):
    #             bkbkcorrw[:, n, m] += weight * correlation_1D(E[:, n], E[:, m], trunc=model.trunc_ind)
    #             dotbkbkcorrw[:, n, m] += weight * correlation_1D(dE[:, n], E[:, m], trunc=model.trunc_ind)
    #     return bkdxcorrw, dotbkdxcorrw, bkbkcorrw, dotbkbkcorrw
