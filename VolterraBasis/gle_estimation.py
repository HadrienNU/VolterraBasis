import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from scipy.stats import describe

from .basis import sum_describe

from .correlation import correlation_1D, correlation_ND, correlation_direct_1D, correlation_direct_ND

from .fkernel import kernel_first_kind_trapz, kernel_first_kind_rect, kernel_first_kind_midpoint, kernel_second_kind_rect, kernel_second_kind_trapz


def solve_linear(G, b):  # Write also a sparse version
    """
    Solve the linear problem Gx = b
    """
    x_np = np.linalg.inv(G.values) @ b.values
    return xr.DataArray(x_np, dims=b.dims)


class Estimator_gle(object):
    """
    The main class for the position dependent memory extraction holding all data.
    """

    def __init__(self, xva_arg, model_class, basis, trunc=1.0, L_obs=None, saveall=True, prefix="", verbose=True, n_jobs=1, **kwargs):
        """
        Create an instance of the Pos_gle class.

        Parameters
        ----------
        xva_arg : xarray dataset (['time', 'x', 'v', 'a']) or list of datasets.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a xarray timeseries
            or a listlike collection of them.
        basis :  scikit-learn transformer to get the element of the basis
                This class should implement, basis() and deriv() function
                and deal with periodicity of the data.
                If a fit() method is defined, it will be called at initialization
        saveall : bool, default=True
            Whether to save all output functions.
        prefix : str
            Prefix for the saved output functions.
        verbose : bool, default=True
            Set verbosity.
        trunc : float, default=1.0
            Truncate all correlation functions and the memory kernel after this
            time value.
        L_obs: str, default given by the model
            Name of the column containing the time derivative of the observable
        """

        # Create all internal variables
        self.saveall = saveall
        self.prefix = prefix
        self.verbose = verbose

        # filenames
        self.corrsfile = "corrs.nc"
        self.kernelfile = "kernel.nc"

        self.data_describe = None

        self._do_check(xva_arg)  # Do some check on the trajectories

        # Save trajectory properties
        if self.xva_list is None:
            return

        # processing input arguments
        self.weights = np.array([xva["time"].shape[0] for xva in self.xva_list], dtype=int)  # Should be the various lenght of trajectory
        self.weightsum = np.sum(self.weights)
        if self.verbose:
            print("Found trajectories with the following lengths:")
            print(self.weights)

        self.n_jobs = int(n_jobs)
        if self.n_jobs <= 1:
            self.loop_over_trajs = self._loop_over_trajs_serial
        else:
            self.loop_over_trajs = self._loop_over_trajs_parallel

        if L_obs is None:  # Default value given by the class of the model
            L_obs = model_class.set_of_obs[-1]

        dim_x = self.xva_list[0].dims["dim_x"]
        dim_obs = self.xva_list[0][L_obs].shape[1]
        self.dt = self.xva_list[0].attrs["dt"]

        trunc_ind = self._compute_trunc_ind(trunc)

        # Fit basis from some description of the data
        describe_data = self.describe_data()

        self.model = model_class(basis, self.dt, dim_x=dim_x, dim_obs=dim_obs, trunc_ind=trunc_ind, L_obs=L_obs, describe_data=describe_data)

        self._do_check_obs(model_class.set_of_obs, self.model.L_obs)  # Check that we have in the trajectories what we need

        # For retrocompatibility, expose model methods for coefficients evaluation
        for func in dir(self.model):
            if callable(getattr(self.model, func)) and not func.startswith("_"):
                if "eval" in func or "compute" in func or "func" in ["basis_vector"]:
                    setattr(self, func, getattr(self.model, func))

    def _do_check(self, xva_arg):
        if xva_arg is not None:
            if isinstance(xva_arg, xr.Dataset):
                self.xva_list = [xva_arg]
            else:
                self.xva_list = xva_arg
            for xva in self.xva_list:
                if "time" not in xva.dims:
                    raise Exception("Time is not a coordinate. Please provide dataset with time, " "or an iterable collection (i.e. list) " "of dataset with time.")
                if "dt" not in xva.attrs:
                    raise Exception("Timestep not in dataset attrs")
        else:
            self.xva_list = None

    def _do_check_obs(self, set_of_obs, L_obs):
        for xva in self.xva_list:
            for col in set_of_obs:
                if col not in xva.data_vars:
                    raise Exception("Please provide time,{} dataset, " "or an iterable collection (i.e. list) " "of time,{} dataset.".format(set_of_obs, set_of_obs))
                if L_obs not in xva.data_vars:
                    raise Exception("Please provide dataset that include {} as variable .".format(L_obs))

    def _compute_trunc_ind(self, trunc):
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
        return self.trunc_ind

    def describe_data(self):
        """
        Return a description of the data
        """
        # Prévoir la sauvegarde du résultats pour 1) ne pas avoir à la calculer à chaque fois 2) pouvoir le sauvegarder
        if self.data_describe is None:
            describe_set = [describe(xva["x"].data) for xva in self.xva_list]
            self.data_describe = describe_set[0]
            for des_data in describe_set[1:]:
                self.data_describe = sum_describe(self.data_describe, des_data)
        return self.data_describe

    def to_gfpe(self, model=None, new_obs_name="dE"):
        """
        Update trajectories to compute derivative of the basis function
        """
        if model is None:
            model = self.model
        self.model.rank_projection = False
        for i in range(len(self.xva_list)):
            _, _, dE = model.basis_vector(self.xva_list[i])
            self.xva_list[i].update({new_obs_name: dE.rename({"dim_basis": "dim_dE"})})
        self.model.L_obs = new_obs_name
        self.model.dim_obs = dE.shape[-1]

    def _loop_over_trajs_serial(self, func, model, **kwargs):
        """
        A generator for iteration over trajectories
        """
        # Et voir alors pour faire une version parallélisé (en distribué)
        array_res = [func(weight, xva, model, **kwargs) for weight, xva in zip(self.weights, self.xva_list)]
        # print(array_res)
        res = [0.0] * len(array_res[0])
        for weight, single_res in zip(self.weights, array_res):
            for i, arr in enumerate(single_res):
                res[i] += arr * weight / self.weightsum
        return res

    def _loop_over_trajs_parallel(self, func, model, **kwargs):
        """
        A generator for iteration over trajectories
        """
        from joblib import Parallel, delayed

        array_res = Parallel(n_jobs=self.n_jobs)(delayed(func)(weight, xva, model, **kwargs) for weight, xva in zip(self.weights, self.xva_list))
        res = [0.0] * len(array_res[0])
        for weight, single_res in zip(self.weights, array_res):
            for i, arr in enumerate(single_res):
                res[i] += arr * weight / self.weightsum
        return res

    def compute_basis_mean(self, basis_type="force"):
        """
        Compute mean value of the basis function
        """
        return self.loop_over_trajs(self._compute_basis_mean, self.model, basis_type=basis_type)[0]

    def compute_gram_force(self):
        if self.verbose:
            print("Calculate gram...")
        avg_gram = self.loop_over_trajs(self._compute_gram, self.model, gram_type="force")[0]
        self.model.gram_force = avg_gram
        if self.verbose:
            print("Found gram:", avg_gram)
        return self.model

    def compute_gram_kernel(self):
        """
        Return gram matrix of the kernel part of the basis.
        """
        if self.verbose:
            print("Calculate kernel gram...")
        self.model.gram_kernel = self.loop_over_trajs(self._compute_gram, self.model, gram_type="kernel")[0]
        if self.model.rank_projection:
            self.model.gram_kernel = np.einsum("lj,jk,mk->lm", self.model.P_range, self.gram_kernel, self.model.P_range)
        return self.model

    def compute_effective_mass(self):
        """
        Return average effective mass computed from equipartition with the velocity.
        """
        if self.verbose:
            print("Calculate effective mass...")
        v2 = self.loop_over_trajs(self._compute_square_vel, self.model)[0]
        self.model.eff_mass = xr.DataArray(np.linalg.inv(v2), dims=("dim_x'", "dim_x"))

        if self.verbose:
            print("Found effective mass:", self.model.eff_mass)
        return self.model

    def compute_pos_effective_mass(self):
        """
        Return position-dependent effective inverse mass
        """
        if self.verbose:
            print("Calculate effective_mass...")
        pos_inv_mass, avg_gram = self.loop_over_trajs(self._compute_square_vel_pos, self.model)
        self.model.inv_mass_coeff = solve_linear(avg_gram, pos_inv_mass)
        return self.model

    def compute_mean_force(self):
        """
        Computes the mean force from the trajectories.
        """
        if self.verbose:
            print("Calculate mean force...")
        avg_disp, avg_gram = self.loop_over_trajs(self._projection_on_basis, self.model)
        self.model.gram_force = avg_gram
        self.model.force_coeff = solve_linear(avg_gram, avg_disp)
        return self.model

    def set_zero_force(self):
        self.model.force_coeff = xr.DataArray(np.zeros((self.model.N_basis_elt_force, self.model.dim_obs)), dims=["dim_basis", self.xva_list[0][self.model.L_obs].dims[1]])

    def compute_corrs(self, large=False, rank_tol=None, **kwargs):
        """
        Compute correlation functions.

        Parameters
        ----------
        large : bool, default=False
            When large is true, it use a slower way to compute correlation that is less demanding in memory
        rank_tol: float, default=None
            Tolerance for rank computation in case of projection onto the range of the basis
         second_order_method:bool, default = True
            If set to False do less computation but prevent to use second_order method in Volterra
        """
        if self.verbose:
            print("Calculate correlation functions...")
        if self.model.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        self.bkdxcorrw, self.dotbkdxcorrw, self.bkbkcorrw, self.dotbkbkcorrw = self.loop_over_trajs(self._correlation_ufunc, self.model, **kwargs)

        if self.model.rank_projection:
            if self.verbose:
                print("Projection on range space...")
            P_range = self.model._set_range_projection(rank_tol, self.bkbkcorrw.isel(time_trunc=0))
            P_range_tranpose = P_range.rename({"dim_basis_old": "dim_basis_old'", "dim_basis": "dim_basis'"})
            tempbkbkcorrw = xr.dot(P_range, self.bkbkcorrw.rename({"dim_basis": "dim_basis_old"}))
            self.bkbkcorrw = xr.dot(P_range_tranpose, tempbkbkcorrw.rename({"dim_basis'": "dim_basis_old'"}))
            # self.bkbkcorrw = np.einsum("lj,ijk,mk->ilm", self.model.P_range, self.bkbkcorrw, self.model.P_range)
            self.bkdxcorrw = xr.dot(P_range, self.bkdxcorrw.rename({"dim_basis": "dim_basis_old"}))
            if self.dotbkdxcorrw is not None:
                self.dotbkdxcorrw = xr.dot(P_range, self.dotbkdxcorrw.rename({"dim_basis": "dim_basis_old"}))
            if isinstance(self.dotbkbkcorrw, xr.DataArray):
                self.dotbkbkcorrw = xr.dot(P_range_tranpose, P_range, self.dotbkbkcorrw.rename({"dim_basis": "dim_basis_old", "dim_basis'": "dim_basis_old'"}))
        if self.saveall:
            xr.Dataset({"bkbk": self.bkbkcorrw, "bkdx": self.bkdxcorrw, "dotbkbk": self.dotbkbkcorrw, "dotbkdx": self.dotbkdxcorrw}, coords={"time_trunc": np.arange(self.bkbkcorrw.shape[-1]) * self.dt}).to_netcdf(self.corrsfile)
        return self.model

    def compute_kernel(self, method="rectangular", k0=None):
        """
        Computes the memory kernel.

        Parameters
        ----------
        method : {"rectangular", "midpoint", "midpoint_w_richardson","trapz","second_kind_rect","second_kind_trapz"}, default=rectangular
            Choose numerical method of inversion of the volterra equation
        k0 : float, default=0.
            If you give a nonzero value for k0, this is used at time zero for the trapz and second kind method. If set to None,
            the F-routine will calculate k0 from the second kind memory equation.
        """
        if self.bkbkcorrw is None or self.bkdxcorrw is None:
            raise Exception("Need correlation functions to compute the kernel.")
        if self.verbose:
            print("Compute memory kernel using {} method".format(method))
        time = np.arange(self.model.trunc_ind) * self.dt
        time_ker = time - time[0]  # Set zero time
        self.model.method = method  # Save used method
        if self.verbose:
            print("Use dt:", self.dt)
        if k0 is None and method in ["trapz", "second_kind_rect", "second_kind_trapz"]:  # Then we should compute initial value from time derivative at zero
            if self.dotbkdxcorrw is None:
                raise Exception("Need correlation with derivative functions to compute the kernel using this method or provide initial value.")
            k0 = solve_linear(self.bkbkcorrw.isel(time_trunc=0), self.dotbkdxcorrw.isel(time_trunc=0)).to_numpy()
            if self.verbose:
                print("K0", k0)
                # print("Gram", self.bkbkcorrw[0, :, :])
                # print("Gram eigs", np.linalg.eigvals(self.bkbkcorrw[0, :, :]))
        if method in ["rect", "rectangular"]:
            kernel = kernel_first_kind_rect(self.bkbkcorrw.to_numpy(), self.bkdxcorrw.to_numpy(), self.dt)
        elif method == "midpoint":  # Deal with not even data lenght
            kernel = kernel_first_kind_midpoint(self.bkbkcorrw.to_numpy(), self.bkdxcorrw.to_numpy(), self.dt)
            time_ker = time_ker[:-1:2]
        elif method == "midpoint_w_richardson":
            ker = kernel_first_kind_midpoint(self.bkbkcorrw.to_numpy(), self.bkdxcorrw.to_numpy(), self.dt)
            ker_3 = kernel_first_kind_midpoint(self.bkbkcorrw.to_numpy()[:, :, ::3], self.bkdxcorrw.to_numpy()[:, :, ::3], 3 * self.dt)
            kernel = (9 * ker[::3][: ker_3.shape[0]] - ker_3) / 8
            time_ker = time_ker[:-3:6]
        elif method == "trapz":
            ker = kernel_first_kind_trapz(k0, self.bkbkcorrw.to_numpy(), self.bkdxcorrw.to_numpy(), self.dt)
            kernel = 0.5 * (ker[1:-1, :, :] + 0.5 * (ker[:-2, :, :] + ker[2:, :, :]))  # Smoothing
            kernel = np.insert(kernel, 0, k0, axis=0)
            time_ker = time_ker[:-1]
        elif method == "second_kind_rect":
            if self.dotbkdxcorrw is None or self.dotbkbkcorrw is None:
                raise Exception("Need correlation with derivative functions to compute the kernel using this method, please use other method.")
            kernel = kernel_second_kind_rect(k0, self.bkbkcorrw.isel(time_trunc=0).to_numpy(), self.dotbkbkcorrw.to_numpy(), self.dotbkdxcorrw.to_numpy(), self.dt)
        elif method == "second_kind_trapz":
            if self.dotbkdxcorrw is None or self.dotbkbkcorrw is None:
                raise Exception("Need correlation with derivative functions to compute the kernel using this method, please use other method.")
            kernel = kernel_second_kind_trapz(k0, self.bkbkcorrw.isel(time_trunc=0).to_numpy(), self.dotbkbkcorrw.to_numpy(), self.dotbkdxcorrw.to_numpy(), self.dt)
        else:
            raise Exception("Method for volterra inversion is not in  {rectangular, midpoint, midpoint_w_richardson,trapz,second_kind_rect,second_kind_trapz}")

        self.model.kernel = xr.DataArray(kernel, dims=("time_kernel", "dim_basis", self.bkdxcorrw.dims[1]), coords={"time_kernel": time_ker})
        if self.saveall:  # TODO: change to xarray save
            xr.Dataset({"kernel": self.model.kernel}).to_netcdf(self.kernelfile)
        return self.model

    def check_volterra_inversion(self, return_diff=False):
        """
        For checking if the volterra equation is correctly inversed
        Compute the integral in volterra equation using trapezoidal rule.
        This only check the volterra of the first kind

        Parameters
        ----------
        return_diff : bool, default = False
            Indicate if you want the result of the intégral or the difference between the result and the expected value
        """
        if self.model.kernel is None:
            raise Exception("Kernel has not been computed.")
        dt = self.dt
        res_int = np.zeros(self.bkdxcorrw.shape)
        for n in range(self.model.kernel.shape[0]):
            to_integrate = np.einsum("jki,ikl->ijl", self.bkbkcorrw[:, :, : n + 1][:, :, ::-1], self.model.kernel[: n + 1, :, :])
            res_int[:, :, n] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
        if return_diff:
            return np.abs(res_int - self.bkdxcorrw)
        else:
            return res_int

    def compute_projected_corrs(self, left_op=None):
        """
        Compute correlation between noise and left_op using the projected correlations
        """
        return self.loop_over_trajs(self._corrs_w_noise, self.model, left_op=left_op)

    @staticmethod
    def _projection_on_basis(weight, xva, model, gram_type="force", **kwargs):
        """
        Do the needed scalar product for one traj
        Return 2 array with dimension:

        avg_disp: (N_basis_elt_force, dim_obs))
        avg_gram: (N_basis_elt_force, N_basis_elt_force)
        """
        E = model.basis_vector(xva, compute_for=gram_type)
        avg_disp = xr.dot(E, xva[model.L_obs]) / weight
        avg_gram = xr.dot(E, E.rename({"dim_basis": "dim_basis'"})) / weight
        return avg_disp, avg_gram

    @staticmethod
    def _compute_basis_mean(weight, xva, model, basis_type="force", **kwargs):
        """
        Do the needed scalar product for one traj
        """
        E = model.basis_vector(xva, compute_for=basis_type)
        return (E.mean(dim="time"),)

    @staticmethod
    def _compute_gram(weight, xva, model, gram_type="force", **kwargs):
        """
        Do the needed scalar product for one traj
        """
        E = model.basis_vector(xva, compute_for=gram_type)
        avg_gram = xr.dot(E, E.rename({"dim_basis": "dim_basis'"})) / weight
        return (avg_gram,)

    @staticmethod
    def _compute_square_vel(weight, xva, model, **kwargs):
        """
        Squared velocity for effective masse
        """
        return (xr.dot(xva["v"], xva["v"].rename({"dim_x": "dim_x'"})) / weight,)

    @staticmethod
    def _compute_square_vel_pos(weight, xva, model, **kwargs):
        """
        Do the needed scalar product for one traj
        """
        E = model.basis_vector(xva, compute_for="kernel")
        avg_disp = xr.dot(E, xva["v"], xva["v"].rename({"dim_x": "dim_x'"})) / weight
        avg_gram = xr.dot(E, E.rename({"dim_basis": "dim_basis'"})) / weight
        return avg_disp, avg_gram

    @staticmethod
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
            func = correlation_1D
        elif method == "direct" and not vectorize:
            func = correlation_direct_ND
        elif method == "direct" and vectorize:
            func = correlation_direct_1D
        else:
            func = correlation_ND
        E_force, E, dE = model.basis_vector(xva)
        # print(E_force, model.force_coeff)
        ortho_xva = xva[model.L_obs] - xr.dot(E_force, model.force_coeff)
        # print(ortho_xva.head(), E.head())
        bkdxcorrw = xr.apply_ufunc(
            func, E, ortho_xva, input_core_dims=[["time"], ["time"]], output_core_dims=[["time_trunc"]], exclude_dims={"time"}, kwargs={"trunc": model.trunc_ind}, dask_gufunc_kwargs={"output_sizes": {"time_trunc": model.trunc_ind}, "allow_rechunk": True}, vectorize=vectorize, dask="parallelized"
        )
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
        if second_order_method:
            dotbkdxcorrw = xr.apply_ufunc(
                func,
                dE,
                ortho_xva,
                input_core_dims=[["time"], ["time"]],
                output_core_dims=[["time_trunc"]],
                exclude_dims={"time"},
                kwargs={"trunc": model.trunc_ind},
                dask_gufunc_kwargs={"output_sizes": {"time_trunc": model.trunc_ind}, "allow_rechunk": True},
                vectorize=vectorize,
                dask="parallelized",
            )
            dotbkbkcorrw = xr.apply_ufunc(
                func,
                dE.rename({"dim_basis": "dim_basis'"}),
                E,
                input_core_dims=[["time"], ["time"]],
                output_core_dims=[["time_trunc"]],
                exclude_dims={"time"},
                kwargs={"trunc": model.trunc_ind},
                dask_gufunc_kwargs={"output_sizes": {"time_trunc": model.trunc_ind}, "allow_rechunk": True},
                vectorize=vectorize,
                dask="parallelized",
            )
        else:
            # We can compute only the first element then, that is faster
            dotbkdxcorrw = xr.dot(dE, ortho_xva).expand_dims({"time_trunc": 1}) / weight
            dotbkbkcorrw = np.array([[0.0]])
        return bkdxcorrw, dotbkdxcorrw, bkbkcorrw, dotbkbkcorrw

    @staticmethod
    def _corrs_w_noise(weight, xva, model, left_op=None, **kwargs):
        """
        Do the needed scalar product for one traj
        """
        return model.compute_corrs_w_noise(xva, left_op)
