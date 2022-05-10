import numpy as np
import xarray as xr
from scipy.integrate import trapezoid

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


class Pos_gle_base(object):
    """
    The main class for the position dependent memory extraction,
    holding all data and the extracted memory kernels.
    """

    def __init__(self, xva_arg, basis, saveall=True, prefix="", verbose=True, trunc=1.0, L_obs="a", **kwargs):
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
        kT : float, default=2.494
            Numerical value for kT.
        trunc : float, default=1.0
            Truncate all correlation functions and the memory kernel after this
            time value.
        L_obs: str, default= "a"
            Name of the column containing the time derivative of the observable
        """
        self.L_obs = L_obs
        self.set_of_obs = ["x", "v", self.L_obs]

        self._do_check(xva_arg)  # Do some check on the trajectories
        self._check_basis(basis)  # Do check on basis

        # Create all internal variables
        self.saveall = saveall
        self.prefix = prefix
        self.verbose = verbose

        # filenames
        self.corrsfile = "corrs.txt"
        self.corrsdxfile = "a-u_corrs.txt"
        self.dcorrsfile = "dE_corrs.txt"
        self.dcorrsdxfile = "dE_a-u_corrs.txt"
        self.kernelfile = "kernel.txt"

        self.bkbkcorrw = None
        self.bkdxcorrw = None
        self.dotbkdxcorrw = None
        self.dotbkbkcorrw = None
        self.force_coeff = None

        self.mass = None
        self.kernel_gram = None

        self.method = None

        self.rank_projection = False
        self.P_range = None

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

    def _check_basis(self, basis):
        """
        Simple checks on the basis class
        """
        if not (callable(getattr(basis, "basis", None))) or not (callable(getattr(basis, "deriv", None))):
            raise Exception("Basis class do not define basis() or deriv() method")
        self.basis = basis
        if not hasattr(self.basis, "n_output_features_"):
            if callable(getattr(self.basis, "fit", None)):
                self.basis = self.basis.fit(np.concatenate([xva["x"].data for xva in self.xva_list], axis=0))  # Fit basis
            else:
                raise Exception("Basis class do have fit() method or do not expose n_output_features_ attribute")

        self.N_basis_elt = self.basis.n_output_features_

    def _set_range_projection(self, rank_tol):
        """
        Set and perfom the projection onto the range of the basis for kernel
        """
        if self.verbose:
            print("Projection on range space...")
        # Check actual rank of the matrix
        B0 = self.bkbkcorrw[0, :, :]
        # Do SVD
        U, S, V = np.linalg.svd(B0, compute_uv=True, hermitian=True)
        # Compute rank from svd
        if rank_tol is None:
            rank_tol = S.max(axis=-1, keepdims=True) * max(B0.shape[-2:]) * np.finfo(S.dtype).eps
        rank = np.count_nonzero(S > rank_tol, axis=-1)
        # print(rank, U.shape, S.shape, V.shape)
        if rank < self.N_basis_elt_kernel:
            if rank != self.N_basis_elt_kernel - self.dim_x:
                print("Warning: rank is different than expected. Current {}, Expected {}. Consider checking your basis or changing the tolerance".format(rank, self.N_basis_elt_kernel - self.dim_x))
            elif rank == 0:
                raise Exception("Rank of basis is null.")
            # Construct projection
            self.P_range = U[:, :rank].T  # # Save the matrix for future use, matrix is rank x N_basis_elt_kernel
            # Faster to do one product in order
            tempbkbkcorrw = np.einsum("ijk,mk->ijm", self.bkbkcorrw, self.P_range)
            self.bkbkcorrw = np.einsum("lj,ijk->ilk", self.P_range, tempbkbkcorrw)
            # self.bkbkcorrw = np.einsum("lj,ijk,mk->ilm", self.P_range, self.bkbkcorrw, self.P_range)
            self.bkdxcorrw = np.einsum("kj,ijd->ikd", self.P_range, self.bkdxcorrw)
            if self.dotbkdxcorrw is not None:
                self.dotbkdxcorrw = np.einsum("kj,ijd->ikd", self.P_range, self.dotbkdxcorrw)
            if self.dotbkbkcorrw is not None:
                self.dotbkbkcorrw = np.einsum("lj,ijk,mk->ilm", self.P_range, self.dotbkbkcorrw, self.P_range)
        else:
            print("No projection onto the range of the basis performed as basis is not deficient.")
            self.P_range = np.identity(self.N_basis_elt_kernel)

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        This is the main method that should be implemented by children class.
        It take as argument a trajectory and should return the value of the basis function depending of the wanted case.
        There is three case that should be implemented.

        "force": for the evaluation and computation of the mean force.

        "pmf": for evaluation of the pmf using integration of the mean force

        "kernel": for the evaluation of the kernel.

        "corrs": for the computation of the correlation function.
        """
        raise NotImplementedError

    def compute_gram(self, kT=1.0):
        if self.verbose:
            print("Calculate gram...")
            print("Use kT:", kT)
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            E = self.basis_vector(xva, compute_for="force")
            avg_gram += np.matmul(E.T, E) / self.weightsum
        self.invgram = kT * np.linalg.pinv(avg_gram)

        if self.verbose:
            print("Found inverse gram:", self.invgram)
        return kT * avg_gram

    def compute_effective_mass(self, kT=1.0):
        """
        Return average effective mass computed from equipartition with the velocity.
        """
        if self.verbose:
            print("Calculate effective mass...")
            print("Use kT:", kT)
        v2sum = 0.0
        for i, xva in enumerate(self.xva_list):
            v2sum += np.einsum("ik,ij->kj", xva["v"], xva["v"])
        v2 = v2sum / self.weightsum
        mass = kT * np.linalg.inv(v2)

        if self.verbose:
            print("Found effective mass:", mass)
        return mass

    def compute_pos_effective_mass(self, kT=1.0):
        """
        Return position-dependent effective mass
        """
        if self.verbose:
            print("Calculate kernel gram...")
        avg_gram = np.zeros((self.N_basis_elt_kernel, self.N_basis_elt_kernel))
        for weight, xva in zip(self.weights, self.xva_list):
            _, E, _ = self.basis_vector(xva, compute_for="corrs")
            avg_gram += np.matmul(E.T, E) / self.weightsum
        self.mass = kT * np.linalg.inv(avg_gram)
        return self.mass

    def compute_kernel_gram(self):
        """
        Return gram matrix of the kernel part of the basis.
        """
        if self.kernel_gram is None:
            if self.verbose:
                print("Calculate kernel gram...")
            avg_gram = np.zeros((self.N_basis_elt_kernel, self.N_basis_elt_kernel))
            for weight, xva in zip(self.weights, self.xva_list):
                E = self.basis_vector(xva, compute_for="kernel")
                if self.rank_projection:
                    E = np.einsum("kj,ijd->ikd", self.P_range, E)
                avg_gram += np.einsum("ikd,ild->kl", E, E) / self.weightsum  # np.matmul(E.T, E)
            self.kernel_gram = avg_gram
        return self.kernel_gram

    def set_zero_force(self):
        self.force_coeff = np.zeros((self.N_basis_elt_force, self.dim_obs))

    def compute_mean_force(self):
        """
        Computes the mean force from the trajectories.
        """
        if self.verbose:
            print("Calculate mean force...")
        avg_disp = np.zeros((self.N_basis_elt_force, self.dim_obs))
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            E = self.basis_vector(xva, compute_for="force")
            avg_disp += np.matmul(E.T, xva[self.L_obs].data) / self.weightsum
            avg_gram += np.matmul(E.T, E) / self.weightsum
        self.force_coeff = np.matmul(np.linalg.inv(avg_gram), avg_disp)

    def compute_corrs(self, large=False, rank_tol=None):
        """
        Compute correlation functions.

        Parameters
        ----------
        large : bool, default=False
            When large is true, it use a slower way to compute correlation that is less demanding in memory
        rank_tol: float, default=None
            Tolerance for rank computation in case of projection onto the range of the basis
        """
        if self.verbose:
            print("Calculate correlation functions...")
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")

        self.bkbkcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.N_basis_elt_kernel))
        self.bkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.dim_obs))

        self.dotbkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.dim_obs))  # Needed for initial value anyway
        self.dotbkbkcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.N_basis_elt_kernel))

        for weight, xva in zip(self.weights, self.xva_list):
            E_force, E, dE = self.basis_vector(xva)
            # print(E_force.shape, E.shape, dE.shape)
            force = np.matmul(E_force, self.force_coeff)
            # print(force.shape, xva[self.L_obs].data.shape)
            # print(self.bkdxcorrw.shape, correlation_ND(E, (xva[self.L_obs].data - force)).shape)
            if not large:
                try:
                    self.bkdxcorrw += weight * correlation_ND(E, (xva[self.L_obs].data - force), trunc=self.trunc_ind)
                    self.dotbkdxcorrw += weight * correlation_ND(dE, (xva[self.L_obs].data - force), trunc=self.trunc_ind)
                    self.bkbkcorrw += weight * correlation_ND(E, trunc=self.trunc_ind)
                    self.dotbkbkcorrw += weight * correlation_ND(dE, E, trunc=self.trunc_ind)
                except MemoryError:  # If too big slow way
                    if self.verbose:
                        print("Too many basis function, compute correlations one by one (slow)")
                    for n in range(E.shape[1]):
                        for d in range(self.dim_obs):
                            self.bkdxcorrw[:, n, d] += weight * correlation_1D(E[:, n], xva[self.L_obs].data[:, d] - force[:, d], trunc=self.trunc_ind)  # Correlate derivative of observable minus mean value
                            self.dotbkdxcorrw[:, n, d] += weight * correlation_1D(dE[:, n], xva[self.L_obs].data[:, d] - force[:, d], trunc=self.trunc_ind)
                        for m in range(E.shape[1]):
                            self.bkbkcorrw[:, n, m] += weight * correlation_1D(E[:, n], E[:, m], trunc=self.trunc_ind)
                            self.dotbkbkcorrw[:, n, m] += weight * correlation_1D(dE[:, n], E[:, m], trunc=self.trunc_ind)
            else:
                for n in range(E.shape[1]):
                    for d in range(self.dim_obs):
                        self.bkdxcorrw[:, n, d] += weight * correlation_1D(E[:, n], xva[self.L_obs].data[:, d] - force[:, d], trunc=self.trunc_ind)  # Correlate derivative of observable minus mean value
                        self.dotbkdxcorrw[:, n, d] += weight * correlation_1D(dE[:, n], xva[self.L_obs].data[:, d] - force[:, d], trunc=self.trunc_ind)
                    for m in range(E.shape[1]):
                        self.bkbkcorrw[:, n, m] += weight * correlation_1D(E[:, n], E[:, m], trunc=self.trunc_ind)
                        self.dotbkbkcorrw[:, n, m] += weight * correlation_1D(dE[:, n], E[:, m], trunc=self.trunc_ind)

        self.bkbkcorrw /= self.weightsum
        self.bkdxcorrw /= self.weightsum
        self.dotbkdxcorrw /= self.weightsum
        self.dotbkbkcorrw /= self.weightsum

        if self.rank_projection:
            self._set_range_projection(rank_tol)

        if self.saveall:
            np.savetxt(self.prefix + self.corrsdxfile, self.bkdxcorrw.reshape(self.trunc_ind, -1))
            np.savetxt(self.prefix + self.corrsfile, self.bkbkcorrw.reshape(self.trunc_ind, -1))
            np.savetxt(self.prefix + self.dcorrsdxfile, self.dotbkdxcorrw.reshape(self.trunc_ind, -1))
            np.savetxt(self.prefix + self.dcorrsfile, self.dotbkbkcorrw.reshape(self.trunc_ind, -1))

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
        print("Compute memory kernel using {} method".format(method))
        time = self.xva_list[0]["time"].data[: self.trunc_ind]
        dt = self.xva_list[0].attrs["dt"]
        self.time = (time - time[0]).reshape(-1, 1)  # Set zero time
        self.method = method  # Save used method
        if self.verbose:
            print("Use dt:", dt)

        if k0 is None and method in ["trapz", "second_kind_rect", "second_kind_trapz"]:  # Then we should compute initial value from time derivative at zero
            if self.dotbkdxcorrw is None:
                raise Exception("Need correlation with derivative functions to compute the kernel using this method or provide initial value.")
            k0 = np.matmul(np.linalg.inv(self.bkbkcorrw[0, :, :]), self.dotbkdxcorrw[0, :, :])
            if self.verbose:
                print("K0", k0)
                # print("Gram", self.bkbkcorrw[0, :, :])
                # print("Gram eigs", np.linalg.eigvals(self.bkbkcorrw[0, :, :]))
        if method in ["rect", "rectangular"]:
            self.kernel = kernel_first_kind_rect(self.bkbkcorrw, self.bkdxcorrw, dt)
        elif method == "midpoint":  # Deal with not even data lenght
            self.kernel = kernel_first_kind_midpoint(self.bkbkcorrw, self.bkdxcorrw, dt)
            self.time = self.time[:-1:2, :]
        elif method == "midpoint_w_richardson":
            ker = kernel_first_kind_midpoint(self.bkbkcorrw, self.bkdxcorrw, dt)
            ker_3 = kernel_first_kind_midpoint(self.bkbkcorrw[::3, :, :], self.bkdxcorrw[::3, :, :], 3 * dt)
            self.kernel = (9 * ker[::3][: ker_3.shape[0]] - ker_3) / 8
            self.time = self.time[:-3:6, :]
        elif method == "trapz":
            ker = kernel_first_kind_trapz(k0, self.bkbkcorrw, self.bkdxcorrw, dt)
            self.kernel = 0.5 * (ker[1:-1, :, :] + 0.5 * (ker[:-2, :, :] + ker[2:, :, :]))  # Smoothing
            self.kernel = np.insert(self.kernel, 0, k0, axis=0)
            self.time = self.time[:-1, :]
        elif method == "second_kind_rect":
            if self.dotbkdxcorrw is None or self.dotbkbkcorrw is None:
                raise Exception("Need correlation with derivative functions to compute the kernel using this method, please use other method.")
            self.kernel = kernel_second_kind_rect(k0, self.bkbkcorrw[0, :, :], self.dotbkbkcorrw, self.dotbkdxcorrw, dt)
        elif method == "second_kind_trapz":
            if self.dotbkdxcorrw is None or self.dotbkbkcorrw is None:
                raise Exception("Need correlation with derivative functions to compute the kernel using this method, please use other method.")
            self.kernel = kernel_second_kind_trapz(k0, self.bkbkcorrw[0, :, :], self.dotbkbkcorrw, self.dotbkdxcorrw, dt)
        else:
            raise Exception("Method for volterra inversion is not in  {rectangular, midpoint, midpoint_w_richardson,trapz,second_kind_rect,second_kind_trapz}")

        if self.saveall:
            np.savetxt(self.prefix + self.kernelfile, np.hstack((self.time, self.kernel.reshape(self.kernel.shape[0], -1))))

        return self.kernel

    def check_volterra_inversion(self):
        """
        For checking if the volterra equation is correctly inversed
        Compute the integral in volterra equation using trapezoidal rule
        """
        time = self.xva_list[0]["time"].data[: self.trunc_ind]
        dt = self.xva_list[0].attrs["dt"]
        res_int = np.zeros(self.bkdxcorrw.shape)
        # res_int[0, :] = 0.5 * dt * to_integrate[0, :]
        # if method == "trapz":
        for n in range(self.trunc_ind):
            to_integrate = np.einsum("ijk,ikl->ijl", self.bkbkcorrw[: n + 1, :, :][::-1, :, :], self.kernel[: n + 1, :, :])
            res_int[n, :] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
            # res_int[n, :] = -1 * simpson(to_integrate, dx=dt, axis=0, even="last")  # res_int[n - 1, :] + 0.5 * dt * (to_integrate[n - 1, :] + to_integrate[n, :])
        # else:
        #     for n in range(self.trunc_ind):
        #         to_integrate = np.einsum("ijk,ik->ij", self.dotbkbkcorrw[: n + 1, :, :][::-1, :, :], self.kernel[: n + 1, :])
        #         res_int[n, :] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
        #     # res_int[n, :] = -1 * simpson(to_integrate, dx=dt, axis=0, even="last")
        #     res_int += np.einsum("jk,ik->ij", self.bkbkcorrw[0, :, :], self.kernel)
        return time - time[0], res_int

    def compute_noise(self, xva, trunc_kernel=None, start_point=0, end_point=None):
        """
        From a trajectory get the noise.

        Parameters
        ----------
        xva : xarray dataset (['time', 'x', 'v', 'a']) .
            Use compute_va() or see its output for format details.
            Input trajectory to compute noise.
        trunc_kernel : int
                Number of datapoint of the kernel to consider.
                Can be used to remove unphysical divergence of the kernel or shortten execution time.
        """
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        if trunc_kernel is None:
            trunc_kernel = self.trunc_ind
        time_0 = xva["time"].data[0]
        xva = xva.isel(time=slice(start_point, end_point))
        time = xva["time"].data - time_0
        dt = time[1] - time[0]
        E_force, E, _ = self.basis_vector(xva)
        if self.rank_projection:
            E = np.einsum("kj,ij->ik", self.P_range, E)
        force = np.matmul(E_force, self.force_coeff)
        if self.method in ["rect", "rectangular", "second_kind_rect"] or self.method is None:
            memory = memory_rect(self.kernel[:trunc_kernel], E, dt)
        elif self.method == "trapz" or self.method == "second_kind_trapz":
            memory = memory_trapz(self.kernel[:trunc_kernel], E, dt)
        else:
            raise ValueError("Cannot compute noise when kernel computed with method {}".format(self.method))
        return time, xva[self.L_obs].data - force - memory, xva[self.L_obs].data, force, memory

    def compute_corrs_w_noise(self, xva, left_op=None):
        """
        Compute correlation between noise and left_op

        Parameters
        ----------
        xva : xarray dataset (['time', 'x', 'v', 'a']) .
            Use compute_va() or see its output for format details.
            Input trajectory to compute noise.
        """
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")

        dt = xva["time"].data[1] - xva["time"].data[0]
        E_force, E, _ = self.basis_vector(xva)
        if self.rank_projection:
            E = np.einsum("kj,ij->ik", self.P_range, E)

        noise = xva[self.L_obs].data - np.matmul(E_force, self.force_coeff)

        if left_op is None:
            left_op = noise
        elif isinstance(left_op, str):
            left_op = xva[left_op]

        if self.method in ["rect", "rectangular", "second_kind_rect"] or self.method is None:
            return self.time, corrs_rect(noise, self.kernel, E, left_op, dt)
        elif self.method == "trapz" or self.method == "second_kind_trapz":
            return self.time[:-1], corrs_trapz(noise, self.kernel, E, left_op, dt)
        else:
            raise ValueError("Cannot compute noise when kernel computed with method {}".format(self.method))

    def force_eval(self, x, coeffs=None):
        """
        Evaluate the force at given points x.
        If coeffs is given, use provided coefficients instead of the force
        """
        if coeffs is None:
            if self.force_coeff is None:
                raise Exception("Mean force has not been computed.")
            coeffs = self.force_coeff
        else:  # Check shape
            if coeffs.shape != (self.N_basis_elt_force, self.dim_obs):
                raise Exception("Wrong shape of the coefficients. Get {} but expect {}.".format(coeffs.shape, (self.N_basis_elt_force, self.dim_obs)))
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="force")
        return np.einsum("ik,kl->il", E, coeffs)  # Return the force as array (nb of evalution point x dim_obs)

    def pmf_eval(self, x, coeffs=None, kT=1.0):
        """
        Compute free energy via integration of the mean force at points x.
        This assume that the effective mass is independent of the position.
        If coeffs is given, use provided coefficients instead of the force coefficients.
        """
        if self.dim_obs > 1:
            print("Warning: Computation of the free energy for dimensions higher than 1 is likely to be incorrect.")
        if coeffs is None:
            if self.force_coeff is None:
                raise Exception("Mean force has not been computed.")
            coeffs = self.force_coeff
        else:  # Check shape
            if coeffs.shape != (self.N_basis_elt_force, self.dim_obs):
                raise Exception("Wrong shape of the coefficients. Get {} but expect {}.".format(coeffs.shape, (self.N_basis_elt_force, self.dim_obs)))
        if self.mass is None:
            self.compute_pos_effective_mass(kT=kT)
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="pmf")
        pmf = -1 * np.einsum("ik,kl->il", E, np.matmul(coeffs, self.mass)) / kT
        return pmf - np.min(pmf)

    def kernel_eval(self, x, coeffs_ker=None):
        """
        Evaluate the kernel at given points x.
        If coeffs_ker is given, use provided coefficients instead of the kernel
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        if coeffs_ker is None:
            coeffs_ker = self.kernel
        else:  # Check shape
            if coeffs_ker.shape != self.kernel.shape:
                raise Exception("Wrong shape of the coefficients. Get {} but expect {}.".format(coeffs_ker.shape, self.kernel.shape))

        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="kernel")
        if self.rank_projection:
            E = np.einsum("kj,ijd->ikd", self.P_range, E)
        return self.time, np.einsum("jkd,ikl->ijld", E, coeffs_ker)  # Return the kernel as array (time x nb of evalution point x dim_obs x dim_x)
