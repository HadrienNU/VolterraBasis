import numpy as np
import xarray as xr


from .fkernel import kernel_first_kind_trapz, kernel_first_kind_rect, kernel_first_kind_midpoint, kernel_second_kind_rect, kernel_second_kind_trapz
from .fkernel import memory_rect, memory_trapz, corrs_rect, corrs_trapz
from .correlation import correlation_1D, correlation_ND
from .trajectories_handler import Trajectories_handler


class Gle_fitter(object):
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

        self.method = None

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
        # Il faut transformer la liste pour avoir à la place un seul xarray avec toutes les données
        # Et on utilise dask si on veut charger les données petits à petit?
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

    def compute_gram(self, kT=1.0):
        if self.verbose:
            print("Calculate gram...")
            print("Use kT:", kT)
        avg_gram = self.trajs.loop_over_traj(self.trajs.compute_gram, self.model, gram_type="force")[0]
        self.invgram = kT * np.linalg.pinv(avg_gram)
        if self.verbose:
            print("Found inverse gram:", self.invgram)
        return kT * avg_gram

    def compute_kernel_gram(self):
        """
        Return gram matrix of the kernel part of the basis.
        """
        if self.verbose:
            print("Calculate kernel gram...")
        self.kernel_gram = self.trajs.loop_over_traj(self.trajs.compute_gram, self.model, gram_type="kernel")[0]
        if self.rank_projection:
            self.kernel_gram = np.einsum("lj,jk,mk->lm", self.P_range, self.kernel_gram, self.P_range)
        return self.kernel_gram

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
        self.eff_mass = kT * np.linalg.inv(v2)

        if self.verbose:
            print("Found effective mass:", self.eff_mass)
        return self.eff_mass

    def compute_pos_effective_mass(self, kT=1.0):
        """
        Return position-dependent effective inverse mass
        """
        if self.verbose:
            print("Calculate kernel gram...")
        pos_inv_mass = np.zeros((self.dim_x, self.N_basis_elt_force, self.dim_x))
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            E = self.basis_vector(xva, compute_for="force")
            pos_inv_mass += np.einsum("ik,ij,il->klj", xva["v"], xva["v"], E) / self.weightsum
            avg_gram += np.matmul(E.T, E) / self.weightsum
        self.inv_mass_coeff = kT * np.dot(np.linalg.inv(avg_gram), pos_inv_mass)
        return self.inv_mass_coeff

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
            avg_disp += xr.dot(E, xva[self.L_obs]) / self.weightsum
            avg_gram += xr.dot(E, E.rename({"dim_basis": "dim_basis'"})) / self.weightsum
        self.force_coeff = np.matmul(np.linalg.inv(avg_gram.data), avg_disp.data)

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
        self.dt = dt
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

    def compute_force_kernel_corrs(self, large=False):
        """
        Compute correlation functions between the force and the kernel.

        Parameters
        ----------
        large : bool, default=False
            When large is true, it use a slower way to compute correlation that is less demanding in memory
        """
        if self.verbose:
            print("Calculate force_kernel correlation functions...")
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        res_corrs = np.zeros((self.trunc_ind, self.dim_obs, self.N_basis_elt_kernel))

        for weight, xva in zip(self.weights, self.xva_list):
            E_force, E, _ = self.basis_vector(xva)
            if self.rank_projection:
                E = np.einsum("kj,ij->ik", self.P_range, E)
            force = np.matmul(E_force, self.force_coeff)
            if not large:
                try:
                    res_corrs += weight * correlation_ND(force, E, trunc=self.trunc_ind)
                except MemoryError:  # If too big slow way
                    if self.verbose:
                        print("Too many basis function, compute correlations one by one (slow)")
                    for n in range(E.shape[1]):
                        for d in range(self.dim_obs):
                            res_corrs[:, d, n] += weight * correlation_1D(force[:, d], E[:, n], trunc=self.trunc_ind)
            else:
                for n in range(E.shape[1]):
                    for d in range(self.dim_obs):
                        res_corrs[:, d, n] += weight * correlation_1D(force[:, d], E[:, n], trunc=self.trunc_ind)

        res_corrs /= self.weightsum
        return res_corrs
