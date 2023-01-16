import numpy as np
import xarray as xr
from scipy.integrate import trapezoid

from .fkernel import kernel_first_kind_trapz, kernel_first_kind_rect, kernel_first_kind_midpoint, kernel_second_kind_rect, kernel_second_kind_trapz


class Estimator_gle(object):
    """
    The main class for the position dependent memory extraction,
    holding all data and the extracted memory kernels.
    """

    def __init__(self, trajs_data, model_class, basis, trunc=1.0, L_obs=None, saveall=True, prefix="", verbose=True, **kwargs):
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
        self.corrsfile = "corrs.txt"
        self.corrsdxfile = "a-u_corrs.txt"
        self.dcorrsfile = "dE_corrs.txt"
        self.dcorrsdxfile = "dE_a-u_corrs.txt"
        self.kernelfile = "kernel.txt"

        self.trajs_data = trajs_data

        if L_obs is None:  # Default value given by the class of the model
            L_obs = model_class.set_of_obs[-1]

        dim_x = self.trajs_data.xva_list[0].dims["dim_x"]
        dim_obs = self.trajs_data.xva_list[0][L_obs].shape[1]
        self.dt = self.trajs_data.xva_list[0].attrs["dt"]

        trunc_ind = self.trajs_data._compute_trunc_ind(trunc)

        # Fit basis from some description of the data
        describe_data = self.trajs_data.describe_data()

        self.model = model_class(basis, self.dt, dim_x=dim_x, dim_obs=dim_obs, trunc_ind=trunc_ind, L_obs=L_obs, describe_data=describe_data)

        self.trajs_data._do_check_obs(model_class.set_of_obs, self.model.L_obs)  # Check that we have in the trajectories what we need

    def compute_gram(self, kT=1.0):
        if self.verbose:
            print("Calculate gram...")
            print("Use kT:", kT)
        avg_gram = self.trajs_data.loop_over_trajs(self.trajs_data._compute_gram, self.model, gram_type="force")[0]
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
        self.kernel_gram = self.trajs_data.loop_over_trajs(self.trajs_data._compute_gram, self.model, gram_type="kernel")[0]
        if self.rank_projection:
            self.kernel_gram = np.einsum("lj,jk,mk->lm", self.model.P_range, self.kernel_gram, self.model.P_range)
        return self.kernel_gram

    # TODO:  changer le calcul de la masse effective (dépendante en position ou non) pour utiliser loop_over_traj

    def compute_effective_mass(self):
        """
        Return average effective mass computed from equipartition with the velocity.
        """
        if self.verbose:
            print("Calculate effective mass...")
        v2sum = 0.0
        for i, xva in enumerate(self.xva_list):
            v2sum += np.einsum("ik,ij->kj", xva["v"], xva["v"])
        v2 = v2sum / self.weightsum
        self.model.eff_mass = np.linalg.inv(v2)

        if self.verbose:
            print("Found effective mass:", self.eff_mass)
        return self.eff_mass

    def compute_pos_effective_mass(self):
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
        self.inv_mass_coeff = np.dot(np.linalg.inv(avg_gram), pos_inv_mass)
        return self.inv_mass_coeff

    def compute_mean_force(self):
        """
        Computes the mean force from the trajectories.
        """
        if self.verbose:
            print("Calculate mean force...")
        avg_disp, avg_gram = self.trajs_data.loop_over_trajs(self.trajs_data._projection_on_basis, self.model)
        # print(avg_gram)
        self.model.force_coeff = np.matmul(np.linalg.inv(avg_gram.data), avg_disp.data)  # TODO
        return self.model

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
        if self.model.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        self.bkdxcorrw, self.dotbkdxcorrw, self.bkbkcorrw, self.dotbkbkcorrw = self.trajs_data.loop_over_trajs(self.trajs_data._correlation_all, self.model)

        if self.model.rank_projection:
            self.model._set_range_projection(rank_tol, self.bkbkcorrw[0, :, :])
            tempbkbkcorrw = np.einsum("ijk,mk->ijm", self.bkbkcorrw, self.model.P_range)
            self.bkbkcorrw = np.einsum("lj,ijk->ilk", self.model.P_range, tempbkbkcorrw)
            # self.bkbkcorrw = np.einsum("lj,ijk,mk->ilm", self.model.P_range, self.bkbkcorrw, self.model.P_range)
            self.bkdxcorrw = np.einsum("kj,ijd->ikd", self.model.P_range, self.bkdxcorrw)
            if self.dotbkdxcorrw is not None:
                self.dotbkdxcorrw = np.einsum("kj,ijd->ikd", self.model.P_range, self.dotbkdxcorrw)
            if isinstance(self.dotbkbkcorrw, xr.DataArray):
                self.dotbkbkcorrw = np.einsum("lj,ijk,mk->ilm", self.model.P_range, self.dotbkbkcorrw, self.model.P_range)

        if self.saveall:
            np.savetxt(self.prefix + self.corrsdxfile, self.bkdxcorrw.reshape(self.bkdxcorrw.shape[0], -1))
            np.savetxt(self.prefix + self.corrsfile, self.bkbkcorrw.reshape(self.bkbkcorrw.shape[0], -1))
            np.savetxt(self.prefix + self.dcorrsdxfile, self.dotbkdxcorrw.reshape(self.dotbkdxcorrw.shape[0], -1))
            np.savetxt(self.prefix + self.dcorrsfile, self.dotbkbkcorrw.reshape(self.dotbkbkcorrw.shape[0], -1))
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
        print("Compute memory kernel using {} method".format(method))
        time = np.arange(self.model.trunc_ind) * self.dt
        self.model.time_kernel = (time - time[0]).reshape(-1, 1)  # Set zero time
        self.model.method = method  # Save used method
        if self.verbose:
            print("Use dt:", self.dt)
        if k0 is None and method in ["trapz", "second_kind_rect", "second_kind_trapz"]:  # Then we should compute initial value from time derivative at zero
            if self.dotbkdxcorrw is None:
                raise Exception("Need correlation with derivative functions to compute the kernel using this method or provide initial value.")
            k0 = np.matmul(np.linalg.inv(self.bkbkcorrw.isel(time_trunc=0)), self.dotbkdxcorrw.isel(time_trunc=0).to_numpy())
            if self.verbose:
                print("K0", k0)
                # print("Gram", self.bkbkcorrw[0, :, :])
                # print("Gram eigs", np.linalg.eigvals(self.bkbkcorrw[0, :, :]))
        if method in ["rect", "rectangular"]:
            kernel = kernel_first_kind_rect(self.bkbkcorrw.to_numpy(), self.bkdxcorrw.to_numpy(), self.dt)
        elif method == "midpoint":  # Deal with not even data lenght
            kernel = kernel_first_kind_midpoint(self.bkbkcorrw.to_numpy(), self.bkdxcorrw.to_numpy(), self.dt)
            self.model.time_kernel = self.model.time_kernel[:-1:2, :]
        elif method == "midpoint_w_richardson":
            ker = kernel_first_kind_midpoint(self.bkbkcorrw.to_numpy(), self.bkdxcorrw.to_numpy(), self.dt)
            ker_3 = kernel_first_kind_midpoint(self.bkbkcorrw.to_numpy()[:, :, ::3], self.bkdxcorrw.to_numpy()[:, :, ::3], 3 * self.dt)
            kernel = (9 * ker[::3][: ker_3.shape[0]] - ker_3) / 8
            self.model.time_kernel = self.model.time_kernel[:-3:6, :]
        elif method == "trapz":
            ker = kernel_first_kind_trapz(k0, self.bkbkcorrw.to_numpy(), self.bkdxcorrw.to_numpy(), self.dt)
            kernel = 0.5 * (ker[1:-1, :, :] + 0.5 * (ker[:-2, :, :] + ker[2:, :, :]))  # Smoothing
            kernel = np.insert(kernel, 0, k0, axis=0)
            self.model.time_kernel = self.model.time_kernel[:-1, :]
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

        if self.saveall:
            np.savetxt(self.prefix + self.kernelfile, np.hstack((self.model.time_kernel, kernel.reshape(kernel.shape[0], -1))))
        self.model.kernel = kernel
        return self.model

    def check_volterra_inversion(self):
        """
        For checking if the volterra equation is correctly inversed
        Compute the integral in volterra equation using trapezoidal rule
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        dt = self.dt
        time = np.arange(self.bkdxcorrw.shape[0]) * dt
        res_int = np.zeros(self.bkdxcorrw.shape)
        # res_int[0, :] = 0.5 * dt * to_integrate[0, :]
        # if method == "trapz":
        for n in range(self.model.trunc_ind):
            to_integrate = np.einsum("ijk,ikl->ijl", self.bkbkcorrw[: n + 1, :, :][::-1, :, :], self.model.kernel[: n + 1, :, :])
            res_int[n, :] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
            # res_int[n, :] = -1 * simpson(to_integrate, dx=dt, axis=0, even="last")  # res_int[n - 1, :] + 0.5 * dt * (to_integrate[n - 1, :] + to_integrate[n, :])
        # else:
        #     for n in range(self.model.trunc_ind):
        #         to_integrate = np.einsum("ijk,ik->ij", self.dotbkbkcorrw[: n + 1, :, :][::-1, :, :], self.kernel[: n + 1, :])
        #         res_int[n, :] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
        #     # res_int[n, :] = -1 * simpson(to_integrate, dx=dt, axis=0, even="last")
        #     res_int += np.einsum("jk,ik->ij", self.bkbkcorrw[0, :, :], self.kernel)
        return time, res_int
