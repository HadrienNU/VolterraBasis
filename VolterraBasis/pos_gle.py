import numpy as np
import pandas as pd
from scipy.integrate import simpson, trapezoid

from .fkernel import kernel_first_kind_trapz, kernel_second_kind, kernel_first_kind_euler, kernel_first_kind_midpoint
from .correlation import correlation1D, correlation_ND


class Pos_gle_base(object):
    """
    The main class for the position dependent memory extraction,
    holding all data and the extracted memory kernels.
    """

    def __init__(self, xva_arg, basis, N_basis_elt, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0, **kwargs):
        """
        Create an instance of the Pos_gle class.

        Parameters
        ----------
        xva_arg : pandas dataframe ()['t', 'x', 'v', 'a']) or list of dataframes.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a pandas timeseries
            or a listlike collection of them. Set xva_arg=None for load mode.
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
        with_const : bool, default=False
            Whatever the constant function is in the basis (to avoid inversion issue with the derivative).
        """
        self._do_check(xva_arg)  # Do some check on the trajectories
        # print(callable(getattr(basis, "basis", None)), callable(getattr(basis, "deriv", None)))
        if not (callable(getattr(basis, "basis", None))) or not (callable(getattr(basis, "deriv", None))):
            raise Exception("Basis class do not define basis() or deriv() method")
        self.basis = basis
        if callable(getattr(self.basis, "fit", None)):
            self.basis = self.basis.fit(np.concatenate([xva["x"].values.reshape(-1, 1) for xva in self.xva_list], axis=0))  # Fit basis

        self.N_basis_elt = N_basis_elt
        # self.N_zeros_kernel = self.N_basis_elt - self.N_basis_elt_force + int(with_const)
        self.saveall = saveall
        self.prefix = prefix
        self.verbose = verbose
        self.kT = kT
        # filenames
        self.corrsfile = "corrs.txt"
        self.corrsdxfile = "a-u_corrs.txt"
        self.dcorrsfile = "dE_corrs.txt"
        self.dcorrsdxfile = "dE_a-u_corrs.txt"
        self.interpfefile = "interp-fe.txt"
        self.histfile = "fe-hist.txt"
        self.kernelfile = "kernel.txt"

        self.bkbkcorrw = None
        self.bkdxcorrw = None
        self.force_coeff = None

        if self.xva_list is None:
            return

        # processing input arguments
        self.weights = np.array([xva.shape[0] for xva in self.xva_list], dtype=int)  # Should be the various lenght of trajectory
        self.weightsum = np.sum(self.weights)
        self.min_x = np.min([np.min(xva["x"]) for xva in self.xva_list])
        self.max_x = np.max([np.max(xva["x"]) for xva in self.xva_list])
        if self.verbose:
            print("Found trajectories with the following lengths:")
            print(self.weights)

        lastinds = np.array([xva.index[-1] for xva in self.xva_list])
        smallest = np.min(lastinds)
        if smallest < trunc:
            if self.verbose:
                print("Warning: Found a trajectory shorter than " "the argument trunc. Override.")
            trunc = smallest
        # Find index of the time truncation
        self.trunc_ind = np.nonzero(self.xva_list[0].index <= trunc)[0][-1]
        if self.verbose:
            print("Trajectories are truncated at lenght {} for dynamic analysis".format(self.trunc_ind))

    def _do_check(self, xva_arg):
        if xva_arg is not None:
            if isinstance(xva_arg, pd.DataFrame):
                self.xva_list = [xva_arg]
            else:
                self.xva_list = xva_arg
            for xva in self.xva_list:
                for col in ["t", "x", "v", "a"]:
                    if col not in xva.columns:
                        raise Exception("Please provide txva data frame, " "or an iterable collection (i.e. list) " "of txva data frames.")
        else:
            self.xva_list = None

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        To be implemented by the infant classes
        """
        raise NotImplementedError

    def compute_gram(self):
        if self.verbose:
            print("Calculate gram...")
            print("Use kT:", self.kT)
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            E = self.basis_vector(xva, compute_for="force")
            avg_gram += np.matmul(E.T, E) / self.weightsum
        self.invgram = self.kT * np.linalg.pinv(avg_gram)

        if self.verbose:
            print("Found inverse gram:", self.invgram)
        return self.kT * avg_gram

    def compute_effective_masse(self):
        if self.verbose:
            print("Calculate effective mass...")
            print("Use kT:", self.kT)
        v2sum = 0.0
        for i, xva in enumerate(self.xva_list):
            v2sum += (xva["v"] ** 2).mean() * self.weights[i]
        v2 = v2sum / self.weightsum
        self.mass = self.kT / v2

        if self.verbose:
            print("Found effective mass:", self.mass)
        return self.mass

    def compute_fe(self, bins="auto", fehist=None, _dont_save_hist=False, average="1d"):
        """
        Computes the free energy from the trajectoy and prepares the cubic spline
        interpolation. You can alternatively provide an histogram.

        Parameters
        ----------

        bins : str, or int, default="auto"
            The number of bins. It is passed to the numpy.histogram routine,
            see its documentation for details.
        fehist : list, default=None
            Provide a (precomputed) histogram in the format as returned by
            numpy.histogram.
        _dont_save_hist : bool, default=False
            Do not save the histogram.
        average: str, default="1d"
            Should be "1d" or "2d", specify the way to do the average for computation of the force
        """
        if self.verbose:
            print("Calculate histogram...")
        if fehist is None:
            fehist = np.histogram(np.concatenate([xva["x"].values for xva in self.xva_list]), bins=bins)

        if self.verbose:
            print("Number of bins:", len(fehist[1]) - 1)

        xfa = (fehist[1][1:] + fehist[1][:-1]) / 2.0
        pf = fehist[0]
        xf = xfa[np.nonzero(pf)]
        fe = -np.log(pf[np.nonzero(pf)]) * self.kT
        # il faudrait plutôt fitter fe directement
        E = self.basis_vector(pd.DataFrame({"x": xf.ravel()}), compute_for="force")
        avg_disp = -1 * np.matmul(np.gradient(fe, xf[1] - xf[0]), E).T  # Ca ne va pas être correct ça
        self.force_coeff = np.matmul(self.invgram, avg_disp)
        # print(self.force_coeff)
        if self.saveall:
            np.savetxt(self.prefix + self.interpfefile, np.vstack((xf, self.dU(xf))).T)
            if not _dont_save_hist:
                np.savetxt(self.prefix + self.histfile, np.vstack((xfa, pf)).T)

    def compute_mean_force(self):
        """
        Computes the mean force from the trajectory.
        """
        if self.verbose:
            print("Calculate mean force...")
        avg_disp = np.zeros((self.N_basis_elt_force))
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            E = self.basis_vector(xva, compute_for="force")
            avg_disp += np.matmul(xva["a"].values, E).T / self.weightsum
            avg_gram += np.matmul(E.T, E) / self.weightsum
        self.force_coeff = np.matmul(np.linalg.inv(avg_gram), avg_disp)

    def compute_corrs(self, large=False):
        """
        Compute correlation functions. When large is true, it use a slower way to compute correlation that is less demanding in memory
        """
        if self.verbose:
            print("Calculate correlation functions...")
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")

        self.bkbkcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.N_basis_elt_kernel))
        self.bkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel))

        self.dotbkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel))  # Needed for initial value anyway
        self.dotbkbkcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.N_basis_elt_kernel))

        for weight, xva in zip(self.weights, self.xva_list):
            E_force, E, dE = self.basis_vector(xva)
            force = np.matmul(E_force, self.force_coeff)
            if not large:
                try:
                    self.bkdxcorrw += weight * correlation_ND(E, (xva["a"].values - force).reshape(-1, 1), trunc=self.trunc_ind)[..., 0]
                    self.dotbkdxcorrw += weight * correlation_ND(dE, (xva["a"].values - force).reshape(-1, 1), trunc=self.trunc_ind)[..., 0]
                    self.bkbkcorrw += weight * correlation_ND(E, trunc=self.trunc_ind)
                    self.dotbkbkcorrw += weight * correlation_ND(dE, E, trunc=self.trunc_ind)
                except MemoryError:  # If too big slow way
                    if self.verbose:
                        print("Too many basis function, compute correlations one by one (slow)")
                    for n in range(E.shape[1]):
                        self.bkdxcorrw[:, n] += weight * correlation1D(E[:, n], xva["a"].values - force, trunc=self.trunc_ind)  # Correlate derivative of observable minus mean value
                        self.dotbkdxcorrw[:, n] += weight * correlation1D(dE[:, n], xva["a"].values - force, trunc=self.trunc_ind)
                        for m in range(E.shape[1]):
                            self.bkbkcorrw[:, n, m] += weight * correlation1D(E[:, n], E[:, m], trunc=self.trunc_ind)
                            self.dotbkbkcorrw[:, n, m] += weight * correlation1D(dE[:, n], E[:, m], trunc=self.trunc_ind)
            else:
                for n in range(E.shape[1]):
                    self.bkdxcorrw[:, n] += weight * correlation1D(E[:, n], xva["a"].values - force, trunc=self.trunc_ind)  # Correlate derivative of observable minus mean value
                    self.dotbkdxcorrw[:, n] += weight * correlation1D(dE[:, n], xva["a"].values - force, trunc=self.trunc_ind)
                    for m in range(E.shape[1]):
                        self.bkbkcorrw[:, n, m] += weight * correlation1D(E[:, n], E[:, m], trunc=self.trunc_ind)
                        self.dotbkbkcorrw[:, n, m] += weight * correlation1D(dE[:, n], E[:, m], trunc=self.trunc_ind)

        self.bkbkcorrw /= self.weightsum
        self.bkdxcorrw /= self.weightsum
        self.dotbkdxcorrw /= self.weightsum
        self.dotbkbkcorrw /= self.weightsum

        if self.saveall:
            np.savetxt(self.prefix + self.corrsdxfile, self.bkdxcorrw)
            np.savetxt(self.prefix + self.corrsfile, self.bkbkcorrw.reshape(-1, self.N_basis_elt_kernel ** 2))
            np.savetxt(self.prefix + self.dcorrsdxfile, self.dotbkdxcorrw)
            np.savetxt(self.prefix + self.dcorrsfile, self.dotbkbkcorrw.reshape(-1, self.N_basis_elt_kernel ** 2))

    def compute_kernel(self, method="rectangular", k0=None):
        """
        Computes the memory kernel.

        Parameters
        ----------
        method : {"rectangular", "midpoint", "midpoint_w_richardson","trapz","second_kind"}, default=rectangular
            Choose numerical method of inversion of the volterra equation
        k0 : float, default=0.
            If you give a nonzero value for k0, this is used at time zero for the trapz and second kind method, if set to None,
            the F-routine will calculate k0 from the second order memory equation.
        """
        if self.bkbkcorrw is None or self.bkdxcorrw is None:
            raise Exception("Need correlation functions to compute the kernel.")

        dt = self.xva_list[0].index[1] - self.xva_list[0].index[0]
        time = self.xva_list[0].index[: self.trunc_ind].to_numpy()
        self.time = (time - time[0]).reshape(-1, 1)  # Set zero time
        self.method = method  # Save used method
        if self.verbose:
            print("Use dt:", dt)
        if k0 is None and method in ["trapz", "second_kind"]:  # Then we should compute initial value from time derivative at zero
            k0 = np.matmul(np.linalg.inv(self.bkbkcorrw[0, :, :]), self.dotbkdxcorrw[0, :])
            if self.verbose:
                print("K0", k0)
                # print("Gram", self.bkbkcorrw[0, self.N_zeros_kernel :, self.N_zeros_kernel :])
                # print("Gram eigs", np.linalg.eigvals(self.bkbkcorrw[0, :, :]))
        if method == "rectangular":
            self.kernel = kernel_first_kind_euler(self.bkbkcorrw, self.bkdxcorrw, dt)
        elif method == "midpoint":  # Deal with not even data lenght
            self.kernel = kernel_first_kind_midpoint(self.bkbkcorrw, self.bkdxcorrw, dt)
            self.time = self.time[:-1:2, :]
        elif method == "midpoint_w_richardson":
            ker = kernel_first_kind_midpoint(self.bkbkcorrw, self.bkdxcorrw, dt)
            ker_3 = kernel_first_kind_midpoint(self.bkbkcorrw[::3, :, :], self.bkdxcorrw[::3, :], 3 * dt)
            self.kernel = (9 * ker[::3][: ker_3.shape[0]] - ker_3) / 8
            self.time = self.time[:-3:6, :]
        elif method == "trapz":
            ker = kernel_first_kind_trapz(k0, self.bkbkcorrw, self.bkdxcorrw, dt)
            self.kernel = 0.5 * (ker[1:-1, :] + 0.5 * (ker[:-2, :] + ker[2:, :]))  # Smoothing
            self.kernel = np.insert(self.kernel, 0, k0, axis=0)
            self.time = self.time[:-1, :]
        elif method == "second_kind":
            self.kernel = kernel_second_kind(k0, self.bkbkcorrw[0, :, :], self.dotbkbkcorrw, self.dotbkdxcorrw, dt)
        else:
            raise Exception("Method for volterra inversion is not in  {rectangular, midpoint, midpoint_w_richardson,trapz,second_kind}")

        if self.saveall:
            np.savetxt(self.prefix + self.kernelfile, np.hstack((self.time, self.kernel)))

        return self.kernel

    def check_volterra_inversion(self):
        """
        For checking if the volterra equation is correctly inversed
        Compute the integral in volterra equation using trapezoidal rule
        """
        dt = self.xva_list[0].index[1] - self.xva_list[0].index[0]
        time = self.xva_list[0].index[: self.trunc_ind].to_numpy()
        res_int = np.zeros(self.bkdxcorrw.shape)
        # res_int[0, :] = 0.5 * dt * to_integrate[0, :]
        # if method == "trapz":
        for n in range(self.trunc_ind):
            to_integrate = np.einsum("ijk,ik->ij", self.bkbkcorrw[: n + 1, :, :][::-1, :, :], self.kernel[: n + 1, :])
            res_int[n, :] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
            # res_int[n, :] = -1 * simpson(to_integrate, dx=dt, axis=0, even="last")  # res_int[n - 1, :] + 0.5 * dt * (to_integrate[n - 1, :] + to_integrate[n, :])
        # else:
        #     for n in range(self.trunc_ind):
        #         to_integrate = np.einsum("ijk,ik->ij", self.dotbkbkcorrw[: n + 1, :, :][::-1, :, :], self.kernel[: n + 1, :])
        #         res_int[n, :] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
        #     # res_int[n, :] = -1 * simpson(to_integrate, dx=dt, axis=0, even="last")
        #     res_int += np.einsum("jk,ik->ij", self.bkbkcorrw[0, :, :], self.kernel)
        return time - time[0], res_int

    def dU(self, x):
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        E = self.basis_vector(pd.DataFrame({"x": x.ravel()}), compute_for="force")
        return -1 * np.matmul(self.force_coeff, E.T)

    def kernel_eval(self, x):
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        E = self.basis_vector(pd.DataFrame({"x": np.asarray(x).ravel()}), compute_for="kernel")
        return self.time, np.matmul(self.kernel, E.T)  # , np.matmul(self.kernel[:, : self.N_basis_elt], self.basis.basis(x.reshape(-1, 1)).T)


class Pos_gle(Pos_gle_base):
    """
    The main class for the position dependent memory extraction,
    holding all data and the extracted memory kernels.
    """

    def __init__(self, xva_arg, basis, N_basis_elt, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0, with_const=False):
        """
        Create an instance of the Pos_gle class.

        Parameters
        ----------
        xva_arg : pandas dataframe ()['t', 'x', 'v', 'a']) or list of dataframes.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a pandas timeseries
            or a listlike collection of them. Set xva_arg=None for load mode.
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
        with_const : bool, default=False
            Whatever the constant function is in the basis (to avoid inversion issue with the derivative).
        """
        Pos_gle_base.__init__(self, xva_arg, basis, N_basis_elt, saveall, prefix, verbose, kT, trunc)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt - int(with_const)
        self.remove_const_ = bool(with_const)

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        # We have to deal with the multidimensionnal case as well
        bk = self.basis.basis(xva["x"].values.reshape(-1, 1))
        # if self.include_const:
        #     bk = np.concatenate((np.ones((bk.shape[0], 1)), bk), axis=1)
        if compute_for == "force":
            return bk
        dbk = self.basis.deriv(xva["x"].values.reshape(-1, 1), remove_const=self.remove_const_)
        if compute_for == "kernel":  # For kernel evaluation
            return dbk
        elif compute_for == "corrs":
            ddbk = self.basis.hessian(xva["x"].values.reshape(-1, 1), remove_const=self.remove_const_)
            E = np.multiply(dbk, xva["v"].values.reshape(-1, 1))
            dE = np.multiply(dbk, xva["a"].values.reshape(-1, 1)) + np.multiply(ddbk, np.power(xva["v"].values, 2).reshape(-1, 1))
            return bk, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")

    def compute_noise(self, xva, trunc_kernel=None):
        """
        From a trajectory get the noise
        """
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        if trunc_kernel is None:
            trunc_kernel = self.trunc_ind
        dt = xva.index[1] - xva.index[0]
        time = xva.index.to_numpy()
        E_force, E, _ = self.basis_vector(xva)
        force = np.matmul(E_force, self.force_coeff)
        memory = np.zeros(force.shape)
        if self.method == "trapz":
            trunc_kernel -= 1
        elif self.method in ["midpoint", "midpoint_w_richardson"]:
            raise ValueError("Cannot compute noise when kernel computed with method {}".format(self.method))
        # else:
        #     pass
        for n in range(trunc_kernel):
            to_integrate = np.einsum("ik,ik->i", E[: n + 1, :][::-1, :], self.kernel[: n + 1, :])
            memory[n] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
            # memory[n] = -1 * to_integrate.sum() * dt
        # Only select self.trunc_ind points for the integration
        for n in range(trunc_kernel, memory.shape[0]):
            to_integrate = np.einsum("ik,ik->i", E[n - trunc_kernel + 1 : n + 1, :][::-1, :], self.kernel[:trunc_kernel, :])
            memory[n] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
            # memory[n] = -1 * to_integrate.sum() * dt
        return time, xva["a"].values - force - memory, xva["a"].values, force, memory


class Pos_gle_with_friction(Pos_gle_base):
    """
    A derived class in which we don't enforce zero friction
    """

    def __init__(self, xva_arg, basis, N_basis_elt, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0, with_const=False):
        Pos_gle.__init__(self, xva_arg, basis, N_basis_elt, saveall, prefix, verbose, kT, trunc)
        self.N_basis_elt = N_basis_elt
        self.N_basis_elt_force = 2 * self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if with_const:
            print("with_const is True, please remove the constant function from the basis to avoid invertibility issue")

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        # We have to deal with the multidimensionnal case as well
        bk = self.basis.basis(xva["x"].values.reshape(-1, 1))
        if compute_for == "force_eval":
            return bk
        dbk = self.basis.deriv(xva["x"].values.reshape(-1, 1))
        if compute_for == "kernel":  # To have the term proportional to velocity
            return dbk
        Evel = np.multiply(dbk, xva["v"].values.reshape(-1, 1))
        E = np.concatenate((bk, Evel), axis=1)
        if compute_for == "force":
            return E
        elif compute_for == "corrs":
            ddbk = self.basis.hessian(xva["x"].values.reshape(-1, 1))
            dE = np.multiply(dbk, xva["a"].values.reshape(-1, 1)) + np.multiply(ddbk, np.power(xva["v"].values, 2).reshape(-1, 1))
            return E, Evel, dE
        else:
            raise ValueError("Basis evaluation goal not specified")

    def dU(self, x):
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        E = self.basis_vector(pd.DataFrame({"x": x.ravel()}), compute_for="force_eval")
        return -1 * np.matmul(self.force_coeff[: self.N_basis_elt], E.T)

    def friction_force(self, x):
        """
        Compute the term of friction, that should be zero
        """
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        E = self.basis_vector(pd.DataFrame({"x": x.ravel()}), compute_for="kernel")
        return np.matmul(self.force_coeff[self.N_basis_elt :], E.T)


class Pos_gle_no_vel_basis(Pos_gle_base):
    """
    A derived class in which we don't enforce the zero values
    """

    def __init__(self, xva_arg, basis, N_basis_elt, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0, with_const=False):
        Pos_gle_base.__init__(self, xva_arg, basis, N_basis_elt, saveall, prefix, verbose, kT, trunc)
        self.N_basis_elt = N_basis_elt
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        # We have to deal with the multidimensionnal case as well
        E = self.basis.basis(xva["x"].values.reshape(-1, 1))
        if compute_for == "force" or compute_for == "kernel":
            return E
        elif compute_for == "corrs":
            dbk = self.basis.deriv(xva["x"].values.reshape(-1, 1))
            dE = np.multiply(dbk, xva["v"].values.reshape(-1, 1))
            return E, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")
