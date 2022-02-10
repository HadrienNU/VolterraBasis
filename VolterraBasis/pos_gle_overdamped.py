import numpy as np
import xarray as xr
from scipy.integrate import trapezoid

from .correlation import correlation1D, correlation_ND

from .pos_gle import Pos_gle_base


class Pos_gle_overdamped(Pos_gle_base):
    """
    The main class for the position dependent memory extraction,
    holding all data and the extracted memory kernels.
    """

    def __init__(self, xva_arg, basis, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0):
        Pos_gle_base.__init__(self, xva_arg, basis, saveall, prefix, verbose, kT, trunc)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")

    def _do_check(self, xva_arg):
        if xva_arg is not None:
            if isinstance(xva_arg, xr.Dataset):
                self.xva_list = [xva_arg]
            else:
                self.xva_list = xva_arg
            for xva in self.xva_list:  # TODO
                for col in ["x", "v"]:
                    if col not in xva.data_vars:
                        raise Exception("Please provide txva data frame, " "or an iterable collection (i.e. list) " "of txva data frames.")
                if "time" not in xva.dims:
                    raise Exception("Time is not a coordinate. Please provide txva dataset, " "or an iterable collection (i.e. list) " "of txva dataset.")
                if "dt" not in xva.attrs:
                    raise Exception("Timestep not in dataset attrs")
        else:
            self.xva_list = None

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        # We have to deal with the multidimensionnal case as well
        E = self.basis.basis(xva["x"].data)
        if compute_for == "force":
            return E
        elif compute_for == "kernel":
            # Extend the basis for multidim value
            return E.reshape(-1, self.N_basis_elt_kernel, 1)
        elif compute_for == "corrs":
            dbk = self.basis.deriv(xva["x"].data)
            dE = np.einsum("nld,nd->nl", dbk, xva["v"].data)
            return E, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")

    def compute_mean_force(self):
        """
        Computes the mean force from the trajectory.
        """
        if self.verbose:
            print("Calculate mean force...")
        avg_disp = np.zeros((self.N_basis_elt_force, self.dim_x))
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            E = self.basis_vector(xva, compute_for="force")
            avg_disp += np.matmul(E.T, xva["v"].data) / self.weightsum
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
        self.bkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.dim_x))

        self.dotbkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.dim_x))  # Needed for initial value anyway
        self.dotbkbkcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.N_basis_elt_kernel))

        for weight, xva in zip(self.weights, self.xva_list):
            E_force, E, dE = self.basis_vector(xva)
            force = np.matmul(E_force, self.force_coeff)
            if not large:
                try:
                    self.bkdxcorrw += weight * correlation_ND(E, (xva["v"].data - force), trunc=self.trunc_ind)
                    self.dotbkdxcorrw += weight * correlation_ND(dE, (xva["v"].data - force), trunc=self.trunc_ind)
                    self.bkbkcorrw += weight * correlation_ND(E, trunc=self.trunc_ind)
                    self.dotbkbkcorrw += weight * correlation_ND(dE, trunc=self.trunc_ind)
                except MemoryError:  # If too big slow way
                    if self.verbose:
                        print("Too many basis function, compute correlations one by one (slow)")
                    for n in range(E.shape[1]):
                        for d in range(self.dim_x):
                            self.bkdxcorrw[:, n, d] += weight * correlation1D(E[:, n], xva["v"].data[:, d] - force[:, d], trunc=self.trunc_ind)  # Correlate derivative of observable minus mean value
                            self.dotbkdxcorrw[:, n, d] += weight * correlation1D(dE[:, n], (xva["v"].data[:, d] - force[:, d]), trunc=self.trunc_ind)
                        for m in range(E.shape[1]):
                            self.bkbkcorrw[:, n, m] += weight * correlation1D(E[:, n], E[:, m], trunc=self.trunc_ind)
                            self.dotbkbkcorrw[:, n, m] += weight * correlation1D(E[:, n], dE[:, m], trunc=self.trunc_ind)
            else:
                for n in range(E.shape[1]):
                    for d in range(self.dim_x):
                        self.bkdxcorrw[:, n, d] += weight * correlation1D(E[:, n], xva["v"].data[:, d] - force[:, d], trunc=self.trunc_ind)  # Correlate derivative of observable minus mean value
                        self.dotbkdxcorrw[:, n, d] += weight * correlation1D(dE[:, n], (xva["v"].data[:, d] - force[:, d]), trunc=self.trunc_ind)
                    for m in range(E.shape[1]):
                        self.bkbkcorrw[:, n, m] += weight * correlation1D(E[:, n], E[:, m], trunc=self.trunc_ind)
                        self.dotbkbkcorrw[:, n, m] += weight * correlation1D(E[:, n], dE[:, m], trunc=self.trunc_ind)

        self.bkbkcorrw /= self.weightsum
        self.bkdxcorrw /= self.weightsum
        self.dotbkdxcorrw /= self.weightsum
        self.dotbkbkcorrw /= self.weightsum

        if self.rank_projection:
            self._set_range_projection(rank_tol)

        if self.saveall:
            np.savetxt(self.prefix + self.corrsdxfile, self.bkdxcorrw)
            np.savetxt(self.prefix + self.corrsfile, self.bkbkcorrw.reshape(-1, self.N_basis_elt ** 2))

    def compute_noise(self, xva, trunc_kernel=None):
        """
        From a trajectory get the noise
        """
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        if trunc_kernel is None:
            trunc_kernel = self.trunc_ind
        time = xva["time"].data
        dt = time[1] - time[0]
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
            to_integrate = np.einsum("ik,ikl->il", E[: n + 1, :][::-1, :], self.kernel[: n + 1, :, :])
            memory[n] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
            # memory[n] = -1 * to_integrate.sum() * dt
        # Only select self.trunc_ind points for the integration
        for n in range(trunc_kernel, memory.shape[0]):
            to_integrate = np.einsum("ik,ikl->il", E[n - trunc_kernel + 1 : n + 1, :][::-1, :], self.kernel[:trunc_kernel, :, :])
            memory[n] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
            # memory[n] = -1 * to_integrate.sum() * dt
        return time, xva["v"].data - force - memory, xva["v"].data, force, memory
