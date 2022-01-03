import numpy as np
import pandas as pd

from .correlation import correlation1D, correlation_ND

from .pos_gle import Pos_gle_base


class Pos_gle_overdamped(Pos_gle_base):
    """
    The main class for the position dependent memory extraction,
    holding all data and the extracted memory kernels.
    """

    def __init__(self, xva_arg, basis, N_basis_elt, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0, with_const=False):
        Pos_gle_base.__init__(self, xva_arg, basis, N_basis_elt, saveall, prefix, verbose, kT, trunc)
        self.N_basis_elt = N_basis_elt
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt

    def _do_check(self, xva_arg):
        if xva_arg is not None:
            if isinstance(xva_arg, pd.DataFrame):
                self.xva_list = [xva_arg]
            else:
                self.xva_list = xva_arg
            for xva in self.xva_list:
                for col in ["t", "x", "v"]:
                    if col not in xva.columns:
                        raise Exception("Please provide txva data frame, " "or an iterable collection (i.e. list) " "of txva data frames.")
        else:
            self.xva_list = None

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
            avg_disp += np.matmul(xva["v"].values, E).T / self.weightsum
            avg_gram += np.matmul(E.T, E) / self.weightsum
        self.force_coeff = np.matmul(np.linalg.inv(avg_gram), avg_disp)

    def compute_corrs(self):
        """
        Compute correlation functions.
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
            try:
                self.bkdxcorrw += weight * correlation_ND(E, (xva["v"].values - force).reshape(-1, 1), trunc=self.trunc_ind)[..., 0]
                self.dotbkdxcorrw += weight * correlation_ND(dE, (xva["v"].values - force).reshape(-1, 1), trunc=self.trunc_ind)[..., 0]
                self.bkbkcorrw += weight * correlation_ND(E, trunc=self.trunc_ind)
                self.dotbkbkcorrw += weight * correlation_ND(dE, trunc=self.trunc_ind)
            except MemoryError:  # If too big slow way
                if self.verbose:
                    print("Too many basis function, compute correlations one by one (slow)")
                for n in range(E.shape[1]):
                    self.bkdxcorrw[:, n] += weight * correlation1D(E[:, n], xva["v"].values - force, trunc=self.trunc_ind)  # Correlate derivative of observable minus mean value
                    self.dotbkdxcorrw[:, n] += weight * correlation1D(dE[:, n], xva["v"].values - force, trunc=self.trunc_ind)
                    for m in range(E.shape[1]):
                        self.bkbkcorrw[:, n, m] += weight * correlation1D(E[:, n], E[:, m], trunc=self.trunc_ind)
                        self.dotbkbkcorrw[:, n, m] += weight * correlation1D(E[:, n], dE[:, m], trunc=self.trunc_ind)

        self.bkbkcorrw /= self.weightsum
        self.bkdxcorrw /= self.weightsum
        self.dotbkdxcorrw /= self.weightsum
        self.dotbkbkcorrw /= self.weightsum

        if self.saveall:
            np.savetxt(self.prefix + self.corrsdxfile, self.bkdxcorrw)
            np.savetxt(self.prefix + self.corrsfile, self.bkbkcorrw.reshape(-1, self.N_basis_elt ** 2))
