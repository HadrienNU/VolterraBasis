import numpy as np

from .pos_gle_base import Pos_gle_base, _convert_input_array_for_evaluation


class Pos_gle(Pos_gle_base):
    """
    The main class for the position dependent memory extraction,
    holding all data and the extracted memory kernels.
    """

    def __init__(self, *args, **kwargs):
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
        """
        Pos_gle_base.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt - int(self.basis.const_removed) * self.dim_x
        self.rank_projection = not self.basis.const_removed

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        # We have to deal with the multidimensionnal case as well
        bk = self.basis.basis(xva["x"].data)
        # if self.include_const:
        #     bk = np.concatenate((np.ones((bk.shape[0], 1)), bk), axis=1)
        if compute_for == "force":
            return bk
        dbk = self.basis.deriv(xva["x"].data)
        if compute_for == "kernel":  # For kernel evaluation
            return dbk
        elif compute_for == "corrs":
            ddbk = self.basis.hessian(xva["x"].data)
            E = np.einsum("nld,nd->nl", dbk, xva["v"].data)
            dE = np.einsum("nld,nd->nl", dbk, xva["a"].data) + np.einsum("nlcd,nc,nd->nl", ddbk, xva["v"].data, xva["v"].data)
            return bk, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")


class Pos_gle_with_friction(Pos_gle_base):
    """
    A derived class in which we don't enforce zero instantaneous friction
    """

    def __init__(self, *args, **kwargs):
        Pos_gle.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = 2 * self.N_basis_elt - int(self.basis.const_removed) * self.dim_x
        self.N_basis_elt_kernel = self.N_basis_elt - int(self.basis.const_removed) * self.dim_x
        self.rank_projection = not self.basis.const_removed

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        # We have to deal with the multidimensionnal case as well
        bk = self.basis.basis(xva["x"].data)
        if compute_for == "force_eval":
            return bk
        dbk = self.basis.deriv(xva["x"].data)
        if compute_for == "kernel":  # To have the term proportional to velocity
            return dbk

        Evel = np.einsum("nld,nd->nl", dbk, xva["v"].data)
        E = np.concatenate((bk, Evel), axis=1)
        if compute_for == "force":
            return E
        elif compute_for == "corrs":
            ddbk = self.basis.hessian(xva["x"].data)
            dE = np.einsum("nld,nd->nl", dbk, xva["a"].data) + np.einsum("nlcd,nc,nd->nl", ddbk, xva["v"].data, xva["v"].data)
            return E, Evel, dE
        else:
            raise ValueError("Basis evaluation goal not specified")

    def force_eval(self, x):
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="force_eval")
        return np.einsum("ik,kl->il", E, self.force_coeff[: self.N_basis_elt])  # -1 * np.matmul(self.force_coeff[: self.N_basis_elt, :], E.T)

    def friction_force_eval(self, x):
        """
        Compute the term of friction, that should be zero
        """
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="kernel")
        return np.einsum("ikd,kl->ild", E, self.force_coeff[self.N_basis_elt :])  # np.matmul(self.force_coeff[self.N_basis_elt :, :], E.T)


class Pos_gle_no_vel_basis(Pos_gle_base):
    """
    Use basis function dependent of the position only
    """

    def __init__(self, *args, **kwargs):
        Pos_gle_base.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")

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


class Pos_gle_const_kernel(Pos_gle_base):
    """
    A derived class in which we the kernel is computed independent of the position
    """

    def __init__(self, *args, **kwargs):
        Pos_gle_base.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.dim_x

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        # We have to deal with the multidimensionnal case as well
        bk = self.basis.basis(xva["x"].data)
        # if self.include_const:
        #     bk = np.concatenate((np.ones((bk.shape[0], 1)), bk), axis=1)
        if compute_for == "force":
            return bk
        if compute_for == "kernel":  # For kernel evaluation
            grad = np.zeros((bk.shape[0], self.dim_x, self.dim_x))
            for i in range(self.dim_x):
                grad[:, i, i] = 1.0
            return grad
        elif compute_for == "corrs":
            return bk, xva["v"].data, xva["a"].data
        else:
            raise ValueError("Basis evaluation goal not specified")


class Pos_gle_hybrid(Pos_gle_base):
    """
    Implement the hybrid projector of arXiv:2202.01922
    """

    def __init__(self, *args, **kwargs):
        Pos_gle_base.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt + 1
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        # We have to deal with the multidimensionnal case as well
        bk = self.basis.basis(xva["x"].data)
        if compute_for == "force":
            return bk
        elif compute_for == "kernel":
            # Extend the basis for multidim value
            E = np.concatenate((np.ones_like(xva["x"].data), bk), axis=1)
            return E.reshape(-1, self.N_basis_elt_kernel, 1)
        elif compute_for == "corrs":
            E = np.concatenate((xva["v"].data, bk), axis=1)
            dbk = np.einsum("nld,nd->nl", self.basis.deriv(xva["x"].data), xva["v"].data)
            dE = np.concatenate((xva["a"].data, dbk), axis=1)
            return bk, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")


class Pos_gle_overdamped(Pos_gle_base):
    """
    Extraction of position dependent memory kernel for overdamped dynamics.
    """

    def __init__(self, xva_arg, basis, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0, L_obs="v"):
        Pos_gle_base.__init__(self, xva_arg, basis, saveall, prefix, verbose, kT, trunc, L_obs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")

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
