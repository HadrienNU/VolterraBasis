import numpy as np

from .pos_gle_base import Pos_gle_base, _convert_input_array_for_evaluation
from .fkernel import solve_ide_rect, solve_ide_trapz


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
        # We have to deal with the multidimensionnal case as well
        bk = self.basis.basis(xva["x"].data)
        # if self.include_const:
        #     bk = np.concatenate((np.ones((bk.shape[0], 1)), bk), axis=1)
        if compute_for == "force":
            return bk
        elif compute_for == "pmf":
            return self.basis.antiderivative(xva["x"].data)
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
        # We have to deal with the multidimensionnal case as well
        bk = self.basis.basis(xva["x"].data)
        if compute_for == "force_eval":
            return bk
        elif compute_for == "pmf":
            return self.basis.antiderivative(xva["x"].data)
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
        """
        Evaluate the force for the position dependent part only
        """
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
        # We have to deal with the multidimensionnal case as well
        E = self.basis.basis(xva["x"].data)
        if compute_for == "force":
            return E
        elif compute_for == "pmf":
            return self.basis.antiderivative(xva["x"].data)
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
        self.N_basis_elt_kernel = self.dim_obs

    def basis_vector(self, xva, compute_for="corrs"):
        # We have to deal with the multidimensionnal case as well
        bk = self.basis.basis(xva["x"].data)
        # if self.include_const:
        #     bk = np.concatenate((np.ones((bk.shape[0], 1)), bk), axis=1)
        if compute_for == "force":
            return bk
        elif compute_for == "pmf":
            return self.basis.antiderivative(xva["x"].data)
        if compute_for == "kernel":  # For kernel evaluation
            grad = np.zeros((bk.shape[0], self.dim_obs, self.dim_obs))
            for i in range(self.dim_obs):
                grad[:, i, i] = 1.0
            return grad
        elif compute_for == "corrs":
            return bk, xva["v"].data, xva["a"].data
        else:
            raise ValueError("Basis evaluation goal not specified")

    def compute_linear_part_force(self):
        """
        Computes the mean force from the trajectories.
        """
        if self.verbose:
            print("Calculate linear part of the force...")
        avg_disp = np.zeros((1, self.dim_obs))
        avg_gram = np.zeros((1, 1))
        for weight, xva in zip(self.weights, self.xva_list):
            E = xva["x"].data
            avg_disp += np.matmul(E.T, self.force_eval(xva)) / self.weightsum
            avg_gram += np.matmul(E.T, E) / self.weightsum
        print(avg_gram)
        self.linear_force_part = avg_disp / avg_gram
        return self.linear_force_part

    def eval_non_linear_force(self, x, coeffs=None):
        """
        Evaluate the force at given points x.
        If coeffs is given, use provided coefficients instead of the force
        """
        return self.force_eval(x, coeffs)  # - self.linear_force_part * x["x"].data  # Return the force as array (nb of evalution point x dim_obs)


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
        # We have to deal with the multidimensionnal case as well
        bk = self.basis.basis(xva["x"].data)
        if compute_for == "force":
            return bk
        elif compute_for == "pmf":
            return self.basis.antiderivative(xva["x"].data)
        elif compute_for == "kernel":
            # Extend the basis for multidim value
            return bk.reshape(-1, self.N_basis_elt_kernel - 1, 1)
        elif compute_for == "corrs":
            E = np.concatenate((xva["v"].data, bk), axis=1)
            dbk = np.einsum("nld,nd->nl", self.basis.deriv(xva["x"].data), xva["v"].data)
            dE = np.concatenate((xva["a"].data, dbk), axis=1)
            return bk, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")

    def get_const_kernel_part(self):
        """
        Return the position independent part of the kernel
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        return self.time, self.kernel[:, 0, :]

    def kernel_eval(self, x):
        """
        Evaluate the position dependant part of the kernel at given points x
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="kernel")
        if self.rank_projection:
            E = np.einsum("kj,ijd->ikd", self.P_range, E)
        return self.time, np.einsum("jkd,ikl->ijld", E, self.kernel[:, 1:, :])  # Return the kernel as array (time x nb of evalution point x dim_obs x dim_x)


class Pos_gle_overdamped(Pos_gle_base):
    """
    Extraction of position dependent memory kernel for overdamped dynamics.
    """

    def __init__(self, xva_arg, basis, saveall=True, prefix="", verbose=True, trunc=1.0, L_obs="v"):
        Pos_gle_base.__init__(self, xva_arg, basis, saveall, prefix, verbose, trunc, L_obs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")

    def basis_vector(self, xva, compute_for="corrs"):
        E = self.basis.basis(xva["x"].data)
        if compute_for == "force":
            return E
        elif compute_for == "pmf":
            return self.basis.antiderivative(xva["x"].data)
        elif compute_for == "kernel":
            return E.reshape(-1, self.N_basis_elt_kernel, self.dim_x)
        elif compute_for == "corrs":
            dbk = self.basis.deriv(xva["x"].data)
            dE = np.einsum("nld,nd->nl", dbk, xva["v"].data)
            return E, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")

    def pmf_eval(self, x, coeffs=None, kT=1.0, set_zero=True):
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
        # if self.eff_mass is None:
        #     self.compute_effective_mass(kT=kT)
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="pmf")
        pmf = -1 * np.einsum("ik,kl->il", E, coeffs) / kT
        return pmf - float(set_zero) * np.min(pmf)


class Pos_gle_overdamped_const_kernel(Pos_gle_base):
    """
    Extraction of position dependent memory kernel for overdamped dynamics
    using position-independent kernel.
    """

    def __init__(self, xva_arg, basis, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0, L_obs="v"):
        Pos_gle_base.__init__(self, xva_arg, basis, saveall, prefix, verbose, kT, trunc, L_obs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.dim_obs

    def basis_vector(self, xva, compute_for="corrs"):
        E = self.basis.basis(xva["x"].data)
        if compute_for == "force":
            return E
        elif compute_for == "pmf":
            return self.basis.antiderivative(xva["x"].data)
        elif compute_for == "kernel":
            return np.ones((E.shape[0], self.N_basis_elt_kernel, self.dim_x))
        elif compute_for == "corrs":
            return E, np.ones((E.shape[0], self.dim_obs)), xva["v"].data
        else:
            raise ValueError("Basis evaluation goal not specified")


class Pos_gle_linear_proj(Pos_gle_base):
    """
    Linear projection of the basis on the basis
    """

    def __init__(self, *args, **kwargs):
        Pos_gle_base.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")

        self.dim_obs = self.N_basis_elt
        self.L_obs = "dE"

        for i in range(len(self.xva_list)):
            E_force, E, dE = self.basis_vector(self.xva_list[i])
            # Ici on échange dim_x avec dim_proj et dim_x devient Nsplines
            # Ca permet de faire une projection linéaire sur la base
            # xvaf.rename_dims({"dim_x": "dim"})
            self.xva_list[i].update({"dE": (["time", "dim_dE"], dE)})

    def basis_vector(self, xva, compute_for="corrs"):
        E = self.basis.basis(xva["x"].data)
        if compute_for == "force":
            return E
        elif compute_for == "pmf":
            return self.basis.antiderivative(xva["x"].data)
        elif compute_for == "kernel":
            return E.reshape(-1, self.N_basis_elt_kernel, self.dim_x)
        elif compute_for == "corrs":
            dbk = self.basis.deriv(xva["x"].data)
            dE = np.einsum("nld,nd->nl", dbk, xva["v"].data)
            return E, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")

    def solve_ide(self, lenTraj, method="trapz", E0=None):
        """
        Solve the integro-differential equation
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        if E0 is None:
            E0 = self.bkbkcorrw[0, :, :]
        else:
            E0 = np.asarray(E0).reshape(self.dim_obs, self.dim_obs)
        dt = self.xva_list[0].attrs["dt"]
        if method == "rect":
            E = solve_ide_rect(self.kernel, E0, self.force_coeff, lenTraj, dt)
        elif method == "trapz":
            E = solve_ide_trapz(self.kernel, E0, self.force_coeff, lenTraj, dt)
        return np.arange(lenTraj) * dt, E
