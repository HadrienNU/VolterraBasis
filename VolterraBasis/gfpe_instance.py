import numpy as np

from .pos_gle_base import Pos_gle_base
from .fkernel import solve_ide_rect, solve_ide_trapz


class Pos_gfpe(Pos_gle_base):
    """
    Linear projection of the basis on the basis. The resulting Volterra equation is a generalized Fokker Planck equation
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
