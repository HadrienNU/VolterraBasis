import numpy as np
from scipy.integrate import simpson

from .pos_gle_base import Pos_gle_base
from .fkernel import solve_ide_rect, solve_ide_trapz


class Pos_gfpe(Pos_gle_base):
    """
    Linear projection of the basis on the basis. The resulting Volterra equation is a generalized Fokker Planck equation
    """

    def __init__(self, xva_arg, basis, saveall=True, prefix="", verbose=True, trunc=1.0, **kwargs):
        Pos_gle_base.__init__(self, xva_arg, basis, saveall, prefix, verbose, trunc, L_obs="v", **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")
        self.rank_projection = False
        self.dim_obs = self.N_basis_elt
        self.L_obs = "dE"

        for i in range(len(self.xva_list)):
            E_force, E, dE = self.basis_vector(self.xva_list[i])
            self.xva_list[i].update({"dE": (["time", "dim_dE"], dE)})

        self.set_zero_force()  # Unless non hermitian, this should be true, we can also compute mean force after if wanted

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

    def set_absorbing_state(self, list_state):
        """
        Set to zero the appropriate element of the memory kernel
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        if type(list_state) in [int, np.int, np.int64]:
            list_state = [list_state]
        for i in list_state:
            self.kernel[:, i, :] = 0.0

    def laplace_transform_kernel(self, n_points):
        """
        Compute the Laplace transform of the kernel matrix
        """
        dt = self.xva_list[0].attrs["dt"]
        mintimelenght = np.min(self.weights) * dt
        s_range = np.linspace(1 / mintimelenght, 1 / dt, n_points)
        laplace = np.zeros((n_points, self.kernel.shape[1], self.kernel.shape[2]))
        for n, s in enumerate(s_range):
            laplace[n, :, :] = simpson(np.exp(-s * self.time) * self.kernel, self.time, axis=0)
        return s_range, laplace

    def occupations(self):
        """
        Compute mean value of the basis function
        """
        occ = 0.0
        weightsum = 0.0
        for xva in self.xva_list:
            E = self.basis.basis(xva["x"].data)
            occ += E.sum(axis=0)
            weightsum += E.shape[0]
        return occ / weightsum

    def solve_gfpe(self, lenTraj, method="trapz", p0=None):
        """
        Solve the integro-differential equation
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        if p0 is None:
            p0 = self.bkbkcorrw[0, :, :]
        else:
            p0 = np.asarray(p0).reshape(self.dim_obs, -1)
        dt = self.xva_list[0].attrs["dt"]
        gram_occ = self.bkbkcorrw[0, :, :] @ np.diag(1.0 / self.occupations())
        # print(gram_occ, np.sum(np.linalg.inv(gram_occ), axis=0), np.sum(gram_occ, axis=0))
        # p0 = gram_occ @ p0
        # kernel = np.einsum("jn,ikj,kl->iln", np.linalg.inv(gram_occ), self.kernel, gram_occ)
        kernel = np.einsum("nj,ikj, kl->inl", np.linalg.inv(gram_occ), self.kernel, gram_occ)
        # kernel = np.einsum("ikj->ijk", self.kernel)
        if method == "rect":
            p = solve_ide_rect(kernel, p0, self.force_coeff, lenTraj, dt)  # TODO it might worth transpose all the code for the kernel
        elif method == "trapz":
            p = solve_ide_trapz(kernel, p0, self.force_coeff, lenTraj, dt)
        return np.arange(lenTraj) * dt, np.squeeze(p)
