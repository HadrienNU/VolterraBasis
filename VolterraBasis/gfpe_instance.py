import numpy as np

from .pos_gle_base import Pos_gle_base
from .fkernel import solve_ide_rect, solve_ide_trapz, solve_ide_trapz_stab


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
        self.inv_occ = np.diag(1 / self.occupations())
        for i in range(len(self.xva_list)):
            E_force, E, dE = self.basis_vector(self.xva_list[i])
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

    def solve_gfpe(self, lenTraj, method="trapz", p0=None, absorbing_states=None, trunc_ind=None):
        """
        Solve the integro-differential equation
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        if p0 is None:
            p0 = np.identity(self.dim_obs)
        else:
            p0 = np.asarray(p0).reshape(self.dim_obs, -1)
        if trunc_ind is None or trunc_ind <= 0:
            trunc_ind = self.trunc_ind

        p0 = self.bkbkcorrw[0, :, :] @ self.inv_occ @ p0
        force_coeff = np.einsum("kj->jk", self.force_coeff)
        kernel = np.einsum("ikj->ijk", self.kernel[:trunc_ind, :, :])

        if not (absorbing_states is None):
            if type(absorbing_states) in [int, np.int, np.int64]:
                list_state = [absorbing_states]
            for i in list_state:  # C'est probablement pas comme ça qu'il faut faire, mais au moins annuler la green function plutôt que <E,E>
                force_coeff[i, :] = 0.0
                kernel[:, i, :] = 0.0
        if method == "rect":
            p = solve_ide_rect(kernel, p0, force_coeff, lenTraj, self.dt)  # TODO it might worth transpose all the code for the kernel
        elif method == "trapz":
            p = solve_ide_trapz(kernel, p0, force_coeff, lenTraj, self.dt)
        elif method == "trapz_stab":
            p = solve_ide_trapz_stab(kernel, p0, force_coeff, lenTraj, self.dt)
        return np.arange(lenTraj) * self.dt, np.squeeze(p)

    def compute_flux(self, p_t, trunc_ind=None):
        """
        Compute the flux for evolution of the probability
        """
        if trunc_ind is None or trunc_ind <= 0:
            trunc_ind = self.trunc_ind
        force_coeff = np.einsum("kj->jk", self.force_coeff)
        kernel = np.einsum("ikj->ijk", self.kernel[:trunc_ind, :, :])
        flux = np.zeros(p_t.shape[0:] + p_t.shape[1:])
        for n in range(p_t.shape[0]):
            m = min(n, kernel.shape[0])
            to_integrate = np.einsum("ikj,ij...->ikj...", kernel[:m, :, :], p_t[n - m : n, ...][::-1, ...])
            flux[n, ...] = np.trapz(to_integrate, dx=self.dt, axis=0) + np.einsum("kj,j...->kj...", force_coeff, p_t[n, ...])  # vb.fkernel.memory_trapz(kernel, p_t[:n, :], dt)[-1, :]
        return flux

    def study_stability(self):
        s_range, laplace = self.laplace_transform_kernel()
        force_coeff = np.einsum("kj->jk", self.force_coeff)
        lapace_ker = np.einsum("ikj->ijk", laplace)
        det_laplace = np.zeros_like(s_range)
        eig_vals = np.zeros((len(s_range), self.dim_obs))
        id = np.identity(self.dim_obs)
        for i, s in enumerate(s_range):
            resol_mat = s * id - force_coeff + lapace_ker[i, :, :]
            det_laplace[i] = np.linalg.det(resol_mat)
            eig_vals[i, :] = np.linalg.eigvals(resol_mat)
        return s_range, det_laplace, eig_vals
