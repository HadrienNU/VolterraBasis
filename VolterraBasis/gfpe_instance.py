import numpy as np
from skfem import LinearForm, BilinearForm, solve_linear
from skfem.helpers import dot
from itertools import product
from scipy.sparse import coo_matrix


from .pos_gle_base import Pos_gle_base
from .pos_gle_fem import ElementFinder
from .fkernel import solve_ide_rect, solve_ide_trapz, solve_ide_trapz_stab


class Pos_gfpe(Pos_gle_base):
    """
    Linear projection of the basis on the basis. The resulting Volterra equation is a generalized Fokker Planck equation
    """

    def __init__(self, xva_arg, basis, saveall=True, prefix="", verbose=True, trunc=1.0, **kwargs):
        Pos_gle_base.__init__(self, xva_arg, basis, saveall, prefix, verbose, trunc, L_obs="v", **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if hasattr(self.basis, "const_removed"):
            if self.basis.const_removed:
                self.basis.const_removed = False
                print("Warning: remove_const on basis function have been set to False.")
        self.rank_projection = False
        self.dim_obs = self.N_basis_elt
        self.L_obs = "dE"
        self._update_traj()
        self.inv_occ = np.diag(1 / self.occupations())

    def _update_traj(self):
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
            flux[n, ...] = -1 * np.trapz(to_integrate, dx=self.dt, axis=0) + np.einsum("kj,j...->kj...", force_coeff, p_t[n, ...])  # vb.fkernel.memory_trapz(kernel, p_t[:n, :], dt)[-1, :]
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


class Pos_gfpe_fem(Pos_gfpe):
    """
    Linear projection of the basis on the basis. The resulting Volterra equation is a generalized Fokker Planck equation
    """

    def _check_basis(self, basis):
        """
        Simple checks on the basis class
        """
        self.basis = basis
        self.N_basis_elt = self.basis.N
        # Find tensorial order of the basis and adapt dimension in consequence
        test = self.basis.elem.gbasis(self.basis.mapping, self.basis.mapping.F(self.basis.mesh.p), 0)[0]
        if len(test.shape) == 3:  # Vectorial basis
            raise ValueError("You should use scalar basis for GFPE")

    def _update_traj(self):
        for weight, xva in zip(self.weights, self.xva_list):
            if "elem" not in xva.data_vars:
                xva.update({"elem": (["time"], self.element_finder(xva["x"].data))})
            globaldE = np.zeros((weight, self.N_basis_elt_kernel))
            loc_groups = xva.groupby("elem")
            group_inds = xva.groupby("elem").groups
            for k, grouped_xva in list(loc_groups):
                _, _, dE, dofs = self.basis_vector(grouped_xva, elem=k)
                inds = np.array(group_inds[k])
                globaldE[inds[:, None], dofs[None, :]] = dE
            xva.update({"dE": (["time", "dim_dE"], globaldE)})

    def element_finder(self, x):
        # At first use, if not implement instancie the element finder
        if not hasattr(self, "element_finder_from_basis"):
            self.element_finder_from_basis = ElementFinder(self.basis.mesh, mapping=self.basis.mapping)  # self.basis.mesh.element_finder(mapping=self.basis.mapping)
        return self.element_finder_from_basis.find(x)

    def basis_vector(self, xva, elem, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        nb_points = xva.dims["time"]
        bk = np.zeros((nb_points, self.basis.Nbfun))  # Check dimension should we add self.dim_x?
        dbk = np.zeros((nb_points, self.basis.Nbfun, self.dim_x))  # Check dimension
        dofs = np.zeros(self.basis.Nbfun, dtype=int)
        loc_value_t = self.basis.mapping.invF(xva["x"].data.T.reshape(self.dim_x, 1, -1), tind=slice(elem, elem + 1))  # Reshape into dim * 1 element * nb of point
        for i in range(self.basis.Nbfun):
            phi_field = self.basis.elem.gbasis(self.basis.mapping, loc_value_t[:, 0, :], i, tind=slice(elem, elem + 1))
            bk[:, i] = phi_field[0].value.flatten()  # The middle indice is the choice of element, ie only one choice here
            dbk[:, i, :] = phi_field[0].grad.T.reshape(-1, self.dim_x)  # dbk via div? # TODO CHECK the transpose
            dofs[i] = self.basis.element_dofs[i, elem]
        if compute_for == "force":
            return bk, dofs
        if compute_for == "kernel":  # For kernel evaluation
            return bk, dofs
        elif compute_for == "corrs":
            dE = np.einsum("nld,nd->nl", dbk, xva["v"].data)
            return bk, bk, dE, dofs
        else:
            raise ValueError("Basis evaluation goal not specified")

    def compute_gram(self, kT=1.0):
        if self.verbose:
            print("Calculate gram...")
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            for k, grouped_xva in list(xva.groupby("elem")):
                E, dofs = self.basis_vector(grouped_xva, elem=k, compute_for="force")  # To check xva[xva["elem"] == k]
                avg_gram[dofs, dofs] += np.matmul(E.T, E) / self.weightsum  #
        self.invgram = kT * np.linalg.pinv(avg_gram)

        if self.verbose:
            print("Found inverse gram:", self.invgram)
        return kT * avg_gram

    def occupations(self):
        """
        Compute mean value of the basis function
        """
        occ = np.zeros(self.N_basis_elt_force)
        weightsum = 0.0
        for weight, xva in zip(self.weights, self.xva_list):
            for k, grouped_xva in list(xva.groupby("elem")):
                E, dofs = self.basis_vector(grouped_xva, elem=k, compute_for="force")  # To check xva[xva["elem"] == k]
                occ[dofs] += np.sum(E, axis=0) / self.weightsum  #
            weightsum += weight
        return occ / weightsum

    def compute_mean_force(self, regularization=0.0):
        """
        Computes the mean force from the trajectories.
        """
        if self.verbose:
            print("Calculate mean force...")
        avg_disp = np.zeros((self.N_basis_elt_force, self.dim_obs))
        # avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        data_gram = np.zeros((self.basis.Nbfun, self.basis.Nbfun, self.basis.nelems))
        indices = np.zeros((self.basis.Nbfun * self.basis.Nbfun * self.basis.nelems, 2), dtype=np.int64)
        for weight, xva in zip(self.weights, self.xva_list):
            for k, grouped_xva in list(xva.groupby("elem")):
                E, dofs = self.basis_vector(grouped_xva, elem=k, compute_for="force")  # To check xva[xva["elem"] == k]
                data_gram[:, :, k] += np.einsum("ij,il->jl", E, E) / self.weightsum
                indices[slice(k * (self.basis.Nbfun * self.basis.Nbfun), (k + 1) * (self.basis.Nbfun * self.basis.Nbfun))] = np.array([[i, j] for j, i in product(dofs, dofs)])
                avg_disp[dofs] += np.einsum("ij,ik->jk", E, grouped_xva[self.L_obs].data) / self.weightsum  # Change to einsum to use vectorial value of E
                # avg_gram[dofs[:, None], dofs[None, :]] += np.einsum("ijk,ilk->jl", E, E) / mymem.weightsum
        # Construct sparse matrix
        avg_gram = coo_matrix((data_gram.flatten("F"), (indices[:, 0], indices[:, 1])), shape=(self.N_basis_elt_force, self.N_basis_elt_force))
        avg_gram.eliminate_zeros()
        self.gram = avg_gram
        if regularization > 0.0:  # Regularization of the gram matrix
            avg_gram += regularization * (BilinearForm(lambda u, v, w: dot(u, v)).assemble(self.basis))
        self.force_coeff = solve_linear(avg_gram.tocsr(), avg_disp)
        # self.force_coeff = np.matmul(np.linalg.pinv(avg_gram), avg_disp)

    def compute_corrs(self, rank_tol=None):
        """
        Compute correlation functions.

        Parameters
        ----------
        rank_tol: float, default=None
            Tolerance for rank computation in case of projection onto the range of the basis
        """
        if self.verbose:
            print("Calculate correlation functions...")
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        # Faire bkbk sparse
        self.bkbkcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.N_basis_elt_kernel))  # TODO: Use sparse array
        self.bkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.dim_obs))
        for weight, xva in zip(self.weights, self.xva_list):
            a_m_force = np.zeros((weight, self.dim_obs))
            globalE = np.zeros((weight, self.N_basis_elt_kernel))
            loc_groups = xva.groupby("elem")
            group_inds = xva.groupby("elem").groups
            for k, grouped_xva in list(loc_groups):
                E_force, E, _, dofs = self.basis_vector(grouped_xva, elem=k)
                inds = np.array(group_inds[k])
                a_m_force[inds, :] = grouped_xva[self.L_obs].data - np.einsum("ij,jd->id", E_force, self.force_coeff[dofs, :])
                globalE[inds[:, None], dofs[None, :]] = E
            for k, grouped_xva in list(loc_groups):
                _, E, _, dofs = self.basis_vector(grouped_xva, elem=k)  # Ca doit sortir un array masqué et on
                # print(E_force.shape, E.shape)
                inds = np.array(group_inds[k])  # Obtenir les indices du groupe
                for n in range(self.trunc_ind):  # # On peut prange ce for avec numba
                    n_remove = np.sum((inds + n) >= weight)
                    last_slice = slice(None, None if n_remove == 0 else -n_remove)
                    self.bkdxcorrw[n, dofs, :] += weight * np.einsum("ij,ik->jk", E[last_slice, :], a_m_force[inds[last_slice] + n, :]) / (weight - n)
                    self.bkbkcorrw[n, dofs, :] += weight * np.einsum("ij,il->jl", E[last_slice, :], globalE[inds[last_slice] + n, :]) / (weight - n)
        self.bkbkcorrw /= self.weightsum
        self.bkdxcorrw /= self.weightsum

        if self.rank_projection:
            self._set_range_projection(rank_tol)

        if self.saveall:
            np.savetxt(self.prefix + self.corrsdxfile, self.bkdxcorrw.reshape(self.trunc_ind, -1))
            np.savetxt(self.prefix + self.corrsfile, self.bkbkcorrw.reshape(self.trunc_ind, -1))
