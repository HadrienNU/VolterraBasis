import numpy as np

#
# class Pos_gfpe(Pos_gle_base):  # pragma: no cover
#     """
#     Linear projection of the basis on the basis. The resulting Volterra equation is a generalized Fokker Planck equation
#     """
#
#     def solve_gfpe(self, lenTraj, method="trapz", p0=None, absorbing_states=None, trunc_ind=None):
#         """
#         Solve the integro-differential equation
#         """
#         if self.kernel is None:
#             raise Exception("Kernel has not been computed.")
#         if p0 is None:
#             p0 = np.identity(self.dim_obs)
#         else:
#             p0 = np.asarray(p0).reshape(self.dim_obs, -1)
#         if trunc_ind is None or trunc_ind <= 0:
#             trunc_ind = self.trunc_ind
#
#         p0 = self.bkbkcorrw[0, :, :] @ self.inv_occ @ p0
#         force_coeff = np.einsum("kj->jk", self.force_coeff)
#         kernel = np.einsum("ikj->ijk", self.kernel[:trunc_ind, :, :])
#
#     def study_stability(self):
#         s_range, laplace = self.laplace_transform_kernel()
#         force_coeff = np.einsum("kj->jk", self.force_coeff)
#         lapace_ker = np.einsum("ikj->ijk", laplace)
#         det_laplace = np.zeros_like(s_range)
#         eig_vals = np.zeros((len(s_range), self.dim_obs))
#         id = np.identity(self.dim_obs)
#         for i, s in enumerate(s_range):
#             resol_mat = s * id - force_coeff + lapace_ker[i, :, :]
#             det_laplace[i] = np.linalg.det(resol_mat)
#             eig_vals[i, :] = np.linalg.eigvals(resol_mat)
#         return s_range, det_laplace, eig_vals
#
#
# class Pos_gfpe_fem(Pos_gfpe):  # pragma: no cover
#     """
#     Linear projection of the basis on the basis. The resulting Volterra equation is a generalized Fokker Planck equation
#     """
#
#     def _update_traj(self):
#         for weight, xva in zip(self.weights, self.xva_list):
#             if "elem" not in xva.data_vars:
#                 xva.update({"elem": (["time"], self.element_finder(xva["x"].data))})
#             globaldE = np.zeros((weight, self.N_basis_elt_kernel))
#             loc_groups = xva.groupby("elem")
#             group_inds = xva.groupby("elem").groups
#             for k, grouped_xva in list(loc_groups):
#                 _, _, dE, dofs = self.basis_vector(grouped_xva, elem=k)
#                 inds = np.array(group_inds[k])
#                 globaldE[inds[:, None], dofs[None, :]] = dE
#             xva.update({"dE": (["time", "dim_dE"], globaldE)})
#
#     def basis_vector(self, xva, elem, compute_for="corrs"):
#         """
#         From one trajectory compute the basis element.
#         """
#         nb_points = xva.dims["time"]
#         bk = np.zeros((nb_points, self.basis.Nbfun))  # Check dimension should we add self.dim_x?
#         dbk = np.zeros((nb_points, self.basis.Nbfun, self.dim_x))  # Check dimension
#         dofs = np.zeros(self.basis.Nbfun, dtype=int)
#         loc_value_t = self.basis.mapping.invF(xva["x"].data.T.reshape(self.dim_x, 1, -1), tind=slice(elem, elem + 1))  # Reshape into dim * 1 element * nb of point
#         for i in range(self.basis.Nbfun):
#             phi_field = self.basis.elem.gbasis(self.basis.mapping, loc_value_t[:, 0, :], i, tind=slice(elem, elem + 1))
#             bk[:, i] = phi_field[0].value.flatten()  # The middle indice is the choice of element, ie only one choice here
#             dbk[:, i, :] = phi_field[0].grad.T.reshape(-1, self.dim_x)  # dbk via div? # TODO CHECK the transpose
#             dofs[i] = self.basis.element_dofs[i, elem]
#         if compute_for == "force":
#             return bk, dofs
#         if compute_for == "kernel":  # For kernel evaluation
#             return bk, dofs
#         elif compute_for == "corrs":
#             dE = np.einsum("nld,nd->nl", dbk, xva["v"].data)
#             return bk, bk, dE, dofs
#         else:
#             raise ValueError("Basis evaluation goal not specified")
