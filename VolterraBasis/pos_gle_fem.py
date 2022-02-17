import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from skfem.assembly.basis import Basis
from skfem import LinearForm, BilinearForm, solve_linear
from skfem.helpers import dot
from scipy.spatial import cKDTree

from .pos_gle import Pos_gle_base


class ElementFinder:
    """
    Class that find the correct element given a location
    """

    def __init__(self, mesh, mapping=None):
        # Get dimension from mesh
        self.dim = mesh.dim()
        # Transform mesh to triangular version if needed
        if self.dim == 1:
            self.max_point = np.max(mesh.p)  # To avoid strict < in np.digitize
            ix = np.argsort(mesh.p)
            self.bins = mesh.p[0, ix[0]]
            self.bins_idx = mesh.t[0]
            self.find = self.find_element_1D
        elif self.dim == 2:
            self.find = self.find_element_2D
            self.mesh_finder = mesh.element_finder(mapping)
        else:
            self.tree = cKDTree(np.mean(mesh.p[:, mesh.t], axis=1).T)
            self.find = self.find_element_ND
            self.mapping = mesh._mapping() if mapping is None else mapping
            if self.dim == 2:  # We should also check the type of element
                self.inside = self.inside_2D
            elif self.dim == 3:
                self.inside = self.inside_3D

    def find_element_1D(self, X):
        """
        Assuming X is nsamples x 1
        """
        maxix = X[:, 0] == self.max_point
        X[maxix, 0] = X[maxix, 0] - 1e-10  # special case in np.digitize
        return np.argmax((np.digitize(X[:, 0], self.bins) - 1)[:, None] == self.bins_idx, axis=1)

    def find_element_2D(self, X):
        return self.mesh_finder(X[:, 0], X[:, 1])

    def find_element_ND(self, X):
        tree_query = self.tree.query(X, 5)[1]
        element_inds = np.empty((X.shape[0],), dtype=np.int)
        for n, point in enumerate(X):  # Try to avoid loop
            i_e = tree_query[n, :]
            X_loc = self.mapping.invF((point.T)[:, None, None], tind=i_e)
            inside = self.inside_2D(X_loc)
            element_inds[n] = i_e[np.argmax(inside, axis=0)]
        return element_inds

    # def find_element_3D(self, X):
    #     ix = self.tree.query(np.array([x, y, z]).T, 5)[1].flatten()
    #     X_loc = self.mapping.invF(np.array([x, y, z])[:, None], ix)
    #     inside = self.inside_3D(X_loc)
    #     return np.array([ix[np.argmax(inside, axis=0)]]).flatten()

    def inside_2D(self, X):  # Do something more general from Refdom?
        """
        Say which point are inside the element
        """
        return (X[0] >= 0) * (X[1] >= 0) * (1 - X[0] - X[1] >= 0)

    def inside_3D(X):
        """
        Say which point are inside the element
        """
        return (X[0] >= 0) * (X[1] >= 0) * (X[2] >= 0) * (1 - X[0] - X[1] - X[2] >= 0)


class Pos_gle_fem_base(Pos_gle_base):
    """
    Memory extraction using finite element basis.
    Finite element are implement using scikit-fem.
    Method "trapz" and "second_kind" are not implemented
    """

    def __init__(self, xva_arg, basis: Basis, element_finder, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0, **kwargs):
        """
        Using Finite element basis functions.

        Parameters
        ----------
        xva_arg : xarray dataset (['time', 'x', 'v', 'a']) or list of datasets.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a xarray timeseries
            or a listlike collection of them.
        basis :  scikitfem Basis class
            The finite element basis
        element_finder: ElementFinder class
            An initialized ElementFinder class to locate element
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
        Pos_gle_base.__init__(self, xva_arg, basis, saveall, prefix, verbose, kT, trunc)
        self.element_finder = element_finder
        for xva in self.xva_list:
            if "elem" not in xva.data_vars:
                xva.update({"elem": (["time"], self.element_finder.find(xva["x"].data))})

    def _check_basis(self, basis):
        """
        Simple checks on the basis class
        """
        self.basis = basis
        self.N_basis_elt = self.basis.N

    def basis_vector(self, xva, elem, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        To be implemented by the children classes
        """

        raise NotImplementedError

    def compute_gram(self):
        if self.verbose:
            print("Calculate gram...")
            print("Use kT:", self.kT)
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            for k, grouped_xva in list(xva.groupby("elem")):
                E, dofs = self.basis_vector(grouped_xva, elem=k, compute_for="force")  # To check xva[xva["elem"] == k]
                avg_gram[dofs, dofs] += np.matmul(E.T, E) / self.weightsum  #
        self.invgram = self.kT * np.linalg.pinv(avg_gram)

        if self.verbose:
            print("Found inverse gram:", self.invgram)
        return self.kT * avg_gram

    def compute_effective_masse(self):
        """
        Return effective mass matrix computed from equipartition with the velocity.
        """
        if self.verbose:
            print("Calculate effective mass...")
            print("Use kT:", self.kT)
        v2sum = 0.0
        for xva in self.xva_list:
            v2sum += np.einsum("ik,ij->kj", xva["v"], xva["v"])
        v2 = v2sum / self.weightsum
        self.mass = self.kT * np.linalg.inv(v2)

        if self.verbose:
            print("Found effective mass:", self.mass)
        return self.mass

    def compute_mean_force(self):
        """
        Computes the mean force from the trajectories.
        """
        if self.verbose:
            print("Calculate mean force...")
        avg_disp = np.zeros((self.N_basis_elt_force))
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))  # TODO: Use sparse array
        for weight, xva in zip(self.weights, self.xva_list):
            for k, grouped_xva in list(xva.groupby("elem")):
                E, dofs = self.basis_vector(grouped_xva, elem=k, compute_for="force")  # To check xva[xva["elem"] == k]
                # print(temp_gram.shape, avg_gram[dofs[:, None], dofs[None, :]].shape)
                avg_gram[dofs[:, None], dofs[None, :]] += np.einsum("ijk,ilk->jl", E, E) / self.weightsum
                avg_disp[dofs] += np.einsum("ijk,ik->j", E, grouped_xva["a"].data) / self.weightsum  # Change to einsum to use vectorial value of E
        self.invgram = self.kT * np.linalg.pinv(avg_gram)
        self.force_coeff = np.matmul(np.linalg.pinv(avg_gram), avg_disp)

    def integrate_vector_field(self, scalar_basis):
        """
        Find phi such that \nabla phi = force
        Parameters
        ----------
        scalar_basis: skfem basis
            Should be a scalar basis with the same quadrature point than internal basis
        """
        # if self.mass is None:
        #     raise Exception("Effective mass has not been computed.")
        if self.verbose:
            print("Integrate force to get potential of mean force...")
        mass = BilinearForm(lambda u, v, w: dot(u, v.grad))
        deriv = BilinearForm(lambda u, v, w: dot(u.grad, v.grad))

        grad_part = deriv.assemble(scalar_basis)
        mass_force = -1 * mass.assemble(self.basis, scalar_basis) @ self.force_coeff
        return solve_linear(grad_part, mass_force)  # TODO: remove effective mass

    def compute_pmf(self, scalar_basis):
        """
        Compute the pmf via int p(x) v(x) dx (eqivalent to the histogram but with finite element basis)
        TODO: Inclure cette fonction dans une classe "equilibre" tel qu'on calcule la force à partir des coeffs du potentiel
        Dans ce cas il faut aussi calculer la matrice de masse sur la base puisque que pour calculer le coeff de force on en a besoin
        """
        if self.verbose:
            print("Calculate potential of mean force...")
        avg_occ = np.zeros((scalar_basis.N))
        for weight, xva in zip(self.weights, self.xva_list):
            for k, grouped_xva in list(xva.groupby("elem")):
                nb_points = grouped_xva.dims["time"]
                u = np.zeros((nb_points, scalar_basis.Nbfun))  # Check dimension should we add self.dim_x?
                dofs = np.zeros(scalar_basis.Nbfun, dtype=int)
                loc_value_t = scalar_basis.mapping.invF(grouped_xva["x"].data.T.reshape(self.dim_x, 1, -1), tind=slice(k, k + 1))  # Reshape into dim * 1 element * nb of point
                # print(k, loc_value_t[0, 0, :])
                for i in range(scalar_basis.Nbfun):
                    u[:, i] = scalar_basis.elem.gbasis(scalar_basis.mapping, loc_value_t[:, 0, :], i, tind=slice(k, k + 1))[0].value  # Include jacobian term
                    dofs[i] = scalar_basis.element_dofs[i, k]
                avg_occ[dofs] += np.sum(u, axis=0) / self.weightsum  # np.einsum("ij->j", u) / self.weightsum  # ???

        mass = BilinearForm(lambda u, v, w: u * v).assemble(scalar_basis)
        self.hist_coeff = solve_linear(mass, avg_occ)
        # Projection of the log of the histogram on the basis
        # Only consider non-zero elements
        self.I_nonzero = self.hist_coeff.nonzero()  # Saved as it can be useful elsewhere
        log_hist = LinearForm(lambda u, w: u * (-1 * (np.log(w.pf) - np.max(np.log(w.pf))))).assemble(scalar_basis, pf=scalar_basis.interpolate(self.hist_coeff))
        self.potential_coeff = solve_linear(mass, log_hist)  # , x=log_hist, I=self.I_nonzero

    def evaluate_mean_force(self, xva):
        """
        Evaluate the mean force on the trajectory.
        """
        force = np.zeros_like(xva["x"].data)
        loc_groups = xva.groupby("elem")
        group_inds = xva.groupby("elem").groups
        for k, grouped_xva in list(loc_groups):
            E, dofs = self.basis_vector(grouped_xva, elem=k, compute_for="force")
            inds = np.array(group_inds[k])
            force[inds, :] = np.einsum("ijk,j->ik", E, self.force_coeff[dofs])
        return force

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
        self.bkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, 1))  # ,1 -> To keep compatibility with fortran kernel code
        for weight, xva in zip(self.weights, self.xva_list):
            a_m_force = np.zeros((weight, self.dim_x))
            globalE = np.zeros((weight, self.N_basis_elt_kernel, self.dim_x))
            loc_groups = xva.groupby("elem")
            group_inds = xva.groupby("elem").groups
            for k, grouped_xva in list(loc_groups):
                E_force, E, _, dofs = self.basis_vector(grouped_xva, elem=k)
                inds = np.array(group_inds[k])
                a_m_force[inds, :] = grouped_xva["a"].data - np.einsum("ijk,j->ik", E_force, self.force_coeff[dofs])
                globalE[inds[:, None], dofs[None, :], :] = E
            for k, grouped_xva in list(loc_groups):
                _, E, _, dofs = self.basis_vector(grouped_xva, elem=k)  # Ca doit sortir un array masqué et on
                # print(E_force.shape, E.shape)
                inds = np.array(group_inds[k])  # Obtenir les indices du groupe
                for n in range(self.trunc_ind):  # # On peut prange ce for avec numba
                    n_remove = np.sum((inds + n) >= weight)
                    last_slice = slice(None, None if n_remove == 0 else -n_remove)
                    self.bkdxcorrw[n, dofs, 0] += weight * np.einsum("ijk,ik->j", E[last_slice, :, :], a_m_force[inds[last_slice] + n, :]) / (weight - n)
                    self.bkbkcorrw[n, dofs, :] += weight * np.einsum("ijk,ilk->jl", E[last_slice, :, :], globalE[inds[last_slice] + n, :]) / (weight - n)
                    # E_shift = E[slice(None, None if n == 0 else -n), :, :]  # slice(None,None if n==0 else -n) shift(E, [-n, 0, 0])
                    # prodbkbk = np.einsum("ijk,ilk->ijl", E_shift, E[n:, :, :])
                    # loc_inds = inds[n:] - inds[slice(None, None if n == 0 else -n)]
                    # for j, loc_ind in enumerate(loc_inds):
                    #     # C'est un peu un groupby on voudrait prodbkdx.groub_by((shift(inds, n, cval=np.NaN) - inds).sum(axis=0) -> comme ça on a des valeurs unique, ensuite on enlève tout ce qui est  > self.trunc_inds
                    #     # Only consider loc_inds < self.trunc_inds
                    #     if loc_ind < self.trunc_ind:
                    #         self.bkbkcorrw[loc_ind, dofs[:, None], dofs[None, :]] += weight * prodbkbk[j] / (weight - loc_ind)
                    #         # self.bkdxcorrw[loc_ind, dofs, 0] += weight * prodbkdx[j] / (weight - loc_ind)

        self.bkbkcorrw /= self.weightsum
        self.bkdxcorrw /= self.weightsum

        if self.rank_projection:
            self._set_range_projection(rank_tol)

        if self.saveall:
            np.savetxt(self.prefix + self.corrsdxfile, self.bkdxcorrw.reshape(self.trunc_ind, -1))
            np.savetxt(self.prefix + self.corrsfile, self.bkbkcorrw.reshape(self.trunc_ind, -1))

    def compute_noise(self, xva, trunc_kernel=None):
        """
        From a trajectory get the noise.

        Parameters
        ----------
        xva : xarray dataset (['time', 'x', 'v', 'a', 'elem']) .
            Use compute_va() or see its output for format details.
            You should have run compute_element_location() on it.
            Input trajectory to compute noise.
        trunc_kernel : int
                Number of datapoint of the kernel to consider.
                Can be used to remove unphysical divergence of the kernel or shortten execution time.
        """
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        if trunc_kernel is None:
            trunc_kernel = self.trunc_ind

        if self.method == "trapz":
            trunc_kernel -= 1
        elif self.method in ["midpoint", "midpoint_w_richardson"]:
            raise ValueError("Cannot compute noise when kernel computed with method {}".format(self.method))
        time = xva["time"].data
        dt = xva.dt

        force = np.zeros(xva["x"].data.shape)
        memory = np.zeros(xva["x"].data.shape)
        if "elem" not in xva.data_vars:
            xva.update({"elem": (["time"], self.element_finder.find(xva["x"].data))})
        loc_groups = xva.groupby("elem")
        group_inds = xva.groupby("elem").groups
        for k, grouped_xva in list(loc_groups):
            E_force, E, _, dofs = self.basis_vector(grouped_xva, elem=k)
            inds = np.array(group_inds[k])  # Obtenir les indices du groupe
            force[inds, :] = np.einsum("ijk,j->ik", E, self.force_coeff[dofs])
            for n in range(memory.shape[0]):  # # On peut prange ce for avec numba
                if n <= trunc_kernel:
                    E_shift = E[slice(None, None if n == 0 else -n), :, :]  # slice(None,None if n==0 else -n) shift(E, [-n, 0, 0])
                    to_integrate = np.einsum("ikd,ik->id", E[: n + 1, :][::-1, :], self.kernel[: n + 1, :, 0])
                    prodbkdx = np.einsum("ijk,ik->ij", E_shift, a_m_force[n:, :])  # On peut virer la fin des array
                else:
                    E_shift = E[slice(None, None if n == 0 else -n), :, :]  # slice(None,None if n==0 else -n) shift(E, [-n, 0, 0])
                    to_integrate = np.einsum("ik,ikl->il", E[n - trunc_kernel + 1 : n + 1, :][::-1, :], self.kernel[:trunc_kernel, :, :])
                loc_inds = inds[n:] - inds[slice(None, None if n == 0 else -n)]
                # print(loc_inds.shape)
                for k, loc_ind in enumerate(loc_inds):
                    if loc_ind < self.trunc_ind:
                        self.memory[loc_ind] += -1 * to_integrate[k] * dt

        for n in range(trunc_kernel):
            to_integrate = np.einsum("ik,ikl->il", E[: n + 1, :][::-1, :], self.kernel[: n + 1, :, :])
            memory[n] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
            # memory[n] = -1 * to_integrate.sum() * dt
        # Only select self.trunc_ind points for the integration
        for n in range(trunc_kernel, memory.shape[0]):
            to_integrate = np.einsum("ik,ikl->il", E[n - trunc_kernel + 1 : n + 1, :][::-1, :], self.kernel[:trunc_kernel, :, :])
            memory[n] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
            # memory[n] = -1 * to_integrate.sum() * dt
        return time, xva["a"].data - force - memory, xva["a"].data, force, memory

    def dU(self, x):
        """
        Evaluate the force at given points x
        """
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        x = np.asarray(x).reshape(-1, self.dim_x)
        output_dU = np.zeros_like(x)
        eval_pos = xr.Dataset({"x": (["time", "dim_x"], x)})
        eval_pos.update({"elem": (["time"], self.element_finder.find(x))})
        loc_groups = eval_pos.groupby("elem")
        group_inds = eval_pos.groupby("elem").groups
        for k, grouped_xva in list(loc_groups):
            E, dofs = self.basis_vector(grouped_xva, elem=k, compute_for="force")  # To check xva[xva["elem"] == k]
            inds = np.array(group_inds[k])
            output_dU[inds, :] = np.einsum("ijk,j->ik", E, self.force_coeff[dofs])
        # E = self.basis_vector(), compute_for="force")
        return -1 * output_dU  # Return the force as array (nb of evalution point x dim_x)

    def kernel_eval(self, x):
        """
        Evaluate the kernel at given points x
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        x = np.asarray(x).reshape(-1, self.dim_x)
        output_kernel = np.zeros((self.kernel.shape[0], x.shape[0], self.dim_x, self.dim_x))
        eval_pos = xr.Dataset({"x": (["time", "dim_x"], x)})
        eval_pos.update({"elem": (["time"], self.element_finder.find(x))})
        loc_groups = eval_pos.groupby("elem")
        group_inds = eval_pos.groupby("elem").groups
        for k, grouped_xva in list(loc_groups):
            E, dofs = self.basis_vector(grouped_xva, elem=k, compute_for="kernel")
            inds = np.array(group_inds[k])
            output_kernel[:, inds, :, :] = np.einsum("jkdf,ik->ijdf", E, self.kernel[:, dofs, 0])
        return self.time, output_kernel  # Return the kernel as array (time x nb of evalution point x dim_x x dim_x)


class Pos_gle_fem(Pos_gle_fem_base):
    """
    The main class for the position dependent memory extraction,
    holding all data and the extracted memory kernels.
    """

    def __init__(self, xva_arg, basis: Basis, element_finder, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0):
        """
        Create an instance of the Pos_gle class.

        Parameters
        ----------
        xva_arg : xarray dataset (['time', 'x', 'v', 'a']) or list of datasets.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a xarray timeseries
            or a listlike collection of them.
        basis :  scikitfem Basis class
            The finite element basis
        element_finder: ElementFinder class
            An initialized ElementFinder class to locate element
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
        Pos_gle_fem_base.__init__(self, xva_arg, basis, element_finder, saveall, prefix, verbose, kT, trunc)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        self.rank_projection = True

    def basis_vector(self, xva, elem, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        nb_points = xva.dims["time"]
        bk = np.zeros((nb_points, self.basis.Nbfun, self.dim_x))  # Check dimension should we add self.dim_x?
        dbk = np.zeros((nb_points, self.basis.Nbfun, self.dim_x, self.dim_x))  # Check dimension
        dofs = np.zeros(self.basis.Nbfun, dtype=int)
        loc_value_t = self.basis.mapping.invF(xva["x"].data.T.reshape(self.dim_x, 1, -1), tind=slice(elem, elem + 1))  # Reshape into dim * 1 element * nb of point
        for i in range(self.basis.Nbfun):
            phi_field = self.basis.elem.gbasis(self.basis.mapping, loc_value_t[:, 0, :], i, tind=slice(elem, elem + 1))
            bk[:, i, :] = phi_field[0].value.T.reshape(-1, self.dim_x)  # The middle indice is the choice of element, ie only one choice here
            dbk[:, i, :, :] = phi_field[0].grad.T.reshape(-1, self.dim_x, self.dim_x)  # dbk via div? # TODO CHECK the transpose
            dofs[i] = self.basis.element_dofs[i, elem]
        if compute_for == "force":
            return bk, dofs
        if compute_for == "kernel":  # For kernel evaluation
            return dbk, dofs
        elif compute_for == "corrs":
            E = np.einsum("nlfd,nd->nlf", dbk, xva["v"].data)
            return bk, E, None, dofs
        else:
            raise ValueError("Basis evaluation goal not specified")


class Pos_gle_fem_equilibrium(Pos_gle_fem_base):
    """
    Class for computation of equilibrium systems.
    """

    def __init__(self, xva_arg, basis: Basis, element_finder, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0):
        """
        Create an instance of the Pos_gle_fem_equilibrium class.

        Parameters
        ----------
        xva_arg : xarray dataset (['time', 'x', 'v', 'a']) or list of datasets.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a xarray timeseries
            or a listlike collection of them.
        basis :  scikitfem Basis class
            The finite element basis
        element_finder: ElementFinder class
            An initialized ElementFinder class to locate element
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
        Pos_gle_fem_base.__init__(self, xva_arg, basis, element_finder, saveall, prefix, verbose, kT, trunc)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        self.rank_projection = True
        # Run some check on basis

    def basis_vector(self, xva, elem, compute_for="corrs"):
        """
        TODO: Implement the basis using scalar Element
        """
        nb_points = xva.dims["time"]
        bk = np.zeros((nb_points, self.basis.Nbfun, self.dim_x))  # Check dimension should we add self.dim_x?
        dbk = np.zeros((nb_points, self.basis.Nbfun, self.dim_x, self.dim_x))  # Check dimension
        dofs = np.zeros(self.basis.Nbfun, dtype=int)
        loc_value_t = self.basis.mapping.invF(xva["x"].data.T.reshape(self.dim_x, 1, -1), tind=slice(elem, elem + 1))  # Reshape into dim * 1 element * nb of point
        for i in range(self.basis.Nbfun):
            phi_field = self.basis.elem.gbasis(self.basis.mapping, loc_value_t[:, 0, :], i, tind=slice(elem, elem + 1))
            bk[:, i, :] = phi_field[0].value.T.reshape(-1, self.dim_x)  # The middle indice is the choice of element, ie only one choice here
            dbk[:, i, :, :] = phi_field[0].grad.T.reshape(-1, self.dim_x, self.dim_x)  # dbk via div?
            dofs[i] = self.basis.element_dofs[i, elem]
        if compute_for == "force":
            return bk, dofs
        if compute_for == "kernel":  # For kernel evaluation
            return dbk, dofs
        elif compute_for == "corrs":
            E = np.einsum("nlfd,nd->nlf", dbk, xva["v"].data)
            return bk, E, None, dofs
        else:
            raise ValueError("Basis evaluation goal not specified")

    def compute_mean_force_from_pmf(self):
        """
        Compute the mean force from the pmf
        """
        self.compute_pmf(self.basis)
        self.compute_effective_masse()
        self.force_coeff = self.mass @ self.potential_coeff


class Pos_gle_fem_app_kernel(Pos_gle_fem_base):
    """
    A class with a different basis for the kernel and the force.
    Can be used as an approximation of the real kernel when dimension of bases is too high
    """

    def __init__(self, xva_arg, basis: Basis, element_finder, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0):
        """
        Create an instance of the Pos_gle class.

        Parameters
        ----------
        xva_arg : xarray dataset (['time', 'x', 'v', 'a']) or list of datasets.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a xarray timeseries
            or a listlike collection of them.
        basis :  scikitfem Basis class
            The finite element basis
        element_finder: ElementFinder class
            An initialized ElementFinder class to locate element
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
        Pos_gle_fem_base.__init__(self, xva_arg, basis, element_finder, saveall, prefix, verbose, kT, trunc)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        self.rank_projection = True

    def basis_vector(self, xva, elem, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        nb_points = xva.dims["time"]
        bk = np.zeros((nb_points, self.basis.Nbfun, self.dim_x))  # Check dimension should we add self.dim_x?
        dbk = np.zeros((nb_points, self.basis.Nbfun, self.dim_x, self.dim_x))  # Check dimension
        dofs = np.zeros(self.basis.Nbfun, dtype=int)
        loc_value_t = self.basis.mapping.invF(xva["x"].data.T.reshape(self.dim_x, 1, -1), tind=slice(elem, elem + 1))  # Reshape into dim * 1 element * nb of point
        for i in range(self.basis.Nbfun):
            phi_field = self.basis.elem.gbasis(self.basis.mapping, loc_value_t[:, 0, :], i, tind=slice(elem, elem + 1))
            bk[:, i, :] = phi_field[0].value.T.reshape(-1, self.dim_x)  # The middle indice is the choice of element, ie only one choice here
            dbk[:, i, :, :] = phi_field[0].grad.T.reshape(-1, self.dim_x, self.dim_x)  # dbk via div? # TODO CHECK the transpose
            dofs[i] = self.basis.element_dofs[i, elem]
        if compute_for == "force":
            return bk, dofs
        if compute_for == "kernel":  # For kernel evaluation
            return dbk, dofs
        elif compute_for == "corrs":
            E = np.einsum("nlfd,nd->nlf", dbk, xva["v"].data)
            return bk, E, None, dofs
        else:
            raise ValueError("Basis evaluation goal not specified")
