import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from skfem.assembly.basis import Basis
from scipy.spatial import cKDTree

from .pos_gle import Pos_gle_base
from .correlation import correlation_ND


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


def compute_element_location(xva, element_finder):
    """
    Computes the location within the mesh of the trajectory.

    Parameters
    ----------
    xvf : xarray dataset (['x'])

    element_finder: Instance of ElementFinder class
    """
    return xva.assign({"elem": (["time"], element_finder.find(xva["x"].data))})


class Pos_gle_fem_base(Pos_gle_base):
    """
    Memory extraction using finite element basis.
    Finite element are implement using scikit-fem.
    Method "trapz" and "second_kind" are not implemented
    """

    def __init__(self, xva_arg, basis: Basis, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0, **kwargs):
        """
        Using Finite element basis functions.

        Parameters
        ----------
        xva_arg : xarray dataset (['time', 'x', 'v', 'a']) or list of datasets.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a xarray timeseries
            or a listlike collection of them.
        basis :  scikitfem Basis class
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
        with_const : bool, default=False
            Whatever the constant function is in the basis (to avoid inversion issue with the derivative).
        """
        self._do_check(xva_arg)  # Do some check on the trajectories
        self.basis = basis
        self.N_basis_elt = self.basis.N

        self.saveall = saveall
        self.prefix = prefix
        self.verbose = verbose
        self.kT = kT
        # filenames
        self.corrsfile = "corrs.txt"
        self.corrsdxfile = "a-u_corrs.txt"
        self.dcorrsfile = "dE_corrs.txt"
        self.dcorrsdxfile = "dE_a-u_corrs.txt"
        self.interpfefile = "interp-fe.txt"
        self.histfile = "fe-hist.txt"
        self.kernelfile = "kernel.txt"

        self.bkbkcorrw = None
        self.bkdxcorrw = None
        self.force_coeff = None

        if self.xva_list is None:
            return

        self.dim_x = self.xva_list[0].dims["dim_x"]

        if self.dim_x != self.basis.elem.dim:
            raise ValueError("Dimension of element different of dimension of the observable")

        # processing input arguments
        self.weights = np.array([xva["time"].shape[0] for xva in self.xva_list], dtype=int)  # Should be the various lenght of trajectory
        self.weightsum = np.sum(self.weights)
        if self.verbose:
            print("Found trajectories with the following lengths:")
            print(self.weights)

        lastinds = np.array([xva["time"][-1] for xva in self.xva_list])
        smallest = np.min(lastinds)
        if smallest < trunc:
            if self.verbose:
                print("Warning: Found a trajectory shorter than " "the argument trunc. Override.")
            trunc = smallest
        # Find index of the time truncation
        print(trunc)
        self.trunc_ind = (self.xva_list[0]["time"] <= trunc).sum().data
        if self.verbose:
            print("Trajectories are truncated at lenght {} for dynamic analysis".format(self.trunc_ind))

    def _do_check(self, xva_arg):
        if xva_arg is not None:
            if isinstance(xva_arg, xr.Dataset):
                self.xva_list = [xva_arg]
            else:
                self.xva_list = xva_arg
            for xva in self.xva_list:
                for col in ["x", "v", "a"]:
                    if col not in xva.data_vars:
                        raise Exception("Please provide txva dataset, " "or an iterable collection (i.e. list) " "of txva dataset.")
                if "elem" not in xva.data_vars:
                    raise Exception("Please compute element location.")
                if "time" not in xva.dims:
                    raise Exception("Time is not a coordinate. Please provide txva dataset, " "or an iterable collection (i.e. list) " "of txva dataset.")
                if "dt" not in xva.attrs:
                    raise Exception("Timestep not in dataset attrs")
        else:
            self.xva_list = None

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
            for k in range(self.basis.nelems):  # loop over element
                if (xva["elem"] == k).any():  # If no data don't proceed
                    E, dofs = self.basis_vector(xva[xva["elem"] == k], elem=k, compute_for="force")  # To check xva[xva["elem"] == k]
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
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            for k in range(self.basis.nelems):  # loop over element
                if (xva["elem"] == k).any():  # If no data don't proceed
                    E, dofs = self.basis_vector(xva[xva["elem"] == k], elem=k, compute_for="force")  # To check xva[xva["elem"] == k]
                    avg_gram[dofs, dofs] += np.matmul(E.T, E) / self.weightsum
                    avg_disp[dofs] += np.matmul(E.T, xva["a"].data) / self.weightsum  # Change to einsum to use vectorial value of E
        self.invgram = self.kT * np.linalg.pinv(avg_gram)
        self.force_coeff = np.matmul(np.linalg.inv(avg_gram), avg_disp)

    def compute_corrs(self):
        """
        Compute correlation functions. When large is true, it use a slower way to compute correlation that is less demanding in memory
        """
        if self.verbose:
            print("Calculate correlation functions...")
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")

        self.bkbkcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.N_basis_elt_kernel))
        self.bkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.dim_x))

        for weight, xva in zip(self.weights, self.xva_list):
            for k in range(self.basis.nelems):  # loop over element
                if (xva["elem"] == k).any():  # If no data don't proceed
                    E_force, E, _, dofs = self.basis_vector(xva)
                    print(E_force.shape, E.shape)
                    force = np.matmul(E_force, self.force_coeff)  # A changer on veut quelque chose du genre self.force_coeff[dofs]
                    self.bkdxcorrw += weight * correlation_ND(E, (xva["a"].data - force), trunc=self.trunc_ind)  # Change correlation ND to use vectorial value of E and xva["a"]
                    self.bkbkcorrw += weight * correlation_ND(E, trunc=self.trunc_ind)

        self.bkbkcorrw /= self.weightsum
        self.bkdxcorrw /= self.weightsum
        self.dotbkdxcorrw /= self.weightsum
        self.dotbkbkcorrw /= self.weightsum

        if self.saveall:
            np.savetxt(self.prefix + self.corrsdxfile, self.bkdxcorrw.reshape(-1, self.N_basis_elt_kernel * self.dim_x))
            np.savetxt(self.prefix + self.corrsfile, self.bkbkcorrw.reshape(-1, self.N_basis_elt_kernel ** 2))
            np.savetxt(self.prefix + self.dcorrsdxfile, self.dotbkdxcorrw.reshape(-1, self.N_basis_elt_kernel * self.dim_x))
            np.savetxt(self.prefix + self.dcorrsfile, self.dotbkbkcorrw.reshape(-1, self.N_basis_elt_kernel ** 2))

    def dU(self, x):
        """
        Evaluate the force at given points x
        """
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        E = self.basis_vector(xr.Dataset({"x": (["time", "dim_x"], np.asarray(x).reshape(-1, self.dim_x))}), compute_for="force")
        return -1 * np.einsum("ik,kl->il", E, self.force_coeff)  # Return the force as array (nb of evalution point x dim_x)

    def kernel_eval(self, x):
        """
        Evaluate the kernel at given points x
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        E = self.basis_vector(xr.Dataset({"x": (["time", "dim_x"], np.asarray(x).reshape(-1, self.dim_x))}), compute_for="kernel")
        return self.time, np.einsum("jkd,ikl->ijld", E, self.kernel)  # Return the kernal as array (time x nb of evalution point x dim_x x dim_x)


class Pos_gle_fem(Pos_gle_fem_base):
    """
    The main class for the position dependent memory extraction,
    holding all data and the extracted memory kernels.
    """

    def __init__(self, xva_arg, basis, saveall=True, prefix="", verbose=True, kT=2.494, trunc=1.0, with_const=False):
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
        with_const : bool, default=False
            Whatever the constant function is in the basis (to avoid inversion issue with the derivative).
        """
        Pos_gle_fem_base.__init__(self, xva_arg, basis, saveall, prefix, verbose, kT, trunc)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt - int(with_const) * self.dim_x
        self.remove_const_ = bool(with_const)

    def basis_vector(self, xva, elem, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        """
        bk = np.zeros((xva.shape[0], self.basis.Nbfun))  # Check dimension should we add self.dim_x?
        dbk = np.zeros((xva.shape[0], self.basis.Nbfun, self.dim_x))  # Check dimension
        dofs = np.zeros(self.basis.Nbfun)
        loc_value_t = self.basis.mapping.invF(xva["x"].data.reshape(self.dim_x, 1, -1), tind=slice(elem, elem + 1))  # Reshape into dim * 1 element * nb of point
        for i in range(self.basis.Nbfun):
            # To use global basis fct instead (probably more general)
            # phi_tp = ubasis.elem.gbasis(ubasis.mapping, loc_value_tp[:, 0, :], j, tind=slice(m, m + 1)).value
            # Check to use gbasis instead
            phi_field = self.basis.elem.lbasis(loc_value_t[:, 0, :], i)
            bk[:, i] = phi_field.value  # The middle indice is the choice of element, ie only one choice here
            # dbk via grad?
            dofs[i] = self.basis.element_dofs[i, elem]
        # if self.include_const:
        #     bk = np.concatenate((np.ones((bk.shape[0], 1)), bk), axis=1)
        if compute_for == "force":
            return bk, dofs
        dbk = self.basis.deriv(xva["x"].data, remove_const=self.remove_const_)
        if compute_for == "kernel":  # For kernel evaluation
            return dbk, dofs
        elif compute_for == "corrs":
            E = np.einsum("nld,nd->nl", dbk, xva["v"].data)
            return bk, E, None, dofs
        else:
            raise ValueError("Basis evaluation goal not specified")

    def compute_noise(self, xva, trunc_kernel=None):
        """
        From a trajectory get the noise.

        Parameters
        ----------
        xva : xarray dataset (['time', 'x', 'v', 'a']) .
            Use compute_va() or see its output for format details.
            Input trajectory to compute noise.
        trunc_kernel : int
                Number of datapoint of the kernel to consider.
                Can be used to remove unphysical divergence of the kernel or shortten execution time.
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
        return time, xva["a"].data - force - memory, xva["a"].data, force, memory