import numpy as np
import xarray as xr
import warnings
from scipy.integrate import simpson
from .basis import describe_from_dim

from .fkernel import memory_rect, memory_trapz, corrs_rect, corrs_trapz
from .fkernel import rect_integral, trapz_integral, simpson_integral
from .fkernel import solve_ide_rect, solve_ide_trapz, solve_ide_trapz_stab


def _convert_input_array_for_evaluation(array, dim_x):
    """
    Take input and return xarray Dataset with correct shape
    """
    if isinstance(array, xr.Dataset):  # TODO add check on dimension of array
        return array
    else:
        x = np.asarray(array).reshape(-1, dim_x)
        return xr.Dataset({"x": (["time", "dim_x"], x)})


def matmulPrange(P_range, E):
    """
    Reduce basis size when needed
    """
    # P_range = xr.DataArray(P_mat, dims=["dim_basis", "dim_basis_old"])
    return xr.dot(P_range, E.rename({"dim_basis": "dim_basis_old"}))


class ModelBase(object):
    """
    The base class for holding the model
    """

    set_of_obs = ["x", "v"]

    def __init__(self, basis, dt, dim_x=1, dim_obs=1, trunc_ind=-1, L_obs="a", describe_data=None, **kwargs):
        """
        Create an instance of the Pos_gle class.

        Parameters
        ----------
        basis :  scikit-learn transformer to get the element of the basis
                This class should implement, basis() and deriv() function
                and deal with periodicity of the data.
                If a fit() method is defined, it will be called at initialization
        trunc : float, default=1.0
            Truncate all correlation functions and the memory kernel after this
            time value.
        L_obs: str, default= "a"
            Name of the column containing the time derivative of the observable
        """
        self.L_obs = L_obs

        self.dim_x = dim_x
        self.dim_obs = dim_obs

        self.dt = dt
        self.trunc_ind = int(trunc_ind)

        self._check_basis(basis, describe_data)  # Do check on basis

        self.force_coeff = None
        self.kernel = None

        self.inv_mass_coeff = None
        self.eff_mass = None

        self.gram_force = None
        self.gram_kernel = None

        self.method = None

        self.rank_projection = False
        self.P_range = None

    def _check_basis(self, basis, describe_data=None):
        """
        Simple checks on the basis class
        """
        if not (callable(getattr(basis, "basis", None))) or not (callable(getattr(basis, "deriv", None))):
            raise Exception("Basis class do not define basis() or deriv() method")
        self.basis = basis
        if not hasattr(self.basis, "n_output_features_"):
            if callable(getattr(self.basis, "fit", None)):
                if describe_data is None:
                    describe_data = describe_from_dim(self.dim_x)
                self.basis = self.basis.fit(describe_data)  # Fit basis with minimal information
            else:
                raise Exception("Basis class do have fit() method or do not expose n_output_features_ attribute")

        self.N_basis_elt = self.basis.n_output_features_

    def setup_basis(self, describe_data):
        """
        We can also fit model from data
        """
        self.dim_x = describe_data.mean.shape[0]

        if callable(getattr(self.basis, "fit", None)):
            if describe_data is None:
                describe_data = describe_from_dim(self.dim_x)
            self.basis = self.basis.fit(describe_data)  # Fit basis with minimal information
        else:
            raise Exception("Basis class do have fit() method or do not expose n_output_features_ attribute")

        self.N_basis_elt = self.basis.n_output_features_

    def _set_range_projection(self, rank_tol, B0):
        """
        Set and perfom the projection onto the range of the basis for kernel
        """
        # Check actual rank of the matrix
        # Do SVD
        U, S, V = np.linalg.svd(B0, compute_uv=True, hermitian=True)
        # Compute rank from svd
        if rank_tol is None:
            rank_tol = S.max(axis=-1, keepdims=True) * max(B0.shape[-2:]) * np.finfo(S.dtype).eps
        rank = np.count_nonzero(S > rank_tol, axis=-1)
        # print(rank, U.shape, S.shape, V.shape)
        if rank < self.N_basis_elt_kernel:
            if rank != self.N_basis_elt_kernel - self.dim_x:
                print("Warning: rank is different than expected. Current {}, Expected {}. Consider checking your basis or changing the tolerance".format(rank, self.N_basis_elt_kernel - self.dim_x))
            elif rank == 0:
                raise Exception("Rank of basis is null.")
            # Construct projection
            P_range = U[:, :rank].T  # # Save the matrix for future use, matrix is rank x N_basis_elt_kernel
            # Faster to do one product in order
        else:
            print("No projection onto the range of the basis performed as basis is not deficient.")
            P_range = np.identity(self.N_basis_elt_kernel)
        self.P_range = xr.DataArray(P_range, dims=["dim_basis", "dim_basis_old"])
        return self.P_range

    def basis_vector(self, xva, compute_for="corrs"):
        """
        From one trajectory compute the basis element.
        This is the main method that should be implemented by children class.
        It take as argument a trajectory and should return the value of the basis function depending of the wanted case.
        There is three case that should be implemented.

        "force": for the evaluation and computation of the mean force.

        "pmf": for evaluation of the pmf using integration of the mean force

        "kernel": for the evaluation of the kernel.

        "corrs": for the computation of the correlation function.
        """
        raise NotImplementedError

    def set_zero_force(self):  # TODO
        self.force_coeff = np.zeros((self.N_basis_elt_force, self.dim_obs))

    def compute_noise(self, xva, trunc_kernel=None, start_point=0, end_point=None):
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
        time_0 = xva["time"].data[0]
        xva = xva.isel(time=slice(start_point, end_point))
        time = xva["time"].data - time_0
        dt = time[1] - time[0]
        E_force, E, _ = self.basis_vector(xva)
        if self.rank_projection:
            E = matmulPrange(self.P_range, E)
        force = xr.dot(E_force, self.force_coeff, dims=["dim_basis", "dim_x"])
        if self.method in ["rect", "rectangular", "second_kind_rect"] or self.method is None:
            memory = memory_rect(self.kernel[:trunc_kernel], E, dt)
        elif self.method == "trapz" or self.method == "second_kind_trapz":
            memory = memory_trapz(self.kernel[:trunc_kernel], E, dt)
        else:
            raise ValueError("Cannot compute noise when kernel computed with method {}".format(self.method))
        return time, xva[self.L_obs] - force - memory, xva[self.L_obs], force, memory

    def compute_corrs_w_noise(self, xva, left_op=None):
        """
        Compute correlation between noise and left_op

        Parameters
        ----------
        xva : xarray dataset (['time', 'x', 'v', 'a']) .
            Use compute_va() or see its output for format details.
            Input trajectory to compute noise.
        """
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")

        dt = xva["time"].data[1] - xva["time"].data[0]
        E_force, E, _ = self.basis_vector(xva)
        if self.rank_projection:
            E = matmulPrange(self.P_range, E)

        noise = (xva[self.L_obs] - xr.dot(E_force, self.force_coeff)).to_numpy()

        if left_op is None:
            left_op_dat = noise
        elif isinstance(left_op, str):
            left_op_dat = xva[left_op]
        elif callable(left_op):
            left_op_dat = left_op(xva)
        elif isinstance(left_op, np.ndarray) or isinstance(left_op, xr.DataArray):
            left_op_dat = left_op

        if self.method in ["rect", "rectangular", "second_kind_rect"] or self.method is None:
            return self.kernel["time_kernel"], corrs_rect(noise, self.kernel, E, left_op_dat, dt)
        elif self.method == "trapz" or self.method == "second_kind_trapz":
            return self.kernel["time_kernel"][:-1], corrs_trapz(noise, self.kernel, E, left_op_dat, dt)
        else:
            raise ValueError("Cannot compute noise when kernel computed with method {}".format(self.method))

    def force_eval(self, x, coeffs=None):
        """
        Evaluate the force at given points x.
        If coeffs is given, use provided coefficients instead of the force
        """
        if coeffs is None:
            if self.force_coeff is None:
                raise Exception("Mean force has not been computed.")
            coeffs = self.force_coeff
        else:  # Check shape
            if coeffs.shape != (self.N_basis_elt_force, self.dim_obs):
                warnings.warn("Wrong shape of the coefficients. Get {} but expect {}.".format(coeffs.shape, (self.N_basis_elt_force, self.dim_obs)))
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="force")
        return xr.dot(E, coeffs).to_numpy()  # Return the force as array (nb of evalution point x dim_obs)

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
                warnings.warn("Wrong shape of the coefficients. Get {} but expect {}.".format(coeffs.shape, (self.N_basis_elt_force, self.dim_obs)))
        if self.eff_mass is None:
            warnings.warn("Effective mass has not been computed, use effetive mass of 1.0")
            self.eff_mass = np.identity(self.dim_x)
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="pmf")
        pmf = -1 * xr.dot(E, coeffs, self.eff_mass).to_numpy() / kT
        return pmf - float(set_zero) * np.min(pmf)

    def inv_mass_eval(self, x, coeffs=None, set_zero=True):
        """
        Compute free energy via integration of the mean force at points x.
        This assume that the effective mass is independent of the position.
        If coeffs is given, use provided coefficients instead of the force coefficients.
        """
        if coeffs is None:
            if self.inv_mass_coeff is None:
                raise Exception("Effective mass has not been computed.")
            coeffs = self.inv_mass_coeff
        else:  # Check shape
            if coeffs.shape != (self.N_basis_elt_force, self.dim_x, self.dim_x):
                raise Exception("Wrong shape of the coefficients. Get {} but expect {}.".format(coeffs.shape, (self.N_basis_elt_force, self.dim_x, self.dim_x)))
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="force")
        return xr.dot(E, coeffs).to_numpy()  # Return the force as array (nb of evalution point x dim_obs x dim_obs)

    def pmf_num_int_eval(self, x, kT=1.0, set_zero=True):
        """
        Compute free energy via integration of the mean force at points x.
        This take into accound the position dependent mass, but the integration is numeric
        """
        force = self.force_eval(x)[:, 0]
        inv_mass = self.inv_mass_eval(x)[:, 0, 0]
        x = np.asarray(x).ravel()
        dx = x[1] - x[0]

        diff_mass = np.gradient(inv_mass) / dx  # Numerical derivative
        grad_pmf = -1 * (force - diff_mass / kT) / inv_mass
        pmf = np.cumsum(grad_pmf) * dx

        return pmf - float(set_zero) * np.min(pmf)

    def kernel_eval(self, x, coeffs_ker=None):
        """
        Evaluate the kernel at given points x.
        If coeffs_ker is given, use provided coefficients instead of the kernel
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        if coeffs_ker is None:
            coeffs_ker = self.kernel
        else:  # Check shape
            if coeffs_ker.shape[1:] != self.kernel.shape[1:]:
                raise Exception("Wrong shape of the coefficients. Get {} but expect {}.".format(coeffs_ker.shape[1:], self.kernel.shape[1:]))

        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="kernel")
        if self.rank_projection:
            E = matmulPrange(self.P_range, E)
        return xr.dot(coeffs_ker, E.rename({"dim_x": "dim_x'"}))

    def evolve_volterra(self, G0, lenTraj, method="trapz", trunc_ind=None):
        """
        Evolve in time the integro-differential equation.
        This assume that the GLE is a linear GLE (i.e. the set of basis function is on the left and right of the equality)

        Parameters
        ----------
        G0 : array
            Initial value of the correlation
        lenTraj : int
            Length of the time evolution
        method : str, default="trapz"
            Method that is used to discretize the continuous Volterra equations
        trunc_ind: int, default= self.trunc_ind
            Truncate the length of the memory to this value
        """
        raise NotImplementedError

    def flux_from_volterra(self, corrs_force, corrs_kernel=None, force_coeff=None, kernel=None, method="trapz", trunc_ind=None):
        """
        From a solution of the Volterra equation, compute the flux term.
        That allow to compute decomposition of the flux
        """
        if corrs_kernel is None:
            corrs_kernel = corrs_force
        if corrs_force.shape[-1] != corrs_kernel.shape[-1]:
            print("# WARNING: Different lenght in time for correlation")
        if force_coeff is None:
            force_coeff = self.force_coeff.to_numpy()
        else:
            force_coeff = np.asarray(force_coeff)
        if kernel is None:
            kernel = self.kernel.to_numpy()
        else:
            kernel = np.asarray(kernel)
        if trunc_ind is not None:
            kernel = kernel[:trunc_ind, :, :]
        force_term = np.matmul(corrs_force, force_coeff)
        res_int = np.zeros((corrs_kernel.shape[-1], corrs_kernel.shape[1], kernel.shape[-1]))
        for n in range(1, corrs_kernel.shape[-1]):
            max_len = min(n, kernel.shape[0])
            if method == "rect":
                res_int[n, :] = -1 * rect_integral(self.dt, corrs_kernel[:, :, n - max_len : n], kernel[:max_len, :, :])
            elif method == "trapz":
                res_int[n, :] = -1 * trapz_integral(self.dt, corrs_kernel[:, :, n - max_len : n], kernel[:max_len, :, :])
            elif method == "simpson":
                res_int[n, :] = -1 * simpson_integral(self.dt, corrs_kernel[:, :, n - max_len : n], kernel[:max_len, :, :])

        return force_term, res_int

    def laplace_transform_kernel(self, s_start=0.0, s_end=None, n_points=None):
        """
        Compute the Laplace transform of the kernel matrix
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        if n_points is None:
            n_points = self.trunc_ind
        if s_end is None:
            s_end = 1.0 / self.dt
        # mintimelenght = self.trunc_ind * dt
        s_range = np.linspace(s_start, s_end, n_points)
        laplace = np.zeros((n_points, self.kernel.shape[1], self.kernel.shape[2]))
        for n, s in enumerate(s_range):
            laplace[n, :, :] = simpson(np.einsum("i,ijk-> ijk", np.exp(-s * self.kernel["time"][:, 0]), self.kernel), self.kernel["time"][:, 0], axis=0)
        return s_range, laplace

    def save_model(self):
        """
        Return DataSet version of the model than can be save to file
        """
        # On doit retourner coeffs de la force, le noyau mémoire, et de quoi générer la base
        coeffs = xr.Dataset(attrs={"dt": self.dt, "trunc_ind": self.trunc_ind, "L_obs": self.L_obs, "dim_x": self.dim_x, "dim_obs": self.dim_obs})  # Ajouter quelques attributs
        if self.force_coeff is not None:
            coeffs.update({"force_coeff": self.force_coeff.rename({"dim_basis": "dim_basis_force"})})
        if self.gram_force is not None:
            coeffs.update({"gram_force": self.gram_force.rename({"dim_basis": "dim_basis_force", "dim_basis'": "dim_basis_force'"})})
        for key, dat in self.__dict__.items():
            if key not in coeffs.attrs and key not in ["basis", "force_coeff", "gram_force", "N_basis_elt_force", "N_basis_elt_kernel"]:  # Eclude some vaiable
                if dat is not None:
                    coeffs.update({key: dat})

        return coeffs

    @classmethod
    def load_model(cls, basis, coeffs, **kwargs):
        """
        Create a model from a save
        """
        # A partir des attibuts du DataSet construire un dictionnaire des arguments à donner
        # kwargs = .. dim_x, dim_obs, trunc_ind, L_obs, describe_data
        cls_attrs = coeffs.attrs
        for key, value in kwargs.items():
            cls_attrs[key] = value
        model = cls(basis, **cls_attrs)

        for key in ["force_coeff", "kernel", "rank_projection", "P_range", "gram_force"]:
            if key in coeffs:
                setattr(model, key, coeffs[key])
        if model.force_coeff is not None:
            model.force_coeff = model.force_coeff.rename({"dim_basis_force": "dim_basis"})
        if model.gram_force is not None:
            model.gram_force = model.gram_force.rename({"dim_basis_force": "dim_basis", "dim_basis_force'": "dim_basis'"})

        return model


class Pos_gle(ModelBase):
    """
    The main class for the position dependent memory extraction,
    holding all data and the extracted memory kernels.
    """

    set_of_obs = ["x", "v", "a"]

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
        ModelBase.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt - int(self.basis.const_removed) * self.dim_x
        self.rank_projection = not self.basis.const_removed

    def basis_vector(self, xva, compute_for="corrs"):

        bk = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        if compute_for == "force":
            return bk
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        dbk = xr.apply_ufunc(self.basis.deriv, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x"]], dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt_kernel, "dim_x": self.dim_x}}, dask="parallelized")
        if compute_for == "kernel":  # For kernel evaluation
            return dbk
        elif compute_for == "corrs":
            ddbk = xr.apply_ufunc(self.basis.hessian, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x", "dim_x'"]], dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt_kernel, "dim_x": self.dim_x, "dim_x'": self.dim_x}}, dask="parallelized")
            E = xr.dot(dbk, xva["v"], dims=["dim_x"])
            dE = xr.dot(dbk, xva["a"], dims=["dim_x"]) + xr.dot(ddbk, xva["v"], xva["v"].rename({"dim_x": "dim_x'"}), dims=["dim_x", "dim_x'"])
            return bk, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")


class Pos_gle_with_friction(Pos_gle):
    """
    A derived class in which we don't enforce zero instantaneous friction
    """

    set_of_obs = ["x", "v", "a"]

    def __init__(self, *args, **kwargs):
        Pos_gle.__init__(self, *args, **kwargs)
        if not self.basis.const_removed:
            print("Warning: The model cannot deal with dask array if the constant is not removed.")
        self.N_basis_elt_force = 2 * self.N_basis_elt - int(self.basis.const_removed) * self.dim_x
        self.N_basis_elt_kernel = self.N_basis_elt - int(self.basis.const_removed) * self.dim_x
        self.rank_projection = not self.basis.const_removed

    def basis_vector(self, xva, compute_for="corrs"):
        # We have to deal with the multidimensionnal case as well
        bk = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        if compute_for == "force_eval":
            return bk
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        dbk = xr.apply_ufunc(self.basis.deriv, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x"]], dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt_kernel, "dim_x": self.dim_x}}, dask="parallelized")
        if compute_for == "kernel":  # To have the term proportional to velocity
            return dbk

        Evel = xr.dot(dbk, xva["v"], dims=["dim_x"])
        E = xr.concat([bk, Evel], dim="dim_basis")
        if compute_for == "force":
            return E
        elif compute_for == "corrs":
            ddbk = xr.apply_ufunc(self.basis.hessian, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x", "dim_x'"]], dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt_kernel, "dim_x": self.dim_x, "dim_x'": self.dim_x}}, dask="parallelized")
            dE = xr.dot(dbk, xva["a"], dims=["dim_x"]) + xr.dot(ddbk, xva["v"], xva["v"].rename({"dim_x": "dim_x'"}), dims=["dim_x", "dim_x'"])
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


class Pos_gle_no_vel_basis(Pos_gle):
    """
    Use basis function dependent of the position only
    """

    set_of_obs = ["x", "v"]

    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")

    def basis_vector(self, xva, compute_for="corrs"):
        E = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        if compute_for == "force":
            return E
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        elif compute_for == "kernel":
            # Extend the basis for multidim value
            return E.expand_dims({"dim_x": self.dim_x}, axis=-1)
        elif compute_for == "corrs":
            dbk = xr.apply_ufunc(self.basis.deriv, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x"]], dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt_kernel, "dim_x": self.dim_x}}, dask="parallelized")
            dE = xr.dot(dbk, xva["v"], dims=["dim_x"])
            return E, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")


class Pos_gle_const_kernel(Pos_gle):
    """
    A derived class in which we the kernel is computed independent of the position
    """

    set_of_obs = ["x", "v", "a"]

    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.dim_obs

    def basis_vector(self, xva, compute_for="corrs"):
        bk = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        if compute_for == "force":
            return bk
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        if compute_for == "kernel":  # For kernel evaluation
            grad = np.zeros((bk.shape[0], self.dim_obs, self.dim_obs))
            for i in range(self.dim_obs):
                grad[:, i, i] = 1.0
            return xr.DataArray(grad, dims=("time", "dim_basis", "dim_x"))
        elif compute_for == "corrs":
            return bk, xva["v"].rename({"dim_x": "dim_basis"}), xva["a"].rename({"dim_x": "dim_basis"})
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


class Pos_gle_hybrid(Pos_gle):
    """
    Implement the hybrid projector of arXiv:2202.01922
    """

    set_of_obs = ["x", "v", "a"]

    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt + 1
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")

    def basis_vector(self, xva, compute_for="corrs"):
        # We have to deal with the multidimensionnal case as well
        bk = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        if compute_for == "force":
            return bk
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        elif compute_for == "kernel":
            # Extend the basis for multidim value
            return bk.expand_dims({"dim_x": self.dim_x}, axis=-1)
            # return bk.reshape(-1, self.N_basis_elt_kernel - 1, 1)
        elif compute_for == "corrs":
            E = xr.concat([xva["v"].rename({"dim_x": "dim_basis"}), bk], dim="dim_basis")
            dbk = xr.dot(xr.apply_ufunc(self.basis.deriv, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x"]], dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt, "dim_x": self.dim_x}}, dask="parallelized"), xva["v"])
            dE = xr.concat([xva["a"].rename({"dim_x": "dim_basis"}), dbk], dim="dim_basis")  # To test
            return bk, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")

    def get_const_kernel_part(self):
        """
        Return the position independent part of the kernel
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        return self.kernel.sel(dim_basis=0)

    def kernel_eval(self, x):
        """
        Evaluate the position dependant part of the kernel at given points x
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="kernel")
        if self.rank_projection:
            E = matmulPrange(self.P_range, E)
        return xr.dot(self.kernel.sel(dim_basis=slice(1, None)), E.rename({"dim_x": "dim_x'"}))
        # return self.kernel["time_kernel"], np.einsum("jkd,ikl->ijld", E, self.kernel[:, 1:, :])  # Return the kernel as array (time x nb of evalution point x dim_obs x dim_x)


class Pos_gle_overdamped(ModelBase):
    """
    Extraction of position dependent memory kernel for overdamped dynamics.
    """

    set_of_obs = ["x", "v"]

    def __init__(self, *args, L_obs="v", rank_projection=False, **kwargs):
        ModelBase.__init__(self, *args, L_obs=L_obs, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")
        self.rank_projection = rank_projection

    def basis_vector(self, xva, compute_for="corrs"):
        E = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        if compute_for == "force":
            return E
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt}}, dask="parallelized")
        elif compute_for == "kernel":
            return E.expand_dims({"dim_x": self.dim_x}, axis=-1)
            # return E.reshape(-1, self.N_basis_elt_kernel, self.dim_x)
        elif compute_for == "corrs":
            dbk = xr.apply_ufunc(self.basis.deriv, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x"]], dask_gufunc_kwargs={"output_sizes": {"dim_basis": self.N_basis_elt_kernel, "dim_x": self.dim_x}}, dask="parallelized")
            dE = xr.dot(dbk, xva["v"], dims=["dim_x"])
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

    def evolve_volterra(self, G0, lenTraj, method="trapz", trunc_ind=None):
        """
        Evolve in time the integro-differential equation.
        This assume that the GLE is a linear GLE (i.e. the set of basis function is on the left and right of the equality)

        Parameters
        ----------
        G0 : array
            Initial value of the correlation
        lenTraj : int
            Length of the time evolution
        method : str, default="trapz"
            Method that is used to discretize the continuous Volterra equations
        trunc_ind: int, default= self.trunc_ind
            Truncate the length of the memory to this value
        """
        if self.force_coeff.shape[0] != self.force_coeff.shape[1] or self.kernel.shape[1] != self.kernel.shape[2]:
            raise ValueError("Cannot evolve volterra equation if the coefficients are not square")

        if G0.shape[0] != self.kernel.shape[-1]:
            raise ValueError("Wrong shape for initial value")

        if trunc_ind is None or trunc_ind <= 0:
            trunc_ind = self.kernel.shape[0]

        G0 = np.asarray(G0)
        coeffs_force = self.force_coeff.to_numpy()
        coeffs_ker = self.kernel.to_numpy()[:trunc_ind, :, :]
        # TODO : A réimplémenter en fortran avec les coeffs en xarray
        # Modifier le code fortran pour séparer l'évolution en temps du calcul de la dérivée ce qui sera plus pratique pour le calcul du flux
        if method == "rect":
            res = solve_ide_rect(coeffs_ker, G0, coeffs_force, lenTraj, self.dt)  # TODO it might worth transpose all the code for the kernel
        elif method == "trapz":
            res = solve_ide_trapz(coeffs_ker, G0, coeffs_force, lenTraj, self.dt)
        elif method == "trapz_stab":
            res = solve_ide_trapz_stab(coeffs_ker, G0, coeffs_force, lenTraj, self.dt)
        return np.arange(lenTraj) * self.dt, res


# class Pos_gle_overdamped_const_kernel(ModelBase):
#     """
#     Extraction of position dependent memory kernel for overdamped dynamics
#     using position-independent kernel.
#     """
#
#     set_of_obs = ["x", "v"]
#
#     def __init__(self, *args, L_obs="v", **kwargs):
#         ModelBase.__init__(self, *args, L_obs=L_obs, **kwargs)
#         self.N_basis_elt_force = self.N_basis_elt
#         self.N_basis_elt_kernel = self.dim_obs
#
#     def basis_vector(self, xva, compute_for="corrs"):
#         E = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="parallelized")
#         if compute_for == "force":
#             return E
#         elif compute_for == "pmf":
#             return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="parallelized")
#         elif compute_for == "kernel":
#             return xr.DataArray(np.ones((E.shape[0], self.N_basis_elt_kernel, self.dim_x)), dims=("time", "dim_basis", "dim_x"))
#         elif compute_for == "corrs":
#             return E, xr.DataArray(np.ones((E.shape[0], self.dim_obs)), dims=("time", "dim_basis")), xva["v"]
#         else:
#             raise ValueError("Basis evaluation goal not specified")
