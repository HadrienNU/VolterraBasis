import numpy as np
import xarray as xr
from scipy.integrate import trapezoid, simpson
from .basis import describe_from_dim


def _convert_input_array_for_evaluation(array, dim_x):
    """
    Take input and return xarray Dataset with correct shape
    """
    if isinstance(array, xr.Dataset):  # TODO add check on dimension of array
        return array
    else:
        x = np.asarray(array).reshape(-1, dim_x)
        return xr.Dataset({"x": (["time", "dim_x"], x)})


class ModelBase(object):
    """
    The base class for holding the model
    """

    def __init__(self, basis, dt, dim_x=1, dim_obs=1, trunc_ind=1.0, L_obs="a", **kwargs):
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
        self._check_basis(basis)  # Do check on basis

        self.bkbkcorrw = None
        self.bkdxcorrw = None
        self.dotbkdxcorrw = None
        self.dotbkbkcorrw = None
        self.force_coeff = None

        self.inv_mass_coeff = None
        self.eff_mass = None
        self.kernel_gram = None

        self.method = None

        self.rank_projection = False
        self.P_range = None

        self.dim_x = dim_x
        self.dim_obs = dim_obs

        self.trunc_ind = trunc_ind

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

    def fit(self, describe_data):
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

    def _set_range_projection(self, rank_tol):
        """
        Set and perfom the projection onto the range of the basis for kernel
        """
        if self.verbose:
            print("Projection on range space...")
        # Check actual rank of the matrix
        B0 = self.bkbkcorrw[0, :, :]
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
            self.P_range = U[:, :rank].T  # # Save the matrix for future use, matrix is rank x N_basis_elt_kernel
            # Faster to do one product in order
            tempbkbkcorrw = np.einsum("ijk,mk->ijm", self.bkbkcorrw, self.P_range)
            self.bkbkcorrw = np.einsum("lj,ijk->ilk", self.P_range, tempbkbkcorrw)
            # self.bkbkcorrw = np.einsum("lj,ijk,mk->ilm", self.P_range, self.bkbkcorrw, self.P_range)
            self.bkdxcorrw = np.einsum("kj,ijd->ikd", self.P_range, self.bkdxcorrw)
            if self.dotbkdxcorrw is not None:
                self.dotbkdxcorrw = np.einsum("kj,ijd->ikd", self.P_range, self.dotbkdxcorrw)
            if self.dotbkbkcorrw is not None:
                self.dotbkbkcorrw = np.einsum("lj,ijk,mk->ilm", self.P_range, self.dotbkbkcorrw, self.P_range)
        else:
            print("No projection onto the range of the basis performed as basis is not deficient.")
            self.P_range = np.identity(self.N_basis_elt_kernel)

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

    def set_zero_force(self):
        self.force_coeff = np.zeros((self.N_basis_elt_force, self.dim_obs))

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
                raise Exception("Wrong shape of the coefficients. Get {} but expect {}.".format(coeffs.shape, (self.N_basis_elt_force, self.dim_obs)))
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="force")
        return np.einsum("ik,kl->il", E, coeffs)  # Return the force as array (nb of evalution point x dim_obs)

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
        if self.eff_mass is None:
            self.compute_effective_mass(kT=kT)
        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="pmf")
        pmf = -1 * np.einsum("ik,kl->il", E, np.matmul(coeffs, self.eff_mass)) / kT
        return pmf - float(set_zero) * np.min(pmf)

    def inv_mass_eval(self, x, coeffs=None, kT=1.0, set_zero=True):
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
        return np.einsum("ik,kld->ild", E, coeffs)  # Return the force as array (nb of evalution point x dim_obs x dim_obs)

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
            if coeffs_ker.shape != self.kernel.shape:
                raise Exception("Wrong shape of the coefficients. Get {} but expect {}.".format(coeffs_ker.shape, self.kernel.shape))

        E = self.basis_vector(_convert_input_array_for_evaluation(x, self.dim_x), compute_for="kernel")
        if self.rank_projection:
            E = np.einsum("kj,ijd->ikd", self.P_range, E)
        return self.time, np.einsum("jkd,ikl->ijld", E, coeffs_ker)  # Return the kernel as array (time x nb of evalution point x dim_obs x dim_x)

    def laplace_transform_kernel(self, n_points=None):
        """
        Compute the Laplace transform of the kernel matrix
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        if n_points is None:
            n_points = self.trunc_ind
        dt = self.dt
        # mintimelenght = self.trunc_ind * dt
        s_range = np.linspace(0.0, 1.0 / dt, n_points)
        laplace = np.zeros((n_points, self.kernel.shape[1], self.kernel.shape[2]))
        for n, s in enumerate(s_range):
            laplace[n, :, :] = simpson(np.einsum("i,ijk-> ijk", np.exp(-s * self.time[:, 0]), self.kernel), self.time[:, 0], axis=0)
        return s_range, laplace

    def check_volterra_inversion(self):
        """
        For checking if the volterra equation is correctly inversed
        Compute the integral in volterra equation using trapezoidal rule
        """
        if self.kernel is None:
            raise Exception("Kernel has not been computed.")
        dt = self.dt
        time = np.arange(self.bkdxcorrw.shape[0]) * dt
        res_int = np.zeros(self.bkdxcorrw.shape)
        # res_int[0, :] = 0.5 * dt * to_integrate[0, :]
        # if method == "trapz":
        for n in range(self.trunc_ind):
            to_integrate = np.einsum("ijk,ikl->ijl", self.bkbkcorrw[: n + 1, :, :][::-1, :, :], self.kernel[: n + 1, :, :])
            res_int[n, :] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
            # res_int[n, :] = -1 * simpson(to_integrate, dx=dt, axis=0, even="last")  # res_int[n - 1, :] + 0.5 * dt * (to_integrate[n - 1, :] + to_integrate[n, :])
        # else:
        #     for n in range(self.trunc_ind):
        #         to_integrate = np.einsum("ijk,ik->ij", self.dotbkbkcorrw[: n + 1, :, :][::-1, :, :], self.kernel[: n + 1, :])
        #         res_int[n, :] = -1 * trapezoid(to_integrate, dx=dt, axis=0)
        #     # res_int[n, :] = -1 * simpson(to_integrate, dx=dt, axis=0, even="last")
        #     res_int += np.einsum("jk,ik->ij", self.bkbkcorrw[0, :, :], self.kernel)
        return time, res_int


class Pos_gle(ModelBase):
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
        ModelBase.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt - int(self.basis.const_removed) * self.dim_x
        self.rank_projection = not self.basis.const_removed

    def basis_vector(self, xva, compute_for="corrs"):

        bk = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        if compute_for == "force":
            return bk
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        dbk = xr.apply_ufunc(self.basis.deriv, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x"]], exclude_dims={"dim_x"}, dask="forbidden")
        if compute_for == "kernel":  # For kernel evaluation
            return dbk
        elif compute_for == "corrs":
            ddbk = xr.apply_ufunc(self.basis.hessian, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x", "dim_x'"]], exclude_dims={"dim_x"}, dask="forbidden")
            E = xr.dot(dbk, xva["v"], dims=["dim_x"])
            dE = xr.dot(dbk, xva["a"], dims=["dim_x"]) + xr.dot(ddbk, xva["v"], xva["v"].rename({"dim_x": "dim_x'"}), dims=["dim_x", "dim_x'"])
            return bk, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")


class Pos_gle_with_friction(ModelBase):
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
        bk = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        if compute_for == "force_eval":
            return bk
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        dbk = xr.apply_ufunc(self.basis.deriv, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x"]], exclude_dims={"dim_x"}, dask="forbidden")
        if compute_for == "kernel":  # To have the term proportional to velocity
            return dbk

        Evel = xr.dot(dbk, xva["v"], dims=["dim_x"])
        E = xr.concat([bk, Evel], dim="dim_basis")
        if compute_for == "force":
            return E
        elif compute_for == "corrs":
            ddbk = xr.apply_ufunc(self.basis.hessian, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x", "dim_x'"]], exclude_dims={"dim_x"}, dask="forbidden")
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


class Pos_gle_no_vel_basis(ModelBase):
    """
    Use basis function dependent of the position only
    """

    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")

    def basis_vector(self, xva, compute_for="corrs"):
        E = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        if compute_for == "force":
            return E
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        elif compute_for == "kernel":
            # Extend the basis for multidim value
            return E.reshape(-1, self.N_basis_elt_kernel, 1)  # TODO: change
        elif compute_for == "corrs":
            dbk = xr.apply_ufunc(self.basis.deriv, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x"]], exclude_dims={"dim_x"}, dask="forbidden")
            dE = xr.dot(dbk, xva["v"], dims=["dim_x"])
            return E, E, dE
        else:
            raise ValueError("Basis evaluation goal not specified")


class Pos_gle_const_kernel(ModelBase):
    """
    A derived class in which we the kernel is computed independent of the position
    """

    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.dim_obs

    def basis_vector(self, xva, compute_for="corrs"):
        bk = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        if compute_for == "force":
            return bk
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        if compute_for == "kernel":  # For kernel evaluation
            grad = np.zeros((bk.shape[0], self.dim_obs, self.dim_obs))
            for i in range(self.dim_obs):
                grad[:, i, i] = 1.0
            return grad  # TODO: update to xarray
        elif compute_for == "corrs":
            return bk, xva["v"], xva["a"]
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


class Pos_gle_hybrid(ModelBase):
    """
    Implement the hybrid projector of arXiv:2202.01922
    """

    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt + 1
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")

    def basis_vector(self, xva, compute_for="corrs"):
        # We have to deal with the multidimensionnal case as well
        bk = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        if compute_for == "force":
            return bk
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        elif compute_for == "kernel":
            # Extend the basis for multidim value
            return bk.reshape(-1, self.N_basis_elt_kernel - 1, 1)
        elif compute_for == "corrs":
            E = xr.concat([xva["v"].rename({"dim_x": "dim_basis"}), bk], dim="dim_basis")
            dbk = np.einsum("nld,nd->nl", xr.apply_ufunc(self.basis.deriv, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x"]], exclude_dims={"dim_x"}, dask="forbidden"), xva["v"].data)
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


class Pos_gle_overdamped(ModelBase):
    """
    Extraction of position dependent memory kernel for overdamped dynamics.
    """

    def __init__(self, *args, L_obs="v", rank_projection=False, **kwargs):
        ModelBase.__init__(self, *args, L_obs=L_obs, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.N_basis_elt
        if self.basis.const_removed:
            self.basis.const_removed = False
            print("Warning: remove_const on basis function have been set to False.")
        self.rank_projection = rank_projection

    def basis_vector(self, xva, compute_for="corrs"):
        E = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        if compute_for == "force":
            return E
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        elif compute_for == "kernel":
            return E.reshape(-1, self.N_basis_elt_kernel, self.dim_x)
        elif compute_for == "corrs":
            dbk = xr.apply_ufunc(self.basis.deriv, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis", "dim_x"]], exclude_dims={"dim_x"}, dask="forbidden")
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


class Pos_gle_overdamped_const_kernel(ModelBase):
    """
    Extraction of position dependent memory kernel for overdamped dynamics
    using position-independent kernel.
    """

    def __init__(self, *args, L_obs="v", **kwargs):
        ModelBase.__init__(self, *args, L_obs=L_obs, **kwargs)
        self.N_basis_elt_force = self.N_basis_elt
        self.N_basis_elt_kernel = self.dim_obs

    def basis_vector(self, xva, compute_for="corrs"):
        E = xr.apply_ufunc(self.basis.basis, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        if compute_for == "force":
            return E
        elif compute_for == "pmf":
            return xr.apply_ufunc(self.basis.antiderivative, xva["x"], input_core_dims=[["dim_x"]], output_core_dims=[["dim_basis"]], exclude_dims={"dim_x"}, dask="forbidden")
        elif compute_for == "kernel":
            return np.ones((E.shape[0], self.N_basis_elt_kernel, self.dim_x))
        elif compute_for == "corrs":
            return E, np.ones((E.shape[0], self.dim_obs)), xva["v"].data
        else:
            raise ValueError("Basis evaluation goal not specified")
