import numpy as np
import xarray as xr


def ft(f, t):
    w = 2.0 * np.pi * np.fft.fftfreq(len(f)) / (t[1] - t[0])
    g = np.fft.fft(f)
    g *= (t[1] - t[0]) * np.exp(-complex(0, 1) * w * t[0])
    return g, w


def ift(f, w, t):
    f *= np.exp(complex(0, 1) * w * t[0])
    g = np.fft.ifft(f)
    g *= len(f) * (w[1] - w[0]) / (2 * np.pi)
    return g


class ColoredNoiseGenerator:
    """
    A class for the generation of colored noise.
    Adapted from https://github.com/jandaldrop/bgle
    Implement correlated noise generator of DOI: 10.1103/PhysRevE.91.032125
    Assume that the kernel is diagonal
    """

    def __init__(self, kernel, t, rng=np.random.normal):
        """
        Create an instance of the ColoredNoiseGenerator class.

        Parameters
        ----------
        kernel : numpy.array
            The correlation function of the noise.
        t : numpy.array
            The time/x values of kernel.
        """

        self.t = t.ravel()
        self.dt = self.t[1] - self.t[0]

        self.N_basis_elt_kernel = kernel.shape[1]
        self.dim = kernel.shape[-1]
        self.kernel = kernel  # np.concatenate([kernel, np.zeros(add_zeros)]) # Calculer add_zeros directement à partir de la puissance de 2 la plus proche

        self.rng = rng

        t_sym = np.concatenate((-self.t[:0:-1], self.t))
        self.sqk = np.zeros((t_sym.shape[0], self.N_basis_elt_kernel, self.dim))

        # TODO: We should take here the cholesky of the matrix and not the square root of each element, that would work only for diagonal kernel
        for d in range(self.dim):
            kernel_sym = np.concatenate((self.kernel[:0:-1, d, d], self.kernel[:, d, d]))
            kernel_ft, w = ft(kernel_sym, t_sym)
            sqk_ft = np.sqrt(kernel_ft)
            self.sqk[:, d, d] = ift(sqk_ft, w, t_sym).real

    def generate(self, size):
        colored_noise = np.empty((max(size, self.sqk.shape[0]), self.dim))
        for d in range(self.dim):
            white_noise = self.rng(size=size)
            colored_noise[:, d] = np.convolve(white_noise, self.sqk[:, d, d], mode="same")
        return colored_noise[:size] * np.sqrt(self.dt)


class KarhunenLoeveNoiseGenerator:
    """
    A class for the generation of colored noise.
    Use Karhunen–Loève decomposition of the kernel
    for non stationary noise
    """

    @staticmethod
    def generate_stationary_covariancematrix(kernel, backward_kernel=None):
        """
        From kernel generate the covariance matrix of the noise, ie the time-dependent kernel matrix
        """
        N_size = kernel.shape[0]
        if backward_kernel is None:
            backward_kernel = kernel[:0:-1]
        kernel_sym = np.concatenate((backward_kernel, kernel))
        cov_mat = np.zeros((N_size, N_size))

        cov_mat[:, 0] = kernel
        for n in range(1, N_size):
            cov_mat[:, n] = kernel_sym[N_size - n - 1 : -n]
        return cov_mat

    def __init__(self, kernel, dt, rng=np.random.normal):
        """
        Create an instance of the KarhunenLoeveNoiseGenerator class.

        Parameters
        ----------
        kernel : numpy.array
            The correlation function of the noise.
        t : numpy.array
            The time values of kernel.
        """
        self.dt = dt

        self.rng = rng
        self.kernel = kernel

        self.lenght = np.min(kernel.shape)

        self.eigvals, self.eigvect = np.linalg.eig(self.kernel)

    def generate(self, size=None):
        if size is None:
            size = self.lenght
        if size > self.lenght:
            raise ValueError("Cannot generate noise for longuer time than the one in the kernel\n Use stationary or periodic assumption to go further")
            # TODO: Use stationary assumption that kernel does not evolve anymore to generate more noise
            # TODO: Use periodic assumption
        white_noise = self.rng(size=self.lenght)
        colored_noise = self.eigvect @ np.diag(np.sqrt(self.eigvals)) @ white_noise
        return colored_noise[:size] * np.sqrt(self.dt)


class PosColoredNoiseGenerator:
    """
    A class for the generation of colored noise.
    Adapted from https://github.com/jandaldrop/bgle
    Implement correlated noise generator of DOI: 10.1103/PhysRevE.91.032125
    """

    def __init__(self, kernel, t, rng=np.random.normal):
        """
        Create an instance of the ColoredNoiseGenerator class.

        Parameters
        ----------
        kernel : numpy.array
            The correlation function of the noise.
        t : numpy.array
            The time/x values of kernel.
        add_zeros : int, default=0
            Add add_zeros number of zeros to the kernel function for numeric Fourier transformation.
        """

        self.t = t.ravel()
        self.dt = self.t[1] - self.t[0]

        self.N_basis_elt_kernel = kernel.shape[1]
        self.dim = kernel.shape[-1]
        self.kernel = kernel  # np.concatenate([kernel, np.zeros(add_zeros)]) # Calculer add_zeros directement à partir de la puissance de 2 la plus proche

        self.rng = rng

        self.t_sym = np.concatenate((-self.t[:0:-1], self.t))
        self.sqk = np.empty((self.t_sym.shape[0], self.N_basis_elt_kernel, self.dim))
        self.kernel_ft = np.empty((self.t_sym.shape[0], self.N_basis_elt_kernel, self.dim))

        # TODO, en fait c'est le cholesky qu'il faut prendre ici
        for k in range(self.N_basis_elt_kernel):
            for d in range(self.dim):
                kernel_sym = np.concatenate((self.kernel[:0:-1, k, d], self.kernel[:, k, d]))
                self.kernel_ft[:, k, d], self.w = ft(kernel_sym, self.t_sym)

    def prepare(self, size):
        """
        Prepare generation
        """
        self.white_noise = self.rng(size=size)
        self.colored_noise = np.empty((max(size, self.sqk.shape[0]), self.dim))

    def generate(self, E_noise, ind):
        """
        Mais c'est super lourd
        """
        sqk_ft = np.sqrt(self.kernel_ft)
        colored_noise = np.empty((max(ind, self.sqk.shape[0]), self.dim))
        sqk_ft = np.sqrt(np.einsum("kl,ikd->idl", E_noise, self.kernel_ft))  # If self.dim > 1, we should use the cholesky instead of the square root
        for d in range(self.dim):
            sqk = ift(sqk_ft[:, d, d], self.w, self.t_sym).real
            white_noise = self.rng(size=self.kernel_ft.shape[0])
            colored_noise[:, d] = np.convolve(white_noise, sqk, mode="same")
        return colored_noise[ind, :] * np.sqrt(self.dt)


class Integrator_gle(object):
    """
    The Class holding the BGLE integrator.
    """

    def __init__(self, pos_gle, coeffs_noise_kernel=None, trunc_kernel=None, rng=np.random.normal, verbose=True, **kwargs):
        """
        On prend comme argument, une classe pos_gle qui contient l'estimation de la force et du kernel
        Des coeffs, coeffs_noise_kernel qui sont les coeffs qu'on veut prendre pour la covariance du bruit
        If pos_gle is None, basis,force_coeff, kernel and dt should be passed as argument
        """
        self.verbose = verbose

        if pos_gle is not None:
            self.basis = pos_gle.basis

            if trunc_kernel is None:
                trunc_kernel = pos_gle.kernel.shape[0]

            self.force_coeff = pos_gle.force_coeff
            self.kernel = pos_gle.kernel[:trunc_kernel]

            self.rank_projection = pos_gle.rank_projection
            self.P_range = pos_gle.P_range

            self.dt = pos_gle.time[1, 0] - pos_gle.time[0, 0]

            self.N_basis_elt_kernel = pos_gle.N_basis_elt_kernel
            self.dim = pos_gle.dim_obs
            if coeffs_noise_kernel is None:
                kernel_gram = pos_gle.compute_kernel_gram()
                coeffs_noise_kernel = np.linalg.inv(kernel_gram) @ pos_gle.bkbkcorrw[0, :, :]
            kernel_noise = np.einsum("kl,ikd->ild", coeffs_noise_kernel, self.kernel)
        else:
            self.basis = kwargs["basis"]
            self.force_coeff = kwargs["force_coeff"]
            self.kernel = kwargs["kernel"]
            self.dt = kwargs["dt"]
            if self.kernel.ndim == 1:
                self.dim = 1
                self.N_basis_elt_kernel = 1
            elif self.kernel.ndim == 2:
                self.dim = self.kernel.shape[-1]
                self.N_basis_elt_kernel = self.dim
            else:
                self.dim = self.kernel.shape[-1]
                self.N_basis_elt_kernel = self.kernel.shape[1]

            self.rank_projection = False
            self.P_range = np.eye(self.dim)
            if coeffs_noise_kernel is None:
                kernel_noise = self.kernel
            else:
                kernel_noise = np.einsum("kl,ikd->ild", coeffs_noise_kernel, self.kernel)

        if self.verbose:
            print("Found dt =", self.dt)

        self.trunc_kernel = self.kernel.shape[0]

        self.noise_generator = ColoredNoiseGenerator(kernel_noise, pos_gle.time[:, 0], rng=rng)

    def _copy_from_estimator(self, pos_gle, trunc_kernel=None):
        """
        Copy everything from the estimator
        """

        self.basis = pos_gle.basis

        if trunc_kernel is None:
            trunc_kernel = pos_gle.kernel.shape[0]

        self.force_coeff = pos_gle.force_coeff
        self.kernel = pos_gle.kernel[:trunc_kernel]

        self.rank_projection = pos_gle.rank_projection
        self.P_range = pos_gle.P_range

        self.dt = pos_gle.time[1, 0] - pos_gle.time[0, 0]

        self.N_basis_elt_kernel = pos_gle.N_basis_elt_kernel
        self.dim = pos_gle.dim_obs

    def initial_conditions(self, xva_arg, n_mem=0):
        """
        Draw random initial start point from another trajectory or a set of trajectories.
        """
        if isinstance(xva_arg, xr.Dataset):
            xva = xva_arg
        else:
            xva = xva_arg[np.random.randint(len(xva_arg))]
        point = np.random.randint(n_mem, xva["time"].shape[0])
        start = xva.isel(time=slice(point - n_mem, point + 1))
        start["time"] = np.arange(n_mem + 1) * self.dt
        return start

    def basis_vector(self, x, v):
        bk = self.basis.basis(x)
        dbk = self.basis.deriv(x)
        E = np.einsum("nld,nd->nl", dbk, v)
        if self.rank_projection:
            E = np.einsum("kj,ij->ik", self.P_range, E)
        return bk, E, dbk

    def _mem_int_red(self, E):
        loc_trunc = min(E.shape[0], self.trunc_kernel - 1)
        start_trunc = max(E.shape[0] - loc_trunc, 0)
        return np.einsum("ik,ikl->l", E[start_trunc + 1 :][::-1, :], self.kernel[1:loc_trunc]) * self.dt + 0.5 * self.dt * E[start_trunc] @ self.kernel[loc_trunc]

    def _f_rk(self, x, v, rmi, fr, alpha, last_E, last_rmi):
        """
        Little speed-up by removing the matrix product between identity and noise
        """
        E_force, E, _ = self.basis_vector(x, v)
        mem = alpha * (last_rmi + 0.5 * self.dt * last_E @ self.kernel[0, :]) + (1.0 - alpha) * (rmi + 0.5 * self.dt * E @ self.kernel[0, :])
        return v, np.matmul(E_force, self.force_coeff) + fr - mem

    def _rk_step(self, x, v, rmi, fr, last_E, last_rmi):
        k1x, k1v = self._f_rk(x, v, rmi, fr, 1.0, last_E, last_rmi)
        k2x, k2v = self._f_rk(x + k1x * self.dt / 2, v + k1v * self.dt / 2, rmi, fr, 0.5, last_E, last_rmi)
        k3x, k3v = self._f_rk(x + k2x * self.dt / 2, v + k2v * self.dt / 2, rmi, fr, 0.5, last_E, last_rmi)
        k4x, k4v = self._f_rk(x + k3x * self.dt, v + k3v * self.dt, rmi, fr, 0.0, last_E, last_rmi)
        a = (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        return x + self.dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0, v + self.dt * a, a

    def run(self, n_steps, x0=None, set_noise_to_zero=False):
        """
        Run a trajectory of length n_steps with initial conditions x0.
        """
        if set_noise_to_zero:
            noise = np.zeros((n_steps, self.dim))
        else:
            noise = self.noise_generator.generate(n_steps)

        trj = xr.Dataset({"x": (["time", "dim_x"], np.zeros((n_steps, self.dim))), "v": (["time", "dim_x"], np.zeros((n_steps, self.dim)))}, coords={"time": np.arange(n_steps) * self.dt}, attrs={"dt": self.dt})

        if x0 is not None:
            n_0 = x0["time"].shape[0]
            # Slicing the DataSet and setting its n_0 first value with
            trj["x"][:n_0] = x0["x"]
            trj["v"][:n_0] = x0["v"]
        else:
            n_0 = 1
        trj["time"] = (n_0 - 1 + np.arange(n_steps)) * self.dt

        E = np.zeros((n_steps, self.kernel.shape[1]))
        rmi = np.zeros(self.dim)
        for ind in range(n_0):  # If needed to initialize
            _, E_step, _ = self.basis_vector(trj["x"].isel(time=[ind]), trj["v"].isel(time=[ind]))
            E[ind, :] = E_step[0, :]
        if n_0 > 1:
            rmi = self._mem_int_red(E[: n_0 - 1])
        x = trj["x"].isel(time=[n_0 - 1]).data
        v = trj["v"].isel(time=[n_0 - 1]).data
        for ind in range(n_0, n_steps):
            # print("----------------", ind, "----------")
            last_rmi = rmi
            last_E = E[ind - 1]
            rmi = self._mem_int_red(E[:ind])
            x, v, a = self._rk_step(x, v, rmi, noise[ind, :], last_E, last_rmi)
            trj["x"][ind] = x[0, :]  # A changer en utilisant loc
            trj["v"][ind] = v[0, :]

            _, E_step, _ = self.basis_vector(x, v)
            E[ind, :] = E_step[0, :]

        return trj


class Integrator_gle_const_kernel(Integrator_gle):
    """
    A derived class in which we the kernel is supposed independent of the position
    """

    # def __init__(self, *args, **kwargs):
    #     """
    #     """
    #     Integrator_gle.__init__(self, *args, **kwargs)
    #
    #     if coeffs_noise_kernel is None:
    #         kernel_gram = pos_gle.compute_kernel_gram()
    #         coeffs_noise_kernel = np.linalg.inv(kernel_gram) @ pos_gle.bkbkcorrw[0, :, :]
    #     kernel_noise = np.einsum("kl,ikd->ild", coeffs_noise_kernel, self.kernel)
    #     self.noise_generator = ColoredNoiseGenerator(kernel_noise, pos_gle.time[:, 0], rng=rng)

    def basis_vector(self, x, v):
        bk = self.basis.basis(x)
        E = v
        if self.rank_projection:
            E = np.einsum("kj,ij->ik", self.P_range, E)
        return bk, E, None

    def _f_rk(self, x, v, rmi, fr, alpha, last_E, last_rmi):
        """
        Little speed-up by removing the matrix product between identity and noise
        """
        E_force, E, _ = self.basis_vector(x, v)
        mem = alpha * (last_rmi + 0.5 * self.dt * last_E @ self.kernel[0, :]) + (1.0 - alpha) * (rmi + 0.5 * self.dt * E @ self.kernel[0, :])
        return v, np.matmul(E_force, self.force_coeff) + fr - mem


class Integrator_posgle(Integrator_gle):
    """
    A test class for development of position-dependent noise
    """

    # def __init__(self, *args, **kwargs):
    #     """
    #     """
    #     Integrator_gle.__init__(self, *args, **kwargs)
    #
    #     if coeffs_noise_kernel is None:
    #         kernel_gram = pos_gle.compute_kernel_gram()
    #         coeffs_noise_kernel = np.linalg.inv(kernel_gram) @ pos_gle.bkbkcorrw[0, :, :]
    #     kernel_noise = np.einsum("kl,ikd->ild", coeffs_noise_kernel, self.kernel)
    #     self.noise_generator = ColoredNoiseGenerator(kernel_noise, pos_gle.time[:, 0], rng=rng)

    def _f_rk(self, x, v, rmi, fr, alpha, last_E, last_rmi):
        E_force, E, E_noise = self.basis_vector(x, v)
        mem = alpha * (last_rmi + 0.5 * self.dt * last_E @ self.kernel[0, :]) + (1.0 - alpha) * (rmi + 0.5 * self.dt * E @ self.kernel[0, :])
        return v, np.matmul(E_force, self.force_coeff) + E_noise[0, :].T @ fr - mem  # Find a way to have the square root of E_noise


class BGLEIntegrator(Integrator_gle_const_kernel):
    """
    The Class holding the BGLE integrator.
    """

    def dU(self, x):
        E_force, _, _ = self.basis_vector(x.reshape(1, -1), np.zeros((1, 1)))
        return -1 * np.matmul(E_force, self.force_coeff)

    def _mem_int_red(self, v):
        if len(v) < len(self.kernel):
            v = np.concatenate([np.zeros(len(self.kernel) - len(v) + 1), v])
        integrand = self.kernel[1:] * v[: len(v) - len(self.kernel[1:]) - 1 : -1]
        return (0.5 * integrand[-1] + np.sum(integrand[:-1])) * self.dt

    def f_rk(self, x, v, rmi, fr, next_w, last_w, last_v, last_rmi):
        nv = v
        na = -next_w * rmi - last_w * last_rmi - 0.5 * next_w * self.kernel[0] * v * self.dt - 0.5 * last_w * self.kernel[0] * last_v * self.dt - self.dU(x) + fr
        return nv, na

    def rk_step(self, x, v, rmi, fr, last_v, last_rmi):
        k1x, k1v = self.f_rk(x, v, rmi, fr, 0.0, 1.0, last_v, last_rmi)
        k2x, k2v = self.f_rk(x + k1x * self.dt / 2, v + k1v * self.dt / 2, rmi, fr, 0.5, 0.5, last_v, last_rmi)
        k3x, k3v = self.f_rk(x + k2x * self.dt / 2, v + k2v * self.dt / 2, rmi, fr, 0.5, 0.5, last_v, last_rmi)
        k4x, k4v = self.f_rk(x + k3x * self.dt, v + k3v * self.dt, rmi, fr, 1.0, 0.0, last_v, last_rmi)
        return x + self.dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0, v + self.dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0, (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0

    def integrate(self, n_steps, x0=0.0, v0=0.0, set_noise_to_zero=False, _custom_noise_array=None, _predef_x=None, _predef_v=None, _n_0=0):

        self.kernel = self.kernel.ravel()
        if set_noise_to_zero:
            noise = np.zeros(n_steps)
        else:
            if _custom_noise_array is None:
                noise = self.noise_generator.generate(n_steps)
            else:
                assert len(_custom_noise_array) == n_steps
                noise = _custom_noise_array
        x, v = x0, v0

        if _predef_v is None:
            self.v_trj = np.zeros(n_steps)
        else:
            assert len(_predef_v) == n_steps
            assert _predef_v[_n_0 - 1] == v
            self.v_trj = _predef_v

        if _predef_x is None:
            self.x_trj = np.zeros(n_steps)
        else:
            assert len(_predef_x) == n_steps
            assert _predef_x[_n_0 - 1] == x
            self.x_trj = _predef_x

        self.t_trj = np.arange(0.0, n_steps * self.dt, self.dt)
        self.v_trj[_n_0] = v
        self.x_trj[_n_0] = x

        self.mem_trj = np.zeros(n_steps)
        self.a_trj = np.zeros(n_steps)

        rmi = 0.0
        for ind in range(_n_0 + 1, n_steps):
            # print("----------------", ind, "----------")
            last_rmi = rmi
            if ind > 1:
                rmi = self._mem_int_red(self.v_trj[:ind])
                last_v = self.v_trj[ind - 1]
            else:
                rmi = 0.0
                last_rmi = 0.0
                last_v = 0.0
            x, v, a = self.rk_step(x, v, rmi, noise[ind], last_v, last_rmi)
            self.v_trj[ind] = v
            self.x_trj[ind] = x
            self.a_trj[ind] = a
            self.mem_trj[ind] = rmi

        return self.x_trj, self.v_trj, self.t_trj, noise[:n_steps], self.mem_trj, self.a_trj
