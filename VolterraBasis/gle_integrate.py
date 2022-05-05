import numpy as np
import xarray as xr

from .fkernel import memory_rect, memory_trapz


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
    Taken from https://github.com/jandaldrop/bgle
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

        self.dim = kernel.shape[-1]
        self.kernel = kernel  # np.concatenate([kernel, np.zeros(add_zeros)]) # Calculer add_zeros directement à partir de la puissance de 2 la plus proche

        self.rng = rng

        t_sym = np.concatenate((-self.t[:0:-1], self.t))
        self.sqk = np.empty((t_sym.shape[0], self.dim))

        for d in range(self.dim):
            kernel_sym = np.concatenate((self.kernel[:0:-1, d], self.kernel[:, d]))
            kernel_ft, w = ft(kernel_sym, t_sym)
            sqk_ft = np.sqrt(kernel_ft)
            self.sqk[:, d] = ift(sqk_ft, w, t_sym).real

    def generate(self, size):
        colored_noise = np.empty((max(size, self.sqk.shape[0]), self.dim))
        for d in range(self.dim):
            white_noise = self.rng(size=size)
            colored_noise[:, d] = np.convolve(white_noise, self.sqk[:, d], mode="same")
        return colored_noise[:size] * np.sqrt(self.dt)


class Integrator_gle(object):
    """
    The Class holding the BGLE integrator.
    """

    def __init__(self, pos_gle, coeffs_noise_kernel=None, trunc_kernel=None, rng=np.random.normal, verbose=True):
        """
        On prend comme argument, une classe pos_gle qui contient l'estimation de la force et du kernel
        Des coeffs, coeffs_noise_kernel qui sont les coeffs qu'on veut prendre pour la covariance du bruit
        """
        self.verbose = verbose
        self.basis = pos_gle.basis

        if trunc_kernel is None:
            self.trunc_kernel = pos_gle.trunc_ind
        else:
            self.trunc_kernel = trunc_kernel

        self.force_coeff = pos_gle.force_coeff
        self.kernel = pos_gle.kernel

        self.rank_projection = pos_gle.rank_projection
        self.P_range = pos_gle.P_range

        self.dt = pos_gle.time[1, 0] - pos_gle.time[0, 0]

        self.dim = pos_gle.dim_obs

        if self.verbose:
            print("Found dt =", self.dt)

        if pos_gle.method in ["rect", "rectangular", "second_kind_rect"] or pos_gle.method is None:
            self.weight_ker_int = 1.0 * self.dt
        elif pos_gle.method in ["trapz", "second_kind_trapz"]:
            self.weight_ker_int = 0.5 * self.dt

        if coeffs_noise_kernel is None:
            coeffs_noise_kernel = np.ones(pos_gle.N_basis_elt_kernel) @ pos_gle.bkbkcorrw[0, :, :]
        kernel_noise = np.einsum("k,ikl->il", coeffs_noise_kernel, pos_gle.kernel)

        self.noise_generator = ColoredNoiseGenerator(kernel_noise, pos_gle.time[:, 0])

    def initial_conditions(self, xva, n_mem=0):
        """
        Draw random initial start point from another trajectory.
        TODO: give set of trajectory
        """
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
        return bk, E

    def _mem_int_red(self, E):
        loc_trunc = min(E.shape[0], self.trunc_kernel - 2)
        start_trunc = max(E.shape[0] - loc_trunc, 0)
        return np.einsum("ik,ikl->l", E[start_trunc + 1 :][::-1, :], self.kernel[1:loc_trunc]) * self.dt + self.weight_ker_int * E[start_trunc] @ self.kernel[loc_trunc]

    def _f_rk(self, x, v, rmi, fr, alpha, last_E, last_rmi):
        E_force, E = self.basis_vector(x, v)
        mem = alpha * (last_rmi + self.weight_ker_int * last_E @ self.kernel[0, :]) + (1 - alpha) * (rmi + self.weight_ker_int * E @ self.kernel[0, :])
        return v, np.matmul(E_force, self.force_coeff) + fr - mem

    def _rk_step(self, x, v, rmi, fr, last_E, last_rmi):
        k1x, k1v = self._f_rk(x, v, rmi, fr, 0.0, last_E, last_rmi)
        k2x, k2v = self._f_rk(x + k1x * self.dt / 2, v + k1v * self.dt / 2, rmi, fr, 0.5, last_E, last_rmi)
        k3x, k3v = self._f_rk(x + k2x * self.dt / 2, v + k2v * self.dt / 2, rmi, fr, 0.5, last_E, last_rmi)
        k4x, k4v = self._f_rk(x + k3x * self.dt, v + k3v * self.dt, rmi, fr, 1.0, last_E, last_rmi)
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

        trj = xr.Dataset({"x": (["time", "dim_x"], np.zeros((n_steps, self.dim))), "v": (["time", "dim_x"], np.zeros((n_steps, self.dim))), "a": (["time", "dim_x"], np.zeros((n_steps, self.dim)))}, coords={"time": np.arange(n_steps) * self.dt}, attrs={"dt": self.dt})
        # Pour les conditions initial, l'idée c'est de remplacer les n_0 premier pas par le contenu de trj0
        if x0 is not None:
            n_0 = x0["time"].shape[0]
            # Slicing the DataSet and setting its n_0 first value with
            trj["x"][:n_0] = x0["x"]
            trj["v"][:n_0] = x0["v"]
            n_0 -= 1
        else:
            n_0 = 0

        E = np.zeros((n_steps, self.kernel.shape[1]))
        for ind in range(n_0):  # If needed to initialize
            _, E_step = self.basis_vector(trj["x"].isel(time=[ind]), trj["v"].isel(time=[ind]))
            E[ind, :] = E_step[0, :]
        rmi = np.zeros(self.dim)
        x = trj["x"].isel(time=[n_0]).data
        v = trj["v"].isel(time=[n_0]).data
        for ind in range(n_0, n_steps - 1):
            # print("----------------", ind, "----------")
            _, E_step = self.basis_vector(x, v)
            E[ind, :] = E_step[0, :]
            last_rmi = rmi
            if ind > 1:
                rmi = self._mem_int_red(E[:ind])
                last_E = E[ind - 1]
            else:
                rmi = np.zeros(self.dim)
                last_rmi = np.zeros(self.dim)
                last_E = np.zeros(self.kernel.shape[1])
            x, v, a = self._rk_step(x, v, rmi, noise[ind, :], last_E, last_rmi)
            trj["x"][ind + 1] = x[0, :]  # A changer en utilisant loc
            trj["v"][ind + 1] = v[0, :]
            trj["a"][ind + 1] = a[0, :]

        return trj, noise[:n_steps]


class Integrator_gle_const_kernel(Integrator_gle):
    """
    A derived class in which we the kernel is supposed independent of the position
    """

    def basis_vector(self, x, v):
        bk = self.basis.basis(x)
        E = v
        if self.rank_projection:
            E = np.einsum("kj,ij->ik", self.P_range, E)
        return bk, E
