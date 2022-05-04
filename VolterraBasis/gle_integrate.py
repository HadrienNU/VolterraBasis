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

    def __init__(self, pos_gle, coeffs_noise_kernel, trunc_kernel=None, rng=np.random.normal, verbose=True):
        """
        On prend comme argument, une classe pos_gle qui contient l'estimation de la force et du kernel
        Des coeffs, coeffs_noise_kernel qui sont les coeffs qu'on veut prendre pour la covariance du bruit
        """
        self.verbose = verbose
        self.pos_gle = pos_gle

        if trunc_kernel is None:
            self.trunc_kernel = self.pos_gle.trunc_ind
        else:
            self.trunc_kernel = trunc_kernel

        self.dt = self.pos_gle.time[1, 0] - self.pos_gle.time[0, 0]

        self.dim = self.pos_gle.dim_obs

        if self.verbose:
            print("Found dt =", self.dt)

        if self.pos_gle.method in ["rect", "rectangular", "second_kind_rect"] or self.pos_gle.method is None:
            self.memory_int = memory_rect
            self.weight_ker_int = 1.0 * self.dt
        elif self.pos_gle.method in ["trapz", "second_kind_trapz"]:
            self.memory_int = memory_trapz
            self.weight_ker_int = 0.5 * self.dt

        kernel_noise = np.einsum("k,ikl->il", coeffs_noise_kernel, self.pos_gle.kernel)

        self.noise_generator = ColoredNoiseGenerator(kernel_noise, self.pos_gle.time[:, 0])

    def initial_conditions(self, xva, n_mem=0):
        """
        Draw random initial start point from another trajectory.
        TODO: give set of trajectory
        """
        point = np.random.randint(n_mem, xva["time"].shape[0])
        start = xva.isel(time=slice(point - n_mem, point + 1))
        start["time"] = np.arange(n_mem + 1) * self.dt
        return start

    def _mem_int_red(self, E):
        return self.memory_int(self.pos_gle.kernel[1 : self.trunc_kernel], E, self.dt)[0, :]  # TODO A changer pour la bonne function

    def _f_rk(self, xv, rmi, fr, alpha, last_E, last_rmi):
        E_force, E, _ = self.pos_gle.basis_vector(xv)
        if self.pos_gle.rank_projection:
            E = np.einsum("kj,ij->ik", self.pos_gle.P_range, E)
        # Calcul du pas suplémentaire du noyau mémoire, j'ai besoin de E aussi
        mem = alpha * (last_rmi + self.weight_ker_int * last_E @ self.pos_gle.kernel[0, :]) + (1 - alpha) * (rmi + self.weight_ker_int * E @ self.pos_gle.kernel[0, :])
        return xv.copy(data={"x": xv["x"], "v": xv["v"], "a": np.matmul(E_force, self.pos_gle.force_coeff) + fr - mem})

    def _rk_step(self, xv, rmi, fr, last_E, last_rmi):
        """
        On doit tout écrire avec des xarray
        """
        xv1 = self._f_rk(xv, rmi, fr, 0.0, last_E, last_rmi)
        xv2 = self._f_rk(xv + xv1 * self.dt / 2, rmi, fr, 0.5, last_E, last_rmi)
        xv3 = self._f_rk(xv + xv2 * self.dt / 2, rmi, fr, 0.5, last_E, last_rmi)
        xv4 = self._f_rk(xv + xv3 * self.dt, rmi, fr, 1.0, last_E, last_rmi)
        xv["a"] = (xv1["a"] + 2.0 * xv2["a"] + 2.0 * xv3["a"] + xv4["a"]) / 6.0
        xv["v"] = xv["v"] + self.dt * xv["a"]
        xv["x"] = xv["x"] + self.dt * (xv1["v"] + 2.0 * xv2["v"] + 2.0 * xv3["v"] + xv4["v"]) / 6.0
        return xv

    def run(self, n_steps, x0=None, set_noise_to_zero=False):
        """
        Il faut prendre en entrée un xarray, xv, qui contient les positions initiales.

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

        E = np.zeros((n_steps, self.pos_gle.kernel.shape[1]))
        for ind in range(n_0):  # If needed to initialize
            _, E_step, _ = self.pos_gle.basis_vector(trj.isel(time=[ind]))
            if self.pos_gle.rank_projection:
                E[ind, :] = np.einsum("kj,ij->ik", self.pos_gle.P_range, E_step)
            else:
                E[ind, :] = E_step
        rmi = np.zeros(self.dim)
        for ind in range(n_0, n_steps - 1):
            # print("----------------", ind, "----------")
            _, E_step, _ = self.pos_gle.basis_vector(trj.isel(time=[ind]))
            if self.pos_gle.rank_projection:
                E[ind, :] = np.einsum("kj,ij->ik", self.pos_gle.P_range, E_step)
            else:
                E[ind, :] = E_step[0, :]
            last_rmi = rmi
            if ind > 1:
                rmi = self._mem_int_red(E[:ind])  # self._mem_int_red(self.v_trj[:ind])  Faire ça ici qu'on ne doit prendre en compte le dernier term de la somme pour pouvoir le changer
                last_E = E[ind - 1]
            else:
                rmi = np.zeros(self.dim)
                last_rmi = np.zeros(self.dim)
                last_E = np.zeros(self.pos_gle.kernel.shape[1])

            xv = self._rk_step(trj.isel(time=[ind]), rmi, noise[ind, :], last_E, last_rmi)
            trj["x"][ind + 1] = xv["x"][0]  # A changer en utilisant loc
            trj["v"][ind + 1] = xv["v"][0]
            trj["a"][ind + 1] = xv["a"][0]

        return trj, noise[:n_steps]
