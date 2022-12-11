import numpy as np
import xarray as xr
from scipy.integrate import trapezoid, simpson

from .fkernel import kernel_first_kind_trapz, kernel_first_kind_rect, kernel_first_kind_midpoint, kernel_second_kind_rect, kernel_second_kind_trapz
from .fkernel import memory_rect, memory_trapz, corrs_rect, corrs_trapz
from .correlation import correlation_1D, correlation_ND


def _convert_input_array_for_evaluation(array, dim_x):
    """
    Take input and return xarray Dataset with correct shape
    """
    if isinstance(array, xr.Dataset):  # TODO add check on dimension of array
        return array
    else:
        x = np.asarray(array).reshape(-1, dim_x)
        return xr.Dataset({"x": (["time", "dim_x"], x)})


class Trajectories_handler(object):
    """
    The main class for the data.
    """

    def __init__(self, xva_arg, trunc=1.0, L_obs="a", verbose=True, **kwargs):
        """
        Create an instance of the Trajectories_handler class.

        Parameters
        ----------
        xva_arg : xarray dataset (['time', 'x', 'v', 'a']) or list of datasets.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a xarray timeseries
            or a listlike collection of them.
        trunc : float, default=1.0
            Truncate all correlation functions and the memory kernel after this
            time value.
        L_obs: str, default= "a"
            Name of the column containing the time derivative of the observable
        verbose : bool, default=True
            Set verbosity.
        """
        self.verbose = verbose
        self.L_obs = L_obs
        self.set_of_obs = ["x", "v", self.L_obs]

        self._do_check(xva_arg)  # Do some check on the trajectories

        # Save trajectory properties
        if self.xva_list is None:
            return

        self.dim_x = self.xva_list[0].dims["dim_x"]
        self.dim_obs = self.xva_list[0][L_obs].shape[1]

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
                for col in self.set_of_obs:
                    if col not in xva.data_vars:
                        raise Exception("Please provide time,{} dataset, " "or an iterable collection (i.e. list) " "of time,{} dataset.".format(self.set_of_obs, self.set_of_obs))
                if "time" not in xva.dims:
                    raise Exception("Time is not a coordinate. Please provide dataset with time, " "or an iterable collection (i.e. list) " "of dataset with time.")
                if "dt" not in xva.attrs:
                    raise Exception("Timestep not in dataset attrs")
        else:
            self.xva_list = None

    def _update_traj_gfpe(self, model):
        for i in range(len(self.xva_list)):
            E_force, E, dE = model.basis_vector(self.xva_list[i])
            self.xva_list[i].update({"dE": (["time", "dim_dE"], dE)})

    def _loop_over_traj_serial(self, func, model):
        """
        A generator for iteration over trajectories
        """
        # Et voir alors pour faire une version parallélisé (en distribué)
        array_res = [func(weight, xva, model) for weight, xva in zip(self.weights, self.xva_list)]
        res = [0.0] * len(array_res[0])
        for weight, single_res in zip(self.weights, array_res):
            for i, arr in enumerate(single_res):
                res[i] += arr * weight / self.weightsum
        return res
        # for weight, xva in zip(self.weights, self.xva_list):
        #     yield weight, xva

    def compute_gram(self, kT=1.0):  # ça devrait plus appartenir au model du coup
        if self.verbose:
            print("Calculate gram...")
            print("Use kT:", kT)
        avg_gram = self.trajs.loop_over_traj(self.trajs._compute_gram, self.model)[0]
        # avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        # for weight, xva in zip(self.weights, self.xva_list):
        #     E = self.basis_vector(xva, compute_for="force")
        #     avg_gram += np.matmul(E.T, E) / self.weightsum
        self.invgram = kT * np.linalg.pinv(avg_gram)

        if self.verbose:
            print("Found inverse gram:", self.invgram)
        return kT * avg_gram

    def compute_effective_mass(self, kT=1.0):
        """
        Return average effective mass computed from equipartition with the velocity.
        """
        if self.verbose:
            print("Calculate effective mass...")
            print("Use kT:", kT)
        v2sum = 0.0
        for i, xva in enumerate(self.xva_list):
            v2sum += np.einsum("ik,ij->kj", xva["v"], xva["v"])
        v2 = v2sum / self.weightsum
        self.eff_mass = kT * np.linalg.inv(v2)

        if self.verbose:
            print("Found effective mass:", self.eff_mass)
        return self.eff_mass

    def compute_pos_effective_mass(self, kT=1.0):
        """
        Return position-dependent effective inverse mass
        """
        if self.verbose:
            print("Calculate kernel gram...")
        pos_inv_mass = np.zeros((self.dim_x, self.N_basis_elt_force, self.dim_x))
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            E = self.basis_vector(xva, compute_for="force")
            pos_inv_mass += np.einsum("ik,ij,il->klj", xva["v"], xva["v"], E) / self.weightsum
            avg_gram += np.matmul(E.T, E) / self.weightsum
        self.inv_mass_coeff = kT * np.dot(np.linalg.inv(avg_gram), pos_inv_mass)
        return self.inv_mass_coeff

    def compute_kernel_gram(self):
        """
        Return gram matrix of the kernel part of the basis.
        """
        if self.kernel_gram is None:
            if self.verbose:
                print("Calculate kernel gram...")
            avg_gram = np.zeros((self.N_basis_elt_kernel, self.N_basis_elt_kernel))
            for weight, xva in zip(self.weights, self.xva_list):
                E = self.basis_vector(xva, compute_for="kernel")
                if self.rank_projection:
                    E = np.einsum("kj,ijd->ikd", self.P_range, E)
                avg_gram += np.einsum("ikd,ild->kl", E, E) / self.weightsum  # np.matmul(E.T, E)
            self.kernel_gram = avg_gram
        return self.kernel_gram

    def compute_mean_force(self):
        """
        Computes the mean force from the trajectories.
        """
        if self.verbose:
            print("Calculate mean force...")
        avg_disp = np.zeros((self.N_basis_elt_force, self.dim_obs))
        avg_gram = np.zeros((self.N_basis_elt_force, self.N_basis_elt_force))
        for weight, xva in zip(self.weights, self.xva_list):
            E = self.basis_vector(xva, compute_for="force")
            avg_disp += np.matmul(E.T, xva[self.L_obs].data) / self.weightsum
            avg_gram += np.matmul(E.T, E) / self.weightsum
        self.force_coeff = np.matmul(np.linalg.inv(avg_gram), avg_disp)

    def compute_corrs(self, large=False, rank_tol=None):
        """
        Compute correlation functions.

        Parameters
        ----------
        large : bool, default=False
            When large is true, it use a slower way to compute correlation that is less demanding in memory
        rank_tol: float, default=None
            Tolerance for rank computation in case of projection onto the range of the basis
        """
        if self.verbose:
            print("Calculate correlation functions...")
        if self.force_coeff is None:
            raise Exception("Mean force has not been computed.")

        self.bkbkcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.N_basis_elt_kernel))
        self.bkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.dim_obs))

        self.dotbkdxcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.dim_obs))  # Needed for initial value anyway
        self.dotbkbkcorrw = np.zeros((self.trunc_ind, self.N_basis_elt_kernel, self.N_basis_elt_kernel))

        for weight, xva in zip(self.weights, self.xva_list):
            E_force, E, dE = self.basis_vector(xva)
            # print(E_force.shape, E.shape, dE.shape)
            force = np.matmul(E_force, self.force_coeff)
            # print(force.shape, xva[self.L_obs].data.shape)
            # print(self.bkdxcorrw.shape, correlation_ND(E, (xva[self.L_obs].data - force)).shape)
            if not large:
                try:
                    self.bkdxcorrw += weight * correlation_ND(E, (xva[self.L_obs].data - force), trunc=self.trunc_ind)
                    self.dotbkdxcorrw += weight * correlation_ND(dE, (xva[self.L_obs].data - force), trunc=self.trunc_ind)
                    self.bkbkcorrw += weight * correlation_ND(E, trunc=self.trunc_ind)
                    self.dotbkbkcorrw += weight * correlation_ND(dE, E, trunc=self.trunc_ind)
                except MemoryError:  # If too big slow way
                    if self.verbose:
                        print("Too many basis function, compute correlations one by one (slow)")
                    for n in range(E.shape[1]):
                        for d in range(self.dim_obs):
                            self.bkdxcorrw[:, n, d] += weight * correlation_1D(E[:, n], xva[self.L_obs].data[:, d] - force[:, d], trunc=self.trunc_ind)  # Correlate derivative of observable minus mean value
                            self.dotbkdxcorrw[:, n, d] += weight * correlation_1D(dE[:, n], xva[self.L_obs].data[:, d] - force[:, d], trunc=self.trunc_ind)
                        for m in range(E.shape[1]):
                            self.bkbkcorrw[:, n, m] += weight * correlation_1D(E[:, n], E[:, m], trunc=self.trunc_ind)
                            self.dotbkbkcorrw[:, n, m] += weight * correlation_1D(dE[:, n], E[:, m], trunc=self.trunc_ind)
            else:
                for n in range(E.shape[1]):
                    for d in range(self.dim_obs):
                        self.bkdxcorrw[:, n, d] += weight * correlation_1D(E[:, n], xva[self.L_obs].data[:, d] - force[:, d], trunc=self.trunc_ind)  # Correlate derivative of observable minus mean value
                        self.dotbkdxcorrw[:, n, d] += weight * correlation_1D(dE[:, n], xva[self.L_obs].data[:, d] - force[:, d], trunc=self.trunc_ind)
                    for m in range(E.shape[1]):
                        self.bkbkcorrw[:, n, m] += weight * correlation_1D(E[:, n], E[:, m], trunc=self.trunc_ind)
                        self.dotbkbkcorrw[:, n, m] += weight * correlation_1D(dE[:, n], E[:, m], trunc=self.trunc_ind)

        self.bkbkcorrw /= self.weightsum
        self.bkdxcorrw /= self.weightsum
        self.dotbkdxcorrw /= self.weightsum
        self.dotbkbkcorrw /= self.weightsum

        if self.rank_projection:
            self._set_range_projection(rank_tol)

        if self.saveall:
            np.savetxt(self.prefix + self.corrsdxfile, self.bkdxcorrw.reshape(self.trunc_ind, -1))
            np.savetxt(self.prefix + self.corrsfile, self.bkbkcorrw.reshape(self.trunc_ind, -1))
            np.savetxt(self.prefix + self.dcorrsdxfile, self.dotbkdxcorrw.reshape(self.trunc_ind, -1))
            np.savetxt(self.prefix + self.dcorrsfile, self.dotbkbkcorrw.reshape(self.trunc_ind, -1))


def projection_force(weight, xva, model):
    """
    Do the needed scalar product for one traj
    """
    E = model.basis_vector(xva, compute_for="force")
    avg_disp = np.matmul(E.T, xva[model.L_obs].data)
    avg_gram = np.matmul(E.T, E)
    return avg_disp, avg_gram


def scalar_product_gram(weight, xva, model):
    """
    Do the needed scalar product for one traj
    """
    E = model.basis_vector(xva, compute_for="force")
    avg_gram = np.matmul(E.T, E)
    return avg_gram


def correlation_all(weight, xva, model):
    """
    Do the correlation
    """
    E_force, E, dE = model.basis_vector(xva)
    force = np.matmul(E_force, model.force_coeff)
    bkdxcorrw = weight * correlation_ND(E, (xva[model.L_obs].data - force), trunc=model.trunc_ind)
    dotbkdxcorrw = weight * correlation_ND(dE, (xva[model.L_obs].data - force), trunc=model.trunc_ind)
    bkbkcorrw = weight * correlation_ND(E, trunc=model.trunc_ind)
    dotbkbkcorrw = weight * correlation_ND(dE, E, trunc=model.trunc_ind)
    return bkdxcorrw, dotbkdxcorrw, bkbkcorrw, dotbkbkcorrw


def correlation_large(weight, xva, model):
    """
    Correlation when too much
    """
    E_force, E, dE = model.basis_vector(xva)
    force = np.matmul(E_force, model.force_coeff)
    dim_obs = force.shape[-1]
    N_basis_elt_kernel = E.shape[1]
    bkbkcorrw = np.zeros((model.trunc_ind, N_basis_elt_kernel, N_basis_elt_kernel))
    bkdxcorrw = np.zeros((model.trunc_ind, N_basis_elt_kernel, dim_obs))

    dotbkdxcorrw = np.zeros((model.trunc_ind, N_basis_elt_kernel, dim_obs))
    dotbkbkcorrw = np.zeros((model.trunc_ind, N_basis_elt_kernel, N_basis_elt_kernel))

    for n in range(N_basis_elt_kernel):
        for d in range(model.dim_obs):
            bkdxcorrw[:, n, d] += weight * correlation_1D(E[:, n], xva[model.L_obs].data[:, d] - force[:, d], trunc=model.trunc_ind)  # Correlate derivative of observable minus mean value
            dotbkdxcorrw[:, n, d] += weight * correlation_1D(dE[:, n], xva[model.L_obs].data[:, d] - force[:, d], trunc=model.trunc_ind)
        for m in range(N_basis_elt_kernel):
            bkbkcorrw[:, n, m] += weight * correlation_1D(E[:, n], E[:, m], trunc=model.trunc_ind)
            dotbkbkcorrw[:, n, m] += weight * correlation_1D(dE[:, n], E[:, m], trunc=model.trunc_ind)
    return bkdxcorrw, dotbkdxcorrw, bkbkcorrw, dotbkbkcorrw
