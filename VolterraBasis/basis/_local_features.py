"""
This the main estimator module
"""
import numpy as np
import scipy.interpolate
import scipy.stats
from sklearn.base import TransformerMixin
from ._data_describe import quick_describe, minimal_describe


def _get_bspline_basis(knots, degree=3, periodic=False):
    """Get spline coefficients for each basis spline."""
    nknots = len(knots)
    y_dummy = np.zeros(nknots)

    knots, coeffs, degree = scipy.interpolate.splrep(knots, y_dummy, k=degree, per=periodic)
    ncoeffs = len(coeffs)
    bsplines = []
    for ispline in range(nknots):
        coeffs = np.asarray([1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)])
        bsplines.append((knots, coeffs, degree))
    return bsplines


class BSplineFeatures(TransformerMixin):
    """
    B splines features
    """

    def __init__(self, n_knots=5, k=3, periodic=False, remove_const=True):
        """
        Parameters
        ----------
            n_knots : int
                Number of knots to use
            k : int
                Degree of the splines
            periodic: bool
                Whatever to use periodic splines or not
        """
        self.periodic = periodic
        self.k = k
        self.n_knots = n_knots  # knots are position along the axis of the knots
        self.const_removed = remove_const

    def fit(self, describe_result, knots=None):
        if isinstance(describe_result, np.ndarray):
            describe_result = minimal_describe(describe_result)
        dim = describe_result.mean.shape[0]
        # TODO determine non uniform position of knots given the datas
        if knots is None:
            knots = np.linspace(describe_result.minmax[0], describe_result.minmax[1], self.n_knots)
        self.bsplines_ = _get_bspline_basis(knots, self.k, periodic=self.periodic)
        self._nsplines = len(self.bsplines_)
        self.n_output_features_ = len(self.bsplines_) * dim
        return self

    def basis(self, X):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, self.n_output_features_))
        for ispline, spline in enumerate(self.bsplines_):
            istart = ispline * dim
            iend = (ispline + 1) * dim
            features[:, istart:iend] = scipy.interpolate.splev(X, spline)
        return features

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        with_const = int(self.const_removed)
        features = np.zeros((nsamples, dim * (self._nsplines - with_const)) + (dim,) * deriv_order)
        if self.k < deriv_order:
            return features
        for ispline, spline in enumerate(self.bsplines_[: len(self.bsplines_) - with_const]):
            istart = (ispline) * dim
            for i in range(dim):
                features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * deriv_order] = scipy.interpolate.splev(X[:, slice(i, i + 1)], spline, der=deriv_order)
        return features

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, self.n_output_features_))
        for ispline, spline in enumerate(self.bsplines_):
            istart = ispline * dim
            iend = (ispline + 1) * dim
            spline_int = scipy.interpolate.splantider(spline, n=order)
            features[:, istart:iend] = scipy.interpolate.splev(X, spline_int)
        return features


def quartic(u, der=0):
    """
    Interpolating function between 0 and 1
    Return values or derivatives
    """
    u2 = 1 - u ** 2
    if der == 0:
        return u2 ** 2
    elif der == 1:
        return -4 * u * u2
    elif der == 2:
        return 12 * u ** 2 - 4


def triweight(u, der=0):
    """
    Interpolating function between 0 and 1
    Return values or derivatives
    """
    u2 = 1 - u ** 2
    if der == 0:
        return u2 ** 3
    elif der == 1:
        return -6 * u * u2 ** 2
    elif der == 2:
        return -6 * u2 ** 2 + 24 * u ** 2 * u2


def tricube(u, der=0):
    """
    Interpolating function between 0 and 1
    Return values or derivatives
    """
    u3 = 1 - u ** 3
    if der == 0:
        return u3 ** 3
    elif der == 1:
        return -9 * u ** 2 * u3 ** 2
    elif der == 2:
        return 54 * u ** 4 * u3 - 18 * u * u3 ** 2


class SmoothIndicatorFeatures(TransformerMixin):
    """
    Indicator function with smooth boundary
    """

    def __init__(self, states_boundary, boundary_type="tricube", periodic=False):
        """
        Parameters
        ----------
            states_boundary : list
                Number of knots to use
            boundary_type : str or callable
                Function to use for the interpolation between zeros and one value
                If this is a callabe function, first argument is between 0-> 1 and 1 -> 0 and second one is the order of the derivative
            periodic: bool
                Whatever to use periodic indicator function. If yes, the last indicator will sbe the same function than the first one
        """
        self.periodic = periodic
        self.states_boundary = states_boundary
        self.n_states = len(states_boundary)  # -1 ? ca dépend si c'est périodique ou pas
        self.const_removed = False
        if boundary_type == "tricube":
            self.boundary = tricube
        elif boundary_type == "triweight":
            self.boundary = triweight
        elif boundary_type == "quartic":
            self.boundary = quartic
        elif callable(boundary_type):
            self.boundary = boundary_type
        else:
            raise ValueError("Not valable boundary")

    def fit(self, describe_result):
        if isinstance(describe_result, np.ndarray):
            describe_result = quick_describe(describe_result)
        dim = describe_result.mean.shape[0]
        if dim > 1:
            raise ValueError("This basis does not support dimension higher than 1. Try to combine it using TensorialBasis2D")
        self.n_output_features_ = len(self.states_boundary) + (not self.periodic)
        return self

    def basis(self, X):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, self.n_output_features_))

        a = min(self.states_boundary[0])  # This way there is no ambiguity on the definition
        b = max(self.states_boundary[0])
        u = (X - a) / (b - a)
        occ = self.boundary(u)
        occ = np.where(u < 0, 1, occ)
        occ = np.where(u > 1, 0, occ)
        features[:, 0:1] = occ
        occ_next = 1 - occ

        for n in range(1, self.n_states):
            # right boundary
            a = min(self.states_boundary[n])  # This way there is no ambiguity on the definition
            b = max(self.states_boundary[n])
            u = (X - a) / (b - a)
            occ = self.boundary(u)
            occ = np.where(u > 1, 0, occ)

            features[:, n : n + 1] = np.where(u < 0, occ_next, occ)
            occ_next = 1 - np.where(u < 0, 1, occ)

        if self.periodic:
            features[:, 0:1] = np.where(occ_next > 0, occ_next, features[:, 0:1])
        else:
            features[:, self.n_states : self.n_states + 1] = occ_next
        return features

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, self.n_output_features_) + (1,) * deriv_order)

        a = min(self.states_boundary[0])  # This way there is no ambiguity on the definition
        b = max(self.states_boundary[0])
        u = (X - a) / (b - a)
        der = self.boundary(u, deriv_order) / (b - a) ** deriv_order
        der = np.where(u < 0, 0, der)
        der = np.where(u > 1, 0, der)

        features[(Ellipsis, slice(0, 1)) + (0,) * deriv_order] = der
        der_next = -der

        for n in range(1, self.n_states):
            # right boundary
            a = min(self.states_boundary[n])  # This way there is no ambiguity on the definition
            b = max(self.states_boundary[n])
            u = (X - a) / (b - a)
            der = self.boundary(u, deriv_order) / (b - a) ** deriv_order
            der = np.where(u > 1, 0, der)

            features[(Ellipsis, slice(n, n + 1)) + (0,) * deriv_order] = np.where(u < 0, der_next, der)
            der_next = -np.where(u < 0, 0, der)

        if self.periodic:
            features[(Ellipsis, slice(0, 1)) + (0,) * deriv_order] = np.where(der_next != 0.0, der_next, features[(Ellipsis, slice(0, 1)) + (0,) * deriv_order])
        else:
            features[(Ellipsis, slice(self.n_states, self.n_states + 1)) + (0,) * deriv_order] = der_next
        return features
        return features

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        raise NotImplementedError


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_range = np.linspace(-10, 10, 10).reshape(-1, 1)
    # basis = BSplineFeatures(6, k=3)
    basis = SmoothIndicatorFeatures([[-5, -2], [2, 3], [5, 7]], "tricube", periodic=True)
    basis.fit(x_range)
    print(x_range.shape)
    print("Basis")
    print(basis.basis(x_range).shape)
    print("Deriv")
    print(basis.deriv(x_range).shape)
    print("Hessian")
    print(basis.hessian(x_range).shape)

    # Plot basis
    x_range = np.linspace(-10, 10, 150).reshape(-1, 1)
    # basis = basis.fit(x_range)
    # basis = LinearFeatures().fit(x_range)
    y = basis.basis(x_range)
    z = basis.deriv(x_range)
    plt.grid()
    for n in range(y.shape[1]):
        plt.plot(x_range[:, 0], y[:, n])
        plt.plot(x_range[:, 0], z[:, n, 0])
    plt.show()
