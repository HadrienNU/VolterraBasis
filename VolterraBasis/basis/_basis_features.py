"""
This the main estimator module
"""
import numpy as np

import scipy.interpolate
import scipy.stats

from sklearn.base import TransformerMixin

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KDTree


def freedman_diaconis(data):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin number.

    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR = scipy.stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
    N = data.size
    bw = (2 * IQR) / np.power(N, 1 / 3)

    datmin, datmax = data.min(), data.max()
    datrng = datmax - datmin
    return int((datrng / bw) + 1)


def _get_bspline_basis(knots, degree=3, periodic=False):
    """Get spline coefficients for each basis spline."""
    nknots = len(knots)
    y_dummy = np.zeros(nknots)

    knots, coeffs, degree = scipy.interpolate.splrep(knots, y_dummy, k=degree, per=periodic)
    ncoeffs = len(coeffs)
    bsplines = []
    for ispline in range(nknots):
        coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
        bsplines.append((knots, coeffs, degree))
    return bsplines


class BSplineFeatures(TransformerMixin):
    def __init__(self, n_knots=5, degree=3, periodic=False):
        self.periodic = periodic
        self.degree = degree
        self.n_knots = n_knots  # knots are position along the axis of the knots

    def fit(self, X, y=None, knots=None):
        # TODO determine position of knots given the datas
        if knots is None:
            knots = np.linspace(np.min(X), np.max(X), self.n_knots)
        self.bsplines_ = _get_bspline_basis(knots, self.degree, periodic=self.periodic)
        self.n_output_features_ = len(self.bsplines_)
        return self

    def basis(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.n_output_features_))
        for ispline, spline in enumerate(self.bsplines_):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = scipy.interpolate.splev(X, spline)
        return features

    def deriv(self, X, deriv_order=1, remove_const=False):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.n_output_features_))
        for ispline, spline in enumerate(self.bsplines_):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = scipy.interpolate.splev(X, spline, der=deriv_order)  # To check
        if remove_const:
            return features[:, :-1]  # Remove the last element
            # proj = np.zeros((nfeatures * (self.n_output_features_ - 1), nfeatures * self.n_output_features_))
            # # Find the correct projection
            # # proj[:, :-1] = np.identity(nfeatures * (self.n_output_features_ - 1))
            # # proj -= np.ones((nfeatures * (self.n_output_features_ - 1), nfeatures * self.n_output_features_)) / (nfeatures * self.n_output_features_)
            # return np.matmul(features, proj.T)  # Do the projection on the image of the derivative
        else:
            return features

    def hessian(self, X, remove_const=False):
        return self.deriv(X, deriv_order=2, remove_const=remove_const)


class SplineFctFeatures(TransformerMixin):
    """
    Using fit of free energy as basis
    """

    def __init__(self, knots, coeffs, k=3, periodic=False):
        self.periodic = periodic
        self.k = k
        self.t = knots  # knots are position along the axis of the knots
        self.c = coeffs

    def fit(self, X, y=None):
        self.spl = scipy.interpolate.BSpline(self.t, self.c, self.k)
        self.n_output_features_ = 1
        return self

    def basis(self, X):
        return self.spl(X)

    def deriv(self, X, deriv_order=1, remove_const=False):
        return self.spl.derivative(deriv_order)(X)

    def hessian(self, X, remove_const=False):
        return self.deriv(X, deriv_order=2, remove_const=remove_const)

    def antiderivative(self, X, order=1):
        return self.spl.antiderivative(order)(X)


class SplineFctWithLinFeatures(TransformerMixin):
    """
    Using fit of free energy as basis
    """

    def __init__(self, knots, coeffs, k=3, periodic=False):
        self.periodic = periodic
        self.k = k
        self.t = knots  # knots are position along the axis of the knots
        self.c = coeffs

    def fit(self, X, y=None):
        self.spl = scipy.interpolate.BSpline(self.t, self.c, self.k)
        return self

    def basis(self, X):
        return np.concatenate((X, self.spl(X)), axis=1)

    def deriv(self, X, deriv_order=1, remove_const=False):
        if deriv_order == 1:
            lin_deriv = np.ones_like(X)
        else:
            lin_deriv = np.zeros_like(X)
        return np.concatenate((lin_deriv, self.spl.derivative(deriv_order)(X)), axis=1)

    def hessian(self, X, remove_const=False):
        return self.deriv(X, deriv_order=2, remove_const=remove_const)

    def antiderivative(self, X, order=1):
        return self.spl.antiderivative(order)(X)


class BinsFeatures(KBinsDiscretizer):
    def __init__(self, n_bins_arg="auto", strategy="uniform"):
        """
        Init class
        """
        super().__init__(encode="onehot-dense", strategy=strategy)
        self.n_bins_arg = n_bins_arg

    def fit(self, X, y=None):
        """
        Determine bin number
        """
        nsamples, dim_x = X.shape
        if self.n_bins_arg == "auto":  # Automatique determination of the number of bins via maximum of sturges and freedman diaconis rules
            # Sturges rules
            self.n_bins = 1 + np.log2(nsamples)
            for d in range(dim_x):
                # Freedman–Diaconis rule
                n_bins = max(self.n_bins, freedman_diaconis(X[:, d]))
            self.n_bins = int(n_bins)
        elif isinstance(self.n_bins_arg, int):
            self.n_bins = self.n_bins_arg
        else:
            raise ValueError("The number of bins must be an integer")
        super().fit(X, y)
        self.n_output_features_ = self.n_bins
        return self

    def basis(self, X):
        return self.transform(X)

    def deriv(self, X, remove_const=False):
        nsamples, nfeatures = X.shape
        return np.zeros((nsamples, nfeatures * self.n_output_features_))

    def hessian(self, X, remove_const=False):
        nsamples, nfeatures = X.shape
        return np.zeros((nsamples, nfeatures * self.n_output_features_))


class LinearElement(object):
    """1D element with linear basis functions.

    Attributes:
        index (int): Index of the element.
        x_l (float): x-coordinate of the left boundary of the element.
        x_r (float): x-coordinate of the right boundary of the element.
    """

    def __init__(self, index, x_left, x_center, x_right):
        self.num_nodes = 2
        self.index = index
        self.x_left = x_left
        self.x_center = x_center
        self.x_right = x_right
        self.center = np.asarray([0.5 * (self.x_right + self.x_left)])
        self.size = 0.5 * (self.x_right - self.x_left)

    def basis_function(self, x):
        x = np.asarray(x)
        return ((x >= self.x_left) & (x < self.x_center)) * (x - self.x_left) / (self.x_center - self.x_left) + ((x >= self.x_center) & (x < self.x_right)) * (self.x_right - x) / (self.x_right - self.x_center)

    def deriv_function(self, x):
        x = np.asarray(x)
        return ((x >= self.x_left) & (x < self.x_center)) / (self.x_center - self.x_left) - ((x >= self.x_center) & (x < self.x_right)) / (self.x_right - self.x_center)


class FEM1DFeatures(TransformerMixin):
    def __init__(self, mesh, periodic=False):
        self.periodic = periodic
        # Add two point for start and end point
        extra_point_start = 2 * mesh.x[0] - mesh.x[1]
        extra_point_end = 2 * mesh.x[-1] - mesh.x[-2]
        x_dat = np.concatenate((np.array([extra_point_start]), mesh.x, np.array([extra_point_end])))
        # Create list of instances of Element
        self.elements = [LinearElement(i, x_dat[i], x_dat[i + 1], x_dat[i + 2]) for i in range(len(x_dat) - 2)]
        self.num_elements = len(self.elements)

    def fit(self, X, y=None):
        self.tree = KDTree(X)
        return self

    def basis(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, self.num_elements))
        for k, element in enumerate(self.elements):
            istart = k  # * nfeatures
            iend = k + 1  # * nfeatures
            features[:, istart:iend] = element.basis_function(X)
        return features

    def deriv(self, X, remove_const=False):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, self.num_elements))
        for k, element in enumerate(self.elements):
            istart = k  # * nfeatures
            iend = k + 1  # * nfeatures
            features[:, istart:iend] = element.deriv_function(X)
        return features


class LinearFeatures(TransformerMixin):
    def __init__(self, to_center=False):
        """"""
        self.centered = to_center

    def fit(self, X, y=None):
        self.n_output_features_ = X.shape[1]
        if self.centered:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros((self.n_output_features_,))
        return self

    def basis(self, X):
        return X - self.mean_

    def deriv(self, X, remove_const=True):
        return np.ones_like(X)

    def hessian(self, X, remove_const=True):
        return np.zeros_like(X)


class PolynomialFeatures(TransformerMixin):
    """
    Wrapper for numpy polynomial series removing the constant polynom
    """

    def __init__(self, deg=1, polynom=np.polynomial.Polynomial):
        self.degree = deg + 1
        self.polynom = polynom

    def basis(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.degree))
        for n in range(0, self.degree):
            istart = n * nfeatures
            iend = (n + 1) * nfeatures
            features[:, istart:iend] = self.polynom.basis(n)(X)
        return features

    def deriv(self, X, deriv_order=1, remove_const=False):
        nsamples, nfeatures = X.shape
        with_const = int(remove_const)
        features = np.zeros((nsamples, nfeatures * (self.degree - with_const)))
        for n in range(with_const, self.degree):
            istart = (n - with_const) * nfeatures
            iend = (n + 1 - with_const) * nfeatures
            features[:, istart:iend] = self.polynom.basis(n).deriv(deriv_order)(X)
        return features

    def hessian(self, X, remove_const=False):
        return self.deriv(X, deriv_order=2, remove_const=remove_const)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.interpolate

    # Plot basis
    x_range = np.linspace(-10, 10, 15).reshape(-1, 1)
    n_elems = 15
    basis = BSplineFeatures(n_elems).fit(x_range)
    print(basis.bsplines_[0][0])
    print(len(basis.bsplines_[0][0]))
    plt.plot(np.arange(n_elems + 4), basis.bsplines_[0][0])
    c = scipy.interpolate.make_interp_spline(x_range[:, 0], x_range[:, 0], t=basis.bsplines_[0][0]).c
    plt.plot(np.arange(n_elems), c, "-x")
    print(c)
    y = basis.deriv(x_range)
    plt.plot(x_range[:, 0], c @ y.T)
    # for n in range(y.shape[1]):
    #     plt.plot(x_range[:, 0], y[:, n])
    plt.show()
