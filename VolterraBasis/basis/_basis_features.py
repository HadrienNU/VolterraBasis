"""
This the main estimator module
"""
import numpy as np

import scipy.interpolate
import scipy.stats
from ._data_describe import quick_describe, minimal_describe

from sklearn.base import TransformerMixin


class LinearFeatures(TransformerMixin):
    """
    Linear function
    """

    def __init__(self, to_center=False):
        """"""
        self.centered = to_center
        self.const_removed = False

    def fit(self, describe_result):
        if isinstance(describe_result, np.ndarray):
            describe_result = minimal_describe(describe_result)
        self.n_output_features_ = describe_result.mean.shape[0]
        if self.centered:
            self.mean_ = describe_result.mean
        else:
            self.mean_ = np.zeros((self.n_output_features_,))
        return self

    def basis(self, X):
        return X - self.mean_

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        grad = np.zeros((nsamples, dim) + (dim,) * deriv_order)
        if deriv_order == 1:
            for i in range(dim):
                grad[:, i, i] = 1.0
        return grad

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        return 0.5 * np.power(X, 2)


class PolynomialFeatures(TransformerMixin):
    """
    Wrapper for numpy polynomial series.
    """

    def __init__(self, deg=1, polynom=np.polynomial.Polynomial, remove_const=True):
        """
        Providing a numpy polynomial class via polynom keyword allow to change polynomial type.
        """
        self.degree = deg + 1
        self.polynom = polynom
        self.const_removed = remove_const

    def fit(self, describe_result):
        if isinstance(describe_result, np.ndarray):
            describe_result = quick_describe(describe_result)
        self.n_output_features_ = describe_result.mean.shape[0] * self.degree
        return self

    def basis(self, X):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, dim * self.degree))
        for n in range(0, self.degree):
            istart = n * dim
            iend = (n + 1) * dim
            features[:, istart:iend] = self.polynom.basis(n)(X)
        return features

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        with_const = int(self.const_removed)
        features = np.zeros((nsamples, dim * (self.degree - with_const)) + (dim,) * deriv_order)
        for n in range(with_const, self.degree):
            istart = (n - with_const) * dim
            for i in range(dim):
                features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * deriv_order] = self.polynom.basis(n).deriv(deriv_order)(X[:, slice(i, i + 1)])
        return features

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, dim * self.degree))
        for n in range(0, self.degree):
            istart = n * dim
            iend = (n + 1) * dim
            features[:, istart:iend] = self.polynom.basis(n).integ(order)(X)
        return features


class FourierFeatures(TransformerMixin):
    """
    Fourier series.
    """

    def __init__(self, order=1, freq=1.0, remove_const=True):
        """
        Parameters
        ----------
        order :  int
            Order of the Fourier series
        freq: float
            Base frequency
        """
        self.order = 2 * order + 1
        self.freq = freq
        self.const_removed = remove_const

    def fit(self, describe_result):
        if isinstance(describe_result, np.ndarray):
            describe_result = quick_describe(describe_result)
        self.n_output_features_ = describe_result.mean.shape[0] * self.order
        return self

    def basis(self, X):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, dim * self.order))
        for n in range(0, self.order):
            istart = n * dim
            iend = (n + 1) * dim
            if n == 0:
                features[:, istart:iend] = np.ones_like(X) / np.sqrt(2 * np.pi)
            elif n % 2 == 0:
                # print(n / 2)
                features[:, istart:iend] = np.cos(n / 2 * X * self.freq) / np.sqrt(np.pi)
            else:
                # print((n + 1) / 2)
                features[:, istart:iend] = np.sin((n + 1) / 2 * X * self.freq) / np.sqrt(np.pi)
        return features

    def deriv(self, X, deriv_order=1):
        if deriv_order == 2:
            return self.hessian(X)
        nsamples, dim = X.shape
        with_const = int(self.const_removed)
        features = np.zeros((nsamples, dim * (self.order - with_const)) + (dim,) * deriv_order)
        for n in range(with_const, self.order):
            istart = (n - with_const) * dim
            for i in range(dim):
                if n % 2 == 0:
                    features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * deriv_order] = -n / 2 * self.freq * np.sin(n / 2 * self.freq * X[:, slice(i, i + 1)]) / np.sqrt(np.pi)
                else:
                    features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * deriv_order] = (n + 1) / 2 * self.freq * np.cos((n + 1) / 2 * self.freq * X[:, slice(i, i + 1)]) / np.sqrt(np.pi)
        return features

    def hessian(self, X):
        nsamples, dim = X.shape
        with_const = int(self.const_removed)
        features = np.zeros((nsamples, dim * (self.order - with_const)) + (dim,) * 2)
        for n in range(with_const, self.order):
            istart = (n - with_const) * dim
            for i in range(dim):
                if n % 2 == 0:
                    features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * 2] = -(((n / 2) * self.freq) ** 2) * np.cos(n / 2 * self.freq * X[:, slice(i, i + 1)]) / np.sqrt(np.pi)
                else:
                    features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * 2] = -(((n + 1) / 2 * self.freq) ** 2) * np.sin((n + 1) / 2 * self.freq * X[:, slice(i, i + 1)]) / np.sqrt(np.pi)
        return features

    def antiderivative(self, X, order=1):
        if order > 1:
            raise NotImplementedError
        nsamples, dim = X.shape
        features = np.zeros((nsamples, dim * self.order))
        for n in range(0, self.order):
            istart = n * dim
            iend = (n + 1) * dim
            if n == 0:
                features[:, istart:iend] = X / np.sqrt(2 * np.pi)
            elif n % 2 == 0:
                # print(n / 2)
                features[:, istart:iend] = np.sin(n / 2 * X * self.freq) / (np.sqrt(np.pi) * n / 2 * self.freq)
            else:
                # print((n + 1) / 2)
                features[:, istart:iend] = -1 * np.cos((n + 1) / 2 * X * self.freq) / (np.sqrt(np.pi) * (n + 1) / 2 * self.freq)
        return features


class SplineFctFeatures(TransformerMixin):
    """
    A single basis function that is given from splines fit of data
    """

    def __init__(self, knots, coeffs, k=3, periodic=False):
        self.periodic = periodic
        self.k = k
        self.t = knots  # knots are position along the axis of the knots
        self.c = coeffs
        self.const_removed = False

    def fit(self, describe_result):
        if isinstance(describe_result, np.ndarray):
            describe_result = quick_describe(describe_result)
        self.spl_ = scipy.interpolate.BSpline(self.t, self.c, self.k)
        self.n_output_features_ = describe_result.mean.shape[0]
        return self

    def basis(self, X):
        return self.spl_(X)

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        grad = np.zeros((nsamples, dim) + (dim,) * deriv_order)
        for i in range(dim):
            grad[(Ellipsis, slice(i, i + 1)) + (i,) * (deriv_order)] = self.spl_.derivative(deriv_order)(X[:, slice(i, i + 1)])
        return grad

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        return self.spl_.antiderivative(order)(X)


class FeaturesCombiner(TransformerMixin):
    """
    Allow to combine features to build composite basis
    """

    def __init__(self, *basis):
        self.basis_set = basis
        self.const_removed = np.any([b.const_removed for b in self.basis_set])  # Check if one of the basis set have the constant removed

    def fit(self, describe_result):
        if isinstance(describe_result, np.ndarray):
            describe_result = scipy.stats.describe(describe_result)
        for b in self.basis_set:
            b.fit(describe_result)
        self.n_output_features_ = np.sum([b.n_output_features_ for b in self.basis_set])
        return self

    def basis(self, X):
        features = self.basis_set[0].basis(X)
        for b in self.basis_set[1:]:
            features = np.concatenate((features, b.basis(X)), axis=1)
        return features

    def deriv(self, X, deriv_order=1):
        grad = self.basis_set[0].deriv(X, deriv_order=deriv_order)
        for b in self.basis_set[1:]:
            # print(grad.shape, b.deriv(X, deriv_order=deriv_order).shape)
            features = np.concatenate((grad, b.deriv(X, deriv_order=deriv_order)), axis=1)
        return features

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        raise NotImplementedError("Don't try this")


# class SplineFctWithLinFeatures(TransformerMixin):
#     """
#     Combine a basis function that is given from splines fit of data with linear function
#     """
#
#     def __init__(self, knots, coeffs, k=3, periodic=False):
#         self.periodic = periodic
#         self.k = k
#         self.t = knots  # knots are position along the axis of the knots
#         self.c = coeffs
#
#     def fit(self, X, y=None):
#         nsamples, dim = X.shape
#         self.spl_ = scipy.interpolate.BSpline(self.t, self.c, self.k)
#         self.n_output_features_ = 2 * dim
#         return self
#
#     def basis(self, X):
#         return np.concatenate((X, self.spl_(X)), axis=1)
#
#     def deriv(self, X, deriv_order=1):
#         if deriv_order == 1:
#             lin_deriv = np.ones_like(X)
#         else:
#             lin_deriv = np.zeros_like(X)
#         return np.concatenate((lin_deriv, self.spl_.derivative(deriv_order)(X)), axis=1)
#
#     def hessian(self, X):
#         return self.deriv(X, deriv_order=2)
#
#     def antiderivative(self, X, order=1):
#         return self.spl_.antiderivative(order)(X)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import describe

    x_range = np.linspace(-10, 10, 30).reshape(-1, 2)
    # b2 = LinearFeatures()
    basis = FourierFeatures(order=3, freq=1.0)
    # b1 = SplineFctFeatures(knots=np.linspace(-1, 1, 8), coeffs=np.logspace(1, 2, 8), k=2)
    # basis = FeaturesCombiner(b1, b2)
    basis.fit(describe(x_range))
    print(x_range.shape)
    print("Basis")
    print(basis.basis(x_range).shape)
    print("Deriv")
    print(basis.deriv(x_range).shape)
    print(basis.deriv(x_range)[0, :, :])
    print("Hessian")
    print(basis.hessian(x_range).shape)

    # Plot basis
    x_range = np.linspace(-2, 2, 50).reshape(-1, 1)
    basis = basis.fit(x_range)
    # basis = LinearFeatures().fit(x_range)
    y = basis.basis(x_range)
    plt.grid()
    for n in range(y.shape[1]):
        plt.plot(x_range[:, 0], y[:, n])
    plt.show()
