"""
This the main estimator module
"""
import numpy as np
import scipy.interpolate
from sklearn.base import TransformerMixin


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
    """
    B splines features
    """

    def __init__(self, n_knots=5, k=3, periodic=False):
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

    def fit(self, X, y=None, knots=None):
        nsamples, dim = X.shape
        # TODO determine non uniform position of knots given the datas
        if knots is None:
            knots = np.linspace(np.min(X), np.max(X), self.n_knots)
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

    def deriv(self, X, deriv_order=1, remove_const=False):
        nsamples, dim = X.shape
        with_const = int(remove_const)
        features = np.zeros((nsamples, dim * (self._nsplines - with_const)) + (dim,) * deriv_order)
        if self.k < deriv_order:
            return features
        for ispline, spline in enumerate(self.bsplines_[: len(self.bsplines_) - with_const]):
            istart = (ispline) * dim
            for i in range(dim):
                features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * deriv_order] = scipy.interpolate.splev(X[:, slice(i, i + 1)], spline, der=deriv_order)
        return features

    def hessian(self, X, remove_const=False):
        return self.deriv(X, deriv_order=2, remove_const=remove_const)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_range = np.linspace(-10, 10, 10).reshape(-1, 2)
    basis = BSplineFeatures(6, k=3)
    basis.fit(x_range)
    print(x_range.shape)
    print("Basis")
    print(basis.basis(x_range).shape)
    print("Deriv")
    print(basis.deriv(x_range, remove_const=True).shape)
    print("Hessian")
    print(basis.hessian(x_range, remove_const=True).shape)

    # Plot basis
    x_range = np.linspace(-2, 2, 50).reshape(-1, 1)
    basis = basis.fit(x_range)
    # basis = LinearFeatures().fit(x_range)
    y = basis.basis(x_range)
    plt.grid()
    for n in range(y.shape[1]):
        plt.plot(x_range[:, 0], y[:, n])
    plt.show()
