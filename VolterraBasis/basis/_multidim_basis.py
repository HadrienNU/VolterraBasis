"""
This the main estimator module
"""
import numpy as np

from sklearn.base import TransformerMixin


class TensorialBasis2D(TransformerMixin):
    """
    Combine two 1D basis to get a 2D basis
    """

    def __init__(self, b1, b2=None):
        """Take two of basis"""
        self.b1 = b1
        if b2 is None:
            self.b2 = b1
        else:
            self.b2 = b2
        # Force rank projection
        self.b1.const_removed = False
        self.b2.const_removed = False
        self.const_removed = False

    def fit(self, X, y=None):
        self.b1.fit(X[:, slice(0, 1)])
        self.b2.fit(X[:, slice(1, 2)])
        self.n_output_features_ = self.b1.n_output_features_ * self.b2.n_output_features_
        return self

    def basis(self, X):
        nsamples, nfeatures = X.shape
        features = np.einsum("nk,nl->nkl", self.b1.basis(X[:, slice(0, 1)]), self.b2.basis(X[:, slice(1, 2)])).reshape(nsamples, -1)
        return features

    def deriv(self, X, deriv_order=1):
        if deriv_order == 2:
            return self.hessian(X)
        nsamples, nfeatures = X.shape
        temp_arr_1 = np.einsum("nk,nl...->nkl...", self.b1.basis(X[:, slice(0, 1)]), self.b2.deriv(X[:, slice(1, 2)], deriv_order=deriv_order))
        grad_1 = temp_arr_1.reshape(nsamples, -1, *temp_arr_1.shape[3:])
        temp_arr_2 = np.einsum("nk...,nl->nkl...", self.b1.deriv(X[:, slice(0, 1)], deriv_order=deriv_order), self.b2.basis(X[:, slice(1, 2)]))
        grad_2 = temp_arr_2.reshape(nsamples, -1, *temp_arr_2.shape[3:])
        return np.concatenate((grad_1, grad_2), axis=-1)  # Il faudrait concatenate le long de la diagonal

    def hessian(self, X):
        nsamples, nfeatures = X.shape
        temp_arr_1 = np.einsum("nk,nl...->nkl...", self.b1.basis(X[:, slice(0, 1)]), self.b2.hessian(X[:, slice(1, 2)]))
        hess_1 = temp_arr_1.reshape(nsamples, -1, *temp_arr_1.shape[3:])
        temp_arr_cross = np.einsum("nkd,nlf->nkldf", self.b1.deriv(X[:, slice(0, 1)]), self.b2.deriv(X[:, slice(1, 2)]))
        hess_cross = temp_arr_cross.reshape(nsamples, -1, *temp_arr_cross.shape[3:])
        temp_arr_2 = np.einsum("nk...,nl->nkl...", self.b1.hessian(X[:, slice(0, 1)]), self.b2.basis(X[:, slice(1, 2)]))
        hess_2 = temp_arr_2.reshape(nsamples, -1, *temp_arr_2.shape[3:])
        print(hess_1.shape, hess_cross.shape, hess_2.shape)
        return np.concatenate((np.concatenate((hess_1, hess_cross), axis=-2), np.concatenate((hess_cross, hess_2), axis=-2)), axis=-1)  # Il faudrait concatenate le long de la diagonal


#
# class TensorialBasis(TransformerMixin):
#     """
#     Combine several 1D basis to get a multidimensionnal basis
#     """
#
#     def __init__(self, *basis: TransformerMixin):
#         """Take set of basis"""
#         if len(basis) <= 1:
#             raise ValueError("Not enough basis in Tensorial Basis")
#         self.basis_set = basis
#         self.dim = len(basis)
#
#     def fit(self, X, y=None):
#         for i, b in enumerate(self.basis_set):
#             b.fit(X[:, slice(i, i + 1)])
#         return self
#
#     def basis(self, X):
#         nsamples, nfeatures = X.shape
#         features = self.basis_set[0].basis(X[:, slice(0, 1)])
#         for i, b in enumerate(self.basis_set[1:]):
#             features = np.einsum("nk,nl->nkl", features, b.basis(X[:, slice(i, i + 1)])).reshape(nsamples, -1)
#         return features
#
#     def deriv(self, X, deriv_order=1):
#         nsamples, nfeatures = X.shape
#         features = self.basis_set[0].basis(X[:, slice(0, 1)])  # (ntimes x nb features)
#         grad = self.basis_set[0].deriv(X[:, slice(0, 1)])  # (ntimes x nb features x 1) ->  (ntimes x nb features x dim_x)
#         # grad = np.concatenate([np.kron(f1.grad, f2.value[np.newaxis, :, :]), np.kron(f1.value[np.newaxis, :, :], f2.grad)], axis=0)
#         for i, b in enumerate(self.basis_set[1:]):
#             # On doit multiplier grad à gauche par ba_f et ajouter une line qui est features (avant kronecker product x grad)
#             ba_f = b.basis(X[:, slice(i, i + 1)])
#             grad_f = self.basis_set[0].deriv(X[:, slice(i, i + 1)])
#
#             grad = np.einsum("nkl,nj-> nkjl", grad, ba_f)
#             # On ajoute ensuite
#             to_concatenate = np.einsum("ni,njk-> nijk", features, grad_f).reshape(nsamples, -1, i + 1)
#             # Puis on concatenate grad et to_concatenate
#             grapd = np.concatenate(grad, to_concatenate)
#             features = np.einsum("nk,nl->nkl", features, ba_f).reshape(nsamples, -1)
#
#         return grad
#
#     def hessian(self, X):
#         return self.deriv(X, deriv_order=2)


if __name__ == "__main__":
    from _local_features import BSplineFeatures

    # Plot basis
    x_range = np.linspace(-10, 10, 30).reshape(-1, 2)
    n_elems = 7
    ten_basis = TensorialBasis2D(BSplineFeatures(n_elems)).fit(x_range)
    print(x_range.shape)
    print("Basis")
    print(ten_basis.basis(x_range).shape)
    print("Deriv")
    print(ten_basis.deriv(x_range).shape)
    print("Hessian")
    print(ten_basis.hessian(x_range).shape)
