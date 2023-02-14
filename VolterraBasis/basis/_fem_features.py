"""
This the main estimator module
"""
import numpy as np

from sklearn.base import TransformerMixin
from scipy.spatial import cKDTree

from skfem.assembly.basis import Basis
from skfem import LinearForm, BilinearForm
from skfem.helpers import dot
from itertools import product
import sparse
from ._data_describe import quick_describe, minimal_describe


class ElementFinder:
    """
    Class that find the correct element given a location
    """

    def __init__(self, mesh, mapping=None):
        # Get dimension from mesh
        self.dim = mesh.dim()
        # Transform mesh to triangular version if needed
        if self.dim == 1:
            self.max_point = np.max(mesh.p)  # To avoid strict < in np.digitize
            ix = np.argsort(mesh.p)
            self.bins = mesh.p[0, ix[0]]
            self.bins_idx = mesh.t[0]
            self.find = self.find_element_1D
        # elif self.dim == 2:
        #     self.find = self.find_element_2D
        #     self.mesh_finder = mesh.element_finder(mapping)
        else:
            self.tree = cKDTree(np.mean(mesh.p[:, mesh.t], axis=1).T)
            self.find = self.find_element_ND
            self.mapping = mesh._mapping() if mapping is None else mapping
            if self.dim == 2:  # We should also check the type of element
                self.inside = self.inside_2D
            elif self.dim == 3:
                self.inside = self.inside_3D

    def find_element_1D(self, X):
        """
        Assuming X is nsamples x 1
        """
        maxix = X[:, 0] == self.max_point
        X[maxix, 0] = X[maxix, 0] - 1e-10  # special case in np.digitize
        return np.argmax((np.digitize(X[:, 0], self.bins) - 1)[:, None] == self.bins_idx, axis=1)

    def find_element_2D(self, X):
        return self.mesh_finder(X[:, 0], X[:, 1])

    def find_element_ND(self, X, _search_all=False):
        tree_query = self.tree.query(X, 5)[1]
        element_inds = np.empty((X.shape[0],), dtype=np.int)
        for n, point in enumerate(X):  # Try to avoid loop
            i_e = tree_query[n, :]
            X_loc = self.mapping.invF((point.T)[:, None, None], tind=i_e)
            inside = self.inside(X_loc)
            element_inds[n] = i_e[np.argmax(inside, axis=0)]
        return element_inds

    def inside_2D(self, X):  # Do something more general from Refdom?
        """
        Say which point are inside the element
        """
        return (X[0] >= -np.finfo(X.dtype).eps) * (X[1] >= -np.finfo(X.dtype).eps) * (1 - X[0] - X[1] >= -np.finfo(X.dtype).eps)

    def inside_3D(X):
        """
        Say which point are inside the element
        """
        return (X[0] >= 0) * (X[1] >= 0) * (X[2] >= 0) * (1 - X[0] - X[1] - X[2] >= -np.finfo(X.dtype).eps)


class FEMFeatures(TransformerMixin):
    """
    Finite elements features
    """

    def __init__(self, basis: Basis):
        # En vrai comme ça ne marche que pour les élements H1 je devrais juste construire la base localement à partir du mesh et de l'élément en vérifiant qu'il dérive bien de H1
        # Mais ça peut marcher aussi pour les éléments globaux, bref on a le droit qu'aux éléments scalaires pour l'instant
        """
        Parameters
        ----------
        """
        self.basis = basis
        self.const_removed = False

    def element_finder(self, x):
        # At first use, if not implement instancie the element finder
        if not hasattr(self, "element_finder_from_basis"):
            self.element_finder_from_basis = ElementFinder(self.basis.mesh, mapping=self.basis.mapping)  # self.basis.mesh.element_finder(mapping=self.basis.mapping)
        return self.element_finder_from_basis.find(x)

    def fit(self, describe_result):
        if isinstance(describe_result, np.ndarray):
            describe_result = minimal_describe(describe_result)
        self.element_finder_from_basis = ElementFinder(self.basis.mesh, mapping=self.basis.mapping)
        # Find tensorial order of the basis and adapt dimension in consequence
        test = self.basis.elem.gbasis(self.basis.mapping, self.basis.mapping.F(self.basis.mesh.p), 0)[0]
        if len(test.shape) == 3:  # Vectorial basis
            self.dim_out_basis = test.shape[0]
            raise NotImplementedError("Unsupported FEM for now")
        elif len(test.shape) == 2:  # Scalar basis
            self.dim_out_basis = 1
        self.n_output_features_ = self.basis.N
        return self

    def basis(self, X):
        nsamples, dim = X.shape
        x = X.T
        cells = self.basis.mesh.element_finder(mapping=self.basis.mapping)(*x)
        pts = self.basis.mapping.invF(x[:, :, np.newaxis], tind=cells)
        phis = np.array([self.basis.elem.gbasis(self.basis.mapping, pts, k, tind=cells)[0] for k in range(self.basis.Nbfun)]).flatten()  # TODO: vérifier la shape
        return sparse.COO(
            (np.tile(np.arange(nsamples), self.basis.Nbfun), self.basis.element_dofs[:, cells].flatten()),
            phis,
            shape=(nsamples, self.n_output_features_),
        )

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        x = X.T
        cells = self.basis.mesh.element_finder(mapping=self.basis.mapping)(*x)  # Il faudrait trouver un moyen de mettre en cache ce résultat
        pts = self.basis.mapping.invF(x[:, :, np.newaxis], tind=cells)
        phis = np.array([self.basis.elem.gbasis(self.basis.mapping, pts, k, tind=cells)[0].grad for k in range(self.basis.Nbfun)])  # TODO: vérifier la shape et en extraire les diverses dimensions
        return sparse.COO(
            (np.tile(np.arange(nsamples), self.basis.Nbfun), self.basis.element_dofs[:, cells].flatten()),
            phis,
            shape=(nsamples, self.n_output_features_, self.dim_x),
        )

    def hessian(self, X):
        raise NotImplementedError

    def antiderivative(self, X, order=1):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, self.n_output_features_))
        return features


if __name__ == "__main__":  # pragma: no cover
    import matplotlib.pyplot as plt

    x_range = np.linspace(-10, 10, 10).reshape(-1, 1)
    # basis = BSplineFeatures(6, k=3)
    basis = FEMFeatures([[-5, -2], [2, 3], [5, 7]], "tricube", periodic=True)
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
