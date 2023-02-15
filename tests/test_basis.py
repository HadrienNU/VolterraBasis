import pytest
import numpy as np
from scipy.stats import describe
import VolterraBasis.basis as bf


@pytest.mark.parametrize(
    "basis,parameters,expected",
    [
        (bf.LinearFeatures, {}, 1),
        (bf.PolynomialFeatures, {"deg": 3, "remove_const": False}, 4),
        (bf.FourierFeatures, {"order": 3, "freq": 1.0, "remove_const": False}, 7),
        (bf.SplineFctFeatures, {"knots": np.linspace(-1, 1, 8), "coeffs": np.logspace(1, 2, 8), "k": 3}, 1),
        (bf.BSplineFeatures, {"n_knots": 8, "k": 3, "remove_const": False}, 8),
        (bf.SmoothIndicatorFeatures, {"states_boundary": [[1.0, 1.5], [2.0, 3.0], [3.7, 4.0]], "periodic": False}, 4),
        (bf.SmoothIndicatorFeatures, {"states_boundary": [[1.0, 1.5], [2.0, 3.0], [3.7, 4.0]], "boundary_type": "quartic", "periodic": True}, 3),
        (bf.SmoothIndicatorFeatures, {"states_boundary": [[1.0, 1.5], [2.0, 3.0]], "boundary_type": "triweight", "periodic": True}, 2),
    ],
)
def test_shape_basis(basis, parameters, expected):
    n_points = 30
    x_range = np.linspace(-10, 10, n_points).reshape(-1, 1)
    basis = basis(**parameters)
    basis.fit(x_range)
    basis.fit(describe(x_range))
    assert basis.basis(x_range).shape == (n_points, expected)
    assert basis.deriv(x_range).shape == (n_points, expected, 1)
    assert basis.hessian(x_range).shape == (n_points, expected, 1, 1)


@pytest.mark.parametrize(
    "basis,parameters,expected",
    [
        (bf.PolynomialFeatures, {"deg": 3, "polynom": np.polynomial.Chebyshev}, 4),
        (bf.FourierFeatures, {"order": 3, "freq": 1.0}, 7),
        (bf.BSplineFeatures, {"n_knots": 8, "k": 3}, 8),
    ],
)
def test_shape_basis_remove_const(basis, parameters, expected):
    n_points = 30
    x_range = np.linspace(-10, 10, n_points).reshape(-1, 1)
    basis = basis(**parameters)
    basis.fit(describe(x_range))
    assert basis.basis(x_range).shape == (n_points, expected)
    assert basis.deriv(x_range).shape == (n_points, expected - 1, 1)
    assert basis.hessian(x_range).shape == (n_points, expected - 1, 1, 1)


@pytest.mark.parametrize(
    "basis,parameters,expected",
    [
        (bf.LinearFeatures, {}, 1),
        (bf.PolynomialFeatures, {"deg": 3, "remove_const": False}, 4),
        (bf.FourierFeatures, {"order": 3, "freq": 1.0, "remove_const": False}, 7),
        (bf.SplineFctFeatures, {"knots": np.linspace(-1, 1, 8), "coeffs": np.logspace(1, 2, 8), "k": 3}, 1),
        (bf.BSplineFeatures, {"n_knots": 8, "k": 3, "remove_const": False}, 8),
    ],
)
def test_antiderivative(basis, parameters, expected):
    n_points = 30
    x_range = np.linspace(-10, 10, n_points).reshape(-1, 1)
    basis = basis(**parameters)
    basis.fit(x_range)
    assert basis.antiderivative(x_range).shape == (n_points, expected)


@pytest.mark.parametrize(
    "basis,parameters,expected",
    [
        (bf.LinearFeatures, {}, 2),
        (bf.PolynomialFeatures, {"deg": 3, "polynom": np.polynomial.Chebyshev, "remove_const": False}, 8),
        (bf.FourierFeatures, {"order": 3, "freq": 1.0, "remove_const": False}, 14),
        # (bf.BSplineFeatures, {"n_knots": 8, "k": 3, "remove_const": False}, 16),
        (bf.TensorialBasis2D, {"b1": bf.BSplineFeatures(5), "b2": bf.PolynomialFeatures(3)}, 20),
    ],
)
def test_shape_basis_2D(basis, parameters, expected):
    x_range = np.linspace(-10, 10, 30).reshape(-1, 2)
    # b2 = LinearFeatures()
    basis = basis(**parameters)
    basis.fit(describe(x_range))
    assert basis.basis(x_range).shape == (15, expected)
    assert basis.deriv(x_range).shape == (15, expected, 2)
    assert basis.hessian(x_range).shape == (15, expected, 2, 2)


def test_features_combiner():
    n_points = 30
    x_range = np.linspace(-10, 10, n_points).reshape(-1, 1)
    b1 = bf.FourierFeatures(order=3, freq=1.0)
    b2 = bf.SplineFctFeatures(knots=np.linspace(-1, 1, 8), coeffs=np.logspace(1, 2, 8), k=2)
    basis = bf.FeaturesCombiner(b1, b2)
    basis.fit(describe(x_range))
    assert basis.basis(x_range).shape == (n_points, 8)
    assert basis.deriv(x_range).shape == (n_points, 7, 1)
    assert basis.hessian(x_range).shape == (n_points, 7, 1, 1)


def test_fem_2d():
    import os
    import skfem

    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_lj.trj"))

    vertices, tri = bf.centroid_driven_mesh(trj[:, 1:3], bins=25)

    m = skfem.MeshTri(vertices.T, tri.T)
    e = skfem.ElementTriP1()  # skfem.ElementTriRT0()  #
    basis_fem = skfem.CellBasis(m, e)
    pts = trj[:150, 1:3]
    n_points = pts.shape[0]
    basis = bf.FEMScalarFeatures(basis_fem)
    basis.fit(pts)
    assert basis.basis(pts).shape == (n_points, basis_fem.N)
    assert basis.deriv(pts).shape == (n_points, basis_fem.N, 2)
