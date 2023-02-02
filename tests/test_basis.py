import pytest
import numpy as np
from scipy.stats import describe
import VolterraBasis.basis as bf


"""
Pour les tests: on peut tester:
- On peut faire un ensemble de tests pour toutes les fonctions de bases (vérification de la shape de sortie)
"""


@pytest.mark.parametrize("basis,parameters,expected", [(bf.LinearFeatures, {}, 1), (bf.FourierFeatures, {"order": 3, "freq": 1.0, "remove_const": False}, 7)])
def test_shape_basis(basis, parameters, expected):
    n_points = 30
    x_range = np.linspace(-10, 10, n_points).reshape(-1, 1)
    basis = basis(**parameters)
    basis.fit(describe(x_range))
    assert basis.basis(x_range).shape == (n_points, expected)
    assert basis.deriv(x_range).shape == (n_points, expected, 1)
    assert basis.hessian(x_range).shape == (n_points, expected, 1, 1)


@pytest.mark.parametrize("basis,parameters,expected", [(bf.FourierFeatures, {"order": 3, "freq": 1.0}, 7)])
def test_shape_basis_remove_const(basis, parameters, expected):
    n_points = 30
    x_range = np.linspace(-10, 10, n_points).reshape(-1, 1)
    basis = basis(**parameters)
    basis.fit(describe(x_range))
    assert basis.basis(x_range).shape == (n_points, expected)
    assert basis.deriv(x_range).shape == (n_points, expected - 1, 1)
    assert basis.hessian(x_range).shape == (n_points, expected - 1, 1, 1)


@pytest.mark.parametrize("basis,parameters,expected", [(bf.LinearFeatures, {}, 2), (bf.FourierFeatures, {"order": 3, "freq": 1.0, "remove_const": False}, 14)])
def test_shape_basis_2D(basis, parameters, expected):
    x_range = np.linspace(-10, 10, 30).reshape(-1, 2)
    # b2 = LinearFeatures()
    basis = basis(**parameters)
    basis.fit(describe(x_range))
    assert basis.basis(x_range).shape == (15, expected)
    assert basis.deriv(x_range).shape == (15, expected, 2)
    assert basis.hessian(x_range).shape == (15, expected, 2, 2)


# def test_features_combiner():
