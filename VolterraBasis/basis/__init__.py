from ._data_describe import DescribeResult, quick_describe, minimal_describe, describe_from_dim, sum_describe
from ._basis_features import LinearFeatures, PolynomialFeatures, FourierFeatures, SplineFctFeatures, FeaturesCombiner
from ._local_features import BSplineFeatures, SmoothIndicatorFeatures
from ._multidim_basis import TensorialBasis2D

try:  # That make depencies on skfem and sparse optionnal
    from ._fem_features import FEMScalarFeatures
except ImportError:
    FEMScalarFeatures = None
    pass
try:
    from ._mesh_utils import *
except ImportError:
    pass


__all__ = ["LinearFeatures", "PolynomialFeatures", "FourierFeatures", "SplineFctFeatures", "FeaturesCombiner", "BSplineFeatures", "SmoothIndicatorFeatures", "TensorialBasis2D", "FEMScalarFeatures"]
