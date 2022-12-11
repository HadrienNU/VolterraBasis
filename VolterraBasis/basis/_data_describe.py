from collections import namedtuple
import numpy as np
import json

DescribeResult = namedtuple("DescribeResult", ("nobs", "minmax", "mean", "variance", "skewness", "kurtosis"))


def quick_describe(X):
    """
    Simply return the dimension of the data
    """
    nobs, dim = X.shape
    return DescribeResult(nobs, (np.zeros(dim), np.zeros(dim)), np.zeros(dim), np.zeros(dim), np.zeros(dim), np.zeros(dim))


def minimal_describe(X):
    """
    Simply return the dimension of the data
    """
    nobs, dim = X.shape
    return DescribeResult(nobs, (np.min(X, axis=0), np.max(X, axis=0)), np.mean(X, axis=0), np.zeros(dim), np.zeros(dim), np.zeros(dim))
