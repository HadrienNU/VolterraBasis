import numpy as np


class PronyFit(object):
    """
    A class to obtain prony series from time-dependent series
    """

    def __init__(self, **kwargs):
        """
        Create an instance of the PronyFit class.

        Parameters
        ----------
        xva_arg : xarray dataset (['time', 'x', 'v', 'a']) or list of datasets.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a xarray timeseries
            or a listlike collection of them.
        """
        pass
        # Ici on doit juste stocker les différents paramètres

    def fit_kernel(self, time_series):
        pass

    def series_eval(self, times):
        """
        Get series evaluated at points t
        You can then use pos_gle.kernel_eval(x, prony_eval) to get kernel at those points
        """
        #
        return
