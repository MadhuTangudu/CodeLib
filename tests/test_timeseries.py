"""
Test suite module for ``knn``.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Sep 25, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

import unittest
import warnings
import re
import sys

from inspect import getsourcefile
from os.path import abspath

import numpy as np
import pandas as pd
import pytest

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.timeseries import FBProphet  # noqa: F841

import metrics  # noqa: F841


# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

path = path + "/data/input/"

# =============================================================================
# --- User defined functions
# =============================================================================


def ignore_warnings(test_func):
    """Suppress warnings."""

    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class Test_FBProphet(unittest.TestCase):
    """Test suite for module ``FBProphet``."""

    def setUp(self):
        """Set up for module ``FBProphet``."""

    def test_FBProphet_predict(self):
        """FBProphet: Test for predict."""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="product_01")
        df_ip["ds"] = pd.to_datetime(df_ip["ds"])
        df_ip.rename(columns={"y":"y_new", "ds":"ds_new"}, inplace=True)
        x_var = ["cost"]
        y_var = "y_new"
        df_train = df_ip.iloc[:-int(len(df_ip)*0.2), :]
        df_test = df_ip.iloc[:int(len(df_ip)*0.2), :]
        mod = FBProphet(df_train, x_var=x_var, epoch="ds_new", y_var=y_var)
        y_hat = mod.predict(df_test[x_var+["ds_new"]])[y_var].tolist()
        y = df_test[y_var].values.tolist()
        acc = round(np.round(metrics.rmse(y, y_hat), 3))
        self.assertLessEqual(acc, 5)
        

    def test_FBProphet_predict_period(self):
        """FBProphet: Test for predict when period is specified."""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="product_01")
        df_ip["ds"] = pd.to_datetime(df_ip["ds"])
        mod = FBProphet(df_ip)
        df_op = mod.predict(period=6)
        self.assertGreaterEqual(len(df_op), 6)

    def test_FBProphet_exception_error(self):
        """FBProphet: Test for Exception error.
            when x_var is not None and period > 0.
        """
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="product_01")
        df_ip["ds"] = pd.to_datetime(df_ip["ds"])
        mod = FBProphet(df_ip, x_var=["cost"])
        with pytest.raises(Exception):
            mod.predict(period=6)

    def test_FBProphet_param_tuning(self):
        """FBProphet: Test for parameter tuning.
        """
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="product_01")
        df_ip["ds"] = pd.to_datetime(df_ip["ds"])
        with pytest.raises(Exception):
            FBProphet(df_ip, k_fold=5)
        with pytest.raises(Exception):
            FBProphet(df_ip, param={"changepoint_prior_scale" : [0.001, 0.01],
                                    "seasonality_prior_scale" : [0.01, 0.1]})


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
