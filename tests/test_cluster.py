"""
Test suite module for ``Cluster``.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 01, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

import unittest
import warnings
import re
import sys

from inspect import getsourcefile
from os.path import abspath

import pandas as pd

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.cluster import Cluster  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

path = path + "/data/input/"

# =============================================================================
# --- User defined functions
# =============================================================================


def ignore_warnings(test_func):
    """Suppress deprecation warnings of pulp."""

    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class TestIntegrationCluster(unittest.TestCase):
    """Test suite for module ``metric``."""

    def setUp(self):
        """Set up for module ``metric``."""

    def test_categorical(self):
        """Cluster: Test for categorical variables."""
        df_ip = pd.read_csv(path + "store.csv")
        clus_sol = Cluster(df=df_ip, x_var=["x1"],
                           max_cluster=6,
                           nrefs=5)
        clus_sol.opt_k()
        self.assertEqual(clus_sol.optimal_k, 4)
        self.assertEqual(clus_sol.method, "gap_max")

    def test_categorical_multiple(self):
        """Cluster: Test for multiple categorical variables."""
        df_ip = pd.read_csv(path + "store.csv")
        clus_sol = Cluster(df=df_ip, x_var=["x1", "x8"],
                           max_cluster=6,
                           nrefs=5)
        clus_sol.opt_k()
        self.assertEqual(clus_sol.optimal_k, 4)
        self.assertEqual(clus_sol.method, "gap_max")

    def test_categorical_continuos(self):
        """Cluster: Test for categorical and continuos variables."""
        df_ip = pd.read_csv(path + "store.csv")
        clus_sol = Cluster(df=df_ip, x_var=["x1", "x3"],
                           max_cluster=6,
                           nrefs=5)
        clus_sol.opt_k()
        self.assertEqual(clus_sol.optimal_k, 5)
        self.assertEqual(clus_sol.method, "gap_max")

    def test_continuos_gap_max(self):
        """Cluster: Test for continuos variables gap_max."""
        df_ip = pd.read_csv(path + "store.csv")
        clus_sol = Cluster(df=df_ip, x_var=["x3", "x4"],
                           max_cluster=5,
                           nrefs=5,
                           method="gap_max")
        clus_sol.opt_k()
        self.assertEqual(clus_sol.optimal_k, 5)
        self.assertEqual(clus_sol.method, "gap_max")

    def test_continuos_one_se(self):
        """Cluster: Test for continuos variables one_se."""
        df_ip = pd.read_csv(path + "store.csv")
        clus_sol = Cluster(df=df_ip, x_var=["x3"],
                           max_cluster=10,
                           nrefs=10)
        clus_sol.opt_k()
        self.assertLessEqual(clus_sol.optimal_k, 10)
        self.assertEqual(clus_sol.method, "one_se")

    def test_gap_max_less_max_clus(self):
        """Cluster: Test for gap_max where optimal k < max_cluster."""
        df_ip = pd.read_csv(path + "store.csv")
        clus_sol = Cluster(df=df_ip, x_var=["x5"],
                           max_cluster=5,
                           nrefs=10,
                           method="gap_max")
        clus_sol.opt_k()
        self.assertLess(clus_sol.optimal_k, 5)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
