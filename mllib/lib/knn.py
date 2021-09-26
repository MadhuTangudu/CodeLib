"""
k-NN module.

**Available routines:**

- class ``KNN``: Builds K-Nearest Neighnour model using cross validation.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Sep 25, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=R0902,R0903,R0913,C0413

from typing import List, Dict, Any

import re
import sys
from inspect import getsourcefile
from os.path import abspath

import pandas as pd
import numpy as np

from sklearn import neighbors as sn
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics as sk_metrics

from sklearn.model_selection import GridSearchCV

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841


class KNN():
    """K-Nearest Neighbour (KNN) module.

    Objective:
        - Build
          `KNN <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_
          model and determine optimal k

    Parameters
    ----------
    df : pandas.DataFrame

        Pandas dataframe containing the `y_var` and `x_var`

    y_var : str

        Dependant variable

    x_var : List[str]

        Independant variables.

    method : str, optional

        Can be either `classify` or `regression` (the default is classify)

    k_fold : int, optional

        Number of cross validations folds (the default is 5)

    param : dict, optional

        KNN parameters (the default is None).
        In case of None, the parameters will default to::

            n_neighbors: max(int(len(df)/(k_fold * 2)), 1)
            weights: ["uniform", "distance"]
            metric: ["euclidean", "manhattan"]

    Returns
    -------
    model : object

        Final optimal model.

    Methods
    -------
    predict

    Example
    -------
    >>> mod = KNN(df=df_ip, y_var=["y"], x_var=["x1", "x2", "x3"])
    >>> df_op = mod.predict(df_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str],
                 method: str = "classify",
                 k_fold: int = 5,
                 param: Dict = None):
        """Initialize variables for module ``KNN``."""
        self.y_var = y_var
        self.x_var = x_var
        self.df = df.reset_index(drop=True)
        self.method = method
        self.model = None
        self.k_fold = k_fold
        if param is None:
            max_k = max(int(len(self.df)/(self.k_fold * 2)), 1)
            param = {"n_neighbors": list(range(1, max_k, 2)),
                     "weights": ["uniform", "distance"],
                     "metric": ["euclidean", "manhattan"]}
        self.param = param
        self._pre_process()
        self.best_params_ = self._fit()
        self.model_summary = None
        self._compute_metrics()

    def _pre_process(self):
        """Pre-process the data, one hot encoding and Normalizing."""
        df_ip_x = pd.get_dummies(self.df[self.x_var])
        self.x_var = list(df_ip_x.columns)
        self.norm = MinMaxScaler()
        self.norm.fit(df_ip_x)
        df_ip_x = pd.DataFrame(self.norm.transform(df_ip_x[self.x_var]))
        df_ip_x.columns = self.x_var
        self.df = self.df[[self.y_var]].join(df_ip_x)

    def _fit(self) -> Dict[str, Any]:
        """Fit KNN model."""
        if self.method == "classify":
            gs = GridSearchCV(estimator=sn.KNeighborsClassifier(),
                              param_grid=self.param,
                              scoring='accuracy',
                              verbose=0,
                              cv=self.k_fold,
                              n_jobs=-1)
        elif self.method == "regression":
            gs = GridSearchCV(estimator=sn.KNeighborsRegressor(),
                              param_grid=self.param,
                              scoring='neg_root_mean_squared_error',
                              verbose=0,
                              cv=self.k_fold,
                              n_jobs=-1)
        gs_op = gs.fit(self.df[self.x_var],
                       self.df[self.y_var])
        opt_k = gs_op.best_params_.get("n_neighbors")
        weight = gs_op.best_params_.get("weights")
        metric = gs_op.best_params_.get("metric")
        if self.method == "classify":
            model = sn.KNeighborsClassifier(n_neighbors=opt_k,
                                            weights=weight,
                                            metric=metric)
        elif self.method == "regression":
            model = sn.KNeighborsRegressor(n_neighbors=opt_k,
                                           weights=weight,
                                           metric=metric)
        self.model = model.fit(self.df[self.x_var],
                               self.df[self.y_var])
        return gs_op.best_params_

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        y = self.df.iloc[:, 0].values.tolist()
        y_hat = list(self.predict(self.df[self.x_var])["y"].values)
        if self.method == "regression":
            model_summary = {"rsq": np.round(metrics.rsq(y, y_hat), 3),
                             "mae": np.round(metrics.mae(y, y_hat), 3),
                             "mape": np.round(metrics.mape(y, y_hat), 3),
                             "rmse": np.round(metrics.rmse(y, y_hat), 3)}
            model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        if self.method == "classify":
            accuracy = np.round(sk_metrics.accuracy_score(y, y_hat), 3)
            f1_score = np.round(sk_metrics.f1_score(y, y_hat,
                                                    average='micro'), 3)
            model_summary = {"accuracy": accuracy,
                             "f1": f1_score}
        self.model_summary = model_summary

    def predict(self, df_predict: pd.DataFrame) -> pd.DataFrame:
        """Predict y_var/target variable.

        Parameters
        ----------
        df_predict : pd.DataFrame

            Pandas dataframe containing `x_var`.

        Returns
        -------
        pd.DataFrame

            Pandas dataframe containing predicted `y_var` and `x_var`.

        """
        df_predict = pd.get_dummies(df_predict)
        df_predict_tmp = pd.DataFrame(columns=self.x_var)
        df_predict = pd.concat([df_predict_tmp, df_predict])
        df_predict = df_predict.fillna(0)
        df_predict = pd.DataFrame(self.norm.transform(df_predict[self.x_var]))
        df_predict.columns = self.x_var
        y_hat = self.model.predict(df_predict)
        df_predict = df_predict.copy()
        df_predict["y"] = y_hat
        df_predict = df_predict[[self.y_var] + self.x_var]
        return df_predict
